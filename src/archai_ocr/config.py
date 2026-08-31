from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

VALID_READING_ORDERS = ("column", "simple")
VALID_DEVICES = ("cpu", "cuda", "mps")

DEFAULT_CONFIG: dict[str, Any] = {
    "weights": {
        "layout_yolo": "weights/layout_yolo.pt",
        "kraken_segmentation": "weights/kraken_segmentation.mlmodel",
        "kraken_recognition": "weights/kraken_recognition.mlmodel",
    },
    "layout": {
        # Must match the layout model's own class vocabulary. The shipped
        # weights use SegmOnto zone names; "MainZone" is the body text.
        # Accepts a single name or a list (e.g. to include MarginTextZone).
        "main_text_class": "MainZone",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "max_regions": 50,
        "crop_padding": 5,
        # "column": cluster regions into columns (left->right), then top->bottom
        #           within each column. Correct for multi-column manuscript pages.
        # "simple": raw top->bottom, then left->right. Reproduces pre-0.2 output.
        "reading_order": "column",
        # Two regions belong to the same column when their horizontal overlap
        # is at least this fraction of the narrower region's width.
        "column_overlap_ratio": 0.5,
        # Crops smaller than this in either dimension are dropped before HTR;
        # degenerate boxes otherwise crash or silently poison recognition.
        "min_region_size": 8,
    },
    "runtime": {
        "output_dir": "outputs",
        "kraken_device": "cpu",
        "write_page_xml": False,
    },
}


@dataclass(frozen=True)
class WeightConfig:
    layout_yolo: Path
    kraken_segmentation: Path
    kraken_recognition: Path


@dataclass(frozen=True)
class LayoutConfig:
    main_text_classes: tuple[str, ...]
    confidence_threshold: float
    iou_threshold: float
    max_regions: int
    crop_padding: int
    reading_order: str
    column_overlap_ratio: float
    min_region_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: Path
    kraken_device: str
    write_page_xml: bool


@dataclass(frozen=True)
class AppConfig:
    weights: WeightConfig
    layout: LayoutConfig
    runtime: RuntimeConfig


class ConfigError(ValueError):
    """Raised when a config value is present but not usable."""


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load config with precedence: defaults < YAML file < environment < explicit overrides."""
    load_dotenv()
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Config file {config_path} must contain a YAML mapping at the top level, "
            f"got {type(loaded).__name__}."
        )

    _reject_unknown_keys(loaded, config_path)

    merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), loaded)
    _apply_env_overrides(merged)
    if overrides:
        _apply_overrides(merged, overrides)

    return _build_app_config(merged, base_dir=config_path.parent)


def validate_required_weights(config: AppConfig) -> None:
    required_paths = [
        ("layout_yolo", config.weights.layout_yolo),
        ("kraken_recognition", config.weights.kraken_recognition),
    ]
    missing = [f"{name}: {path}" for name, path in required_paths if not path.exists()]
    if missing:
        detail = "; ".join(missing)
        raise FileNotFoundError(
            "Missing required model weight(s). Please place them in weights/ or update config. "
            f"Details: {detail}"
        )


def _reject_unknown_keys(loaded: dict[str, Any], config_path: Path) -> None:
    """Fail loudly on typo'd keys instead of silently ignoring them."""
    unknown_sections = set(loaded) - set(DEFAULT_CONFIG)
    if unknown_sections:
        raise ConfigError(
            f"Unknown config section(s) in {config_path}: {sorted(unknown_sections)}. "
            f"Valid sections: {sorted(DEFAULT_CONFIG)}."
        )
    for section, values in loaded.items():
        if not isinstance(values, dict):
            raise ConfigError(f"Config section '{section}' in {config_path} must be a mapping.")
        unknown_keys = set(values) - set(DEFAULT_CONFIG[section])
        if unknown_keys:
            raise ConfigError(
                f"Unknown key(s) in config section '{section}' of {config_path}: "
                f"{sorted(unknown_keys)}. Valid keys: {sorted(DEFAULT_CONFIG[section])}."
            )


def _build_app_config(raw: dict[str, Any], base_dir: Path) -> AppConfig:
    weights = raw.get("weights", {})
    layout = raw.get("layout", {})
    runtime = raw.get("runtime", {})

    def layout_default(key: str) -> Any:
        return DEFAULT_CONFIG["layout"][key]

    reading_order = str(layout.get("reading_order", layout_default("reading_order"))).strip().lower()
    if reading_order not in VALID_READING_ORDERS:
        raise ConfigError(
            f"layout.reading_order must be one of {list(VALID_READING_ORDERS)}, got {reading_order!r}."
        )

    device = str(runtime.get("kraken_device", DEFAULT_CONFIG["runtime"]["kraken_device"])).strip().lower()
    if device not in VALID_DEVICES and not device.startswith("cuda:"):
        raise ConfigError(
            f"runtime.kraken_device must be one of {list(VALID_DEVICES)} (or 'cuda:N'), got {device!r}."
        )

    confidence = _bounded_float(layout, "confidence_threshold", 0.0, 1.0)
    iou = _bounded_float(layout, "iou_threshold", 0.0, 1.0)
    overlap = _bounded_float(layout, "column_overlap_ratio", 0.0, 1.0)
    max_regions = _positive_int(layout, "max_regions")
    min_region_size = _positive_int(layout, "min_region_size")
    crop_padding = _non_negative_int(layout, "crop_padding")

    return AppConfig(
        weights=WeightConfig(
            layout_yolo=_resolve_path(
                weights.get("layout_yolo", DEFAULT_CONFIG["weights"]["layout_yolo"]), base_dir
            ),
            kraken_segmentation=_resolve_path(
                weights.get("kraken_segmentation", DEFAULT_CONFIG["weights"]["kraken_segmentation"]),
                base_dir,
            ),
            kraken_recognition=_resolve_path(
                weights.get("kraken_recognition", DEFAULT_CONFIG["weights"]["kraken_recognition"]),
                base_dir,
            ),
        ),
        layout=LayoutConfig(
            main_text_classes=_normalize_classes(
                layout.get("main_text_class", layout_default("main_text_class"))
            ),
            confidence_threshold=confidence,
            iou_threshold=iou,
            max_regions=max_regions,
            crop_padding=crop_padding,
            reading_order=reading_order,
            column_overlap_ratio=overlap,
            min_region_size=min_region_size,
        ),
        runtime=RuntimeConfig(
            output_dir=_resolve_path(
                runtime.get("output_dir", DEFAULT_CONFIG["runtime"]["output_dir"]), base_dir
            ),
            kraken_device=device,
            write_page_xml=_as_bool(
                runtime.get("write_page_xml", DEFAULT_CONFIG["runtime"]["write_page_xml"])
            ),
        ),
    )


def _normalize_classes(value: Any) -> tuple[str, ...]:
    """Accept either a single class name or a list of them."""
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
    else:
        raise ConfigError(f"layout.main_text_class must be a string or a list of strings, got {value!r}.")
    cleaned = tuple(name.strip() for name in names if str(name).strip())
    if not cleaned:
        raise ConfigError("layout.main_text_class must name at least one layout class.")
    return cleaned


def _bounded_float(section: dict[str, Any], key: str, low: float, high: float) -> float:
    raw = section.get(key, DEFAULT_CONFIG["layout"][key])
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"layout.{key} must be a number, got {raw!r}.") from exc
    if not low <= value <= high:
        raise ConfigError(f"layout.{key} must be between {low} and {high}, got {value}.")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = _coerce_int(section, key)
    if value <= 0:
        raise ConfigError(f"layout.{key} must be greater than 0, got {value}.")
    return value


def _non_negative_int(section: dict[str, Any], key: str) -> int:
    value = _coerce_int(section, key)
    if value < 0:
        raise ConfigError(f"layout.{key} must be 0 or greater, got {value}.")
    return value


def _coerce_int(section: dict[str, Any], key: str) -> int:
    raw = section.get(key, DEFAULT_CONFIG["layout"][key])
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"layout.{key} must be an integer, got {raw!r}.") from exc


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in overrides.items():
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Canonical env var name -> (section, key, caster). Aliases below keep the names
# that shipped in .env.example working; previously they were silently ignored.
_ENV_MAP: dict[str, tuple[str, str, Callable[[str], Any]]] = {
    "ARCHAI_OCR_LAYOUT_YOLO": ("weights", "layout_yolo", str),
    "ARCHAI_OCR_KRAKEN_SEGMENTATION": ("weights", "kraken_segmentation", str),
    "ARCHAI_OCR_KRAKEN_RECOGNITION": ("weights", "kraken_recognition", str),
    "ARCHAI_OCR_MAIN_TEXT_CLASS": ("layout", "main_text_class", str),
    "ARCHAI_OCR_CONFIDENCE_THRESHOLD": ("layout", "confidence_threshold", float),
    "ARCHAI_OCR_IOU_THRESHOLD": ("layout", "iou_threshold", float),
    "ARCHAI_OCR_MAX_REGIONS": ("layout", "max_regions", int),
    "ARCHAI_OCR_CROP_PADDING": ("layout", "crop_padding", int),
    "ARCHAI_OCR_READING_ORDER": ("layout", "reading_order", str),
    "ARCHAI_OCR_COLUMN_OVERLAP_RATIO": ("layout", "column_overlap_ratio", float),
    "ARCHAI_OCR_MIN_REGION_SIZE": ("layout", "min_region_size", int),
    "ARCHAI_OCR_OUTPUT_DIR": ("runtime", "output_dir", str),
    "ARCHAI_OCR_KRAKEN_DEVICE": ("runtime", "kraken_device", str),
    "ARCHAI_OCR_WRITE_PAGE_XML": ("runtime", "write_page_xml", _parse_bool),
}

# Names documented in .env.example that the loader never actually read.
_ENV_ALIASES: dict[str, str] = {
    "ARCHAI_LAYOUT_YOLO_WEIGHTS": "ARCHAI_OCR_LAYOUT_YOLO",
    "ARCHAI_KRAKEN_SEG_WEIGHTS": "ARCHAI_OCR_KRAKEN_SEGMENTATION",
    "ARCHAI_KRAKEN_REC_WEIGHTS": "ARCHAI_OCR_KRAKEN_RECOGNITION",
    "ARCHAI_OUTPUT_DIR": "ARCHAI_OCR_OUTPUT_DIR",
}


def _apply_env_overrides(config: dict[str, Any]) -> None:
    for env_name, (section, key, caster) in _ENV_MAP.items():
        raw = os.getenv(env_name)
        if raw is None:
            raw = _lookup_alias(env_name)
        if raw is None:
            continue
        try:
            config.setdefault(section, {})[key] = caster(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Environment variable {env_name}={raw!r} is not a valid {section}.{key}."
            ) from exc


def _lookup_alias(canonical: str) -> str | None:
    for alias, target in _ENV_ALIASES.items():
        if target == canonical:
            value = os.getenv(alias)
            if value is not None:
                return value
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_bool(value)
    return bool(value)
