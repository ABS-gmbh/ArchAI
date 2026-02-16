from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import copy
import os

import yaml
from dotenv import load_dotenv


DEFAULT_CONFIG: dict[str, Any] = {
    "weights": {
        "layout_yolo": "weights/layout_yolo.pt",
        "kraken_segmentation": "weights/kraken_segmentation.mlmodel",
        "kraken_recognition": "weights/kraken_recognition.mlmodel",
    },
    "layout": {
        "main_text_class": "main_text",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.5,
        "max_regions": 50,
        "crop_padding": 5,
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
    main_text_class: str
    confidence_threshold: float
    iou_threshold: float
    max_regions: int
    crop_padding: int


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


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    load_dotenv()
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), loaded)
    _apply_env_overrides(merged)
    if overrides:
        _apply_overrides(merged, overrides)

    base_dir = config_path.parent
    return _build_app_config(merged, base_dir=base_dir)


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


def _build_app_config(raw: dict[str, Any], base_dir: Path) -> AppConfig:
    weights = raw.get("weights", {})
    layout = raw.get("layout", {})
    runtime = raw.get("runtime", {})

    return AppConfig(
        weights=WeightConfig(
            layout_yolo=_resolve_path(weights.get("layout_yolo", DEFAULT_CONFIG["weights"]["layout_yolo"]), base_dir),
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
            main_text_class=str(layout.get("main_text_class", DEFAULT_CONFIG["layout"]["main_text_class"])),
            confidence_threshold=float(
                layout.get("confidence_threshold", DEFAULT_CONFIG["layout"]["confidence_threshold"])
            ),
            iou_threshold=float(layout.get("iou_threshold", DEFAULT_CONFIG["layout"]["iou_threshold"])),
            max_regions=int(layout.get("max_regions", DEFAULT_CONFIG["layout"]["max_regions"])),
            crop_padding=int(layout.get("crop_padding", DEFAULT_CONFIG["layout"]["crop_padding"])),
        ),
        runtime=RuntimeConfig(
            output_dir=_resolve_path(runtime.get("output_dir", DEFAULT_CONFIG["runtime"]["output_dir"]), base_dir),
            kraken_device=str(runtime.get("kraken_device", DEFAULT_CONFIG["runtime"]["kraken_device"])),
            write_page_xml=_as_bool(runtime.get("write_page_xml", DEFAULT_CONFIG["runtime"]["write_page_xml"])),
        ),
    )


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


def _apply_env_overrides(config: dict[str, Any]) -> None:
    env_map: dict[str, tuple[str, str, Any]] = {
        "ARCHAI_OCR_LAYOUT_YOLO": ("weights", "layout_yolo", str),
        "ARCHAI_OCR_KRAKEN_SEGMENTATION": ("weights", "kraken_segmentation", str),
        "ARCHAI_OCR_KRAKEN_RECOGNITION": ("weights", "kraken_recognition", str),
        "ARCHAI_OCR_MAIN_TEXT_CLASS": ("layout", "main_text_class", str),
        "ARCHAI_OCR_CONFIDENCE_THRESHOLD": ("layout", "confidence_threshold", float),
        "ARCHAI_OCR_IOU_THRESHOLD": ("layout", "iou_threshold", float),
        "ARCHAI_OCR_MAX_REGIONS": ("layout", "max_regions", int),
        "ARCHAI_OCR_CROP_PADDING": ("layout", "crop_padding", int),
        "ARCHAI_OCR_OUTPUT_DIR": ("runtime", "output_dir", str),
        "ARCHAI_OCR_KRAKEN_DEVICE": ("runtime", "kraken_device", str),
        "ARCHAI_OCR_WRITE_PAGE_XML": ("runtime", "write_page_xml", _parse_bool),
    }

    for env_name, (section, key, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        config.setdefault(section, {})
        config[section][key] = caster(raw)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _parse_bool(value)
    return bool(value)
