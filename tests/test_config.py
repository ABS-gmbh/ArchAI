from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from archai_ocr.config import ConfigError, load_config, validate_required_weights


def test_defaults_applied_for_absent_keys(config_file: Path) -> None:
    config = load_config(config_file)
    assert config.layout.confidence_threshold == 0.25
    assert config.layout.max_regions == 50
    assert config.runtime.kraken_device == "cpu"


def test_reading_order_defaults_to_column(config_file: Path) -> None:
    assert load_config(config_file).layout.reading_order == "column"


def test_relative_paths_resolve_against_the_config_file(config_file: Path, weights_dir: Path) -> None:
    config = load_config(config_file)
    assert config.weights.layout_yolo == (weights_dir / "layout_yolo.pt").resolve()
    assert config.runtime.output_dir == (config_file.parent / "outputs").resolve()


def test_explicit_overrides_beat_file_and_env(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_OCR_MAIN_TEXT_CLASS", "from_env")
    config = load_config(config_file, overrides={"layout.main_text_class": "from_cli"})
    assert config.layout.main_text_classes == ("from_cli",)


def test_env_beats_config_file(config_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCHAI_OCR_MAIN_TEXT_CLASS", "from_env")
    assert load_config(config_file).layout.main_text_classes == ("from_env",)


@pytest.mark.parametrize(
    ("alias", "canonical_attr"),
    [
        ("ARCHAI_OUTPUT_DIR", "output_dir"),
    ],
)
def test_documented_env_aliases_are_honoured(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, alias: str, canonical_attr: str
) -> None:
    """Regression: these names ship in .env.example but were previously ignored."""
    target = tmp_path / "aliased-out"
    monkeypatch.setenv(alias, str(target))
    config = load_config(config_file)
    assert getattr(config.runtime, canonical_attr) == target.resolve()


def test_weights_env_alias_is_honoured(
    config_file: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "other.pt"
    target.write_bytes(b"")
    monkeypatch.setenv("ARCHAI_LAYOUT_YOLO_WEIGHTS", str(target))
    assert load_config(config_file).weights.layout_yolo == target.resolve()


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"typo_section": {"a": 1}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown config section"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"layout": {"confidence_treshold": 0.4}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("layout", "confidence_threshold", 1.5, "between 0.0 and 1.0"),
        ("layout", "iou_threshold", -0.1, "between 0.0 and 1.0"),
        ("layout", "max_regions", 0, "greater than 0"),
        ("layout", "crop_padding", -1, "0 or greater"),
        ("layout", "reading_order", "sideways", "reading_order must be one of"),
        ("runtime", "kraken_device", "quantum", "kraken_device must be one of"),
    ],
)
def test_invalid_values_are_rejected(
    tmp_path: Path, section: str, key: str, value: object, match: str
) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({section: {key: value}}), encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_cuda_device_with_index_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"runtime": {"kraken_device": "cuda:1"}}), encoding="utf-8")
    assert load_config(path).runtime.kraken_device == "cuda:1"


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(path)


def test_validate_required_weights_names_what_is_missing(config_file: Path, weights_dir: Path) -> None:
    (weights_dir / "layout_yolo.pt").unlink()
    with pytest.raises(FileNotFoundError, match="layout_yolo"):
        validate_required_weights(load_config(config_file))


def test_segmentation_weights_are_optional(config_file: Path, weights_dir: Path) -> None:
    (weights_dir / "kraken_segmentation.mlmodel").unlink()
    validate_required_weights(load_config(config_file))


def test_default_main_text_class_matches_the_shipped_model_vocabulary(tmp_path: Path) -> None:
    """Regression: the old default 'main_text' matches no class the shipped
    layout weights emit (they use SegmOnto zones), so every run silently wrote
    an empty transcription and exited 0."""
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({}), encoding="utf-8")
    assert load_config(path).layout.main_text_classes == ("MainZone",)


def test_main_text_class_accepts_a_list(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({"layout": {"main_text_class": ["MainZone", "MarginTextZone"]}}),
        encoding="utf-8",
    )
    assert load_config(path).layout.main_text_classes == ("MainZone", "MarginTextZone")


def test_main_text_class_rejects_an_empty_list(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"layout": {"main_text_class": []}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="at least one layout class"):
        load_config(path)


def test_main_text_class_rejects_a_non_string(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"layout": {"main_text_class": 5}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="string or a list of strings"):
        load_config(path)
