from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

# Make src/ importable without requiring an editable install.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _clear_archai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env / shell exports from leaking into tests."""
    for key in list(os_environ_keys()):
        if key.startswith(("ARCHAI_OCR_", "ARCHAI_")):
            monkeypatch.delenv(key, raising=False)


def os_environ_keys() -> list[str]:
    import os

    return list(os.environ)


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    """A directory with plausible (empty) weight files so validation passes."""
    d = tmp_path / "weights"
    d.mkdir()
    for name in ("layout_yolo.pt", "kraken_recognition.mlmodel", "kraken_segmentation.mlmodel"):
        (d / name).write_bytes(b"")
    return d


@pytest.fixture
def config_file(tmp_path: Path, weights_dir: Path) -> Path:
    """A minimal valid config written next to the fake weights."""
    payload = {
        "weights": {
            "layout_yolo": "weights/layout_yolo.pt",
            "kraken_recognition": "weights/kraken_recognition.mlmodel",
            "kraken_segmentation": "weights/kraken_segmentation.mlmodel",
        },
        "layout": {"main_text_class": "main_text"},
        "runtime": {"output_dir": "outputs"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.fixture
def page_image(tmp_path: Path) -> Path:
    path = tmp_path / "page_001.png"
    Image.new("RGB", (1000, 1400), "white").save(path)
    return path
