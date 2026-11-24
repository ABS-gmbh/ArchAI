"""Which model files and packages the reading pipeline lacks.

Without the layout weights, /api/predict answers 500 and the segmented OCR
engine cannot run, so pages are read with GLM-OCR alone (if Ollama serves it).
Without the CATMuS Medieval recogniser or the kraken package, the segmented
engine has nothing to read lines with. None of this stops the server, which
still serves chat, retrieval and GLM-OCR: startup logs a warning
(log_missing_models), and /api/health reports two optional checks
(app/routers/health.py) that the workspace's banner shows.

missing_layout() and missing_recogniser() give short names (file names,
settings, packages) safe to return to an unauthenticated caller; the startup
log adds where the server looked.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from app.config import settings

_LOGGER = logging.getLogger(__name__)


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def _layout_paths() -> list[str]:
    return [settings.emanuskript_model_path, settings.catmus_model_path, settings.zone_model_path]


def _recogniser_path() -> Path:
    from app.services.ocr_backends import _resolve_model_path

    return _resolve_model_path(settings.kraken_catmus_model_path)


def missing_layout() -> list[str]:
    """What layout segmentation lacks: weight file names (in MODEL_DIR) and the ultralytics package."""
    missing = [os.path.basename(path) for path in _layout_paths() if not os.path.isfile(path)]
    if not _installed("ultralytics"):
        missing.append("the ultralytics package")
    return missing


def missing_recogniser() -> list[str]:
    """What line recognition lacks: the CATMuS Medieval recogniser's file name and the kraken package."""
    missing = []
    if not _recogniser_path().is_file():
        missing.append(os.path.basename(settings.kraken_catmus_model_path))
    if not _installed("kraken"):
        missing.append("the kraken package")
    return missing


def missing_models() -> list[str]:
    """Everything missing, each with where the server looked or the setting that supplies it (for the log)."""
    missing: list[str] = []
    layout = [path for path in _layout_paths() if not os.path.isfile(path)]
    if layout:
        names = ", ".join(os.path.basename(path) for path in layout)
        missing.append(f"layout weights {names} in {os.path.dirname(layout[0])!r} (MODEL_DIR)")
    if not _recogniser_path().is_file():
        missing.append(
            f"the CATMuS Medieval recogniser {settings.kraken_catmus_model_path!r} "
            "(KRAKEN_CATMUS_MODEL_PATH; scripts/fetch_kraken_models.sh downloads it)"
        )
    for package, purpose in (("ultralytics", "layout segmentation"), ("kraken", "line recognition")):
        if not _installed(package):
            missing.append(f"the {package} package ({purpose})")
    return missing


def log_missing_models() -> None:
    missing = missing_models()
    if missing:
        _LOGGER.warning(
            "Layout segmentation and the segmented OCR engine cannot run; pages are read with GLM-OCR only, "
            "if Ollama serves it. Missing: %s. See docs/model_setup.md.",
            "; ".join(missing),
        )
