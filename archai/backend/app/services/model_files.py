"""Which model files and packages the reading pipeline lacks, said once at startup.

Without the layout weights, /api/predict answers 500 and the segmented OCR
engine cannot run, so pages are read with GLM-OCR alone (if Ollama serves it).
Without the CATMuS Medieval recogniser or the kraken package, the segmented
engine has nothing to read lines with. None of this stops the server, which
still serves chat, retrieval and GLM-OCR, so it is a warning, not a refusal.
"""

from __future__ import annotations

import importlib.util
import logging
import os

from app.config import settings

_LOGGER = logging.getLogger(__name__)


def _installed(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def missing_models() -> list[str]:
    """What is missing, each with the setting or command that supplies it."""
    from app.services.ocr_backends import _resolve_model_path

    missing: list[str] = []
    layout = [
        path
        for path in (settings.emanuskript_model_path, settings.catmus_model_path, settings.zone_model_path)
        if not os.path.isfile(path)
    ]
    if layout:
        names = ", ".join(os.path.basename(path) for path in layout)
        missing.append(f"layout weights {names} in {os.path.dirname(layout[0])!r} (MODEL_DIR)")
    recogniser = _resolve_model_path(settings.kraken_catmus_model_path)
    if not recogniser.is_file():
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
