"""The settings scripts/fetch_kraken_models.sh tells an operator to set exist.

Its closing hints named ARCHAI_KRAKEN_MODELS_DIR and
ARCHAI_KRAKEN_CREMMA_LAT_MODEL_PATH. Nothing reads either: the backend's
settings have no prefix (KRAKEN_MODELS_DIR, KRAKEN_CREMMA_LAT_MODEL_PATH), and
the command-line pipeline reads ARCHAI_OCR_*. An operator who set them saw no
effect and no error.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.config import Settings

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "fetch_kraken_models.sh"


def _cli_variables() -> set[str]:
    """Environment variables the command-line pipeline (src/archai_ocr/config.py) reads, aliases included."""
    source = (REPO / "src" / "archai_ocr" / "config.py").read_text(encoding="utf-8")
    return set(re.findall(r'"(ARCHAI_[A-Z0-9_]+)"\s*:', source))


def _printed_assignments() -> list[str]:
    """NAME=value lines in the script's heredocs: the settings it tells the operator to use."""
    return re.findall(r"^\s+([A-Z][A-Z0-9_]+)=", SCRIPT.read_text(encoding="utf-8"), re.MULTILINE)


def test_the_script_prints_some_settings() -> None:
    assert _printed_assignments()


def test_every_printed_setting_is_read_by_the_backend_or_the_cli() -> None:
    backend = {name.upper() for name in Settings.model_fields}
    known = backend | _cli_variables()
    # Shell variables the script assigns for itself are not settings.
    script_locals = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", SCRIPT.read_text(encoding="utf-8"), re.MULTILINE))
    printed = [name for name in _printed_assignments() if name not in script_locals]
    assert printed
    assert sorted(set(printed) - known) == []
