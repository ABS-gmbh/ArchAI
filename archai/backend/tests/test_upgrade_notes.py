"""docs/upgrade-notes.md names only settings that exist, with the defaults they have.

Every name in backticks that looks like a setting (UPPER_SNAKE_CASE) must be
read by the backend (a Settings field or an os.getenv / os.environ.get) or by
the frontend (process.env.X or env.X in its sources or next.config.ts, or a
name next.config.ts refuses). The defaults in the "New settings" table must
equal the Settings defaults. A setting the notes call removed must be read
nowhere. Like test_env_example.py for .env.example.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from app.config import Settings

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]
NOTES = ROOT / "docs" / "upgrade-notes.md"
FRONTEND = ROOT / "archai" / "frontend"

_NAME = re.compile(r"`([A-Z][A-Z0-9_]{2,})`")
_ENV_LOOKUP = re.compile(r"os\.(?:getenv|environ\.get)\(\s*[\"']([A-Z][A-Z0-9_]+)[\"']")
_FRONTEND_ENV = re.compile(r"\benv\.([A-Z][A-Z0-9_]+)")
_TABLE_ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]+)` \| `([^`]*)` \|", re.MULTILINE)
# Named in the notes as no longer listed, and read by no version.
REMOVED = {"SAIA_MODEL_PROBE"}


def _backend_names() -> set[str]:
    names = {name.upper() for name in Settings.model_fields}
    for path in (BACKEND / "app").rglob("*.py"):
        names.update(_ENV_LOOKUP.findall(path.read_text(encoding="utf-8")))
    return names


def _frontend_names() -> set[str]:
    names: set[str] = set()
    config = (FRONTEND / "next.config.ts").read_text(encoding="utf-8")
    names.update(_FRONTEND_ENV.findall(config))
    names.update(re.findall(r"\"(NEXT_PUBLIC_[A-Z0-9_]+)\"", config))  # the names the build refuses
    for path in (FRONTEND / "src").rglob("*.ts*"):
        if not path.name.endswith(".test.ts"):
            names.update(_FRONTEND_ENV.findall(path.read_text(encoding="utf-8")))
    return names


def _notes() -> str:
    return NOTES.read_text(encoding="utf-8")


def test_every_setting_named_exists() -> None:
    named = set(_NAME.findall(_notes()))
    assert len(named) > 30, "the notes name fewer settings than expected; did the format change?"
    unknown = named - REMOVED - _backend_names() - _frontend_names()
    assert sorted(unknown) == []


def test_the_new_settings_have_the_defaults_stated() -> None:
    rows = _TABLE_ROW.findall(_notes())
    assert len(rows) > 20
    wrong = []
    for name, literal in rows:
        field = Settings.model_fields[name.lower()]
        if ast.literal_eval(literal) != field.default:
            wrong.append(f"{name}: notes say {literal}, Settings has {field.default!r}")
    assert wrong == []


def test_a_setting_named_as_removed_is_read_nowhere() -> None:
    assert REMOVED <= set(_NAME.findall(_notes()))
    assert sorted(REMOVED & (_backend_names() | _frontend_names())) == []


def test_the_notes_are_linked_from_the_readme_and_the_deployment_guide() -> None:
    for page in (ROOT / "README.md", ROOT / "docs" / "deployment.md"):
        assert "upgrade-notes.md" in page.read_text(encoding="utf-8"), page.name
