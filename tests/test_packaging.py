"""pyproject.toml declares what archai_ocr imports, and nothing it does not (audit OCR-019)."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

# Distribution name -> the module it provides, where the two differ.
MODULE_OF = {"Pillow": "PIL", "PyYAML": "yaml", "python-dotenv": "dotenv"}


def _distribution(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0]


def _third_party_imports(directory: Path) -> set[str]:
    found: set[str] = set()
    for path in directory.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {name for name in found if name not in sys.stdlib_module_names and name != "archai_ocr"}


def test_runtime_dependencies_match_the_packages_imports() -> None:
    """opencv-python-headless was declared, 60 MB, and never imported."""
    declared = {
        MODULE_OF.get(_distribution(req), _distribution(req)) for req in PYPROJECT["project"]["dependencies"]
    }
    assert declared == _third_party_imports(ROOT / "src" / "archai_ocr")


def test_the_showcase_script_dependencies_are_installable() -> None:
    """scripts/build_thesis_showcase_payloads.py needs requests and openai, which were declared nowhere."""
    showcase = {_distribution(req) for req in PYPROJECT["project"]["optional-dependencies"]["showcase"]}
    assert {"requests", "openai"} <= showcase
