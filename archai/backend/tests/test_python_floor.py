"""The backend runs on the Python its pyproject.toml declares (requires-python >= 3.10).

logging_setup.py and run_retention.py imported datetime.UTC, which exists only
from Python 3.11, so on 3.10 the app failed at import although pip installed
it. This parses app/ and scripts/ with the 3.10 grammar and refuses the
3.11-only standard-library names most likely to slip in.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
SOURCES = sorted([*(BACKEND / "app").rglob("*.py"), *(BACKEND / "scripts").rglob("*.py")])

# module -> names added in 3.11
_NEW_IN_311 = {
    "datetime": {"UTC"},
    "typing": {"Self", "LiteralString", "Never", "assert_never", "reveal_type", "Required", "NotRequired"},
    "enum": {"StrEnum"},
    "asyncio": {"TaskGroup", "timeout"},
}


def _floor() -> tuple[int, int]:
    declared = tomllib.loads((BACKEND / "pyproject.toml").read_text())["project"]["requires-python"]
    major, minor = re.search(r">=\s*(\d+)\.(\d+)", declared).groups()
    return int(major), int(minor)


def test_the_declared_floor_is_3_10() -> None:
    assert _floor() == (3, 10)


def test_sources_parse_with_the_declared_grammar() -> None:
    for path in SOURCES:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=_floor())


def test_no_3_11_only_standard_library_names() -> None:
    found = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _NEW_IN_311:
                found += [f"{path.name}: from {node.module} import {a.name}" for a in node.names if a.name in _NEW_IN_311[node.module]]
            elif isinstance(node, ast.Import):
                found += [f"{path.name}: import {a.name}" for a in node.names if a.name == "tomllib"]
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.attr in _NEW_IN_311.get(node.value.id, ())
            ):
                found.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert found == []
