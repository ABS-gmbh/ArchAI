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


def _guarded_tomllib_imports(tree: ast.AST) -> set[int]:
    """`import tomllib` nodes inside `try: ... except ModuleNotFoundError/ImportError: import tomli`.

    That is the backport's documented pattern, and runs on 3.10 with tomli installed.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        falls_back = any(
            isinstance(handler.type, ast.Name)
            and handler.type.id in {"ModuleNotFoundError", "ImportError"}
            and any(isinstance(stmt, ast.Import) and any(a.name == "tomli" for a in stmt.names) for stmt in handler.body)
            for handler in node.handlers
        )
        if falls_back:
            guarded.update(id(stmt) for stmt in node.body if isinstance(stmt, ast.Import))
    return guarded


def test_no_3_11_only_standard_library_names() -> None:
    found = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        guarded = _guarded_tomllib_imports(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _NEW_IN_311:
                found += [f"{path.name}: from {node.module} import {a.name}" for a in node.names if a.name in _NEW_IN_311[node.module]]
            elif isinstance(node, ast.Import) and id(node) not in guarded:
                found += [f"{path.name}: import {a.name}" for a in node.names if a.name == "tomllib"]
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.attr in _NEW_IN_311.get(node.value.id, ())
            ):
                found.append(f"{path.name}:{node.lineno} {node.value.id}.{node.attr}")
    assert found == []


def test_a_bare_tomllib_import_is_still_refused() -> None:
    bare = ast.parse("import tomllib\n")
    guarded = ast.parse("try:\n    import tomllib\nexcept ModuleNotFoundError:\n    import tomli as tomllib\n")
    assert _guarded_tomllib_imports(bare) == set()
    assert len(_guarded_tomllib_imports(guarded)) == 1
