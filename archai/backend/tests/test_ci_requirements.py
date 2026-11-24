"""CI's test install comes from pyproject.toml, less the inference stack.

.github/workflows/ci.yml installed a hand-copied list of the backend's
dependencies, which had to be edited alongside pyproject.toml. The list is now
printed by scripts/ci_requirements.py; these tests pin what it must contain.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "scripts"))

import ci_requirements  # noqa: E402

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = pytest.importorskip("tomli")


def _names(requirements: list[str]) -> set[str]:
    return {ci_requirements._name(requirement) for requirement in requirements}


def test_every_runtime_dependency_but_the_inference_stack_is_installed() -> None:
    project = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = _names(project["dependencies"])
    printed = _names(ci_requirements.ci_requirements())
    assert "ultralytics" in declared
    assert "ultralytics" not in printed
    assert declared - {"ultralytics"} <= printed


def test_the_test_tools_and_opencv_are_installed() -> None:
    printed = _names(ci_requirements.ci_requirements())
    assert {"pytest", "httpx", "opencv-python-headless"} <= printed


def test_names_are_normalised() -> None:
    assert ci_requirements._name("uvicorn[standard]>=0.30.0") == "uvicorn"
    assert ci_requirements._name("python_jose[cryptography]>=3.3.0") == "python-jose"
    assert ci_requirements._name("numpy<2.0") == "numpy"


def test_every_module_the_app_imports_unguarded_is_declared() -> None:
    """A top-level import of a third-party module must have a declared package behind it."""
    provided_by = {
        # module: the distributions that provide it, declared or transitive
        "PIL": {"pillow"},
        "jose": {"python-jose"},
        "dotenv": {"python-dotenv"},
        "pydantic": {"pydantic-settings", "fastapi"},
        "pydantic_settings": {"pydantic-settings"},
        "starlette": {"fastapi"},
        "yaml": {"uvicorn"},
    }
    project = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = _names(project["dependencies"])
    stdlib = set(sys.stdlib_module_names)
    missing: list[str] = []
    for path in (BACKEND / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only: imports inside functions or try blocks are optional
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top in stdlib or top in {"app", "__future__"}:
                    continue
                candidates = provided_by.get(top, {top.lower().replace("_", "-")})
                if not candidates & declared:
                    missing.append(f"{path.relative_to(BACKEND)}: {top}")
    assert missing == []
