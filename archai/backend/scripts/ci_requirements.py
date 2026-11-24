"""Print the backend's requirements for a test install without the inference stack.

CI runs the suite without ultralytics, which brings torch and several hundred
megabytes; tests that need it are marked requires_models. It used to install
a hand-copied list of pyproject.toml's dependencies, which drifted whenever a
dependency was added. This prints that list from pyproject.toml itself:
[project] dependencies less the inference stack, plus the `dev` extra
(pytest, httpx) and the `headless` extra (OpenCV, which otherwise arrives
through ultralytics). Usage::

    python scripts/ci_requirements.py > requirements-ci.txt
    python -m pip install -r requirements-ci.txt

On Python 3.10, which has no tomllib, install tomli first.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
# Installed only with the models: layout inference (and torch, through it).
INFERENCE_STACK = {"ultralytics"}
EXTRAS = ("dev", "headless")


def _name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
    if match is None:
        raise ValueError(f"cannot read a package name from {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def ci_requirements(pyproject: Path = PYPROJECT) -> list[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    wanted = [req for req in project["dependencies"] if _name(req) not in INFERENCE_STACK]
    for extra in EXTRAS:
        wanted.extend(project["optional-dependencies"][extra])
    return wanted


def main() -> int:
    sys.stdout.write("".join(f"{requirement}\n" for requirement in ci_requirements()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
