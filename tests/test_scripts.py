"""Shell entry points work from any directory (audit OCR-023)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_run_demo_finds_the_package_and_config_from_another_directory(tmp_path: Path) -> None:
    """It set PYTHONPATH=src and config.example.yaml relative to the caller's directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text('#!/usr/bin/env bash\necho "PYTHONPATH=${PYTHONPATH}"\necho "ARGS=$*"\n')
    fake_python.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_demo.sh"), "page.png"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"PYTHONPATH={REPO_ROOT / 'src'}:" in result.stdout
    assert f"--config {REPO_ROOT / 'config.example.yaml'}" in result.stdout
    assert "--image page.png" in result.stdout
