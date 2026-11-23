"""Shell entry points work from any directory (audit OCR-023)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

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


def _showcase_script() -> Any:
    import importlib.util
    import sys

    name = "build_thesis_showcase_payloads"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses resolve their annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bundle(root: Path, *, sha256: dict[str, str] | None = None) -> Path:
    import json

    (root / "figure").mkdir(parents=True)
    (root / "figure" / "data.json").write_text('{"ok": true}', encoding="utf-8")
    (root / "README.md").write_text("bundle\n", encoding="utf-8")
    manifest: dict[str, object] = {"files": ["manifest.json", "README.md", "figure/data.json"]}
    if sha256 is not None:
        manifest["sha256"] = sha256
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_verify_accepts_a_bundle_that_matches_its_manifest(tmp_path: Path) -> None:
    import hashlib

    digest = hashlib.sha256(b"bundle\n").hexdigest()
    bundle = _bundle(tmp_path / "b", sha256={"README.md": digest})
    assert _showcase_script().verify_bundle(bundle) == []


def test_verify_reports_every_kind_of_problem(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", sha256={"README.md": "0" * 64})
    (bundle / "figure" / "data.json").write_text("{not json", encoding="utf-8")
    (bundle / "extra.png").write_bytes(b"x")
    manifest = (bundle / "manifest.json").read_text(encoding="utf-8")
    (bundle / "manifest.json").write_text(manifest.replace('"README.md"]', '"README.md", "gone.txt"]'))
    (bundle / "manifest.json").write_text(
        manifest.replace('"figure/data.json"]', '"figure/data.json", "gone.txt"]')
    )

    problems = _showcase_script().verify_bundle(bundle)

    assert "listed but missing: gone.txt" in problems
    assert any(p.startswith("not valid JSON: figure/data.json") for p in problems)
    assert "present but not in the manifest: extra.png" in problems
    assert "content changed: README.md" in problems


def test_verify_mode_never_generates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--verify was silently ignored: the full networked generation ran and overwrote the bundle."""
    script = _showcase_script()
    monkeypatch.setattr(script, "ARTIFACT_ROOT", _bundle(tmp_path / "b"))

    def generate() -> int:
        raise AssertionError("--verify must not generate")

    monkeypatch.setattr(script, "generate", generate)
    assert script.main(["--verify"]) == 0


def test_an_unknown_flag_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _showcase_script().main(["--verfy"])
    assert exit_info.value.code == 2


def _check_packages() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_packages", REPO_ROOT / "scripts" / "check_packages.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_package_check_is_not_a_test_module() -> None:
    """Audit OCR-020: test_packages.py at the root was collectable and asserted nothing."""
    assert not (REPO_ROOT / "test_packages.py").exists()
    assert "/Users/" not in (REPO_ROOT / "scripts" / "check_packages.py").read_text(encoding="utf-8")


def test_the_package_check_exits_1_when_a_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """It printed the error and exited 0."""
    module = _check_packages()

    def broken() -> None:
        raise ImportError("No module named 'cv2'")

    monkeypatch.setattr(module, "CHECKS", {"ok": lambda: None, "opencv": broken})
    assert module.main() == 1
    monkeypatch.setattr(module, "CHECKS", {"ok": lambda: None})
    assert module.main() == 0


def test_region_ocr_availability_is_checked_not_assumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit OCR-022: both engines were recorded as not installed, unconditionally."""
    import importlib.util

    script = _showcase_script()
    installed = {"kraken"}
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: Any) -> Any:
        if name in {"kraken", "calamari_ocr"}:
            return object() if name in installed else None
        return real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert script.missing_region_ocr_packages() == ["calamari region OCR unavailable: package not installed"]
    installed.add("calamari_ocr")
    assert script.missing_region_ocr_packages() == []


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize(
    "path",
    ["archai.sqlite", "data/archai.sqlite-wal", "archai/backend/.cache/cache.sqlite", "runs/pipeline.db"],
)
def test_sqlite_databases_are_ignored_anywhere(path: str) -> None:
    """Audit HYG-18: only archai/backend/app/archai.sqlite was ignored, by the backend's own .gitignore."""
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=REPO_ROOT, capture_output=True, check=False
    )
    assert result.returncode == 0, f"{path} is not ignored"
