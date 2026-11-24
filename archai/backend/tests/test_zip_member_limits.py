"""A batch ZIP cannot create files without bound (security review C6).

Empty members cost no extraction budget: a 5.6 MB ZIP of 60,000 empty .png
members created 60,000 files in 3.5 s, and the default upload size allows
about 11 million.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from app.routers import predict


def _zip(path: Path, names: list[str], data: bytes = b"") -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        for name in names:
            zf.writestr(name, data)
    return path


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def test_an_archive_over_the_member_limit_is_refused_before_anything_is_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict.settings, "max_zip_members", 10, raising=False)
    archive = _zip(tmp_path / "many.zip", [f"{i:04d}.png" for i in range(11)])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="members"):
        predict._safe_extract_zip(str(archive), str(out))
    assert _files(out) == []
    archive = _zip(tmp_path / "ten.zip", [f"{i:04d}.png" for i in range(10)])
    predict._safe_extract_zip(str(archive), str(out))
    assert len(_files(out)) == 10


def test_members_that_are_not_images_are_not_written(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "mixed.zip", ["page.png", "notes.txt", "run.sh", "deep/page2.JPG"], b"x")
    out = tmp_path / "out"
    out.mkdir()
    predict._safe_extract_zip(str(archive), str(out))
    assert sorted(p.name for p in _files(out)) == ["page.png", "page2.JPG"]


def test_every_member_costs_a_minimum_share_of_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict.settings, "max_extracted_megabytes", 1)
    monkeypatch.setattr(predict.settings, "max_zip_members", 100_000, raising=False)
    archive = _zip(tmp_path / "empty.zip", [f"{i:05d}.png" for i in range(300)])  # 300 x 4 KiB > 1 MiB
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="MB"):
        predict._safe_extract_zip(str(archive), str(out))
    assert len(_files(out)) < 300
