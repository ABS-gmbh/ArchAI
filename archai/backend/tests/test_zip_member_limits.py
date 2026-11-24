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


def _directory_only_zip(path: Path, members: int, stated: int = 1) -> Path:
    """Only a central directory of *members* entries, no member data; the end record states *stated*."""
    import struct

    with open(path, "wb") as out:
        size = 0
        for i in range(members):
            name = b"%07d.png" % i
            entry = struct.pack("<4s4B4HL2L5H2L", b"PK\x01\x02", 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, len(name), 0, 0, 0, 0, 0, 0) + name
            out.write(entry)
            size += len(entry)
        out.write(struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, stated, stated, size, 0, 0))
    return path


def test_an_end_record_that_understates_the_members_is_refused_before_zipfile_parses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Second review, B: the end record said 1 member over a directory of a
    # million; zipfile built them all (2.5 s, +528 MB) before the count check.
    monkeypatch.setattr(predict.settings, "max_zip_members", 10, raising=False)
    archive = _directory_only_zip(tmp_path / "lying.zip", members=1000)

    def never_opened(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("zipfile.ZipFile opened an archive the end record already condemns")

    monkeypatch.setattr(predict.zipfile, "ZipFile", never_opened)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(predict.BatchArchiveRefused, match="directory"):
        predict._safe_extract_zip(str(archive), str(out))
    assert _files(out) == []


def test_a_real_archive_at_the_member_limit_with_long_names_still_extracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict.settings, "max_zip_members", 20, raising=False)
    names = [f"{'Bibliothèque nationale de France/' * 6}folio-{i:03d}r.png" for i in range(20)]
    archive = _zip(tmp_path / "long.zip", names, b"x")
    out = tmp_path / "out"
    out.mkdir()
    predict._safe_extract_zip(str(archive), str(out))
    assert len(_files(out)) == 20
