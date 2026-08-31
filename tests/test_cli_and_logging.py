from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from archai_ocr import __version__
from archai_ocr.cli import EXIT_OK, EXIT_PARTIAL_FAILURE, EXIT_USER_ERROR, _collect_images, build_parser, main
from archai_ocr.logging_utils import log_stage, setup_logging

# ----------------------------------------------------------------- arguments --


def test_image_argument_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_image_accepts_multiple_paths() -> None:
    args = build_parser().parse_args(["--image", "a.png", "b.png"])
    assert args.image == ["a.png", "b.png"]


def test_invalid_reading_order_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--image", "a.png", "--reading-order", "diagonal"])


def test_version_flag_reports_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


# ------------------------------------------------------------ input collection --


def test_collect_images_expands_a_directory(tmp_path: Path) -> None:
    for name in ("b.png", "a.jpg", "notes.txt"):
        path = tmp_path / name
        if path.suffix == ".txt":
            path.write_text("x", encoding="utf-8")
        else:
            Image.new("RGB", (10, 10)).save(path)
    found = _collect_images([str(tmp_path)], recursive=False)
    assert [p.name for p in found] == ["a.jpg", "b.png"]  # sorted, non-images excluded


def test_collect_images_is_not_recursive_by_default(tmp_path: Path) -> None:
    Image.new("RGB", (10, 10)).save(tmp_path / "top.png")
    nested = tmp_path / "sub"
    nested.mkdir()
    Image.new("RGB", (10, 10)).save(nested / "deep.png")
    assert [p.name for p in _collect_images([str(tmp_path)], recursive=False)] == ["top.png"]
    assert {p.name for p in _collect_images([str(tmp_path)], recursive=True)} == {"top.png", "deep.png"}


def test_collect_images_deduplicates(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    Image.new("RGB", (10, 10)).save(image)
    assert len(_collect_images([str(image), str(image)], recursive=False)) == 1


def test_collect_images_errors_on_directory_with_no_images(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No supported images"):
        _collect_images([str(tmp_path / "empty")], recursive=False)


# ------------------------------------------------------------------ exit codes --


def test_dry_run_succeeds_without_weights(
    config_file: Path, page_image: Path, weights_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (weights_dir / "layout_yolo.pt").unlink()  # dry-run must not need models
    code = main(["--image", str(page_image), "--config", str(config_file), "--dry-run"])
    assert code == EXIT_OK
    assert str(page_image) in capsys.readouterr().out


def test_missing_weights_is_a_user_error_not_a_crash(
    config_file: Path, page_image: Path, weights_dir: Path
) -> None:
    (weights_dir / "layout_yolo.pt").unlink()
    assert main(["--image", str(page_image), "--config", str(config_file)]) == EXIT_USER_ERROR


def test_missing_config_is_a_user_error(page_image: Path, tmp_path: Path) -> None:
    assert main(["--image", str(page_image), "--config", str(tmp_path / "nope.yaml")]) == EXIT_USER_ERROR


def test_missing_image_is_a_user_error(config_file: Path, tmp_path: Path) -> None:
    assert main(["--image", str(tmp_path / "gone.png"), "--config", str(config_file)]) == EXIT_USER_ERROR


def test_continue_on_error_reports_partial_failure(
    config_file: Path, page_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import archai_ocr.cli as cli

    def boom(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(cli, "process_image", boom)
    code = main(["--image", str(page_image), "--config", str(config_file), "--continue-on-error"])
    assert code == EXIT_PARTIAL_FAILURE


def test_without_continue_on_error_it_stops_at_the_first_failure(
    config_file: Path, page_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import archai_ocr.cli as cli

    calls = []

    def boom(image_path: Path, *_args: object, **_kwargs: object) -> Path:
        calls.append(image_path)
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(cli, "process_image", boom)
    second = page_image.parent / "page_002.png"
    Image.new("RGB", (10, 10)).save(second)
    code = main(["--image", str(page_image), str(second), "--config", str(config_file)])
    assert code == EXIT_USER_ERROR
    assert len(calls) == 1


# -------------------------------------------------------------------- logging --


def test_setup_logging_does_not_touch_the_root_logger() -> None:
    root = logging.getLogger()
    sentinel = logging.StreamHandler()
    root.addHandler(sentinel)
    try:
        setup_logging("INFO", stream=io.StringIO())
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)


def test_setup_logging_rejects_a_bad_level() -> None:
    with pytest.raises(ValueError, match="Invalid log level"):
        setup_logging("VERBOSE")


def test_arbitrary_extras_reach_the_json_payload() -> None:
    buf = io.StringIO()
    logger = setup_logging("INFO", stream=buf)
    logger.info("evt", extra={"stage": "htr", "unlisted_field": 7})
    payload = json.loads(buf.getvalue().strip())
    assert payload["stage"] == "htr"
    assert payload["unlisted_field"] == 7


def test_non_ascii_is_not_escaped() -> None:
    buf = io.StringIO()
    logger = setup_logging("INFO", stream=buf)
    logger.info("evt", extra={"text": "Æthelred cœur"})
    assert "Æthelred cœur" in buf.getvalue()


def test_non_serializable_extra_does_not_break_logging() -> None:
    buf = io.StringIO()
    logger = setup_logging("INFO", stream=buf)
    logger.info("evt", extra={"obj": object()})
    payload = json.loads(buf.getvalue().strip())
    assert isinstance(payload["obj"], str)


def test_log_stage_records_failure() -> None:
    buf = io.StringIO()
    logger = setup_logging("INFO", stream=buf)
    with pytest.raises(RuntimeError), log_stage(logger, "htr"):
        raise RuntimeError("x")
    end = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert end["message"] == "stage.end"
    assert end["ok"] is False


def test_log_stage_records_success_and_duration() -> None:
    buf = io.StringIO()
    logger = setup_logging("INFO", stream=buf)
    with log_stage(logger, "layout"):
        pass
    end = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert end["ok"] is True
    assert "duration_s" in end
