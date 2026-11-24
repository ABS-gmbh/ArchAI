"""Settings that would start a broken or open server stop startup instead.

Second security review (2026-11-24): `API_TOKEN="   "` started unauthenticated
with only the "not set" warning; `CORS_ORIGINS=*` (with credentials allowed)
started silently; MAX_JSON_MEGABYTES, OCR_TRACE_STREAM_MAX_OPEN,
OCR_RUN_RESERVATION_MAX_LIVE, MAX_ZIP_MEMBERS and MAX_CONCURRENT_BATCHES at 0
or below were clamped to 1 without a word, although docs/deployment.md says an
out-of-range value stops startup; and OCR_RUN_RESERVATION_TTL_SECONDS=99999999999999
started, then every reservation raised OverflowError.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

AT_LEAST_ONE = [
    "max_json_megabytes",
    "max_upload_megabytes",
    "max_extracted_megabytes",
    "max_zip_members",
    "max_concurrent_batches",
    "max_active_batches",
    "task_ttl_minutes",
    "ocr_max_decode_pixels",
    "ocr_run_reservation_ttl_seconds",
    "ocr_run_reservation_max_live",
    "ocr_trace_stream_max_open",
]


@pytest.mark.parametrize("field", AT_LEAST_ONE)
@pytest.mark.parametrize("value", [0, -1])
def test_a_limit_below_one_stops_startup_naming_the_setting(field: str, value: int) -> None:
    with pytest.raises(ValidationError, match=field.upper()):
        Settings(**{field: value})


@pytest.mark.parametrize("field", AT_LEAST_ONE)
def test_one_is_accepted(field: str) -> None:
    assert getattr(Settings(**{field: 1}), field) == 1


def test_a_blank_api_token_stops_startup() -> None:
    for blank in ("   ", "\t", " \n "):
        with pytest.raises(ValidationError, match="API_TOKEN"):
            Settings(api_token=blank)
    assert Settings(api_token="").api_token == ""
    assert Settings(api_token="  " + "t" * 30 + " ").api_token == "t" * 30


@pytest.mark.parametrize("origins", ["*", "http://localhost:3000,*", " * "])
def test_a_wildcard_cors_origin_stops_startup(origins: str) -> None:
    with pytest.raises(ValidationError, match="CORS_ORIGINS"):
        Settings(cors_origins=origins)


def test_the_default_cors_origins_start() -> None:
    assert "http://localhost:3000" in Settings().cors_origins_list


def test_a_reservation_ttl_too_large_for_a_date_stops_startup() -> None:
    with pytest.raises(ValidationError, match="OCR_RUN_RESERVATION_TTL_SECONDS"):
        Settings(ocr_run_reservation_ttl_seconds=99999999999999)
    assert Settings(ocr_run_reservation_ttl_seconds=86400).ocr_run_reservation_ttl_seconds == 86400
