"""The ?token= of the two event streams counts, and is hidden, only under its own name.

Found by a security review: "?%74oken=<value>" authenticated on the stream
routes, since the query string is decoded before it is read, while uvicorn's
access log redaction matched only the literal "token=", so that spelling put
the API token in the log in clear.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.db import pipeline_db
from app.logging_setup import RedactTokenFilter

TOKEN = "token-for-tests-only-0123456789"


def _access_line(path: str) -> str:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:5000", "GET", path, "1.1", 200), None
    )
    RedactTokenFilter().filter(record)
    return record.getMessage()


@pytest.mark.parametrize(
    "query",
    [
        f"%74oken={TOKEN}",
        f"x=1&%74%6F%6B%65%6E={TOKEN}",
        f"to%6Ben={TOKEN}&y=2",
        f"TOKEN={TOKEN}",
        f"token={TOKEN}",
        f"a=1;token={TOKEN}",
    ],
)
def test_no_spelling_of_the_key_puts_the_token_in_the_access_log(query: str) -> None:
    line = _access_line(f"/api/ocr/trace/stream/r1?{query}")
    assert TOKEN not in line
    assert "[redacted]" in line


def test_other_parameters_are_left_as_they_are() -> None:
    assert _access_line("/api/index/search?query=tokens%3Dmany&k=3").endswith('"GET /api/index/search?query=tokens%3Dmany&k=3 HTTP/1.1" 200')


def test_only_the_literal_key_authenticates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_token", TOKEN)
    client = TestClient(main.app)
    run_id = pipeline_db.create_run(asset_ref="page")
    pipeline_db.update_run_fields(run_id, status="COMPLETED", current_stage="DONE")
    assert client.get(f"/api/ocr/trace/stream/{run_id}?token={TOKEN}").status_code == 200
    assert client.get(f"/api/ocr/trace/stream/{run_id}?%74oken={TOKEN}").status_code == 401


# Second security review (2026-11-24): the redaction stopped at a double quote,
# so "?a=\"&token=<value>" kept the token in the access log. uvicorn's parsers
# both pass a raw quote into the query (httptools 0.9 and h11 0.16, checked
# live), and h11 a raw "#" too; only the token's holder can craft such a URL.
@pytest.mark.parametrize("before", ['a="', "a=#", "a='", 'a="x"#y', "a=%22"])
def test_a_quote_or_hash_before_the_token_does_not_stop_the_redaction(before: str) -> None:
    line = _access_line(f"/api/ocr/trace/stream/r1?{before}&token={TOKEN}")
    assert TOKEN not in line
    assert line.endswith(' HTTP/1.1" 200')


def test_a_whole_line_is_redacted_too() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, f'127.0.0.1:5000 - "GET /api/x?a="&token={TOKEN} HTTP/1.1" 200', None, None
    )
    RedactTokenFilter().filter(record)
    assert TOKEN not in record.getMessage()
    assert record.getMessage().endswith(' HTTP/1.1" 200')
