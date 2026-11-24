"""What the run's step list says after the auto-index step.

A live read with the embedding provider down ended on "Indexed 0 chunks."
under a finished run: it looked like success, while nothing on the page could
be searched or cited.
"""

from __future__ import annotations

import pytest

from app.services import rag_store, trace_pipeline


def _events_after_index(monkeypatch: pytest.MonkeyPatch, result: dict) -> list[tuple[str, str, str]]:
    events: list[tuple[str, str, str]] = []
    monkeypatch.setattr(trace_pipeline._app_settings, "rag_auto_index", True)
    monkeypatch.setattr(trace_pipeline, "log_event", lambda run_id, stage, event, message=None: events.append((stage, event, message or "")))
    monkeypatch.setattr(rag_store, "index_run", lambda run_id: result)
    trace_pipeline._auto_index_run("run-1")
    return events


@pytest.mark.parametrize(
    ("status", "words"),
    [
        ("embedding_unavailable", "embedding service is unavailable"),
        ("blocked_by_quality_gate", "quality check"),
        ("empty", "Nothing to index"),
    ],
)
def test_a_run_that_stored_nothing_is_reported_as_not_indexed(monkeypatch: pytest.MonkeyPatch, status: str, words: str) -> None:
    events = _events_after_index(monkeypatch, {"chunks_indexed": 0, "status": status})
    assert [stage for stage, _event, _message in events] == ["INDEX_RUNNING", "INDEX_SKIPPED"]
    stage, event, message = events[-1]
    assert event == "INFO" and words in message
    assert "Indexed 0" not in message


def test_an_indexed_run_still_reports_its_chunk_count(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _events_after_index(monkeypatch, {"chunks_indexed": 4, "status": "ok"})
    assert events[-1] == ("INDEX_DONE", "END", "Indexed 4 chunks.")


def test_entities_alone_count_as_indexed(monkeypatch: pytest.MonkeyPatch) -> None:
    # index_run answers "ok" when only entity documents were stored.
    events = _events_after_index(monkeypatch, {"chunks_indexed": 0, "entities_indexed": 2, "status": "ok"})
    assert events[-1][0] == "INDEX_DONE"
