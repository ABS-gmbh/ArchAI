"""Chat answers cite evidence by a short label, and say what each label stands for.

The prompt asked for "(asset_ref=..., run_id=..., chunk_id=..., offsets=...)",
and models wrote exactly that, sometimes with the dots, five UUIDs in a row: a
reader could not use it. Each evidence block now opens with [1], [2], ... in the
order shown, the model cites by label, and every answer carries ``sources``:
what each label stands for (chunk ids, offsets, the page's lines) and whether
the answer cites it.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.config import settings
from app.services import chat_ai, rag_store
from app.services.rag_store import RetrievalResult
from tests.test_chat_api import _FakeClient, _FakeStreamClient

TEXT = "Caldus li roys\nqui molt fu ber\nEt sages hom\nde grant valor\nSon fil maria\n"


def _hit(start: int, end: int, idx: int, run_id: str = "run-1") -> dict[str, Any]:
    return {"chunk_id": f"{run_id}:{idx}", "run_id": run_id, "asset_ref": "page-1", "chunk_idx": idx,
            "start_offset": start, "end_offset": end, "text": TEXT[start:end], "distance": None}


LINE3 = TEXT.index("Et sages")
HITS = [_hit(LINE3, TEXT.index("Son fil"), 1), _hit(0, LINE3, 0, run_id="run-2")]


def test_each_block_opens_with_its_label_in_the_order_shown() -> None:
    evidence = rag_store.format_evidence_blocks(HITS)
    assert evidence.index("label: [1]") < evidence.index("chunk_id: run-1:1") < evidence.index("label: [2]")
    sources = rag_store.evidence_sources(HITS, {"run-1": TEXT, "run-2": TEXT})
    assert [(s["label"], s["chunk_ids"]) for s in sources] == [("1", ["run-1:1"]), ("2", ["run-2:0"])]
    assert (sources[0]["line_start"], sources[0]["line_end"]) == (3, 4)  # "Et sages hom" .. "de grant valor"
    assert (sources[1]["line_start"], sources[1]["line_end"]) == (1, 2)


def test_lines_are_left_out_when_the_text_is_not_the_runs() -> None:
    sources = rag_store.evidence_sources(HITS, {"run-1": "another text altogether, longer than the chunk is"})
    assert "line_start" not in sources[0] and "line_start" not in sources[1]


def test_merged_neighbours_share_one_label() -> None:
    sources = rag_store.evidence_sources([_hit(0, LINE3, 0), _hit(TEXT.index("qui"), TEXT.index("de grant"), 1)], {"run-1": TEXT})
    assert len(sources) == 1 and sources[0]["chunk_ids"] == ["run-1:0", "run-1:1"]
    assert (sources[0]["line_start"], sources[0]["line_end"]) == (1, 3)


@pytest.mark.parametrize(
    ("answer", "cited"),
    [("Caldus was king [2].", {"2"}), ("He married his son [1][2].", {"1", "2"}), ("See [1, 2].", {"1", "2"}), ("No citation.", set())],
)
def test_cited_labels_are_read_from_the_answer(answer: str, cited: set[str]) -> None:
    evidence = chat_ai.RetrievedEvidence("x", {}, rag_store.evidence_sources(HITS))
    assert {s["label"] for s in chat_ai._cited_sources(answer, evidence) if s["cited"]} == cited


def _setup(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    monkeypatch.setattr(chat_ai, "_create_client", lambda: client)
    monkeypatch.setattr(settings, "chat_rag_model", "meta-llama-3.1-8b-instruct")
    monkeypatch.setattr(settings, "paleography_verification_enabled", False)
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])
    monkeypatch.setattr("app.db.pipeline_db.get_run", lambda run_id: {"ocr_text": TEXT})
    monkeypatch.setattr("app.services.rag_store.retrieve_entities", lambda *a, **k: RetrievalResult(mode="dense"))
    monkeypatch.setattr("app.services.rag_store.retrieve_chunks", lambda *a, **k: RetrievalResult(HITS[:1], mode="hybrid"))


def test_the_prompt_asks_for_labels_not_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="Caldus [1].")
    _setup(monkeypatch, fake)
    chat_ai.create_chat_completion(messages=[{"role": "user", "content": "Who is the king?"}], context={"ocr_run_id": "run-1"})
    system = fake.calls[-1]["messages"][0]["content"]
    assert "by its label alone" in system and "asset_ref=..." not in system


def test_the_answer_carries_its_sources_streamed_or_not(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(monkeypatch, _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="Sages hom [1]."))
    result = chat_ai.create_chat_completion(messages=[{"role": "user", "content": "Who?"}], context={"ocr_run_id": "run-1"})
    assert result["sources"] == [
        {"label": "1", "run_id": "run-1", "chunk_ids": ["run-1:1"], "start_offset": HITS[0]["start_offset"], "end_offset": HITS[0]["end_offset"],
         "chunks": [{"chunk_id": "run-1:1", "start_offset": HITS[0]["start_offset"], "end_offset": HITS[0]["end_offset"]}],
         "line_start": 3, "line_end": 4, "cited": True}
    ]
    _setup(monkeypatch, _FakeStreamClient(["meta-llama-3.1-8b-instruct"], deltas=["Sages ", "hom."]))
    done = list(chat_ai.stream_chat_completion(messages=[{"role": "user", "content": "Who?"}], context={"ocr_run_id": "run-1"}))[-1]
    assert done["type"] == "done" and done["sources"][0]["label"] == "1" and done["sources"][0]["cited"] is False


def test_the_answer_event_carries_the_sources_before_the_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The workspace shows the answer at "answer" and the verdict later: its [n] buttons need the sources then."""
    _setup(monkeypatch, _FakeStreamClient(["meta-llama-3.1-8b-instruct"], deltas=["Sages ", "hom [1]."]))
    events = list(chat_ai.stream_chat_completion(messages=[{"role": "user", "content": "Who?"}], context={"ocr_run_id": "run-1"}))
    answer = next(event for event in events if event["type"] == "answer")
    assert answer["sources"][0]["label"] == "1" and answer["sources"][0]["cited"] is True
    assert answer["sources"][0]["line_start"] == 3


def test_the_answer_comes_apart_from_its_appendices(monkeypatch: pytest.MonkeyPatch) -> None:
    """``answer`` is the answer alone; ``text`` keeps the [Retrieval] notice and [Verification] block for old clients."""
    _setup(monkeypatch, _FakeClient(["meta-llama-3.1-8b-instruct"], completion_text="Sages hom [1]."))
    monkeypatch.setattr("app.services.rag_store.retrieve_chunks", lambda *a, **k: RetrievalResult(HITS[:1], mode="lexical", notes=["dense ranking unavailable"]))
    result = chat_ai.create_chat_completion(messages=[{"role": "user", "content": "Who?"}], context={"ocr_run_id": "run-1"})
    assert result["answer"] == "Sages hom [1]."
    assert result["text"].startswith("Sages hom [1].") and "[Retrieval]" in result["text"]
    _setup(monkeypatch, _FakeStreamClient(["meta-llama-3.1-8b-instruct"], deltas=["Sages ", "hom."]))
    monkeypatch.setattr("app.services.rag_store.retrieve_chunks", lambda *a, **k: RetrievalResult(HITS[:1], mode="lexical", notes=["dense ranking unavailable"]))
    done = list(chat_ai.stream_chat_completion(messages=[{"role": "user", "content": "Who?"}], context={"ocr_run_id": "run-1"}))[-1]
    assert done["answer"] == "Sages hom." and "[Retrieval]" in done["text"]
