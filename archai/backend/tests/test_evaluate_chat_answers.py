"""scripts/evaluate_chat_answers.py: the labels, the scoring and the offline run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.config import settings
from app.db import pipeline_db
from app.services import rag_store
from tests.test_retrieval_scope import _embed

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("evaluate_chat_answers", BACKEND / "scripts" / "evaluate_chat_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_item_is_labelled_and_matches_the_retrieval_questions(ev) -> None:
    items = ev.load_items(BACKEND / "eval/chat/answers.json")
    questions = json.loads((BACKEND / "eval/retrieval/questions.json").read_text(encoding="utf-8"))
    references = ev.load_references(BACKEND / "eval/quality_gate/readings.json")
    answerable = [i for i in items if i["kind"] == "answerable"]
    absent = [i for i in items if i["kind"] == "not_in_sources"]
    assert [i["question"] for i in answerable] == [q["question"] for q in questions["questions"]]
    assert [i["question"] for i in absent] == [q["question"] for q in questions["not_in_sources"]]
    assert len({i["id"] for i in items}) == len(items)
    for item in answerable:
        assert references[item["page_id"]].count(item["answer_span"]) == 1, item["id"]
        assert item["expected"].strip() and item["accept"], item["id"]
        assert all(group and all(term == term.lower() for term in group) for group in item["accept"]), item["id"]
        # The expected answer itself must pass its own accept groups.
        assert ev.score_answer(item, item["expected"], set())["outcome"] == "correct", item["id"]


ANSWERABLE = {"id": "a", "kind": "answerable", "accept": [["sabbath", "saturday"], ["holy", "hallow"]]}
ABSENT = {"id": "n", "kind": "not_in_sources"}


@pytest.mark.parametrize(
    ("text", "outcome"),
    [
        ("Keep the Sabbath day holy (chunk old-french:7).", "correct"),
        ("Keep the Saturday holy.", "correct"),
        ("Keep the Sabbath.", "wrong"),
        ("The sources for this page do not contain the answer.", "abstained"),
        # Opens by refusing, but the terms match later: neither correct nor abstained.
        ("The text does not contain a direct answer. It says keep the Sabbath holy [1].", "hedged"),
        ("The provided text does not mention a holy Sabbath.", "hedged"),
        # A refusal in the answer itself is not a refusal of the sources.
        ("No: the Sabbath is not given to work; keep it holy.", "correct"),
    ],
)
def test_answerable_outcomes(ev, text: str, outcome: str) -> None:
    assert ev.score_answer(ANSWERABLE, text, {"old-french:7"})["outcome"] == outcome


def test_with_labels_a_citation_counts_for_a_cited_label_covering_the_answer(ev) -> None:
    sources = [{"label": "1", "chunk_ids": ["old-french:6", "old-french:7"], "cited": True}, {"label": "2", "chunk_ids": ["old-french:2"], "cited": False}]
    assert ev.score_answer(ANSWERABLE, "Keep the Sabbath holy [1].", {"old-french:7"}, sources)["cited"] is True
    sources[0]["cited"], sources[1]["cited"] = False, True
    assert ev.score_answer(ANSWERABLE, "Keep the Sabbath holy [2].", {"old-french:7"}, sources)["cited"] is False


def test_a_citation_counts_only_for_a_chunk_holding_the_answer(ev) -> None:
    assert ev.score_answer(ANSWERABLE, "Keep the Sabbath holy (old-french:7).", {"old-french:7"})["cited"] is True
    assert ev.score_answer(ANSWERABLE, "Keep the Sabbath holy (old-french:2).", {"old-french:7"})["cited"] is False


@pytest.mark.parametrize(
    ("text", "outcome"),
    [
        ("The text does not mention the fourth commandment.", "abstained"),
        ("That is not in the sources for this page.", "abstained"),
        ("There is no information about the gates.", "abstained"),
        ("The page doesn't say.", "abstained"),
        ("The fourth commandment is to honour your father and mother.", "answered"),
    ],
)
def test_not_in_sources_outcomes(ev, text: str, outcome: str) -> None:
    assert ev.score_answer(ABSENT, text, set())["outcome"] == outcome


def test_tally_reports_n_per_kind_and_page(ev) -> None:
    rows = [
        {"kind": "answerable", "page_id": "p", "outcome": "correct"},
        {"kind": "answerable", "page_id": "q", "outcome": "wrong"},
        {"kind": "not_in_sources", "page_id": "p", "outcome": "abstained"},
    ]
    table = ev.tally(rows, "outcome")
    assert table["answerable"] == {"n": 2, "counts": {"correct": 1, "wrong": 1}, "per_page": {"p": {"correct": 1}, "q": {"wrong": 1}}}
    assert table["not_in_sources"]["n"] == 1


def test_the_offline_run_uses_a_throwaway_store_and_the_chat_path(ev, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    # Everything build_store changes is restored after the test.
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "unused.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "chroma_persist_dir", settings.chroma_persist_dir)
    monkeypatch.setattr(settings, "rag_embedding_model", "test-embedder")
    monkeypatch.setattr(rag_store, "_CLIENT", None)
    monkeypatch.setattr(rag_store, "_embed", _embed)
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])
    monkeypatch.setattr(ev.tempfile, "TemporaryDirectory", lambda prefix: _Dir(tmp_path / "store"))

    assert ev.main(["--embedder", "provider", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    report = out["report"]
    assert set(report["index_status"].values()) == {"ok"}
    assert report["retrieval"]["answerable"]["n"] == 62
    assert report["retrieval"]["not_in_sources"]["n"] == 17
    assert report["not_in_sources_instruction"]["answerable"]["counts"] == {"True": 62}
    holds = report["evidence_holds_answer"]["answerable"]["counts"]
    assert sum(holds.values()) == 62 and holds.get("True", 0) > 31
    assert (tmp_path / "store" / "archai.sqlite").is_file()


class _Dir:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, *exc) -> None:
        return None


def test_e5_local_embeds_here_and_instructs_the_query(ev, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    seen: list[str] = []

    class _FakeModel:
        def __init__(self, name: str, model_kwargs: dict) -> None:
            assert name == ev.E5_LOCAL_MODEL
            assert model_kwargs == {"dtype": "float32"}  # the float16 checkpoint is slow on a CPU

        def encode(self, texts, batch_size, normalize_embeddings):
            seen.extend(texts)
            return _embed(list(texts))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_FakeModel))
    monkeypatch.setenv("ARCHAI_DB_PATH", str(tmp_path / "unused.sqlite"))
    monkeypatch.setattr(pipeline_db, "_DB_READY", False)
    monkeypatch.setattr(settings, "chroma_persist_dir", settings.chroma_persist_dir)
    monkeypatch.setattr(settings, "rag_embedding_model", settings.rag_embedding_model)
    monkeypatch.setattr(rag_store, "_CLIENT", None)
    monkeypatch.setattr(rag_store, "_embed", rag_store._embed)
    monkeypatch.setattr("app.db.pipeline_db.list_mention_links_for_run", lambda run_id: [])

    references = ev.load_references(BACKEND / "eval/quality_gate/readings.json")
    runs = ev.build_store(tmp_path, references, {"old-french"}, embedder="e5-local")
    assert runs["old-french"]["index_status"] == "ok"
    assert settings.rag_embedding_model == ev.E5_LOCAL_MODEL
    assert seen and not any(text.startswith("Instruct:") for text in seen)  # documents as they are

    items = [i for i in ev.load_items(BACKEND / "eval/chat/answers.json") if i["page_id"] == "old-french"][:1]
    rows = ev.evaluate_retrieval(items, runs, references)
    assert rows[0]["retrieval"] == "ok"
    assert seen[-1] == f"Instruct: {rag_store.E5_RETRIEVAL_TASK}\nQuery: {items[0]['question']}"


def test_the_verifier_report_lines_assessments_up_with_the_scored_answers(ev) -> None:
    rows = [{"id": "a1", "outcome": "correct"}, {"id": "a2", "outcome": "wrong"}, {"id": "n1", "outcome": "answered"}]
    checks = [
        {"id": "a1", "assessment": "supported", "seconds": 1.0, "verifier": "v"},
        {"id": "a2", "assessment": "unsupported", "seconds": 3.0, "verifier": "v"},
        {"id": "n1", "assessment": "partially_supported", "seconds": 2.0, "verifier": "v"},
    ]
    report = ev.verifier_report([], rows, checks)
    assert report["by_outcome"] == {"correct": {"supported": 1}, "wrong": {"unsupported": 1}, "answered": {"partially_supported": 1}}
    assert report["seconds_median"] == 2.0 and report["verifiers"] == ["v"]


def test_requests_are_paced_under_ten_in_ten_seconds(ev, monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(ev.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ev.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    ev._SENT.clear()
    for _ in range(9):
        ev._pace()
    assert clock[0] == 0.0  # nine go out at once
    ev._pace()
    assert clock[0] >= 10.0  # the tenth waits for the window
