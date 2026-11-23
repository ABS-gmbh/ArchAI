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
    ],
)
def test_answerable_outcomes(ev, text: str, outcome: str) -> None:
    assert ev.score_answer(ANSWERABLE, text, {"old-french:7"})["outcome"] == outcome


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
