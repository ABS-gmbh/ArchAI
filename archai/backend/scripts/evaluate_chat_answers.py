"""Evaluate chat answers on the reference pages: grounded, cited, or "not in the sources".

Usage::

    python scripts/evaluate_chat_answers.py                       # retrieval only, no chat model
    python scripts/evaluate_chat_answers.py --live --out answers.jsonl
    python scripts/evaluate_chat_answers.py --score answers.jsonl

``eval/chat/answers.json`` holds hand-labelled items about the five reference
pages of ``eval/quality_gate/readings.json`` (see ``eval/chat/README.md``):

- ``answerable``: a question, the span of the page that answers it, the
  expected answer in English and ``accept``, groups of terms an answer must
  contain (every group, any term of a group).
- ``not_in_sources``: a question on the same page and topic that the page
  does not answer. The right answer says so.

Each reference page is loaded into a throwaway pipeline DB and vector store in
a temporary directory, as one gate-approved run chunked as the pipeline chunks.
Questions then go through the chat path itself, ``chat_ai._prepare_messages``,
with the run as context, so retrieval, the evidence blocks and the grounding
instructions are the production ones.

Without ``--live`` no chat model is called. The report then says, per item,
whether the evidence put in front of the model holds the answer span (for an
answerable item) and whether the "not in the sources" instruction is present.
With ``--live``, ``chat_ai.create_chat_completion`` answers each question
(verification off, so the answer itself is scored) and every answer is written
to ``--out`` for review. ``--score`` re-scores a saved file.

An answer is scored by its text:

- answerable items: ``correct`` if every accept group matches; ``abstained``
  if it says the sources do not contain the answer and the terms do not
  match; otherwise ``wrong``. ``cited`` if it names the chunk_id of a chunk
  holding the answer span.
- not-in-sources items: ``abstained`` if it says the sources do not contain
  the answer, else ``answered``.

Term matching is a heuristic. Read the saved answers before quoting a rate.
Counts are reported as k/n per kind and page, with no headline percentage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

ABSTAIN_RE = re.compile(
    r"(?:do(?:es)?\s*n[o']?t|cannot|can't|did\s*n[o']?t)\s+(?:contain|mention|say|state|include|answer|provide|specify|address|tell|give)"
    r"|not\s+(?:mentioned|stated|found|specified|addressed|given|provided|recorded)"
    r"|not\s+in\s+the\s+(?:sources?|evidence|text|page|transcription|passage)"
    r"|no\s+(?:information|mention|evidence|passage|indication)",
    re.IGNORECASE,
)


def load_items(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def load_references(path: Path) -> dict[str, str]:
    return {page["page_id"]: page["reference"] for page in json.loads(path.read_text(encoding="utf-8"))["pages"]}


def answer_chunks(chunks: list[dict[str, Any]], reference: str, span: str) -> set[str]:
    """chunk_ids of the chunks holding *span* of *reference* (overlapping it, if none holds it whole)."""
    start = reference.index(span)
    end = start + len(span)
    holding = {c["chunk_id"] for c in chunks if c["start_offset"] <= start and end <= c["end_offset"]}
    return holding or {c["chunk_id"] for c in chunks if c["start_offset"] < end and start < c["end_offset"]}


def evidence_chunk_ids(evidence: str) -> set[str]:
    """Every chunk id the evidence blocks cite; a merged block lists several."""
    ids: set[str] = set()
    for line in str(evidence).splitlines():
        if line.startswith("chunk_id: "):
            ids.update(part.strip() for part in line[len("chunk_id: "):].split(",") if part.strip())
    return ids


def score_answer(item: dict[str, Any], text: str, relevant_chunk_ids: set[str]) -> dict[str, Any]:
    abstained = bool(ABSTAIN_RE.search(text or ""))
    if item["kind"] == "not_in_sources":
        return {"outcome": "abstained" if abstained else "answered"}
    lowered = (text or "").lower()
    correct = all(any(term.lower() in lowered for term in group) for group in item["accept"])
    outcome = "correct" if correct else ("abstained" if abstained else "wrong")
    return {"outcome": outcome, "cited": any(chunk_id in (text or "") for chunk_id in relevant_chunk_ids)}


def build_store(tmp: Path, references: dict[str, str], pages: set[str], *, embedder: str) -> dict[str, dict[str, Any]]:
    """One searchable run per page in a throwaway DB and vector store under *tmp*."""
    os.environ["ARCHAI_DB_PATH"] = str(tmp / "archai.sqlite")
    from app.config import settings
    from app.db import pipeline_db
    from app.services import rag_store
    from app.services.chunking import build_window_chunks

    pipeline_db._DB_READY = False
    settings.chroma_persist_dir = str(tmp / "chroma")
    rag_store._CLIENT = None
    if embedder == "local":
        settings.rag_embedding_model = ""
    runs: dict[str, dict[str, Any]] = {}
    for page_id in sorted(pages):
        reference = references[page_id]
        run_id = pipeline_db.create_run(asset_ref=f"eval:{page_id}", asset_sha256=f"eval-{page_id}")
        windows = build_window_chunks(
            reference,
            window_lines=int(settings.rag_chunk_window_lines),
            overlap_lines=min(int(settings.rag_chunk_overlap_lines), int(settings.rag_chunk_window_lines) - 1),
        )
        chunks = pipeline_db.insert_chunks(run_id, [{**w, "chunk_id": f"{page_id}:{w['idx']}"} for w in windows])
        pipeline_db.update_run_fields(run_id, search_allowed=1, ocr_text=reference)
        indexed = rag_store.index_run(run_id)
        runs[page_id] = {"run_id": run_id, "chunks": chunks, "index_status": indexed["status"]}
    return runs


def _context(run: dict[str, Any]) -> dict[str, Any]:
    return {"ocr_run_id": run["run_id"]}


def evaluate_retrieval(items, runs, references) -> list[dict[str, Any]]:
    from app.services import chat_ai

    rows = []
    for item in items:
        run = runs[item["page_id"]]
        messages = [{"role": "user", "content": item["question"]}]
        prepared, evidence = chat_ai._prepare_messages(messages, _context(run))
        system = prepared[0]["content"]
        retrieval = dict(getattr(evidence, "retrieval", {}) or {})
        row = {"id": item["id"], "page_id": item["page_id"], "kind": item["kind"], "retrieval": retrieval.get("status")}
        if item["kind"] == "answerable":
            relevant = answer_chunks(run["chunks"], references[item["page_id"]], item["answer_span"])
            row["evidence_holds_answer"] = bool(relevant & evidence_chunk_ids(evidence))
        else:
            row["evidence_blocks"] = evidence.count("[OCR_CHUNK_EVIDENCE]")
        row["not_in_sources_instruction"] = "sources for this page do not contain the answer" in system
        rows.append(row)
    return rows


def evaluate_live(items, runs, references, out: Path) -> list[dict[str, Any]]:
    from app.config import settings
    from app.services import chat_ai

    settings.paleography_verification_enabled = False
    rows = []
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            run = runs[item["page_id"]]
            result = chat_ai.create_chat_completion(
                messages=[{"role": "user", "content": item["question"]}], model=None, context=_context(run)
            )
            record = {
                "id": item["id"],
                "question": item["question"],
                "answer": result["text"],
                "model": result.get("model"),
                "retrieval": result.get("retrieval"),
                "relevant_chunk_ids": sorted(
                    answer_chunks(run["chunks"], references[item["page_id"]], item["answer_span"])
                )
                if item["kind"] == "answerable"
                else [],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append(record)
    return rows


def score_saved(items: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in items}
    rows = []
    for record in records:
        item = by_id[record["id"]]
        scored = score_answer(item, record["answer"], set(record.get("relevant_chunk_ids") or []))
        rows.append({"id": item["id"], "page_id": item["page_id"], "kind": item["kind"], **scored})
    return rows


def tally(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    """Counts of *field*'s values per kind, overall and per page."""
    out: dict[str, dict[str, Any]] = {}
    for kind in ("answerable", "not_in_sources"):
        picked = [row for row in rows if row["kind"] == kind and field in row]
        if not picked:
            continue
        counts: dict[str, int] = defaultdict(int)
        per_page: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in picked:
            counts[str(row[field])] += 1
            per_page[row["page_id"]][str(row[field])] += 1
        out[kind] = {"n": len(picked), "counts": dict(counts), "per_page": {p: dict(c) for p, c in sorted(per_page.items())}}
    return out


def _print_tally(title: str, table: dict[str, dict[str, Any]]) -> None:
    print(title)
    for kind, entry in table.items():
        cells = ", ".join(f"{value} {count}/{entry['n']}" for value, count in sorted(entry["counts"].items()))
        print(f"  {kind} (n={entry['n']}): {cells}")
        for page, counts in entry["per_page"].items():
            n = sum(counts.values())
            print(f"    {page}: " + ", ".join(f"{value} {count}/{n}" for value, count in sorted(counts.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--items", type=Path, default=BACKEND / "eval" / "chat" / "answers.json")
    parser.add_argument("--readings", type=Path, default=BACKEND / "eval" / "quality_gate" / "readings.json")
    parser.add_argument(
        "--embedder", choices=("local", "provider"), default="local", help="local: ChromaDB's default model; provider: the configured one"
    )
    parser.add_argument("--live", action="store_true", help="ask the configured chat model and score its answers")
    parser.add_argument("--out", type=Path, help="with --live, where to save the answers (JSONL)")
    parser.add_argument("--score", type=Path, help="score answers saved by an earlier --live run")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    items = load_items(args.items)
    if args.score:
        records = [json.loads(line) for line in args.score.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = score_saved(items, records)
        report = {"outcome": tally(rows, "outcome"), "cited": tally(rows, "cited")}
    else:
        references = load_references(args.readings)
        if args.live and not args.out:
            parser.error("--live needs --out, so the answers can be read")
        with tempfile.TemporaryDirectory(prefix="archai-chat-eval-") as tmp:
            runs = build_store(Path(tmp), references, {item["page_id"] for item in items}, embedder=args.embedder)
            if args.live:
                records = evaluate_live(items, runs, references, args.out)
                rows = score_saved(items, records)
                report = {"outcome": tally(rows, "outcome"), "cited": tally(rows, "cited")}
            else:
                rows = evaluate_retrieval(items, runs, references)
                report = {
                    "retrieval": tally(rows, "retrieval"),
                    "evidence_holds_answer": tally(rows, "evidence_holds_answer"),
                    "not_in_sources_instruction": tally(rows, "not_in_sources_instruction"),
                }
            report["index_status"] = {page: run["index_status"] for page, run in runs.items()}
    if args.json:
        print(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2))
        return 0
    for name, table in report.items():
        if name == "index_status":
            print("index status: " + ", ".join(f"{page} {status}" for page, status in table.items()))
        else:
            _print_tally(name, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
