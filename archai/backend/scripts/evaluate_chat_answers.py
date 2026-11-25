"""Evaluate chat answers on the reference pages: grounded, cited, or "not in the sources".

Usage::

    python scripts/evaluate_chat_answers.py                       # retrieval only, no chat model
    python scripts/evaluate_chat_answers.py --embedder e5-local   # the same, with e5 on this machine
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

``--embedder`` picks the embedding space: ``local`` (ChromaDB's MiniLM, the
default), ``provider`` (the configured model through the provider) or
``e5-local`` (multilingual-e5-large-instruct run here with sentence-transformers,
the configured model family without the provider). With ``e5-local`` the store
embeds through that model, and queries get the e5-instruct instruction exactly
as with the provider, since ``rag_embedding_model`` is set to its name.

Without ``--live`` no chat model is called. The report then says, per item,
whether the evidence put in front of the model holds the answer span (for an
answerable item) and whether the "not in the sources" instruction is present.
With ``--live``, ``chat_ai.create_chat_completion`` answers each question
(verification off, so the answer itself is scored) and every answer is written
to ``--out`` for review. ``--score`` re-scores a saved file.

An answer is scored by its text:

- answerable items: ``correct`` if every accept group matches and the answer
  does not open by saying the sources lack the answer; ``hedged`` if it opens
  that way but the terms match (read these: some quote and give the answer,
  some only refuse); ``abstained`` if it says the sources do not contain the
  answer and the terms do not match; otherwise ``wrong``. ``cited`` if it cites a label ([1], [2], ...)
  whose evidence block covers a chunk holding the answer span; for answers
  saved before labels (no ``sources`` in the record), if it names the
  chunk_id of such a chunk.
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
import time
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


# A sentence that says the sources do not hold the answer: narrower than
# ABSTAIN_RE, which also matches an answer such as "No: he is not given to
# return".
SOURCE_REFUSAL_RE = re.compile(
    r"\b(?:text|texts|sources?|evidence|manuscript|passages?|page|transcription|excerpts?)\b[^.!?]{0,60}?"
    r"\b(?:do(?:es)?\s*n[o']?t|cannot|can't)\s+(?:contain|mention|provide|say|state|specify|address|answer|include|give|tell)"
    r"|\bnot\s+in\s+the\s+(?:sources?|evidence|text|page|transcription|passage)",
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


def score_answer(
    item: dict[str, Any], text: str, relevant_chunk_ids: set[str], sources: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    abstained = bool(ABSTAIN_RE.search(text or ""))
    if item["kind"] == "not_in_sources":
        return {"outcome": "abstained" if abstained else "answered"}
    lowered = (text or "").lower()
    correct = all(any(term.lower() in lowered for term in group) for group in item["accept"])
    # An answer that opens by saying the sources do not hold the answer is not
    # scored correct on its terms alone: "does not contain" holds "not", and a
    # restated question holds its own words. If the terms are there it is
    # "hedged" (some go on to quote and give the answer, some only refuse:
    # read them), otherwise "abstained".
    opening = re.split(r"(?<=[.!?])\s", (text or "").strip(), maxsplit=1)[0]
    if SOURCE_REFUSAL_RE.search(opening):
        outcome = "hedged" if correct else "abstained"
    else:
        outcome = "correct" if correct else ("abstained" if abstained else "wrong")
    if sources is None:  # saved before citation labels
        cited = any(chunk_id in (text or "") for chunk_id in relevant_chunk_ids)
    else:
        cited = any(s.get("cited") and relevant_chunk_ids & set(s.get("chunk_ids") or []) for s in sources)
    return {"outcome": outcome, "cited": cited}


E5_LOCAL_MODEL = "intfloat/multilingual-e5-large-instruct"
# The checkpoint is float16, and sentence-transformers 6 loads it as such: on a
# CPU that encodes about 5 times slower than float32, with the same vectors
# (cosine 1.000 on the evaluation VM).
_FLOAT32 = {"dtype": "float32"}


def _use_local_e5() -> None:
    """Embed through multilingual-e5-large-instruct on this machine instead of the provider."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("error: --embedder e5-local needs sentence-transformers installed") from exc
    from app.config import settings
    from app.services import rag_store

    model = SentenceTransformer(E5_LOCAL_MODEL, model_kwargs=_FLOAT32)
    settings.rag_embedding_model = E5_LOCAL_MODEL
    rag_store._embed = lambda texts: [
        [float(x) for x in v] for v in model.encode(list(texts), batch_size=16, normalize_embeddings=True)
    ]


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
    elif embedder == "e5-local":
        _use_local_e5()
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


REQUESTS: dict[str, int] = defaultdict(int)
# The chat route's default (ChatCompletionRequest.temperature); the workspace sends no other.
TEMPERATURE = 0.2


def _count_requests() -> None:
    """Count every HTTP request the provider client sends, by path (embeddings, chat/completions, models).

    The openai package sends through httpx or, in recent versions, its fork
    httpx2; both clients are counted.
    """
    import importlib

    for name in ("httpx", "httpx2"):
        try:
            client_class = importlib.import_module(name).Client
        except (ImportError, AttributeError):
            continue
        real_send = client_class.send

        def send(self, request, *args, _real=real_send, **kwargs):  # type: ignore[no-untyped-def]
            path = str(request.url.path).rstrip("/").rsplit("/v1/", 1)[-1]
            REQUESTS[f"{CURRENT_PHASE[0]}:{path}"] += 1
            return _real(self, request, *args, **kwargs)

        client_class.send = send  # type: ignore[method-assign]


CURRENT_PHASE = ["index"]
USAGE: list[dict[str, Any]] = []
_SENT: list[float] = []


def _pace(limit: int = 9, window: float = 10.0) -> None:
    """Wait so that at most *limit* chat requests go out in any *window* seconds (the provider allows 10)."""
    now = time.monotonic()
    while _SENT and now - _SENT[0] >= window:
        _SENT.pop(0)
    if len(_SENT) >= limit:
        time.sleep(window - (now - _SENT[0]) + 0.05)
    _SENT.append(time.monotonic())
# --no-thinking: ask Qwen 3.x-style models to answer without a reasoning pass
# (chat_template_kwargs.enable_thinking=false). The app does not send this yet.
NO_THINKING = [False]


def _record_usage() -> None:
    """Keep the provider's token usage of every non-streamed chat completion (``USAGE``), to give tokens per second."""
    from app.services import chat_ai

    real_create_client = chat_ai._create_client

    def create_client():  # type: ignore[no-untyped-def]
        # No automatic retries, so the request counter equals the calls made;
        # requests are paced under the provider's 10 per 10 seconds instead.
        client = real_create_client().with_options(max_retries=0)
        completions = client.chat.completions
        real_create = completions.create

        def create(*args, **kwargs):  # type: ignore[no-untyped-def]
            _pace()
            if NO_THINKING[0]:
                extra = dict(kwargs.get("extra_body") or {})
                extra["chat_template_kwargs"] = {**dict(extra.get("chat_template_kwargs") or {}), "enable_thinking": False}
                kwargs["extra_body"] = extra
            response = real_create(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None and not kwargs.get("stream"):
                USAGE.append({"model": kwargs.get("model"), "completion_tokens": getattr(usage, "completion_tokens", None),
                              "prompt_tokens": getattr(usage, "prompt_tokens", None)})
            return response

        completions.create = create
        return client

    chat_ai._create_client = create_client


def _cache_embeddings(path: Path) -> None:
    """Keep every embedding by model and text in *path*, so repeated runs retrieve the same evidence and spend no embedding request twice."""
    from app.config import settings
    from app.services import rag_store

    cache: dict[str, list[float]] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    real = rag_store._embed

    def embed(texts: list[str]) -> list[list[float]]:
        # What the app itself would send for this call (batches of 64), cached or not.
        REQUESTS[f"{CURRENT_PHASE[0]}:embeddings, as the app sends them"] += -(-len(texts) // rag_store._EMBED_BATCH_SIZE)
        model = str(settings.rag_embedding_model or "local")
        keys = [f"{model}\n{text}" for text in texts]
        missing = [text for text, key in zip(texts, keys) if key not in cache]
        if missing:
            for text, vector in zip(missing, real(missing)):
                cache[f"{model}\n{text}"] = [float(x) for x in vector]
            path.write_text(json.dumps(cache), encoding="utf-8")
        return [cache[key] for key in keys]

    rag_store._embed = embed


def evaluate_live(
    items, runs, references, out: Path, *, model: str | None = None, verify: bool = False, resume: bool = False, stream_first: int = 0
) -> list[dict[str, Any]]:
    from app.config import settings
    from app.services import chat_ai

    settings.paleography_verification_enabled = verify
    done = {json.loads(line)["id"]: json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()} if resume and out.is_file() else {}
    rows = [done[item["id"]] for item in items if item["id"] in done]
    with out.open("a" if resume else "w", encoding="utf-8") as handle:
        for item in items:
            if item["id"] in done:
                continue
            CURRENT_PHASE[0] = "question"
            run = runs[item["page_id"]]
            started = time.perf_counter()
            first_token = None
            usage_before = len(USAGE)
            messages = [{"role": "user", "content": item["question"]}]
            if items.index(item) < stream_first:
                # Streamed, as the workspace asks: the time to the first token is what a reader waits.
                result = {}
                for event in chat_ai.stream_chat_completion(messages=messages, model=model, temperature=TEMPERATURE, context=_context(run)):
                    if event.get("type") == "delta" and event.get("delta") and first_token is None:
                        first_token = time.perf_counter() - started
                    if event.get("type") == "done":
                        result = event
            else:
                result = chat_ai.create_chat_completion(messages=messages, model=model, temperature=TEMPERATURE, context=_context(run))
            record = {
                "id": item["id"],
                "question": item["question"],
                # The answer proper (``answer``), without the [Verification] and
                # [Retrieval] appendices ``text`` keeps; older backends had text only.
                "answer": result.get("answer") or result["text"],
                "text": result["text"],
                "model": result.get("model"),
                "verification": result.get("verification"),
                "retrieval": result.get("retrieval"),
                "sources": result.get("sources"),
                "seconds": round(time.perf_counter() - started, 2),
                "first_token_seconds": round(first_token, 2) if first_token is not None else None,
                "streamed": items.index(item) < stream_first,
                "temperature": TEMPERATURE,
                "thinking": "off" if NO_THINKING[0] else "model default",
                "usage": USAGE[usage_before:],
                "relevant_chunk_ids": sorted(
                    answer_chunks(run["chunks"], references[item["page_id"]], item["answer_span"])
                )
                if item["kind"] == "answerable"
                else [],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            rows.append(record)
    return rows


def verify_saved(items, runs, records: list[dict[str, Any]], out: Path, *, verifier: str) -> list[dict[str, Any]]:
    """Run *verifier* on answers saved by an earlier --live run, with the evidence that answer saw; nothing is re-answered."""
    from app.config import settings
    from app.services import chat_ai

    settings.paleography_verification_enabled = True
    settings.paleography_verification_model = verifier
    by_id = {item["id"]: item for item in items}
    checks = []
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            item = by_id[record["id"]]
            context = _context(runs[item["page_id"]])
            CURRENT_PHASE[0] = "verify"
            _prepared, evidence = chat_ai._prepare_messages([{"role": "user", "content": item["question"]}], context)
            started = time.perf_counter()
            verdict = chat_ai._run_paleography_verification(
                question=item["question"], draft_text=record["answer"], context=context, evidence_text=evidence,
                answer_model=str(record.get("model") or ""),
            ) or {}
            check = {
                "id": record["id"],
                "answer_model": record.get("model"),
                "verifier": verdict.get("model_used"),
                "assessment": verdict.get("assessment"),
                "self_check": verdict.get("self_check"),
                "seconds": round(time.perf_counter() - started, 2),
                "corrected_answer": verdict.get("corrected_answer"),
                "notes": verdict.get("notes"),
            }
            handle.write(json.dumps(check, ensure_ascii=False) + "\n")
            handle.flush()
            checks.append(check)
    return checks


def verifier_report(items, answer_rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> dict[str, Any]:
    """How the verifier's assessments line up with the scored answers: flags on wrong ones, mark-downs on correct ones."""
    outcome = {row["id"]: row["outcome"] for row in answer_rows}
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for check in checks:
        table[outcome[check["id"]]][str(check["assessment"])] += 1
    seconds = sorted(float(c["seconds"]) for c in checks)
    return {
        "by_outcome": {k: dict(v) for k, v in table.items()},
        "seconds_median": seconds[len(seconds) // 2] if seconds else None,
        "seconds_p90": seconds[min(len(seconds) - 1, int(0.9 * len(seconds)))] if seconds else None,
        "verifiers": sorted({str(c["verifier"]) for c in checks}),
    }


def score_saved(items: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in items}
    rows = []
    for record in records:
        item = by_id[record["id"]]
        scored = score_answer(item, record["answer"], set(record.get("relevant_chunk_ids") or []), record.get("sources"))
        rows.append({"id": item["id"], "page_id": item["page_id"], "kind": item["kind"], **scored})
    return rows


def timing(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Seconds per answer (median, 90th percentile), seconds to the first streamed token, and tokens per second."""

    def quantile(values: list[float], q: float) -> float | None:
        values = sorted(values)
        return round(values[min(len(values) - 1, int(q * len(values)))], 2) if values else None

    seconds = [float(r["seconds"]) for r in records if r.get("seconds") is not None]
    first = [float(r["first_token_seconds"]) for r in records if r.get("first_token_seconds") is not None]
    rates = []
    for r in records:
        tokens = sum(int(u.get("completion_tokens") or 0) for u in r.get("usage") or [] if u.get("model") == r.get("model"))
        if tokens and r.get("seconds") and not r.get("streamed"):
            rates.append(tokens / float(r["seconds"]))
    return {
        "answers": len(seconds),
        "seconds_median": quantile(seconds, 0.5),
        "seconds_p90": quantile(seconds, 0.9),
        "first_token_n": len(first),
        "first_token_median": quantile(first, 0.5),
        "first_token_p90": quantile(first, 0.9),
        "tokens_per_second_median": quantile(rates, 0.5),
        "models": sorted({str(r.get("model")) for r in records}),
    }


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
        "--embedder",
        choices=("local", "provider", "e5-local"),
        default="local",
        help=f"local: ChromaDB's default model; provider: the configured one; e5-local: {E5_LOCAL_MODEL} on this machine",
    )
    parser.add_argument("--live", action="store_true", help="ask the configured chat model and score its answers")
    parser.add_argument("--out", type=Path, help="with --live, where to save the answers (JSONL)")
    parser.add_argument("--score", type=Path, help="score answers saved by an earlier --live run")
    parser.add_argument("--model", help="with --live, the chat model to ask (default: the configured one)")
    parser.add_argument("--verify", action="store_true", help="with --live, run the configured verifier on each answer")
    parser.add_argument("--resume", action="store_true", help="with --live, keep the answers already in --out and ask only the rest")
    parser.add_argument("--embed-cache", type=Path, help="keep embeddings by text in this file and reuse them")
    parser.add_argument("--stream-first", type=int, default=0, help="with --live, stream the first N questions to time the first token")
    parser.add_argument("--limit", type=int, help="with --live, ask only the first N questions")
    parser.add_argument("--verify-saved", type=Path, help="run --verifier on the answers of this saved file (no new answers); results to --out")
    parser.add_argument("--verifier", help="with --verify-saved, the verifier model")
    parser.add_argument("--no-thinking", action="store_true", help="with --live, send enable_thinking=false (Qwen 3.x models)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    items = load_items(args.items)
    if args.score:
        records = [json.loads(line) for line in args.score.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = score_saved(items, records)
        report = {"outcome": tally(rows, "outcome"), "cited": tally(rows, "cited"), "timing": timing(records)}
    else:
        references = load_references(args.readings)
        if args.live and not args.out:
            parser.error("--live needs --out, so the answers can be read")
        with tempfile.TemporaryDirectory(prefix="archai-chat-eval-") as tmp:
            _count_requests()
            if args.embed_cache:
                if args.embedder == "e5-local":
                    _use_local_e5()
                _cache_embeddings(args.embed_cache)
            runs = build_store(Path(tmp), references, {item["page_id"] for item in items}, embedder=args.embedder)
            if args.verify_saved:
                if not (args.out and args.verifier):
                    parser.error("--verify-saved needs --verifier and --out")
                _record_usage()
                NO_THINKING[0] = args.no_thinking
                saved = [json.loads(line) for line in args.verify_saved.read_text(encoding="utf-8").splitlines() if line.strip()]
                checks = verify_saved(items, runs, saved, args.out, verifier=args.verifier)
                report = {"verifier": verifier_report(items, score_saved(items, saved), checks)}
                rows = checks
            elif args.live:
                _record_usage()
                NO_THINKING[0] = args.no_thinking
                if args.limit:
                    items = items[: args.limit]
                records = evaluate_live(
                    items, runs, references, args.out, model=args.model, verify=args.verify, resume=args.resume, stream_first=args.stream_first
                )
                rows = score_saved(items, records)
                report = {"outcome": tally(rows, "outcome"), "cited": tally(rows, "cited"), "timing": timing(records)}
            else:
                rows = evaluate_retrieval(items, runs, references)
                report = {
                    "retrieval": tally(rows, "retrieval"),
                    "evidence_holds_answer": tally(rows, "evidence_holds_answer"),
                    "not_in_sources_instruction": tally(rows, "not_in_sources_instruction"),
                }
            report["index_status"] = {page: run["index_status"] for page, run in runs.items()}
            report["requests"] = dict(REQUESTS)
    if args.json:
        print(json.dumps({"report": report, "rows": rows}, ensure_ascii=False, indent=2))
        return 0
    for name, table in report.items():
        if name == "index_status":
            print("index status: " + ", ".join(f"{page} {status}" for page, status in table.items()))
        elif name == "verifier":
            print("verifier: " + json.dumps(table, ensure_ascii=False))
        elif name == "timing":
            print("timing: " + ", ".join(f"{key} {value}" for key, value in table.items()))
        elif name == "requests":
            print("provider requests: " + ", ".join(f"{key} {count}" for key, count in sorted(table.items())))
        else:
            _print_tally(name, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
