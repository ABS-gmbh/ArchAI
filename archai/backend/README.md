# ArchAI Backend

FastAPI backend of the ArchAI application (`archai/`).

## Responsibilities

- document OCR and extraction endpoints
- evidence span persistence and retrieval
- grounded chat and authority-linking services
- analytics and model-routing endpoints

## Structure

| Path | Purpose |
|---|---|
| `app/main.py` | FastAPI app bootstrap and router registration |
| `app/routers/` | API route modules |
| `app/services/` | OCR, chat, retrieval, and support services |
| `app/agents/` | Agent-oriented orchestration components |
| `app/schemas/` | API request/response models |
| `tests/` | Backend tests |
| `.env.example` | Runtime environment template |

`app/routers/ocr.py` holds the OCR endpoints and the reading of a page
(segmentation, recognisers, retries). What is done with the text afterwards
lives in services the router calls:

| Module | What it does |
|---|---|
| `app/services/language_detection.py` | The page's language: langdetect checked against Latin and Old/Middle French anchors |
| `app/services/mention_extraction.py` | Entity mentions (people, places, works, dates) in diplomatic text |
| `app/services/mention_report.py` | The mention-extraction and consolidated reports of a traced run |
| `app/services/trace_pipeline.py` | Chunks, mentions, authority linking, the search decision and indexing of a run |

`tests/test_ocr_router_characterisation.py` checks these against what the
router produced before they were moved out of it.

## Setup

```bash
cd archai/backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
cd archai/backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/api/health
```

## Test

```bash
cd archai/backend
pip install -e ".[dev]"
python -m pytest tests -q
```

CI runs the whole suite on every push, without the inference stack
(ultralytics, torch, kraken): `python -m pytest -q -m "not requires_models"`.
Mark a test that needs model weights or that stack `@pytest.mark.requires_models`.
The suite runs offline: `tests/conftest.py` refuses every connection that is
not loopback, so stub the SAIA, GWDG, Ollama or Wikidata client in the test.

Five tests are expected failures (`xfail(strict=True)`), each with its reason:
four expect the SAIA OCR agent to try Qwen first while it reads with InternVL
only, and one expects it to request deterministic decoding, which it does not.

## Full-page OCR

`POST /ocr/extract_full_page` - what the workspace calls - segments the page
with the layout models, then reads it line by line with Kraken, CATMuS Medieval
first. Text outside every text column (marginalia, folio numbers, a library
stamp) comes back in `secondary_lines`, apart from the body in `text`. GLM-OCR
reads the whole page instead when the request asks for `glmocr`, when
`FULL_PAGE_OCR_ENGINE=glmocr`, or as a fallback when a page yields no text
regions or no text; `ocr_engine` in the response says which engine read it, and
an `OCR_ENGINE_FALLBACK:<reason>` warning says why a fallback happened.

Character error rate on the five reference pages in `eval/recognizers/`, and
over all of them (micro-averaged):

| Engine | Old French | Latin | Latin, Romanesque | Middle French | Old French verse | All |
|---|---|---|---|---|---|---|
| segmented, CATMuS Medieval (default) | 6.5% | 6.6% | 18.0% | 4.1% | 7.2% | 7.6% |
| segmented, CREMMA Medieval | 9.9% | 14.6% | 46.9% | 19.6% | 11.8% | 15.1% |
| segmented, McCATMuS | 34.3% | 38.8% | 41.0% | 40.0% | 35.4% | 36.5% |
| GLM-OCR, whole page | 25.6% | 27.6% | 29.9% | 24.5% | 29.2% | 27.7% |

Kraken reads each line crop in plain grayscale, as its models are trained; see
`_preprocess_kraken_crop_with_metadata` for what an earlier contrast stretch,
denoise and deskew cost.

Reproduce with the page images (matched by SHA-256, any file names):

```bash
python scripts/benchmark_recognizers.py --images /path/to/pages
```

## Measure accuracy

`scripts/evaluate_ocr.py` scores OCR output against reference transcriptions
and reports CER and WER, micro- and macro-averaged, with bootstrap confidence
intervals resampled over pages:

```bash
python scripts/evaluate_ocr.py --manifest eval/manifest.jsonl
python scripts/evaluate_ocr.py --manifest eval/manifest.jsonl --policy lenient --json
```

Each manifest line names a page and two text files, resolved relative to the
manifest:

```json
{"page_id": "fmb-cb-0001_001r", "reference": "gold/fmb-cb-0001_001r.gt.txt", "hypothesis": "runs/fmb-cb-0001_001r.txt"}
```

References should be diplomatic transcriptions, one manuscript line per line.
The `diplomatic` policy (default) keeps case, punctuation and abbreviation
marks; `lenient` also folds case and drops punctuation. Scoring two machine
runs against each other measures their agreement, not their accuracy.

Retrieval metrics (recall@k, precision@k, MRR, nDCG@k) live in
`app/services/evaluation.py` alongside the OCR metrics.

## Retrieval

Chunk retrieval is hybrid. The dense ranking from the embedding model is fused
by Reciprocal Rank Fusion with BM25 over character 4-grams of the
abbreviation-expanded search key (`app/services/lexical_retrieval.py`), so a
query meets OCR text that spells the same words differently. Each hit reports
`dense_rank`, `lexical_rank` and `fusion_score` next to the usual fields; set
`RAG_LEXICAL_FUSION=false` to fall back to dense retrieval only.

`scripts/benchmark_retrieval.py` measures this. No relevance judgments are
needed: it looks up passages from one OCR reading of a page in a different
reading of the same page, locating the right chunk by character alignment. It
reads either a pipeline DB or the reference pages of
`eval/quality_gate/readings.json`, and reports how many lookups each page gave:
a DB can hold one page read many times (the live one gives 1,451 of its 1,457
lookups from one page), so use `--readings` for a spread of pages.

```bash
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json
python scripts/benchmark_retrieval.py --db app/archai.sqlite
```

On the four reference pages long enough to chunk (222 lookups), with
ChromaDB's default model:

| | MRR | Hit@1 | Hit@5 |
|---|---|---|---|
| dense | 0.707 | 0.572 | 0.896 |
| BM25 over character 4-grams | 0.886 | 0.829 | 0.959 |
| fused (RRF, k = 60, equal weights) | 0.863 | 0.770 | **0.968** |

BM25 alone ranks the passage first more often, but the fusion puts it in the
top 5, the chunks chat passes to the model, more often than either ranker.

Passage lookup quotes the page. `eval/retrieval/questions.json` holds 35
questions a reader might ask about the same pages, in English, each with the
span of the reference that answers it; `--questions` asks them of every
reading (151 lookups):

```bash
python scripts/benchmark_retrieval.py --questions eval/retrieval/questions.json --embedder e5-local
```

| Questions | MRR | Hit@1 | Hit@5 |
|---|---|---|---|
| dense, e5 | 0.590 | 0.417 | 0.821 |
| dense, ChromaDB's default model | 0.449 | 0.272 | 0.722 |
| BM25 over character 4-grams | 0.382 | 0.305 | 0.490 |
| fused, e5 | 0.620 | 0.464 | 0.815 |

Here the dense side carries the fusion. `RAG_LEXICAL_WEIGHT` sets BM25's weight
in the fusion (the dense ranking's is 1); `--lexical-weight` and `--rrf-k` try
other settings on the benchmark. At 2, passage lookup gains (MRR 0.863 to
0.903) but questions with e5 do not (MRR -0.011, Hit@5 -0.007), so it stays 1.
latin-apocalypse has four chunks, so every ranker puts its answer in the top
5 there; MRR is the measure that separates them on that page.

The configured embedding model, `multilingual-e5-large-instruct`, expects every
query to state its task: `Instruct: <task>\nQuery: <text>`. Documents are
embedded as they are. `rag_store.embedding_input` adds the instruction to
queries only, so stored vectors stay valid. The task is the model card's own
retrieval sentence. On passage lookup its effect depends on the page: it
raised dense MRR from 0.510 to 0.628 on the live DB (one page), and left it
unchanged on the reference pages (0.690 bare, 0.683 with it) and on the
questions (0.593 bare, 0.590 with it). `--embedder
e5-local` runs the model on this machine (sentence-transformers) when the
provider cannot be reached:

```bash
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --embedder e5-local
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --embedder e5-local --no-query-instruction
```

### What is searched

Only searchable runs (`pipeline_db.searchable_runs`):

- **The quality gate allowed search on the run.** The pipeline records the
  decision on the run (`search_allowed`) before it indexes, and every path into
  the vector store reads it back, including the auto-indexing a chat query
  triggers. Runs from before the column fall back to their latest quality
  report; a run with neither is never searched.
- **Without a run filter, only the newest such run of each page.** A page is
  its image hash, or its `asset_ref` when none was recorded. Indexing a re-run
  deletes the earlier runs' vectors. A chat scoped to an older run still
  searches that run.

One embedding space: the configured `RAG_EMBEDDING_MODEL`, or ChromaDB's
default when it is empty. When the provider cannot embed, nothing is indexed
into another space, and queries are answered from BM25 alone, which reads the
chunks from the pipeline database. Results carry `mode` (`hybrid`, `dense`,
`lexical`, `none`) and `notes` saying why a ranker was left out.

Two maintenance scripts bring an existing store in line. Both only report
unless given `--apply`:

```bash
# Grade runs from before the decision was recorded with the current gate.
python scripts/regrade_runs.py --apply
# Delete vectors retrieval can no longer use; optionally drop other spaces.
python scripts/prune_rag_index.py --apply --drop-other-spaces
```

## Environment

Copy `.env.example` to `.env` (or `.env.local`) and set API keys before running
chat/OCR model-backed routes.
