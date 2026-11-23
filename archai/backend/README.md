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

## Configure

For a server, see [`docs/deployment.md`](../../docs/deployment.md): run one
worker, what to set, what to put in front, what to back up.

```bash
cd archai/backend
cp .env.example .env
```

Settings are read from `.env`, `.env.local` and the environment: the
environment overrides `.env.local`, which overrides `.env`. `app/config.py`
holds every setting and its default. A value of the wrong type or out of range stops startup with an
error that names the variable. The ones to look at first:

| Variable | What it does |
|---|---|
| `CHAT_AI_API_KEY` | SAIA / Chat AI key for chat, translation and provider embeddings |
| `GLMOCR_OLLAMA_HOST` | Ollama server running `glm-ocr`, for GLM-OCR and the segmented engine's fallback |
| `ANALYTICS_USERNAME`, `ANALYTICS_PASSWORD` | Analytics login; with either empty the login refuses with 503 |
| `JWT_SECRET` | Signs analytics tokens; empty, a random secret is made per process and tokens end at restart |
| `ARCHAI_DB_PATH`, `CHROMA_PERSIST_DIR` | Pipeline database and vector store; by default inside this directory |
| `MAX_UPLOAD_MEGABYTES`, `TASK_TTL_MINUTES` | Upload limit, and how long layout task files are kept |
| `CHAT_AI_TIMEOUT_SECONDS` | How long the chat client waits for an answer (default 180) |
| `LOG_LEVEL`, `LOG_FORMAT` | Log level, and `text` or `json` lines on stderr |
| `ENABLE_TEST_OCR_FIXTURES` | Canned answers for four test images; keep it `false` |
| `ENABLE_API_DOCS` | `true` serves `/docs`, `/redoc` and `/openapi.json`; set `false` on a server |
| `API_TOKEN` | Off when empty. When set (24+ characters), every route but `/api/health` needs `X-API-Token: <token>` or `Authorization: Bearer <token>`; the two event streams also accept `?token=` |

## Run

```bash
cd archai/backend
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

The frontend expects the backend at 127.0.0.1:8010; `NEXT_PUBLIC_BACKEND_URL` (the browser's direct URL) and `BACKEND_URL` (the Next server's `/api` proxy) override it; `BACKEND_API_TOKEN` gives that proxy the backend's `API_TOKEN`.

Health endpoint:

```bash
curl http://127.0.0.1:8010/api/health
```

`status` is `ok` when every check passes; `degraded` (HTTP 200) when Ollama is
unreachable or no chat key is configured, so GLM-OCR or chat will fail; and
`unavailable` (HTTP 503) when the pipeline database cannot be used. Each entry
under `checks` (`database`, `ollama`, `chat_api_key`) has `ok` and a short
`detail`; the endpoint is unauthenticated, so error text and hosts go to the
log, not the response. The chat key's presence is checked, not its validity.

## Retention

Pipeline runs accumulate: every analysis of a page adds a run with its chunks,
mentions, decisions and vectors. `scripts/prune_runs.py` removes runs past a
window. Without `--apply` it only reports, so run it that way first to preview
a policy:

```bash
python scripts/prune_runs.py --older-than-days 180            # what would go, and why the rest stays
python scripts/prune_runs.py --older-than-days 180 --apply    # delete those runs
```

A run is kept if it was updated within the window, has not finished, has no
page identity (no image hash and no asset reference), is its page's newest run,
newest completed run or the run search serves, has an OCR benchmark reference,
carries an annotation
(an evidence span posted to `/api/evidence/spans`; `--include-annotated`
overrides this) or is named in the thesis showcase bundle. Deleting a run
removes its vectors first, then the run and everything recorded under it;
shared authority records stay. `--vacuum` compacts the database afterwards and
locks it while it runs.

The running server can do the same on a schedule, and does not by default.
With `RUN_RETENTION_DAYS` set to a number of days (0, the default, turns it
off), every 15-minute sweep deletes at most `RUN_RETENTION_MAX_PER_SWEEP` (50)
expired runs, oldest first, never an annotated one, logging each deletion and a
summary. The first sweep after startup only logs the runs the next sweep will
delete.

## Type checking

`pyproject.toml` has a non-strict mypy configuration for `app/` (audit HYG-15).
It is not a CI check; whether to make it one is still open. From this directory:

```bash
mypy
```

Baseline on 2026-11-23 (mypy 2.4, Python 3.10 target): 89 errors in 26 files,
none of them a confirmed runtime defect. By kind: 49 `arg-type` (mostly a `str`
passed where a `Literal` is declared: script hints, languages and statuses in
the OCR schemas), 2 stale `# type: ignore` comments (in chat_ai and rag_store),
11 `assignment`, 10
`attr-defined` (`object` values iterated, numpy and Pillow stubs), and 17 others.
By module, leaving the stale comments out: saia_ocr_agent 13, ocr_agent 11,
rag_store 8, label_analysis_agent 7, ocr_text_parsing 6, ocr_backends 5,
routers/ocr 5, trace_pipeline 4, ocr_image_prep 4, pipeline_db 3, and 1 or 2
in each of 15 other modules. Untyped function bodies are not checked, so the
count covers annotated code only.

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

`scripts/smoke_api.py` checks a running server over real HTTP: every route
with valid, invalid and oversized input, streams, a client disconnecting and
`/api/health` under 20 open streams. Start a backend on scratch data with no
`.env` (the script's docstring has the command), then:

```bash
python scripts/smoke_api.py --base http://127.0.0.1:8765 --db /tmp/smoke/archai.sqlite
```

It exits 1 if any route answers differently from what it expects. Never
point it at a live backend: it writes runs, tasks and evidence spans.

Four tests are expected failures (`xfail(strict=True)`): they expect the SAIA
OCR agent to try Qwen first, while it reads with InternVL only.

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

Which model reads a page: `select_backend_plan` puts CATMuS Medieval first
under every language hint, and with none. Year, script family and place are
not consulted, and backend election (`OCR_BACKEND_ELECTION`) is off. A later
model reads a line only when every earlier one fails or reads nothing. So
for all five pages the routed model is CATMuS Medieval, and it is also the
best reading of each page. Three references (Latin Romanesque, Middle French,
Old French verse) were transcribed with a CATMuS draft beside the image, which
flatters CATMuS on those pages (see `eval/quality_gate/README.md`). On the two
pages read off the image alone, it still leads (6.5% and 6.6%).

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
top 5 more often than either ranker. Chat now passes the top 8
(`RAG_TOP_K`), with overlapping hits merged into one evidence block; see
`eval/chat/README.md`.

Passage lookup quotes the page. `eval/retrieval/questions.json` holds
questions a reader might ask about the same pages, in English, each with the
span of the reference that answers it; `--questions` asks them of every
reading. The table is the first 35 (`--question-set 2026-10`, 151 lookups):

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
With ChromaDB's default model (no `RAG_EMBEDDING_MODEL`), BM25 weighs 2
instead (`RAG_LEXICAL_WEIGHT_LOCAL`); in that configuration `RAG_LEXICAL_WEIGHT`
is not read, so set `RAG_LEXICAL_WEIGHT_LOCAL` to change it. Against weight 1 on that model, MRR rises
+0.037 [+0.011, +0.063] over 270 question lookups (both question sets, see
`eval/retrieval/README.md`) and +0.040 [+0.019, +0.063] over 222 passage lookups.
latin-apocalypse has four chunks, so every ranker puts its answer in the top
5 there; MRR is the measure that separates them on that page.

There is no relevance floor. On 17 questions the pages do not answer (77
lookups), the top hit scores no lower than on the answerable ones: the AUC is
0.44 for dense cosine, 0.47 for its z-score and 0.51 for BM25 with ChromaDB's
model (`--separation`, see `eval/retrieval/README.md`). e5 is not measured yet.

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
