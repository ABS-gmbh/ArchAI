# ArchAI Vendor Layout Backend

FastAPI backend for the `archai/vendor/layout` workspace.

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

## Setup

```bash
cd archai/vendor/layout/backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
cd archai/vendor/layout/backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/api/health
```

## Test

```bash
cd archai/vendor/layout/backend
pip install -e ".[dev]"
python -m pytest tests -q
```

CI runs the dependency-light subset (evaluation, chunking, medieval text,
lexical retrieval, retrieval benchmark) on every push; the rest of the suite
needs the inference stack.

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

`scripts/benchmark_retrieval.py` measures this on the pipeline DB. No relevance
judgments are needed: it looks up passages from one OCR reading of a page in a
different reading of the same page, locating the right chunk by character
alignment.

```bash
python scripts/benchmark_retrieval.py --db app/archai.sqlite
```

## Environment

Copy `.env.example` to `.env` (or `.env.local`) and set API keys before running
chat/OCR model-backed routes.
