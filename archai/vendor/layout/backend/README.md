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

CI runs the dependency-light subset (evaluation, chunking, medieval text) on
every push; the rest of the suite needs the inference stack.

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

## Environment

Copy `.env.example` to `.env` (or `.env.local`) and set API keys before running
chat/OCR model-backed routes.
