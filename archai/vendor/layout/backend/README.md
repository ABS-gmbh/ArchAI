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

## Environment

Copy `.env.example` to `.env` (or `.env.local`) and set API keys before running
chat/OCR model-backed routes.
