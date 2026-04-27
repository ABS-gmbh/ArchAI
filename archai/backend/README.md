# ArchAI Prototype Backend

This package is the prototype backend track for the `archai/` workspace.

It exposes a small FastAPI service that supports:

- image ingest (`/ingest/image`)
- async pipeline job status (`/jobs/{job_id}`)
- evidence span lookup (`/evidence/span/{span_id}`)
- basic citation-aware chat response (`/chat`)

## Layout

| Path | Purpose |
|---|---|
| `src/archai_backend/api/main.py` | FastAPI app and routes |
| `src/archai_backend/store/db.py` | SQLite storage and repository helpers |
| `src/archai_backend/config/` | Paths and logging config |
| `scripts/` | Placeholder runner scripts |
| `tests/` | Placeholder test directory |

## Setup

```bash
cd archai/backend
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Run

```bash
cd archai/backend
uvicorn archai_backend.api.main:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```
