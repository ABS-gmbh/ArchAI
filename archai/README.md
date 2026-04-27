# ArchAI Workspace (`archai/`)

This directory is the monorepo workspace inside the root ArchAI repository.

It contains three tracks:

1. `backend/` and `frontend/`: prototype implementation track.
2. `vendor/layout/`: active full-stack document + chat + OCR workspace.
3. Shared workspace assets/config/data/docs/docker files.

## Directory Guide

| Path | Purpose | Status |
|---|---|---|
| `assets/` | Prompt templates and model manifests | Active |
| `config/` | Model configuration JSON | Active |
| `data/` | Workspace data folders (`raw`, `processed`, `derived`, `indexes`, `exports`) | Active |
| `docs/` | Workspace architecture and schema docs | In progress |
| `docker/` | Dockerfiles and compose setup | Active |
| `backend/` | Prototype FastAPI backend package `archai_backend` | Prototype |
| `frontend/` | Prototype Vue/Vite UI | Prototype |
| `vendor/layout/` | Main integrated document workspace (FastAPI + Next.js) | Active |

## Development Entry Points

- Prototype backend: [`backend/README.md`](backend/README.md)
- Prototype frontend: [`frontend/README.md`](frontend/README.md)
- Vendor layout app: [`vendor/layout/README.md`](vendor/layout/README.md)

## Make Targets

Use the local workspace `Makefile` for common commands:

```bash
cd archai
make help
```
