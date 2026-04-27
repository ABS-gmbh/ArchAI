# ArchAI

ArchAI is a unified research and engineering repository for medieval manuscript OCR,
layout analysis, grounded document chat, and thesis showcase artifacts.

This repository intentionally keeps all project tracks in one place on `master`:

- Root OCR CLI pipeline (`src/archai_ocr`)
- Monorepo workspace (`archai/`) with scaffolded backend/frontend and vendor integrations
- Thesis showcase outputs (`artifacts/thesis_showcase`)

## Repository Contents

| Path | Purpose | Status |
|---|---|---|
| `pyproject.toml` | Root Python package for OCR CLI (`archai`, `archai-ocr`) | Active |
| `src/archai_ocr/` | OCR pipeline modules (layout, crop, Kraken recognition, text assembly) | Active |
| `scripts/` | OCR/model/thesis helper scripts | Active |
| `config.example.yaml` | Root OCR defaults (weights/layout/runtime) | Active |
| `.env.example` | Root env defaults for model paths/output | Active |
| `data/` | Project data and evidence resources | Active |
| `docs/` | Root technical documentation | Active |
| `artifacts/thesis_showcase/` | Thesis figure payloads, screenshots, manifests | Active |
| `archai/` | Workspace monorepo (frontend/backend/vendor/docker/docs) | Mixed |
| `weights/` | Local model files (ignored runtime assets) | Runtime |
| `outputs/` | Local OCR run outputs/debug files (ignored runtime assets) | Runtime |

Detailed map: [`docs/repository_map.md`](docs/repository_map.md)

## Quick Start

### 1) Root OCR pipeline

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
python -m archai_ocr.cli --image sample.png --config config.example.yaml
```

You can also use installed entrypoints:

- `archai --image sample.png --config config.example.yaml`
- `archai-ocr --image sample.png --config config.example.yaml`

### 2) Vendor Document + Chat workspace (`archai/vendor/layout`)

Backend:

```bash
cd archai/vendor/layout/backend
pip install -e .
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd archai/vendor/layout/frontend
npm install
npm run dev
```

### 3) Vite Vue workspace (`archai/frontend`)

```bash
cd archai/frontend
npm install
npm run dev
```

## Thesis Artifacts

`artifacts/thesis_showcase/` holds reproducible figure datasets and screenshots for
Figures 24/25/26/29/30. Use:

```bash
python scripts/build_thesis_showcase_payloads.py --verify
```

to validate bundle integrity.

## Repository Hygiene

Tracked:

- Source code
- Config templates (`*.example`, docs, manifests)
- Curated thesis artifacts

Ignored:

- Virtual environments and local caches
- Runtime outputs/models (`outputs/`, `weights/`, `.tasks/`, local DB/runtime dirs)
- Local agent/editor workspaces (`.claude/`)

## Branch And Naming Policy

- Canonical branch: `master`
- Project/repository identity: `ArchAI`

GitHub rename command (requires authenticated `gh`):

```bash
gh repo rename -R mohamedbasuony/thesis-project ArchAI --yes
git remote set-url origin https://github.com/mohamedbasuony/ArchAI.git
git push -u origin master
```
