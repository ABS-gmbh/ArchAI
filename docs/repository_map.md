# ArchAI Repository Map

This document explains the structure of the full `ArchAI` repository and clarifies
which parts are active, experimental, or runtime-only.

## Top-Level Layout

| Path | Description | Classification |
|---|---|---|
| `.env.example` | Root environment defaults for OCR weights/output paths | Config template |
| `config.example.yaml` | Root OCR pipeline configuration | Config template |
| `pyproject.toml` | Root Python package metadata (`archai`) | Build config |
| `src/archai_ocr/` | Root OCR pipeline implementation | Active code |
| `scripts/` | Utility scripts for OCR runs, Kraken setup, thesis artifact builds | Active code |
| `data/` | Shared evidence/data files | Project data |
| `docs/` | Root docs (`model_setup`, this map) | Documentation |
| `artifacts/thesis_showcase/` | Thesis figures, manifests, and payloads | Curated artifacts |
| `archai/` | Monorepo workspace for backend/frontend/vendor tracks | Mixed |
| `weights/` | Local OCR/model binaries | Runtime-only (ignored) |
| `outputs/` | Generated OCR outputs and debug traces | Runtime-only (ignored) |

## Root OCR Pipeline (`src/archai_ocr`)

| Module | Purpose |
|---|---|
| `cli.py` | Main CLI entrypoint and stage orchestration |
| `config.py` | YAML/env configuration loading and validation |
| `logging_utils.py` | Stage-aware logging helpers |
| `pipeline/layout_yolo.py` | Layout detection (YOLO) |
| `pipeline/crop_regions.py` | Region crop extraction |
| `pipeline/htr_kraken.py` | Kraken OCR/HTR recognition |
| `pipeline/assemble_text.py` | Reading-order TXT assembly |
| `utils/coco_writer.py` | COCO output helpers |
| `utils/image_io.py` | Image validation and IO utilities |

## Workspace Monorepo (`archai/`)

| Path | Description | Status |
|---|---|---|
| `archai/assets/` | Prompt/model manifests and assets | Active |
| `archai/config/` | Model/config JSON | Active |
| `archai/data/` | Workspace raw/processed/index/export data dirs | Active |
| `archai/docs/` | Workspace-specific docs | Partial |
| `archai/docker/` | Dockerfiles and compose definitions | Active |
| `archai/backend/` | Prototype FastAPI backend package (`archai_backend`) | Active (prototype) |
| `archai/frontend/` | Vite + Vue frontend scaffold/prototype | Active (prototype) |
| `archai/vendor/layout/` | Full-featured document + chat + OCR workspace | Active |

## Vendor Layout Workspace (`archai/vendor/layout`)

| Path | Purpose |
|---|---|
| `backend/app/` | FastAPI backend, routers, services, agents, schemas |
| `backend/tests/` | API/service regression tests |
| `backend/scripts/` | Smoke checks and utilities |
| `frontend/src/` | Next.js frontend workspace UI |
| `compare/data/` | Model-comparison scripts and generated summaries |
| `APP_DOCUMENTATION.md` | Detailed app-level documentation |
| `MODEL_COMBINATION_GUIDE.md` | Combined-model pipeline guide |

## Thesis Artifact Bundle (`artifacts/thesis_showcase`)

| Path | Description |
|---|---|
| `manifest.json` | Bundle metadata |
| `figure24/` | OCR comparison crops and outputs |
| `figure25/` | Grounded vs plain answer comparison |
| `figure26/` | Limitation examples and summaries |
| `figure29/` | Cross-language processing summary |
| `figure30/` | Entity linking examples and analysis notes |
| `figures_rendered/` | Rendered figure images (PNG/SVG) |
| `scripts/` | Artifact helper scripts |

## Runtime And Local-Only Paths

The following locations are expected to change locally and are not part of stable source control:

- `outputs/`
- `weights/`
- `archai/vendor/layout/backend/.tasks/`
- `archai/vendor/layout/backend/outputs/`
- `archai/vendor/layout/backend/app/.data/`
- `archai/vendor/layout/backend/archai.db`
- local virtual environments (`venv/`, `.venv/`, `archai/backend/venv/`)
