# ArchAI Repository Map

This document explains the structure of the full `ArchAI` repository and clarifies
which parts are active, experimental, or runtime-only.

## Top-Level Layout

| Path | Description | Classification |
|---|---|---|
| `archai/` | The web application: FastAPI backend and Next.js frontend | Active code |
| `src/archai_ocr/` | Offline batch pipeline: YOLO layout, Kraken HTR, TXT and PAGE XML | Active code |
| `tests/` | Tests of `src/archai_ocr` | Tests |
| `scripts/` | Kraken model setup and sanity check, demo run, zone coverage, thesis artifact builds | Active code |
| `artifacts/thesis_showcase/` | Thesis figures, manifests and payloads | Curated artifacts |
| `docs/` | Model setup and this map | Documentation |
| `AUDIT_FINDINGS.md`, `ACCURACY_ROADMAP.md` | Audit of the codebase and the accuracy backlog, with fixed items marked | Documentation |
| `.github/workflows/ci.yml` | CI: `src/archai_ocr` lint, types and tests; the backend's test suite; secret scan | Config |
| `pyproject.toml` | Package metadata for `src/archai_ocr` (`archai`, `archai-ocr` commands) | Build config |
| `.env.example`, `config.example.yaml` | Batch pipeline configuration templates | Config template |
| `test_packages.py` | Manual check that OpenCV, ChromaDB and NumPy import and run | Utility |
| `weights/` | Local OCR model binaries (only `README.md` is tracked) | Runtime-only |
| `outputs/`, `data/` | Generated OCR outputs and logs | Runtime-only (ignored) |

## Batch Pipeline (`src/archai_ocr`)

| Module | Purpose |
|---|---|
| `cli.py` | Main CLI entrypoint and stage orchestration |
| `config.py` | YAML/env configuration loading and validation |
| `logging_utils.py` | Stage-aware logging helpers |
| `pipeline/layout_yolo.py` | Layout detection (YOLO) |
| `pipeline/zones.py`, `pipeline/zone_lines.py` | Zone selection and the lines inside each zone |
| `pipeline/crop_regions.py` | Region crop extraction |
| `pipeline/htr_kraken.py` | Kraken OCR/HTR recognition |
| `pipeline/assemble_text.py` | Reading-order TXT assembly |
| `pipeline/page.py`, `pipeline/page_xml.py` | Page model and PAGE XML output |
| `utils/coco_writer.py` | COCO output helpers |
| `utils/image_io.py` | Image validation and IO utilities |

The web backend does not import this package; the two read pages independently.

## Application (`archai/`)

| Path | Purpose |
|---|---|
| `backend/app/` | FastAPI backend: routers, services, agents, schemas, pipeline database |
| `backend/tests/` | Backend test suite, run by CI on every pull request |
| `backend/scripts/` | Benchmarks (recognisers, entities, retrieval), calibration and maintenance scripts |
| `backend/eval/` | Reference readings, entity gold standard and recogniser manifests the benchmarks use |
| `frontend/src/` | Next.js workspace UI |
| `tools/layout_comparison/` | Research scripts that compared the YOLO layout models on CVAT exports; not part of the running app |
| `docs/architecture.md` | How the backend, frontend and tools fit |
| `Makefile` | `make backend`, `make frontend` |

Until October 2026 the application lived under `archai/vendor/layout/`; it began as a
copy of [emanuskript/layout](https://github.com/emanuskript/layout).

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

- `outputs/`, `data/`
- `weights/`
- `archai/backend/.tasks/`
- `archai/backend/outputs/`
- `archai/backend/app/.data/`
- `archai/backend/app/archai.sqlite` (pipeline database)
- `archai/backend/weights/`, `archai/backend/models/` (model files)
- `archai/backend/data/evidence/` (OCR evidence log; path set by `OCR_EVIDENCE_PATH`)
- local virtual environments (`venv/`, `.venv/`)
