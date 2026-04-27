# ArchAI Workspace Architecture

The `archai/` workspace has two implementation layers and one vendor-integrated layer.

## 1) Prototype Layer

- `archai/backend`: lightweight FastAPI prototype (`archai_backend`)
- `archai/frontend`: lightweight Vue/Vite prototype

This layer is used for fast iteration on core contracts (ingest/jobs/evidence/chat).

## 2) Vendor-Integrated Layer

- `archai/vendor/layout/backend`: full backend stack (routers/services/agents)
- `archai/vendor/layout/frontend`: Next.js document workspace UI

This layer contains the full document + chat workflow used for integrated testing
and thesis demonstrations.

## 3) Shared Resources

- `archai/assets`: prompts and model manifests
- `archai/config`: model config metadata
- `archai/data`: raw/processed/derived/index/export paths
- `archai/docker`: containerized run definitions

## Data Flow (Integrated Path)

1. User uploads/selects a document page in the frontend.
2. Frontend sends OCR/chat requests to backend APIs.
3. Backend orchestrates OCR, evidence spans, retrieval, and model calls.
4. Grounded answers and citations are returned to frontend panels.
5. Evidence/provenance artifacts are persisted for inspection/export.
