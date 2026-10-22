# ArchAI Application Architecture

- `archai/backend`: FastAPI service (routers, services, agents, pipeline database)
- `archai/frontend`: Next.js document workspace UI
- `archai/tools/layout_comparison`: offline research scripts for the layout models; the backend's
  `app/core/model_runner.py` and `image_batch_classes.py` are ported from them

The offline batch pipeline, `src/archai_ocr`, is a separate package with its own tests; the
backend does not import it.

## Data Flow

1. The user uploads or selects a document page in the frontend.
2. The frontend sends OCR and chat requests to the backend.
3. The backend segments the page, reads it, gates the text on quality, extracts and links
   entities, and records every stage of the run in the pipeline database.
4. Retrieval fuses dense and lexical rankings over the runs the quality gate allowed.
5. Grounded answers come back with citations to the evidence spans they rest on.
