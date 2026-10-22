# ArchAI application (`archai/`)

The web application: upload manuscript pages, read them line by line, check and
link the entities, and ask questions over the result with cited evidence.

| Path | What it is |
|---|---|
| [`backend/`](backend/README.md) | FastAPI service: segmentation, OCR backends, proofreading guard, quality gate, entity extraction and authority linking, hybrid retrieval and grounded chat |
| [`frontend/`](frontend/README.md) | Next.js workspace UI (`/workspace`) |
| [`tools/layout_comparison/`](tools/layout_comparison/MODEL_COMBINATION_GUIDE.md) | Research scripts that compared the YOLO layout models on CVAT exports; not part of the running app |
| [`docs/architecture.md`](docs/architecture.md) | How the pieces fit |

The offline batch pipeline (YOLO layout, Kraken, TXT and PAGE XML) is a
separate package, `src/archai_ocr` at the repository root.

## Run

```bash
cd archai
make backend     # FastAPI on 127.0.0.1:8000
make frontend    # Next.js on 127.0.0.1:3001
```

Setup, environment variables and tests: [`backend/README.md`](backend/README.md)
and [`frontend/README.md`](frontend/README.md).

## Origin

The application began as a copy of
[emanuskript/layout](https://github.com/emanuskript/layout) (the layout
comparison scripts still carry its `/home/hasan/layout` paths) and lived under
`archai/vendor/layout/` until it was moved here. The upstream commit it was
copied from was not recorded; see AUDIT_FINDINGS.md HYG-13.
