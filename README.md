# ArchAI

Medieval manuscript OCR/HTR: YOLO layout analysis, Kraken recognition, multi-agent
post-OCR verification, authority linking, and grounded document chat.

[![CI](https://github.com/ABS-gmbh/ArchAI/actions/workflows/ci.yml/badge.svg)](https://github.com/ABS-gmbh/ArchAI/actions/workflows/ci.yml)

## What is actually in here

The repository holds three tracks with very different maturity. Read this table
before picking an entry point.

| Path | What it is | Status |
|---|---|---|
| `src/archai_ocr/` | Command-line pipeline: YOLO layout → crop → Kraken HTR → `.txt` | **Working**, tested, type-checked |
| `archai/vendor/layout/backend/` | FastAPI service: OCR backends, agent verification, authority linking, RAG chat (~28k LOC) | **Working**, partial test coverage |
| `archai/vendor/layout/frontend/` | Next.js UI for the above | Working |
| `archai/frontend/` | Vue/Vite prototype UI | Prototype |
| `archai/backend/` | Scaffolding: 41 of 47 modules are `# TODO: implement` | **Mock — fabricates output** |
| `artifacts/thesis_showcase/` | Reproducible figure payloads and manifests cited by the thesis | Reference data |

> `archai/backend` is a mock. Its `/ingest/image` endpoint does not run OCR — it
> sleeps through fake stage names and inserts hardcoded English sentences as
> document spans. It refuses to start unless `ARCHAI_ALLOW_MOCK_BACKEND=1` is set.
> Use `archai/vendor/layout/backend` for the real service.

## Quick start — OCR command line

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Place model weights in `weights/` (see [`weights/README.md`](weights/README.md)), then:

```bash
archai --image page.jpg --config config.example.yaml
```

Validate configuration and inputs without loading any model:

```bash
archai --image page.jpg --config config.example.yaml --dry-run
```

Process a directory, continuing past individual failures:

```bash
archai --image pages/ --recursive --continue-on-error
```

Exit codes: `0` success, `1` configuration or input error, `2` some pages failed.

### The layout class must match your model

`layout.main_text_class` has to name a class your layout model actually emits.
The shipped weights use the [SegmOnto](https://segmonto.github.io/) zone
vocabulary, where body text is `MainZone`:

```
DigitizationArtefactZone  DropCapitalZone  GraphicZone  MainZone  MarginTextZone
MusicZone  NumberingZone  QuireMarksZone  RunningTitleZone  StampZone  TitlePageZone
```

Matching is case-insensitive, and a list is accepted to include more than body
text, e.g. `main_text_class: [MainZone, MarginTextZone]`. If the configured class
matches nothing the model emits, the run fails with the list of available classes
rather than silently writing an empty file.

### Reading order

`layout.reading_order: column` (the default) groups regions into columns, orders
columns left to right, then reads each column top to bottom. This is required for
multi-column manuscript pages — a global top-to-bottom sort interleaves the
columns, producing a transcription that alternates between them line by line.

Set `reading_order: simple` to reproduce the pre-0.2.0 global sort exactly.

### Configuration precedence

`defaults` < `config.yaml` < environment variables < command-line flags.

Relative paths in the config file resolve against **the directory containing that
config file**, not the working directory. Unknown keys and out-of-range values are
rejected at load time. See [`.env.example`](.env.example) for the environment
variables.

## Quick start — document workspace

Backend:

```bash
cd archai/vendor/layout/backend
pip install -e .
cp .env.example .env    # then fill in ANALYTICS_USERNAME / ANALYTICS_PASSWORD / JWT_SECRET
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The analytics endpoints fail closed with HTTP 503 until credentials are
configured; there are deliberately no default credentials. Generate a secret with:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Frontend:

```bash
cd archai/vendor/layout/frontend && npm install && npm run dev
```

## Development

```bash
ruff check src tests        # lint
ruff format src tests      # format
mypy                       # strict type check
pytest tests -q            # 84 tests, no model weights required
```

The `archai_ocr` test suite runs without model weights or Kraken installed, so it
works in CI. The vendor backend suite needs weights and the full inference stack:

```bash
cd archai/vendor/layout/backend && pytest tests -q
```

CI runs lint, format, strict typing and tests on Python 3.11 and 3.12,
byte-compiles the vendor backend, and scans tracked files for credential
literals. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Reproducibility notes

- `YOLO_AUTOINSTALL=false` is set before ultralytics is imported. Ultralytics
  otherwise pip-installs optional packages at inference time, mutating the
  environment mid-run and requiring network access.
- Canned OCR fixtures exist in the vendor backend for demos and regression tests
  (`app/services/test_ocr_overrides.py`). They match three specific images by
  SHA-256 and return hand-written transcriptions. They are **off by default** and
  require `ENABLE_TEST_OCR_FIXTURES=true`; when active they log a warning naming
  the digest.
- Transcriptions are Unicode NFC-normalized so accented forms compare equal
  across regions.

## Thesis artifacts

`artifacts/thesis_showcase/` holds reproducible figure datasets for Figures
24/25/26/29/30. Verify bundle integrity with:

```bash
python scripts/build_thesis_showcase_payloads.py --verify
```

## Repository hygiene

Tracked: source, config templates (`*.example`), docs, manifests, curated thesis
artifacts.

Ignored: virtual environments, caches, model weights (`weights/`), run outputs
(`outputs/`, `output/`, `tmp/`), the LaTeX writing workspace (`fixes/`), local
databases, and all `.env` files.

## License

MIT — see [LICENSE](LICENSE).
