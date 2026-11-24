# ArchAI

Medieval manuscript OCR/HTR: YOLO layout analysis, Kraken recognition, multi-agent
post-OCR verification, authority linking, and grounded document chat.

[![CI](https://github.com/ABS-gmbh/ArchAI/actions/workflows/ci.yml/badge.svg)](https://github.com/ABS-gmbh/ArchAI/actions/workflows/ci.yml)

## What is actually in here

The repository holds two products. Read this table before picking an entry point.

| Path | What it is | Status |
|---|---|---|
| `archai/` | The web application: FastAPI backend (`archai/backend`: OCR backends, quality gate, entity extraction and authority linking, hybrid retrieval, grounded chat) and Next.js frontend (`archai/frontend`) | **Working**; backend suite runs in CI |
| `src/archai_ocr/` | Offline batch pipeline: YOLO layout → crop → Kraken HTR → TXT and PAGE XML | **Working**, tested, type-checked |
| `artifacts/thesis_showcase/` | Reproducible figure payloads and manifests cited by the thesis | Reference data |

> The application lived under `archai/vendor/layout/` until it was moved to
> `archai/`. An earlier mock backend that fabricated pipeline stages, and a Vue
> prototype of TODO stubs, have been removed. See `docs/repository_map.md`.

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

Matching is case-insensitive and a list is accepted. If the configured class
matches nothing the model can emit, the run fails with the list of available
classes rather than silently writing an empty file. To transcribe marginalia,
use the secondary lane below rather than adding `MarginTextZone` here: listed
here, glosses are spliced into the body's reading order.

### Outputs

Each page gets a directory `outputs/<stem>/` containing:

| File | Contents |
|---|---|
| `<stem>.txt` | The body text (main-text zones) in reading order. |
| `<stem>.secondary.txt` | Text outside the body, one section per zone type. Written only when there is some. |
| `<stem>.page.xml` | With `--page-xml` (or `runtime.write_page_xml`): the whole page as [PAGE XML](https://github.com/PRImA-Research-Lab/PAGE-XML) 2019 - every zone, each line's polygon, baseline and confidence, and the reading order. |
| `layout_coco.json` | Every detected zone as COCO, one category per zone class. |
| `crops/` | The region images recognition ran on; PAGE region ids match their file names. |

The PAGE file opens in eScriptorium and Transkribus. Correcting it there and
exporting is the practical way to build the ground truth this pipeline's accuracy
can be measured against. Zone types are written in the `custom="structure
{type:MarginTextZone;}"` form those tools read.

### Text outside the body

Marginalia, running titles, foliation and quire marks
(`layout.secondary_text_class`) are transcribed separately from the body, so a
gloss never lands in the middle of a sentence. They use a lower detection
threshold (`layout.secondary_confidence_threshold`, 0.10). Zones mostly inside
the body are left to it, since their lines are already recognised there.
`--no-secondary-text` turns the lane off.

These zones are often under 100 px high, too small for Kraken's page
segmenter, which found no usable line in any of 20 such zones on ten real
pages. They are instead cut into lines where their ink profile falls to near
white between two lines. Folio numbers, shelfmarks and short references come
out readable ("B.109.", "28", "Apoc / 21.3"). Tightly written multi-line notes
in small hands still come out poorly, and colour bars or decoration the
detector mistakes for text produce short junk strings; the PAGE output keeps
their geometry for correction. The body text is unchanged: on those ten pages
every `.txt` is byte-identical to the previous release.

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

Upgrading an installation from the version of 2026-11-09? Read
[`docs/upgrade-notes.md`](docs/upgrade-notes.md) first: what to do before the
upgrade, the new settings, and what changes for users and scripts.

Backend:

```bash
cd archai/backend
pip install -e .
cp .env.example .env    # then fill in CHAT_AI_API_KEY, and ANALYTICS_USERNAME / ANALYTICS_PASSWORD / JWT_SECRET
uvicorn app.main:app --host 127.0.0.1 --port 8010
```

The models the reading pipeline needs (layout weights, Kraken recognisers,
GLM-OCR) are set up as [`docs/model_setup.md`](docs/model_setup.md) describes;
the backend's startup log names any that are missing.

The frontend expects the backend at 127.0.0.1:8010; `NEXT_PUBLIC_BACKEND_URL` (the browser's direct URL) and `BACKEND_URL` (the Next server's `/api` proxy) override it; `BACKEND_API_TOKEN` gives that proxy the backend's `API_TOKEN`.

Without `CHAT_AI_API_KEY` the backend starts, but chat, translation and SAIA OCR
answer that the key is not configured, and `/api/health` reports `degraded`.
GLM-OCR needs an Ollama server with `glm-ocr` at `GLMOCR_OLLAMA_HOST`.
`archai/backend/README.md` lists the other settings.

The analytics endpoints fail closed with HTTP 503 until credentials are
configured; there are deliberately no default credentials. Generate a secret with:

```bash
python -c 'import secrets; print(secrets.token_urlsafe(48))'
```

Frontend:

```bash
cd archai/frontend && npm ci && npm run dev -- --hostname 127.0.0.1
```

## Development

```bash
ruff check src tests        # lint
ruff format src tests      # format
mypy                       # strict type check
pytest tests -q            # no model weights or Kraken required
```

The `archai_ocr` test suite runs without model weights or Kraken installed. So
does the backend's, offline, apart from tests marked `requires_models`:

```bash
cd archai/backend && pytest -q -m "not requires_models"
```

CI runs lint, format, strict typing and tests of `archai_ocr` on Python 3.11 and
3.12, the backend test suite, the frontend's lint, type check, font coverage
check and production build, and a scan of tracked files for credential
literals. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Reproducibility notes

- `YOLO_AUTOINSTALL=false` is set before ultralytics is imported. Ultralytics
  otherwise pip-installs optional packages at inference time, mutating the
  environment mid-run and requiring network access.
- Canned OCR fixtures exist in the backend for demos and regression tests
  (`app/services/ocr_test_fixtures.py`). They match three specific images by
  SHA-256 and return hand-written transcriptions. They are **off by default** and
  require `ENABLE_TEST_OCR_FIXTURES=true`; when active they log a warning naming
  the digest.
- Transcriptions are Unicode NFC-normalized so accented forms compare equal
  across regions.

## Thesis artifacts

`artifacts/thesis_showcase/` holds reproducible figure datasets for Figures
24/25/26/29/30. Check the bundle against its manifest, offline, with:

```bash
python scripts/build_thesis_showcase_payloads.py --verify
```

It checks that every file `manifest.json` lists exists, that every listed JSON
file parses, and that no file is present that the manifest omits; when the
manifest has a `sha256` map of path to digest, it also checks each file's
content. It exits 1 on any problem. The committed manifest has no `sha256` map,
so content is not checked, and the committed bundle currently reports 18
problems:

- 6 files listed but missing: `figure24/outputs/*.txt`. The root `.gitignore`
  ignores `outputs/`, so they were never committed.
- 1 screenshot listed under a name one character different from the file in
  `figure30/screenshots/`.
- 10 files present but not listed: `figures_rendered/`, added after the
  manifest was written.

Without `--verify` the script regenerates the bundle and overwrites it. That
needs the backend running at `ARCHAI_BACKEND_URL` (default
`http://127.0.0.1:8010`), the `showcase` extra (`pip install -e ".[showcase]"`)
and, for the chat figures, a SAIA key in `archai/backend/.env`, which bills
real completions.

## Repository hygiene

Tracked: source, config templates (`*.example`), docs, manifests, curated thesis
artifacts.

Ignored: virtual environments, caches, model weights (`weights/`), run outputs
(`outputs/`, `output/`, `tmp/`), the LaTeX writing workspace (`fixes/`), local
databases, and all `.env` files.

## License

MIT — see [LICENSE](LICENSE).
