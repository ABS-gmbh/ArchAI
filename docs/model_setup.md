# Model setup

What the two products need besides their Python packages, and where each file
goes. The backend starts without any of it: it then serves chat and retrieval
and reads pages with GLM-OCR alone, and its startup log names what is missing
("Layout segmentation and the segmented OCR engine cannot run ... Missing: ...").

## The backend (`archai/backend`)

A page is read by the segmented engine: three YOLO layout models find the text
regions and lines, and a Kraken recogniser reads each line. GLM-OCR, through
Ollama, is its fallback.

| What | Size | Where it goes | Setting |
|---|---|---|---|
| Layout weights `best_emanuskript_segmentation.pt`, `best_catmus.pt`, `best_zone_detection.pt` | about 570 MB | `archai/backend/models/` | `MODEL_DIR` |
| Kraken recognisers (CATMuS Medieval, McCATMuS, CREMMA Medieval, CATMuS Print) | about 70 MB | `weights/kraken_models/` at the repository root | `KRAKEN_MODELS_DIR`, `KRAKEN_CATMUS_MODEL_PATH`, ... |
| GLM-OCR `glm-ocr:latest` | about 2.2 GB | an Ollama server | `GLMOCR_OLLAMA_HOST` |

### 1. Packages

`pip install -e .` installs ultralytics (layout) but not kraken (line
recognition), which the segmented engine needs.

Installing kraken into that environment is not yet a single command.
`pip install kraken` resolves kraken 7.2 with NumPy 2.4, while the backend's
`pyproject.toml` pins `numpy<2.0`. The combination the evaluation VM runs is in
`scripts/eval_vm/setup.sh`: Python 3.11, torch 2.9.0, kraken 6.0.3,
ultralytics 8.4.13 and NumPy 2.0.2.

### 2. Layout weights

The repository names no public source for the three layout weights (audit
HYG-13; see `scripts/eval_vm/README.md`). Copy them from an existing
installation into `archai/backend/models/`, or point `MODEL_DIR` at the
directory that holds them. Without them, `/api/predict/*` answers 500 and pages
are read with GLM-OCR only.

### 3. Kraken recognisers

From the repository root, with the `kraken` command installed:

```bash
bash scripts/fetch_kraken_models.sh --dry-run   # what it would fetch
bash scripts/fetch_kraken_models.sh
```

It downloads the recognisers with `kraken get` (Zenodo DOIs) and CREMMA
Medieval directly. It then prints the `ln -sfn` commands that give each one the
file name the settings expect, in `weights/kraken_models/`; run them. The
backend resolves a relative model path against the repository root first, so
the defaults find them there. Set `KRAKEN_MODELS_DIR` or a
`KRAKEN_*_MODEL_PATH` only to put them elsewhere.

For medieval manuscripts the backend reads every page with CATMuS Medieval
first. McCATMuS is trained on prints and on hands from the late 16th century
onwards. On the five medieval reference pages in
`archai/backend/eval/recognizers/` it reads at 36.5% CER, against CATMuS
Medieval's 7.6%.

### 4. GLM-OCR (optional)

Install Ollama, run `ollama pull glm-ocr`, and set `GLMOCR_OLLAMA_HOST` (default
`http://localhost:11434`). `/api/health` reports `ollama` as reachable or not.

## The command-line pipeline (`src/archai_ocr`)

`pip install -e ".[dev]"` at the repository root installs kraken and
ultralytics. The pipeline needs one layout model and one Kraken recogniser in
`weights/`; [`weights/README.md`](../weights/README.md) lists the file names,
which layout classes the model must emit, and the `ARCHAI_OCR_*` variables
that point elsewhere. The recognisers come from the same
`scripts/fetch_kraken_models.sh`. For medieval manuscripts, link CATMuS Medieval
as `weights/kraken_recognition.mlmodel`:

```bash
ln -sfn "$PWD/weights/kraken_models/catmus_medieval.mlmodel" weights/kraken_recognition.mlmodel
```

Other Kraken models: `kraken list`, `kraken show <doi>` and `kraken get <doi>`.
Then point `ARCHAI_OCR_KRAKEN_RECOGNITION` at the file. A generic model is a
practical default; a model trained on your collection's script and period
reads it better.
