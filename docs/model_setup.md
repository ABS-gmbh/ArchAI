# Kraken Model Setup

`Extract with Kraken` requires a **Kraken recognition model**.  
Without it, extraction is disabled in the UI.

## Why this is required

- Layout segmentation (YOLO) finds regions.
- Kraken recognition converts segmented lines to text.
- Without recognition weights, no `.txt` output can be produced.

## Quick Setup

From repository root:

```bash
bash scripts/fetch_kraken_models.sh
```

Preview selection without downloading:

```bash
bash scripts/fetch_kraken_models.sh --dry-run
```

The script:

- runs `kraken list`/`kraken show` to discover candidates
- selects a robust generic default recognition model
- selects an optional general segmentation model
- downloads with `kraken get`
- prints the downloaded model file paths and wiring commands

## Default model choice rationale

The script prioritizes models described as:

- generic/mixed/multi-use
- covering handwritten + printed (and typewritten if available)
- broader language/script coverage

Current expected default recognition candidate is:

- `10.5281/zenodo.13788177` (McCATMuS generic model; broad format/language coverage)

Current expected default segmentation candidate is:

- `10.5281/zenodo.14602569` (Kraken general print+handwriting segmentation model)

## Wiring models into the app

### Option 1: Environment variables (recommended)

Set:

- `ARCHAI_KRAKEN_REC_WEIGHTS` (required)
- `ARCHAI_KRAKEN_SEG_WEIGHTS` (optional)
- `ARCHAI_OUTPUT_DIR` (optional; default `outputs`)

Example:

```bash
ARCHAI_KRAKEN_REC_WEIGHTS="/absolute/path/to/recognition.mlmodel"
ARCHAI_KRAKEN_SEG_WEIGHTS="/absolute/path/to/segmentation.mlmodel"
ARCHAI_OUTPUT_DIR="/Users/mobasuony/Desktop/Thesis project/outputs"
```

### Option 2: Link/copy into `weights/`

Use these filenames:

- `weights/kraken_recognition.mlmodel` (required)
- `weights/kraken_segmentation.mlmodel` (optional)

The app defaults resolve to these paths when env vars are not set.

## Override with a different Kraken model

You can choose another model from the repository:

```bash
kraken list --recognition
kraken show <doi-or-id>
kraken get <doi-or-id>
```

Then point `ARCHAI_KRAKEN_REC_WEIGHTS` to that downloaded model path.

## Important note

A “generic” model is a practical default, not universally optimal.  
For best quality on specific collections, switch to a model closer to your manuscript domain (script/language/time period/layout).
