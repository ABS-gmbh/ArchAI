# Archai OCR Backend

Backend-only OCR/HTR pipeline for manuscript pages.  
Input is a single page image (`png/jpg/tif`), output is a single `.txt` file in reading order.

## Pipeline

1. Layout detection with Ultralytics YOLO (`weights/layout_yolo.pt`)
2. Crop detected text regions in reading order
3. Kraken HTR on each crop:
   - Preferred: baseline segmentation model (`weights/kraken_segmentation.mlmodel`)
   - Fallback: legacy binarize + page segmentation when segmentation model is missing
   - Recognition model (`weights/kraken_recognition.mlmodel`)
4. Assemble one text file: `outputs/<image_stem>/<image_stem>.txt`

Intermediate debugging artifacts are written per image run:

- `layout_coco.json`
- `crops/region_*.png`
- optional `crops/region_*.xml` (PAGE-XML-like output)

## Requirements

- Python 3.11+
- Local model weights:
  - `weights/layout_yolo.pt`
  - `weights/kraken_segmentation.mlmodel` (optional; fallback available)
  - `weights/kraken_recognition.mlmodel`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python -m archai_ocr.cli --image sample.png --config config.example.yaml
```

On success, the command prints the final text path.

Equivalent entrypoint:

```bash
archai-ocr --image sample.png --config config.example.yaml
```

## Configuration

Default config is `config.example.yaml`.  
You can override selected values via CLI:

- `--outdir outputs/other`
- `--main-class main_text`

`.env` is supported for environment overrides (prefixed with `ARCHAI_OCR_`), for example:

```bash
ARCHAI_OCR_CONFIDENCE_THRESHOLD=0.35
ARCHAI_OCR_WRITE_PAGE_XML=true
```

## Demo Script

```bash
bash scripts/run_demo.sh sample.png config.example.yaml
```
