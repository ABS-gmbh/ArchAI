# `archai_ocr` Package

Root OCR/HTR pipeline package used by the repository-level CLI.

## Pipeline Stages

1. `pipeline/layout_yolo.py` - detect text layout regions
2. `pipeline/crop_regions.py` - crop regions in reading order
3. `pipeline/htr_kraken.py` - recognize text per crop with Kraken
4. `pipeline/assemble_text.py` - merge region text into final TXT output

## Supporting Modules

- `config.py` for typed config loading/validation
- `logging_utils.py` for structured stage logging
- `utils/image_io.py` and `utils/coco_writer.py` for IO/output helpers

## Entry Point

Use `archai_ocr.cli:main` via:

- `python -m archai_ocr.cli ...`
- `archai ...`
- `archai-ocr ...`
