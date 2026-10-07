# Model weights

This folder holds local model weights. Everything here is gitignored except this
file — the weights themselves are large and machine-local.

## Expected files

| File | Required | Purpose |
|---|---|---|
| `layout_yolo.pt` | yes | YOLO layout/zone detector |
| `kraken_recognition.mlmodel` | yes | Kraken HTR recognition model |
| `kraken_segmentation.mlmodel` | no | Kraken baseline segmentation; falls back to legacy `pageseg` if absent |

Place real files here, or symlink models you have already downloaded:

```bash
ln -sfn "/path/to/recognition.mlmodel" weights/kraken_recognition.mlmodel
ln -sfn "/path/to/segmentation.mlmodel" weights/kraken_segmentation.mlmodel
```

Note that a symlink into a per-user directory (for example
`~/Library/Application Support/htrmopo/...`) is not portable across machines. For
a reproducible run, copy the model file in or point the config at an explicit path.

## Which recognition model

For medieval manuscripts, point `kraken_recognition.mlmodel` at CATMuS Medieval
(`scripts/fetch_kraken_models.sh` downloads it):

```bash
ln -sfn "$PWD/weights/kraken_models/catmus_medieval.mlmodel" weights/kraken_recognition.mlmodel
```

McCATMuS is trained on documents from the late 16th century onwards. On the two
medieval reference pages in `archai/vendor/layout/backend/eval/recognizers/`,
the CLI read them at 34.7% and 36.3% character error rate with McCATMuS, and at
6.1% and 9.6% with CATMuS Medieval.

## Layout classes must match your detector

The pipeline filters detections by class name, so `layout.main_text_class` in your
config must name a class the detector actually emits. Check yours with:

```bash
python -c "from ultralytics import YOLO; print(YOLO('weights/layout_yolo.pt').names)"
```

Models trained on the [SegmOnto](https://segmonto.github.io/) vocabulary emit zone
names such as `MainZone`, `MarginTextZone`, `DropCapitalZone` — body text is
`MainZone`, which is the shipped default. A class name that matches nothing now
raises an error listing the available classes, instead of silently producing an
empty transcription.

## Configuring paths by environment

Canonical variables:

- `ARCHAI_OCR_LAYOUT_YOLO`
- `ARCHAI_OCR_KRAKEN_RECOGNITION`
- `ARCHAI_OCR_KRAKEN_SEGMENTATION`
- `ARCHAI_OCR_OUTPUT_DIR`

These legacy names are still honoured as aliases:

- `ARCHAI_LAYOUT_YOLO_WEIGHTS`
- `ARCHAI_KRAKEN_REC_WEIGHTS`
- `ARCHAI_KRAKEN_SEG_WEIGHTS`
- `ARCHAI_OUTPUT_DIR`
