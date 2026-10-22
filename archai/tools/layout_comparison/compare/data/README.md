# Model Comparison Scripts

This directory contains scripts to compare old models vs new models vs ground truth annotations.

## Files

- `original_annotations.py`: Parses CVAT XML annotations and converts to COCO format
- `old_models.py`: Runs old models (Line, Border, Zones) and converts predictions to COCO
- `new_models.py`: Runs new models (emanuskript, catmus, zone) and converts predictions to COCO
- `compare.py`: Main script that orchestrates the comparison and calculates metrics

## Setup

1. Install required dependencies:
```bash
pip install pycocotools numpy pillow matplotlib ultralytics
```

2. Ensure model files are in the project root:
   - Old models: `best_line_detection_yoloe (1).pt`, `border_model_weights.pt`, `zones_model_weights.pt`
   - New models: `best_emanuskript_segmentation.pt`, `best_catmus.pt`, `best_zone_detection.pt`

## Usage

Run the main comparison script:

```bash
cd /home/hasan/layout/compare/data
python compare.py
```

The script will:
1. Load ground truth annotations from `Aleyna 1 (2024)/Annotations/annotations.xml`
2. Run old models on all images in `Aleyna 1 (2024)/Images`
3. Run new models on all images
4. Calculate metrics (mAP@50, mAP@[.50:.95], Precision, Recall)
5. Create side-by-side visualizations for each image

## Output

Results are saved to `results/` directory:
- `ground_truth.json`: Ground truth in COCO format
- `old_models_merged.json`: Old models predictions
- `new_models_merged.json`: New models predictions
- `metrics.json`: Calculated metrics for both model sets
- `visualizations/`: Side-by-side comparison images

## Metrics

The comparison calculates:
- **mAP@50**: Mean Average Precision at IoU=0.50
- **mAP@[.50:.95]**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95
- **Precision**: Approximated from mAP@50
- **Recall**: Maximum recall with 100 detections
- **F1 Score**: Harmonic mean of Precision and Recall

## Notes

- The CVAT XML parser handles RLE (Run-Length Encoding) format masks
- Category alignment is performed automatically to match ground truth categories
- Images are processed sequentially - batch processing may take time
- Visualizations show: Original+GT | Old Models | New Models side-by-side

