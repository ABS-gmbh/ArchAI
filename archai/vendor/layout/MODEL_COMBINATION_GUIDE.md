# Model Combination Guide

## Overview

This guide explains how to combine predictions from three YOLO models to produce a unified COCO-format output with only the classes defined in `coco_class_mapping`.

## The Three Models

### 1. **best_emanuskript_segmentation.pt**
- **Type**: Segmentation model
- **Classes**: 21 classes including:
  - Border, Table, Diagram, Music
  - Main script black/coloured
  - Variant script black/coloured
  - Plain initial (coloured/highlighted/black)
  - Historiated, Inhabited, Embellished
  - Page Number, Quire Mark, Running header, Catchword, Gloss, Illustrations

### 2. **best_catmus.pt**
- **Type**: Segmentation model
- **Classes**: 19 classes including:
  - DefaultLine, InterlinearLine
  - MainZone, MarginTextZone
  - DropCapitalZone, GraphicZone, MusicZone
  - NumberingZone, QuireMarksZone, RunningTitleZone
  - StampZone, TitlePageZone

### 3. **best_zone_detection.pt**
- **Type**: Detection model
- **Classes**: 11 zone classes:
  - MainZone, MarginTextZone
  - DropCapitalZone, GraphicZone, MusicZone
  - NumberingZone, QuireMarksZone, RunningTitleZone
  - StampZone, TitlePageZone, DigitizationArtefactZone

## How It Works

### Step 1: Run Model Predictions
Each model is run independently on the input image:
```python
# Emanuskript model
emanuskript_results = model.predict(image_path, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20])

# Catmus model  
catmus_results = model.predict(image_path, classes=[1,7])  # DefaultLine and InterlinearLine

# Zone model
zone_results = model.predict(image_path)  # All classes
```

Predictions are saved to JSON files in separate folders.

### Step 2: Combine Predictions (ImageBatch Class)

The `ImageBatch` class (`utils/image_batch_classes.py`) handles:

1. **Loading Images**: Loads the image and gets dimensions
2. **Loading Annotations**: Loads predictions from all 3 JSON files
3. **Unifying Names**: Maps class names using `catmus_zones_mapping`:
   - `DefaultLine` → `Main script black`
   - `InterlinearLine` → `Gloss`
   - `MainZone` → `Column`
   - `DropCapitalZone` → `Plain initial- coloured`
   - etc.

4. **Filtering Annotations**: 
   - Removes overlapping annotations based on spatial indexing
   - Uses overlap thresholds (0.3-0.8 depending on class)
   - Handles conflicts between different model predictions

5. **COCO Format Conversion**: Converts to COCO JSON format

### Step 3: Filter to coco_class_mapping

Only annotations with classes in `coco_class_mapping` are kept (25 classes total).

## Key Functions

### `predict_annotations()` (in `utils/data.py`)
- Runs a single model on an image
- Saves predictions to JSON
- Used by Celery tasks for async processing

### `unify_predictions()` (in `utils/data.py`)
- Combines predictions from all three models
- Uses `ImageBatch` to process and filter
- Returns COCO format JSON
- Imports annotations into database

### `ImageBatch` class (in `utils/image_batch_classes.py`)
- Main class for combining predictions
- Methods:
  - `load_images()`: Load image files
  - `load_annotations()`: Load predictions from JSON files
  - `unify_names()`: Map class names to coco_class_mapping
  - `filter_annotations()`: Remove overlapping annotations
  - `return_coco_file()`: Generate COCO JSON

## Usage Example

```python
from ultralytics import YOLO
from utils.image_batch_classes import ImageBatch

# 1. Run models (or use predict_annotations function)
# ... save predictions to JSON files ...

# 2. Combine predictions
image_batch = ImageBatch(
    image_folder="path/to/images",
    catmus_labels_folder="path/to/catmus/predictions",
    emanuskript_labels_folder="path/to/emanuskript/predictions",
    zone_labels_folder="path/to/zone/predictions"
)

image_batch.load_images()
image_batch.load_annotations()
image_batch.unify_names()

# 3. Get COCO format
coco_json = image_batch.return_coco_file()
```

## Running the Test Script

```bash
python3 test_combined_models.py
```

This will:
1. Run all three models on `bnf-naf-10039__page-001-of-004.jpg`
2. Combine and filter predictions
3. Save results to `combined_predictions.json`
4. Print a summary of detected classes

## Output Format

The final output is a COCO-format JSON file with:
- **images**: Image metadata (id, width, height, filename)
- **categories**: List of category definitions (25 classes from coco_class_mapping)
- **annotations**: List of annotations with:
  - `id`: Annotation ID
  - `image_id`: Associated image ID
  - `category_id`: Class ID from coco_class_mapping
  - `segmentation`: Polygon coordinates
  - `bbox`: Bounding box [x, y, width, height]
  - `area`: Polygon area

## Class Mapping

The `catmus_zones_mapping` in `image_batch_classes.py` maps:
- Catmus/Zone model classes → coco_class_mapping classes
- Example: `DefaultLine` → `Main script black`
- Example: `MainZone` → `Column`

Only classes that map to `coco_class_mapping` are included in the final output.

