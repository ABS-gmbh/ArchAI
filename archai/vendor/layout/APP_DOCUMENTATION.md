# Combined Manuscript Models Web Application

> **Gradio-based web interface for manuscript layout analysis using three combined ML models: emanuskript, catmus, and zone.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Configuration](#configuration)
- [Core Components](#core-components)
  - [Analytics System](#analytics-system)
  - [COCO Annotation Handling](#coco-annotation-handling)
  - [Image Processing](#image-processing)
  - [Visualization](#visualization)
- [API Reference](#api-reference)
- [Class Definitions](#class-definitions)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This application provides a web-based interface for analyzing manuscript images using three specialized ML models. It combines predictions from multiple models into a unified COCO-format output, with support for:

- Single image processing
- Batch processing via ZIP upload
- Interactive class filtering
- Downloadable annotations (JSON/ZIP)
- Built-in visitor analytics

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Web Interface                      │
├─────────────┬─────────────────┬─────────────────────────────────┤
│ Single Image│  Batch (ZIP)    │         Analytics Tab           │
│    Tab      │     Tab         │                                 │
├─────────────┴─────────────────┴─────────────────────────────────┤
│                      Processing Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  emanuskript │  │    catmus    │  │     zone     │          │
│  │    model     │  │    model     │  │    model     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│              Combination & Filtering (ImageBatch)                │
├─────────────────────────────────────────────────────────────────┤
│                    COCO JSON Output                              │
├─────────────────────────────────────────────────────────────────┤
│                  SQLite Analytics Database                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Inference** | Runs 3 models simultaneously and combines predictions |
| **COCO Format Output** | Standard COCO JSON annotations for interoperability |
| **Interactive Filtering** | Filter results by 25 predefined classes |
| **Batch Processing** | Process entire ZIP archives of images |
| **Visual Overlays** | Color-coded annotations with semi-transparent fills |
| **Statistics** | Per-image and aggregate statistics with graphs |
| **Analytics** | Built-in visitor tracking with privacy compliance |
| **Cookie Consent** | GDPR-compliant consent banner |

---

## Configuration

### Constants

| Variable | Value | Description |
|----------|-------|-------------|
| `SCRIPT_DIR` | `os.path.dirname(__file__)` | Application root directory |
| `ANALYTICS_DB` | `analytics.db` | SQLite database path |

### Server Settings (Launch)

```python
demo.launch(
    server_name="0.0.0.0",    # Bind to all interfaces
    server_port=8000,          # HTTP port
    max_threads=4,             # Concurrent request limit
    debug=False,               # Production mode
    share=False,               # No public Gradio link
)
```

---

## Core Components

### Analytics System

#### Database Schema

**`visits` table:**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key (auto-increment) |
| `visitor_id` | TEXT | SHA256 hash of IP + User-Agent (16 chars) |
| `ip_address` | TEXT | Client IP (from headers) |
| `user_agent` | TEXT | Full User-Agent string |
| `browser` | TEXT | Parsed browser name + version |
| `os` | TEXT | Parsed OS name + version |
| `device` | TEXT | Device family |
| `is_mobile` | INTEGER | 1 = mobile, 0 = desktop |
| `is_bot` | INTEGER | 1 = bot, 0 = human |
| `page` | TEXT | Page identifier |
| `action` | TEXT | Event type (`page_view`, `single_image_process`, `batch_process`) |
| `timestamp` | DATETIME | Event timestamp |
| `session_id` | TEXT | MD5 hash for session grouping |
| `country` | TEXT | Reserved for geo-IP |
| `city` | TEXT | Reserved for geo-IP |

**`cookie_consents` table:**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `visitor_id` | TEXT | Unique visitor ID |
| `consent_given` | INTEGER | Consent status |
| `consent_timestamp` | DATETIME | When consent was given |

#### Key Functions

| Function | Purpose |
|----------|---------|
| `init_analytics_db()` | Creates tables if not exist |
| `generate_visitor_id(ip, ua)` | SHA256 hash → 16-char ID |
| `log_visit(request, action)` | Records visitor event |
| `get_analytics_summary()` | Returns DataFrames for dashboard |
| `get_visitor_chart()` | Generates 14-day bar chart PNG |

---

### COCO Annotation Handling

#### Filter Function

```python
def _filter_coco_by_classes(coco_json: Dict, allowed_classes: List[str]) -> Dict
```

Filters COCO annotations to only include specified classes. Also removes images with no remaining annotations.

#### Merge Function

```python
def _merge_coco_list(coco_list: List[Dict]) -> Dict
```

Merges multiple single-image COCO dictionaries into one batch output. Reassigns `image_id` and `annotation_id` to ensure uniqueness.

#### Statistics Functions

| Function | Returns |
|----------|---------|
| `_stats_from_coco(coco_json)` | `Dict[str, int]` - class counts |
| `_stats_table(stats, image_name)` | `pd.DataFrame` - tabular view |
| `_stats_graph(stats, title)` | `str` - path to PNG chart |

---

### Image Processing

#### Single Image Pipeline

```python
def process_single_image(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    final_classes: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]
```

**Flow:**
1. Save input to temp file
2. Run `run_models_parallel()` (3 models in parallel via ProcessPoolExecutor)
3. Run `combine_and_filter_predictions()`
4. Filter to selected classes
5. Draw annotations
6. Generate statistics
7. Return original, annotated, stats table, stats graph

#### Batch Processing Pipeline

```python
def process_batch_images(
    zip_file,
    conf_threshold: float,
    iou_threshold: float,
    final_classes: List[str],
) -> Tuple[List, str, pd.DataFrame, pd.DataFrame, str]
```

**Validation includes:**
- Skip `__MACOSX` directories
- Skip hidden files (`.DS_Store`, `._*`, etc.)
- Verify image validity via PIL
- Skip empty/corrupted files

---

### Visualization

#### Color Mapping

Each of the 25 classes has a predefined hex color:

```python
CLASS_COLOR_MAP = {
    'Border': '#8B0000',           # Dark red
    'Table': '#006400',            # Dark green
    'Diagram': '#00008B',          # Dark blue
    'Main script black': '#FF0000', # Bright red
    # ... (25 total classes)
}
```

#### Drawing Function

```python
def _draw_coco_on_image(
    image_path: str,
    coco_json: Dict,
    allowed_classes: List[str],
) -> np.ndarray
```

**Features:**
- Semi-transparent polygon/rectangle fills (α=0.3)
- Opaque edge outlines (α=0.8)
- Smart label positioning to avoid overlap
- White background labels with colored borders

---

## API Reference

### Gradio Endpoints

| Button/Trigger | Handler | Inputs | Outputs |
|----------------|---------|--------|---------|
| "Run 3 Models + Combine" | `process_single_image` | image, conf, iou, classes | original, annotated, table, graph |
| "Process ZIP" | `process_batch_images` | zip, conf, iou, classes | gallery, status, per_image, summary, graph |
| "Download Annotations (JSON)" | `download_single_annotations` | — | JSON file |
| "Download Image" | `download_single_image` | annotated image | JPEG file |
| "Download COCO Annotations" | `download_batch_annotations` | — | JSON file |
| "Download Results (ZIP)" | `download_batch_zip` | gallery | ZIP file |
| "Refresh Analytics" | `refresh_analytics` | — | summary, tables, chart |

### Download Functions

| Function | Output Format |
|----------|---------------|
| `download_single_annotations()` | `combined_single_YYYYMMDD_HHMMSS.json` |
| `download_single_image(img)` | `combined_single_image_YYYYMMDD_HHMMSS.jpg` |
| `download_batch_annotations()` | `combined_batch_YYYYMMDD_HHMMSS.json` |
| `download_batch_zip(gallery)` | `combined_batch_results_YYYYMMDD_HHMMSS.zip` |

---

## Class Definitions

### 25 Final Classes

Sourced from `utils/image_batch_classes.coco_class_mapping`:

| Category | Classes |
|----------|---------|
| **Layout** | Border, Column, Table, Diagram |
| **Script** | Main script black, Main script coloured, Variant script black, Variant script coloured |
| **Initials** | Historiated, Inhabited, Zoo - Anthropomorphic, Embellished, Plain initial - coloured, Plain initial - Highlighted, Plain initial - Black |
| **Navigation** | Page Number, Quire Mark, Running header, Catchword |
| **Content** | Gloss, Illustrations |
| **Zones** | GraphicZone, MusicLine, MusicZone, Music |

---

## Usage

### Starting the Server

```bash
cd /home/cloud/layout
python app.py
```

Access at: `http://localhost:8000`

### Single Image Processing

1. Upload an image (JPEG/PNG/TIFF/BMP/WebP)
2. Adjust confidence threshold (default: 0.25)
3. Adjust IoU threshold (default: 0.3)
4. Select/deselect classes to visualize
5. Click "Run 3 Models + Combine"
6. Download results via buttons

### Batch Processing

1. Create ZIP with images (nested folders supported)
2. Upload ZIP file
3. Configure thresholds and classes
4. Click "Process ZIP"
5. Browse results in gallery
6. Download merged annotations or complete ZIP

### Analytics Dashboard

1. Navigate to "📈 Analytics" tab
2. Click "🔄 Refresh Analytics"
3. View:
   - Summary statistics
   - 14-day visitor trend chart
   - Recent visitors table
   - Daily aggregates
   - Browser breakdown

---

## Dependencies

### Python Packages

```
gradio
numpy
Pillow
matplotlib
pandas
sqlite3 (stdlib)
user-agents
```

### Internal Modules

| Module | Purpose |
|--------|---------|
| `utils.image_batch_classes` | `coco_class_mapping` dict, `ImageBatch` class |
| `test_combined_models` | `combine_and_filter_predictions()` |
| `app.py` | `run_models_parallel()` - parallel model execution |

### File Structure

```
layout/
├── app.py                    # Main application
├── analytics.db              # SQLite visitor database (auto-created)
├── utils/
│   └── image_batch_classes.py
├── test_combined_models.py   # Model inference logic
└── APP_DOCUMENTATION.md      # This file
```

---

## Privacy & Compliance

- **Anonymization**: IPs are hashed (SHA256, truncated)
- **Session Tracking**: MD5 hash based on IP + hourly bucket
- **Cookie Banner**: GDPR-compliant consent mechanism
- **Data Retention**: No automatic purging (manual cleanup required)

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No classes selected | `gr.Error("Please select at least one class.")` |
| Invalid image in ZIP | Logged to console, skipped, error count shown |
| Empty ZIP | Returns empty gallery with status message |
| Analytics DB error | Silently caught, logged to stdout |

---

## Performance Notes

- **Threading**: Limited to 4 concurrent Gradio requests
- **Multiprocessing**: 3 worker processes for parallel model inference
- **Memory**: Each image creates temporary files (auto-cleaned)
- **Batch Size**: No hard limit; images processed with parallel models
- **Chart Generation**: Uses `matplotlib.use("Agg")` for headless rendering

### Parallel Processing Architecture

The application uses Python's `multiprocessing` module with `spawn` method for safe parallel execution from Gradio's threaded environment:

```python
# Process pool initialized at startup (before Gradio queue)
mp.set_start_method('spawn', force=True)

# 3 worker processes, one per model
_model_pool = ProcessPoolExecutor(max_workers=3)

# Models cached per-process after first load
_worker_models = {}  # Global in each worker
```

**Benefits:**
- Avoids GIL contention with YOLO models
- Workers stay alive between requests (model caching)
- Safe spawning from main thread before Gradio starts
- 5-minute timeout per model prediction

---

*Documentation generated for `app.py` — Layout Analysis Web Application*




