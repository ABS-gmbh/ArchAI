from typing import Tuple, Dict, List

import gradio as gr
import numpy as np
from PIL import Image
import zipfile
import os
import tempfile
import json
from datetime import datetime, timedelta
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sqlite3
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

matplotlib.use("Agg")  # Use non-interactive backend

# =====================================================================
# MULTIPROCESSING SETUP - Must be before any Gradio code
# =====================================================================
# Set spawn method for safe multiprocessing from threads
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from utils.image_batch_classes import coco_class_mapping, ImageBatch
from test_combined_models import combine_and_filter_predictions

# Script directory for model paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths (resolved at module level)
MODEL_PATHS = {
    'emanuskript': os.path.join(SCRIPT_DIR, "best_emanuskript_segmentation.pt"),
    'catmus': os.path.join(SCRIPT_DIR, "best_catmus.pt"),
    'zone': os.path.join(SCRIPT_DIR, "best_zone_detection.pt"),
}

# Model-specific classes to filter
MODEL_CLASSES = {
    'emanuskript': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20],
    'catmus': [1, 7],  # DefaultLine and InterlinearLine
    'zone': None,  # All classes
}

# Global dict to cache models in worker processes
_worker_models = {}


def _worker_init():
    """Initialize worker process - models loaded lazily on first use."""
    global _worker_models
    _worker_models = {}


def _run_single_model(args):
    """
    Run a single YOLO model prediction in worker process.
    Models are cached per-process to avoid reloading.
    """
    global _worker_models
    model_name, model_path, image_path, output_dir, classes = args
    
    # Lazy load model (cached per worker process)
    if model_name not in _worker_models:
        from ultralytics import YOLO
        _worker_models[model_name] = YOLO(model_path)
    
    model = _worker_models[model_name]
    
    # Create output directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Run prediction
    predict_kwargs = {
        'device': 'cpu',
        'iou': 0.3,
        'augment': False,
        'stream': False,
    }
    if classes is not None:
        predict_kwargs['classes'] = classes
    
    results = model.predict(image_path, **predict_kwargs)
    
    # Save results to JSON
    image_id = Path(image_path).stem
    json_path = os.path.join(model_dir, f"{image_id}.json")
    with open(json_path, 'w') as f:
        f.write(results[0].to_json())
    
    return model_name, model_dir


# Create process pool at module level (before Gradio starts)
# This avoids the hang when spawning from Gradio's threads
_model_pool = None


def _get_model_pool():
    """Get or create the model pool (lazy initialization)."""
    global _model_pool
    if _model_pool is None:
        _model_pool = ProcessPoolExecutor(
            max_workers=3,
            initializer=_worker_init,
        )
    return _model_pool


def run_models_parallel(image_path: str, output_dir: str) -> Dict[str, str]:
    """
    Run all 3 YOLO models in parallel using pre-initialized process pool.
    Returns dict of {model_name: labels_folder_path}.
    """
    pool = _get_model_pool()
    
    # Prepare args for each model
    model_args = [
        ('emanuskript', MODEL_PATHS['emanuskript'], image_path, output_dir, MODEL_CLASSES['emanuskript']),
        ('catmus', MODEL_PATHS['catmus'], image_path, output_dir, MODEL_CLASSES['catmus']),
        ('zone', MODEL_PATHS['zone'], image_path, output_dir, MODEL_CLASSES['zone']),
    ]
    
    # Submit all jobs
    futures = {pool.submit(_run_single_model, args): args[0] for args in model_args}
    
    # Collect results
    results = {}
    for future in as_completed(futures):
        model_name = futures[future]
        try:
            name, dir_path = future.result(timeout=300)  # 5 min timeout per model
            results[name] = dir_path
            print(f"  ✓ {model_name} model completed", flush=True)
        except Exception as e:
            print(f"  ✗ {model_name} model failed: {e}", flush=True)
            raise
    
    return results

# =====================================================================
# ANALYTICS CONFIG
# =====================================================================
ANALYTICS_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.db")
ANALYTICS_PASSWORD = "layout2024"  # Change this to your desired password
ANALYTICS_USERNAME = "admin"

# =====================================================================
# CONFIG
# =====================================================================

# Cookie consent banner - Fully inline JavaScript for Gradio compatibility
# Note: All analytics are now handled via Nginx logs -> Loki -> Grafana
COOKIE_BANNER_HTML = """
<style>
.cookie-banner-box {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #fff;
    padding: 20px 30px;
    z-index: 99999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.3);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    flex-wrap: wrap;
    gap: 15px;
    border-top: 3px solid #e94560;
}
.cookie-banner-box p {
    margin: 0;
    flex: 1;
    min-width: 300px;
    font-size: 14px;
    line-height: 1.5;
}
.cookie-banner-box .cookie-btns {
    display: flex;
    gap: 10px;
}
.cookie-banner-box .cbtn {
    padding: 12px 24px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s ease;
}
.cookie-banner-box .cbtn-accept {
    background: #e94560;
    color: white;
}
.cookie-banner-box .cbtn-accept:hover {
    background: #ff6b6b;
}
.cookie-banner-box .cbtn-decline {
    background: transparent;
    color: #fff;
    border: 2px solid #fff;
}
.cookie-banner-box .cbtn-decline:hover {
    background: rgba(255,255,255,0.1);
}
</style>

<div class="cookie-banner-box" id="gdpr-cookie-banner">
    <p>🍪 We use cookies and analytics to track usage and improve this service. 
    This includes storing anonymous visitor information (IP hash, browser, device type).
    By clicking "Accept", you consent to analytics tracking.</p>
    <div class="cookie-btns">
        <button class="cbtn cbtn-decline" onclick="document.cookie='analytics_consent=declined;expires='+new Date(Date.now()+31536000000).toUTCString()+';path=/';this.closest('.cookie-banner-box').style.display='none';">Decline</button>
        <button class="cbtn cbtn-accept" onclick="document.cookie='analytics_consent=accepted;expires='+new Date(Date.now()+31536000000).toUTCString()+';path=/';this.closest('.cookie-banner-box').style.display='none';">Accept Cookies</button>
    </div>
</div>

<script>
(function(){
    if(document.cookie.indexOf('analytics_consent=')>=0){
        var b=document.getElementById('gdpr-cookie-banner');
        if(b)b.style.display='none';
    }
})();
</script>
"""

# Final combined classes exposed to the user (25 classes)
FINAL_CLASSES = list(coco_class_mapping.keys())


# =====================================================================
# COCO / COMBINATION HELPERS
# =====================================================================

def _filter_coco_by_classes(coco_json: Dict, allowed_classes: List[str]) -> Dict:
    """Return a new COCO dict filtered to given class names."""
    id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    allowed_ids = {cid for cid, name in id_to_name.items() if name in allowed_classes}

    filtered_categories = [c for c in coco_json["categories"] if c["id"] in allowed_ids]
    filtered_annotations = [
        a for a in coco_json["annotations"] if a["category_id"] in allowed_ids
    ]

    used_image_ids = {a["image_id"] for a in filtered_annotations}
    filtered_images = [img for img in coco_json["images"] if img["id"] in used_image_ids]

    return {
        **coco_json,
        "categories": filtered_categories,
        "annotations": filtered_annotations,
        "images": filtered_images,
    }


# Predefined dark, vibrant colors for each class (RGB tuples, 0-1 range)
CLASS_COLOR_MAP = {
    'Border': '#8B0000',  # Dark red
    'Table': '#006400',  # Dark green
    'Diagram': '#00008B',  # Dark blue
    'Main script black': '#FF0000',  # Bright red
    'Main script coloured': '#FF4500',  # Orange red
    'Variant script black': '#8B008B',  # Dark magenta
    'Variant script coloured': '#FF1493',  # Deep pink
    'Historiated': '#FFD700',  # Gold
    'Inhabited': '#FF8C00',  # Dark orange
    'Zoo - Anthropomorphic': '#32CD32',  # Lime green
    'Embellished': '#FF00FF',  # Magenta
    'Plain initial- coloured': '#00CED1',  # Dark turquoise
    'Plain initial - Highlighted': '#00BFFF',  # Deep sky blue
    'Plain initial - Black': '#000000',  # Black
    'Page Number': '#DC143C',  # Crimson
    'Quire Mark': '#9932CC',  # Dark orchid
    'Running header': '#228B22',  # Forest green
    'Catchword': '#B22222',  # Fire brick
    'Gloss': '#4169E1',  # Royal blue
    'Illustrations': '#FF6347',  # Tomato
    'Column': '#2E8B57',  # Sea green
    'GraphicZone': '#8A2BE2',  # Blue violet
    'MusicLine': '#20B2AA',  # Light sea green
    'MusicZone': '#4682B4',  # Steel blue
    'Music': '#1E90FF',  # Dodger blue
}


def _draw_coco_on_image(
    image_path: str,
    coco_json: Dict,
    allowed_classes: List[str],
) -> np.ndarray:
    """Draw combined COCO annotations on the image with dark, visible colors."""
    from matplotlib.patches import Polygon, Rectangle
    import matplotlib.colors as mcolors

    coco_filtered = _filter_coco_by_classes(coco_json, allowed_classes)
    id_to_name = {c["id"]: c["name"] for c in coco_filtered["categories"]}

    if not coco_filtered["images"]:
        return np.array(Image.open(image_path).convert("RGB"))

    img_info = coco_filtered["images"][0]
    img_id = img_info["id"]
    anns = [a for a in coco_filtered["annotations"] if a["image_id"] == img_id]

    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.imshow(img)
    ax.axis("off")

    # Use predefined colors or fallback to dark colors
    def get_class_color(class_name: str) -> str:
        """Get color for a class, with fallback."""
        if class_name in CLASS_COLOR_MAP:
            return CLASS_COLOR_MAP[class_name]
        # Fallback: use hash-based color for unknown classes
        import hashlib
        hash_val = int(hashlib.md5(class_name.encode()).hexdigest()[:8], 16)
        hue = (hash_val % 360) / 360.0
        # Use dark, saturated colors
        rgb = mcolors.hsv_to_rgb([hue, 0.9, 0.7])
        return mcolors.rgb2hex(rgb)

    # Track label positions to avoid overlap
    label_positions = []
    
    def find_label_position(x, y, w, h, existing_positions, img_width, img_height):
        """Find a good position for label to avoid overlap."""
        # Try top-left corner first
        label_w, label_h = 150, 30  # Approximate label size
        candidates = [
            (x, y - label_h - 5),  # Above top-left
            (x, y),  # Top-left corner
            (x + w - label_w, y),  # Top-right corner
            (x, y + h + 5),  # Below bottom-left
        ]
        
        for pos_x, pos_y in candidates:
            # Check if position is within image bounds
            if pos_x < 0 or pos_y < 0 or pos_x + label_w > img_width or pos_y + label_h > img_height:
                continue
            
            # Check overlap with existing labels
            overlap = False
            for ex_x, ex_y in existing_positions:
                if abs(pos_x - ex_x) < label_w * 0.8 and abs(pos_y - ex_y) < label_h * 0.8:
                    overlap = True
                    break
            
            if not overlap:
                return pos_x, pos_y
        
        # If all positions overlap, use top-left anyway
        return x, y
    
    img_width, img_height = img.size
    
    for ann in anns:
        name = id_to_name[ann["category_id"]]
        color_hex = get_class_color(name)
        # Convert hex to RGB for matplotlib
        color_rgb = mcolors.hex2color(color_hex)

        segs = ann.get("segmentation", [])
        if segs and isinstance(segs, list) and len(segs[0]) >= 6:
            coords = segs[0]
            xs = coords[0::2]
            ys = coords[1::2]
            # Add semi-transparent fill
            poly = Polygon(
                list(zip(xs, ys)),
                closed=True,
                edgecolor=color_rgb,
                facecolor=color_rgb,
                linewidth=2.5,
                alpha=0.3,  # Semi-transparent fill
            )
            ax.add_patch(poly)
            # Also add edge outline with higher opacity
            poly_edge = Polygon(
                list(zip(xs, ys)),
                closed=True,
                edgecolor=color_rgb,
                facecolor="none",
                linewidth=2.5,
                alpha=0.8,  # More opaque edge
            )
            ax.add_patch(poly_edge)
            
            # Find good label position
            min_x, min_y = min(xs), min(ys)
            label_x, label_y = find_label_position(
                min_x, min_y, max(xs) - min_x, max(ys) - min_y,
                label_positions, img_width, img_height
            )
            label_positions.append((label_x, label_y))
            
            # Label with black text on white background for visibility
            ax.text(
                label_x,
                label_y,
                name,
                color='black',  # Black text for visibility
                fontsize=9,  # Slightly smaller to fit better
                fontweight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color_rgb,
                    linewidth=2,
                    alpha=0.7,  # Fully opaque label background
                ),
                zorder=10,  # Ensure labels are on top
            )
        else:
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            # Add semi-transparent fill
            rect = Rectangle(
                (x, y),
                w,
                h,
                edgecolor=color_rgb,
                facecolor=color_rgb,
                linewidth=2.5,
                alpha=0.3,  # Semi-transparent fill
            )
            ax.add_patch(rect)
            # Also add edge outline with higher opacity
            rect_edge = Rectangle(
                (x, y),
                w,
                h,
                edgecolor=color_rgb,
                facecolor="none",
                linewidth=2.5,
                alpha=0.8,  # More opaque edge
            )
            ax.add_patch(rect_edge)
            
            # Find good label position
            label_x, label_y = find_label_position(
                x, y, w, h, label_positions, img_width, img_height
            )
            label_positions.append((label_x, label_y))
            
            # Label with black text on white background for visibility
            ax.text(
                label_x,
                label_y,
                name,
                color='black',  # Black text for visibility
                fontsize=9,  # Slightly smaller to fit better
                fontweight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color_rgb,
                    linewidth=2,
                    alpha=0.7,  # Fully opaque label background
                ),
                zorder=10,  # Ensure labels are on top
            )

    plt.tight_layout()
    tmp_path = os.path.join(tempfile.gettempdir(), "tmp_vis.png")
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return np.array(Image.open(tmp_path).convert("RGB"))


def _stats_from_coco(coco_json: Dict) -> Dict[str, int]:
    """Return {class_name: count} from COCO annotations."""
    id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    counts: Dict[str, int] = {}
    for ann in coco_json["annotations"]:
        name = id_to_name.get(ann["category_id"], f"cls_{ann['category_id']}")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _stats_table(stats: Dict[str, int], image_name: str | None = None) -> pd.DataFrame:
    if not stats:
        return pd.DataFrame(columns=["Class", "Count"])
    rows = [{"Class": k, "Count": v} for k, v in sorted(stats.items())]
    df = pd.DataFrame(rows)
    if image_name:
        df.insert(0, "Image", image_name)
    return df


def _stats_graph(stats: Dict[str, int], title: str) -> str:
    if not stats:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        classes = list(sorted(stats.keys()))
        counts = [stats[c] for c in classes]
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(classes)), counts, color="steelblue")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(title)
        for b, c in zip(bars, counts):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height(),
                str(c),
                ha="center",
                va="bottom",
            )

    out_path = os.path.join(
        tempfile.gettempdir(),
        f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run_combined_on_path(
    image_path: str, conf: float, iou: float
) -> Tuple[Dict, np.ndarray]:
    """
    Run the three models on image_path, combine via ImageBatch helpers, and return:
    - combined COCO json (already filtered to coco_class_mapping)
    - annotated image (all final classes drawn)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run 3 models in parallel using pre-initialized process pool
        labels_folders = run_models_parallel(image_path, tmp_dir)

        # Combine & filter to coco_class_mapping
        coco_json = combine_and_filter_predictions(
            image_path, labels_folders, output_json_path=None
        )

        annotated = _draw_coco_on_image(image_path, coco_json, FINAL_CLASSES)

    return coco_json, annotated


# =====================================================================
# COCO MERGE FOR BATCH
# =====================================================================


def _merge_coco_list(coco_list: List[Dict]) -> Dict:
    """Merge multiple single-image COCO dicts into one."""
    merged = {
        "info": {"description": "Combined predictions", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # fixed categories from coco_class_mapping
    for name, cid in coco_class_mapping.items():
        merged["categories"].append({"id": cid, "name": name, "supercategory": ""})

    ann_id = 1
    img_id = 1
    for coco in coco_list:
        local_img_id_map = {}
        for img in coco["images"]:
            new_img = dict(img)
            new_img["id"] = img_id
            merged["images"].append(new_img)
            local_img_id_map[img["id"]] = img_id
            img_id += 1

        for ann in coco["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = local_img_id_map.get(ann["image_id"], ann["image_id"])
            merged["annotations"].append(new_ann)
            ann_id += 1

    return merged


# =====================================================================
# SINGLE IMAGE HANDLER
# =====================================================================


def process_single_image(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    final_classes: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    """Gradio handler for single image."""
    if image is None:
        return None, None, pd.DataFrame(columns=["Class", "Count"]), None

    if not final_classes:
        raise gr.Error("Please select at least one class.")

    # Save image to temp path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.fromarray(image.astype("uint8")).save(tmp.name)
        img_path = tmp.name

    coco_json, annotated_all = _run_combined_on_path(
        img_path, conf_threshold, iou_threshold
    )
    filtered_coco = _filter_coco_by_classes(coco_json, final_classes)
    annotated = _draw_coco_on_image(img_path, filtered_coco, final_classes)

    stats = _stats_from_coco(filtered_coco)
    stats_table = _stats_table(stats, image_name="image")
    stats_graph = _stats_graph(stats, "Single Image Statistics")

    os.unlink(img_path)

    # store for downloads
    global _single_coco_json
    _single_coco_json = filtered_coco

    return image, annotated, stats_table, stats_graph


_single_coco_json: Dict | None = None
_batch_coco_list: List[Dict] = []


def download_single_annotations() -> str | None:
    if not _single_coco_json:
        return None
    path = os.path.join(
        tempfile.gettempdir(),
        f"combined_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(path, "w") as f:
        json.dump(_single_coco_json, f, indent=2)
    return path


def download_single_image(annotated: np.ndarray) -> str | None:
    if annotated is None:
        return None
    path = os.path.join(
        tempfile.gettempdir(),
        f"combined_single_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
    )
    Image.fromarray(annotated.astype("uint8")).save(path, "JPEG", quality=95)
    return path


# =====================================================================
# BATCH HANDLERS
# =====================================================================


def process_batch_images(
    zip_file,
    conf_threshold: float,
    iou_threshold: float,
    final_classes: List[str],
):
    global _batch_coco_list
    _batch_coco_list = []

    if zip_file is None:
        return [], "Please upload a ZIP file.", pd.DataFrame(), pd.DataFrame(), None
    if not final_classes:
        raise gr.Error("Please select at least one class.")

    gallery = []
    per_image_tables = []
    total_stats: Dict[str, int] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_file.name, "r") as zf:
            zf.extractall(tmp_dir)

        # Valid image extensions
        IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp', '.gif'}
        
        # Patterns to skip (hidden files, system files, etc.)
        SKIP_PATTERNS = [
            '._',  # macOS resource forks
            '.DS_Store',  # macOS metadata
            'Thumbs.db',  # Windows thumbnail cache
            'desktop.ini',  # Windows folder settings
            '~$',  # Temporary files
        ]
        
        image_paths = []
        for root, dirs, files in os.walk(tmp_dir):
            # Skip __MACOSX directories (macOS metadata)
            if '__MACOSX' in root or '__MACOSX' in dirs:
                dirs[:] = [d for d in dirs if d != '__MACOSX']
                continue
            
            # Skip hidden directories (starting with .)
            dirs[:] = [d for d in dirs if not d.startswith('.')]
                
            for fn in files:
                # Skip hidden/system files
                if any(fn.startswith(pattern) for pattern in SKIP_PATTERNS):
                    continue
                
                # Skip files starting with . (hidden files on Unix/Linux)
                if fn.startswith('.'):
                    continue
                    
                # Check file extension (case-insensitive)
                file_ext = os.path.splitext(fn)[1].lower()
                if file_ext not in IMAGE_EXTENSIONS:
                    continue
                
                full_path = os.path.join(root, fn)
                
                # Skip if not a regular file (e.g., symlinks, directories)
                if not os.path.isfile(full_path):
                    continue
                
                # Skip empty files
                if os.path.getsize(full_path) == 0:
                    print(f"WARNING ⚠️ Skipping empty file: {full_path}")
                    continue
                
                # Verify it's actually a valid image file by trying to open it
                try:
                    # Try to open and validate the image
                    with Image.open(full_path) as test_img:
                        # Verify it's a valid image format
                        test_img.verify()
                    
                    # Reopen for format check (verify() closes the file)
                    with Image.open(full_path) as test_img:
                        # Check if we can actually read the image data
                        test_img.load()
                        # Check if image has valid dimensions
                        if test_img.size[0] == 0 or test_img.size[1] == 0:
                            print(f"WARNING ⚠️ Skipping image with invalid dimensions: {full_path}")
                            continue
                    
                    # All checks passed, add to list
                    image_paths.append(full_path)
                except Exception as e:
                    # Skip invalid image files
                    print(f"WARNING ⚠️ Skipping invalid/non-image file: {full_path} (Error: {str(e)})")
                    continue

        processed_count = 0
        error_count = 0
        error_messages = []
        
        for path in sorted(image_paths):
            fn = os.path.basename(path)
            try:
                coco_json, _ = _run_combined_on_path(path, conf_threshold, iou_threshold)
                filtered = _filter_coco_by_classes(coco_json, final_classes)
                _batch_coco_list.append(filtered)

                annotated = _draw_coco_on_image(path, filtered, final_classes)
                gallery.append((annotated, fn))

                stats = _stats_from_coco(filtered)
                per_image_tables.append(_stats_table(stats, image_name=fn))
                for k, v in stats.items():
                    total_stats[k] = total_stats.get(k, 0) + v
                
                processed_count += 1
            except Exception as e:
                error_count += 1
                error_msg = f"Error processing {fn}: {str(e)}"
                error_messages.append(error_msg)
                print(f"WARNING ⚠️ {error_msg}")
                import traceback
                traceback.print_exc()
                continue

    per_image_df = (
        pd.concat(per_image_tables, ignore_index=True) if per_image_tables else pd.DataFrame()
    )
    summary_df = _stats_table(total_stats)
    graph_path = _stats_graph(total_stats, "Batch Statistics")

    # Build status message
    status_parts = [f"Processed {processed_count} images successfully."]
    if error_count > 0:
        status_parts.append(f"Skipped {error_count} files with errors.")
        if error_messages:
            status_parts.append(f"Errors: {', '.join(error_messages[:3])}")  # Show first 3 errors
    status = " ".join(status_parts)
    return gallery, status, per_image_df, summary_df, graph_path


def download_batch_annotations() -> str | None:
    if not _batch_coco_list:
        return None
    merged = _merge_coco_list(_batch_coco_list)
    path = os.path.join(
        tempfile.gettempdir(),
        f"combined_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    return path


def download_batch_zip(gallery_images: List[Tuple[str, np.ndarray]]) -> str | None:
    if not gallery_images:
        return None
    merged = _merge_coco_list(_batch_coco_list)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(
        tempfile.gettempdir(), f"combined_batch_results_{ts}.zip"
    )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # images
        for idx, (img_np, caption) in enumerate(gallery_images, start=1):
            img = Image.fromarray(img_np.astype("uint8"))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            zf.writestr(f"images/{caption}", buf.getvalue())
        # annotations
        zf.writestr("annotations.json", json.dumps(merged, indent=2))
    return zip_path


# =====================================================================
# ANALYTICS TAB HANDLERS
# =====================================================================

def verify_analytics_password(username: str, password: str) -> Tuple[bool, str]:
    """Verify username and password for analytics access."""
    if username == ANALYTICS_USERNAME and password == ANALYTICS_PASSWORD:
        return True, "✅ Access granted! Loading analytics..."
    return False, "❌ Invalid credentials. Please try again."


def get_analytics_data(days_filter: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Fetch analytics data from SQLite database."""
    try:
        conn = sqlite3.connect(ANALYTICS_DB)
        
        # Calculate date filter
        if days_filter > 0:
            date_cutoff = (datetime.now() - timedelta(days=days_filter)).strftime('%Y-%m-%d')
            date_filter = f"WHERE DATE(timestamp) >= '{date_cutoff}'"
        else:
            date_filter = ""
        
        # Get visits by country
        country_query = f"""
            SELECT 
                COALESCE(NULLIF(country, ''), 'Unknown') as country,
                COUNT(*) as visits,
                COUNT(DISTINCT visitor_id) as unique_visitors
            FROM visits
            {date_filter}
            GROUP BY country
            ORDER BY visits DESC
        """
        country_df = pd.read_sql_query(country_query, conn)
        
        # Get visits by day
        daily_query = f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as visits,
                COUNT(DISTINCT visitor_id) as unique_visitors
            FROM visits
            {date_filter}
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """
        daily_df = pd.read_sql_query(daily_query, conn)
        
        # Get recent visits
        recent_query = f"""
            SELECT 
                timestamp,
                COALESCE(NULLIF(country, ''), 'Unknown') as country,
                COALESCE(NULLIF(city, ''), 'Unknown') as city,
                browser,
                os,
                device,
                page,
                action
            FROM visits
            {date_filter}
            ORDER BY timestamp DESC
            LIMIT 100
        """
        recent_df = pd.read_sql_query(recent_query, conn)
        
        # Get summary stats
        today = datetime.now().strftime('%Y-%m-%d')
        summary_query = f"""
            SELECT 
                COUNT(*) as total_visits,
                COUNT(DISTINCT visitor_id) as unique_visitors,
                COUNT(DISTINCT COALESCE(NULLIF(country, ''), 'Unknown')) as countries,
                SUM(CASE WHEN DATE(timestamp) = '{today}' THEN 1 ELSE 0 END) as today_visits,
                SUM(CASE WHEN is_bot = 1 THEN 1 ELSE 0 END) as bot_visits
            FROM visits
            {date_filter}
        """
        summary_df = pd.read_sql_query(summary_query, conn)
        summary = summary_df.iloc[0].to_dict() if len(summary_df) > 0 else {}
        
        conn.close()
        return country_df, daily_df, recent_df, summary
        
    except Exception as e:
        print(f"Analytics DB error: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}


def create_world_map(country_df: pd.DataFrame) -> str:
    """Create a world map visualization of visitors by country using matplotlib."""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        if country_df.empty or len(country_df) == 0:
            ax.text(0.5, 0.5, "No visitor data available", ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            # Create a horizontal bar chart of countries (better than empty map)
            df = country_df.head(15).copy()  # Top 15 countries
            
            # Sort by visits ascending for horizontal bar (so highest is at top)
            df = df.sort_values('visits', ascending=True)
            
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
            
            y_pos = np.arange(len(df))
            bars = ax.barh(y_pos, df['visits'], color=colors, edgecolor='navy', linewidth=0.5)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df['country'], fontsize=11)
            ax.set_xlabel('Number of Visits', fontsize=12)
            ax.set_title('🌍 Top Countries by Visits', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for i, (bar, visits, unique) in enumerate(zip(bars, df['visits'], df['unique_visitors'])):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{int(visits)} ({int(unique)} unique)',
                       va='center', fontsize=9, color='#333')
            
            # Add grid
            ax.xaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
            # Style
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Expand x-axis to fit labels
            max_val = df['visits'].max()
            ax.set_xlim(0, max_val * 1.3)
        
        plt.tight_layout()
        map_path = os.path.join(tempfile.gettempdir(), f"analytics_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return map_path
        
    except Exception as e:
        print(f"Map creation error: {e}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f"Error creating visualization: {str(e)}", ha='center', va='center', fontsize=12)
        ax.axis('off')
        map_path = os.path.join(tempfile.gettempdir(), "analytics_map_error.png")
        fig.savefig(map_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return map_path


def create_daily_chart(daily_df: pd.DataFrame) -> str:
    """Create a bar chart of daily visits."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if daily_df.empty:
        ax.text(0.5, 0.5, "No visit data available", ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        # Reverse to show oldest first
        df = daily_df.iloc[::-1].copy()
        x = range(len(df))
        
        bars = ax.bar(x, df['visits'], color='steelblue', alpha=0.8, label='Total Visits')
        ax.plot(x, df['unique_visitors'], color='orange', marker='o', linewidth=2, label='Unique Visitors')
        
        ax.set_xticks(x)
        ax.set_xticklabels(df['date'], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Count')
        ax.set_title('Daily Visits (Last 30 Days)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(tempfile.gettempdir(), f"daily_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return chart_path


def refresh_analytics(days_filter: int) -> Tuple[str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refresh all analytics data and visualizations."""
    country_df, daily_df, recent_df, summary = get_analytics_data(days_filter)
    
    # Create summary text
    if summary:
        summary_text = f"""
### 📊 Analytics Summary

| Metric | Value |
|--------|-------|
| **Total Visits** | {summary.get('total_visits', 0):,} |
| **Unique Visitors** | {summary.get('unique_visitors', 0):,} |
| **Countries** | {summary.get('countries', 0)} |
| **Visits Today** | {summary.get('today_visits', 0):,} |
| **Bot Visits** | {summary.get('bot_visits', 0):,} |
"""
    else:
        summary_text = "### No analytics data available"
    
    # Create visualizations
    map_path = create_world_map(country_df)
    chart_path = create_daily_chart(daily_df)
    
    return summary_text, map_path, chart_path, country_df, daily_df, recent_df


def analytics_login(username: str, password: str) -> Tuple[gr.update, gr.update, str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Handle analytics login and load data if successful."""
    is_valid, message = verify_analytics_password(username, password)
    
    if is_valid:
        # Load analytics data
        summary_text, map_path, chart_path, country_df, daily_df, recent_df = refresh_analytics(30)
        return (
            gr.update(visible=False),  # Hide login panel
            gr.update(visible=True),   # Show analytics content
            summary_text,
            map_path,
            chart_path,
            country_df,
            daily_df,
            recent_df
        )
    else:
        return (
            gr.update(visible=True),   # Keep login panel visible
            gr.update(visible=False),  # Keep analytics hidden
            message,
            None,
            None,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )


# =====================================================================
# GRADIO UI
# =====================================================================


with gr.Blocks() as demo:
    gr.HTML(COOKIE_BANNER_HTML)
    gr.Markdown("## Combined Manuscript Models (emanuskript + catmus + zone)")
    
    with gr.Tabs():
        # Single image
        with gr.TabItem("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Input Image", type="numpy")
                    with gr.Accordion("Detection Settings", open=True):
                        with gr.Row():
                            conf_threshold = gr.Slider(
                                label="Confidence Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.25,
                            )
                            iou_threshold = gr.Slider(
                                label="IoU Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.3,
                            )
                    with gr.Accordion("Final Classes (combined)", open=True):
                        final_classes_box = gr.CheckboxGroup(
                            label="Final Classes",
                            choices=FINAL_CLASSES,
                            value=FINAL_CLASSES,
                        )
                        with gr.Row():
                            select_all_btn = gr.Button("Select All", size="sm")
                            unselect_all_btn = gr.Button("Unselect All", size="sm")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        detect_btn = gr.Button(
                            "Run 3 Models + Combine", variant="primary"
                        )
                        
                with gr.Column():
                    output_image = gr.Image(
                        label="Combined Result (final classes)", type="numpy"
                    )
                    with gr.Row():
                        single_download_json_btn = gr.Button(
                            "📄 Download Annotations (JSON)", size="sm"
                        )
                        single_download_image_btn = gr.Button(
                            "🖼️ Download Image", size="sm"
                        )
                    single_json_output = gr.File(
                        label="Single JSON", visible=True, height=50
                    )
                    single_image_output = gr.File(
                        label="Single Image", visible=True, height=50
                    )
                    with gr.Accordion("📊 Statistics", open=False):
                        with gr.Tabs():
                            with gr.TabItem("Table"):
                                single_stats_table = gr.Dataframe(
                                    label="Detection Statistics",
                                    headers=["Class", "Count"],
                                    wrap=True,
                                )
                            with gr.TabItem("Graph"):
                                single_stats_graph = gr.Image(
                                    label="Statistics Graph", type="filepath"
                                )

        # Batch tab
        with gr.TabItem("Batch Processing (ZIP)"):
            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(
                        label="Upload ZIP with images", file_types=[".zip"]
                    )
                    with gr.Accordion("Detection Settings", open=True):
                        with gr.Row():
                            batch_conf_threshold = gr.Slider(
                                label="Confidence Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.25,
                            )
                            batch_iou_threshold = gr.Slider(
                                label="IoU Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.05,
                                value=0.3,
                            )
                    with gr.Accordion("Final Classes (combined)", open=True):
                        batch_final_classes_box = gr.CheckboxGroup(
                            label="Final Classes",
                            choices=FINAL_CLASSES,
                            value=FINAL_CLASSES,
                        )
                        with gr.Row():
                            batch_select_all_btn = gr.Button("Select All", size="sm")
                            batch_unselect_all_btn = gr.Button(
                                "Unselect All", size="sm"
                            )
                    batch_status = gr.Textbox(
                        label="Processing Status",
                        value="Ready to process ZIP file...",
                        interactive=False,
                        max_lines=3,
                    )
                    with gr.Row():
                        clear_batch_btn = gr.Button("Clear")
                        process_batch_btn = gr.Button(
                            "Process ZIP", variant="primary"
                        )
                        
                with gr.Column():
                    batch_gallery = gr.Gallery(
                        label="Batch Results (combined)",
                        show_label=True,
                        columns=2,
                        rows=2,
                        height="auto",
                        type="numpy",
                    )
                    with gr.Row():
                        download_json_btn = gr.Button(
                            "📄 Download COCO Annotations (JSON)", size="sm"
                        )
                        download_zip_btn = gr.Button(
                            "📦 Download Results (ZIP)", size="sm"
                        )
                    json_file_output = gr.File(
                        label="Batch JSON", visible=True, height=50
                    )
                    zip_file_output = gr.File(
                        label="Batch ZIP", visible=True, height=50
                    )
                    with gr.Accordion("📊 Statistics", open=False):
                        with gr.Tabs():
                            with gr.TabItem("Per Image"):
                                batch_stats_table = gr.Dataframe(
                                    label="Per Image Statistics", wrap=True
                                )
                            with gr.TabItem("Overall Summary"):
                                batch_stats_summary_table = gr.Dataframe(
                                    label="Overall Statistics", wrap=True
                                )
                            with gr.TabItem("Graph"):
                                batch_stats_graph = gr.Image(
                                    label="Batch Statistics Graph", type="filepath"
                                )

        # Analytics tab (password protected)
        with gr.TabItem("📈 Analytics"):
            with gr.Column() as analytics_login_panel:
                gr.Markdown("### 🔐 Analytics Login")
                gr.Markdown("Enter credentials to access visitor analytics.")
                with gr.Row():
                    analytics_username = gr.Textbox(
                        label="Username",
                        placeholder="Enter username",
                        scale=1
                    )
                    analytics_password = gr.Textbox(
                        label="Password",
                        placeholder="Enter password",
                        type="password",
                        scale=1
                    )
                analytics_login_btn = gr.Button("🔓 Login", variant="primary")
                analytics_login_status = gr.Markdown("")
            
            with gr.Column(visible=False) as analytics_content:
                with gr.Row():
                    analytics_summary = gr.Markdown("### Loading analytics...")
                    with gr.Column():
                        analytics_days_filter = gr.Slider(
                            label="Days to show",
                            minimum=1,
                            maximum=365,
                            value=30,
                            step=1
                        )
                        analytics_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                
                with gr.Tabs():
                    with gr.TabItem("🗺️ World Map"):
                        analytics_map = gr.Image(label="Visitors by Country", type="filepath")
                    
                    with gr.TabItem("📊 Daily Chart"):
                        analytics_daily_chart = gr.Image(label="Daily Visits", type="filepath")
                    
                    with gr.TabItem("🌍 By Country"):
                        analytics_country_table = gr.Dataframe(
                            label="Visits by Country",
                            headers=["country", "visits", "unique_visitors"],
                            wrap=True
                        )
                    
                    with gr.TabItem("📅 By Day"):
                        analytics_daily_table = gr.Dataframe(
                            label="Daily Statistics",
                            headers=["date", "visits", "unique_visitors"],
                            wrap=True
                        )
                    
                    with gr.TabItem("🕐 Recent Visits"):
                        analytics_recent_table = gr.Dataframe(
                            label="Recent Visits (Last 100)",
                            wrap=True
                        )
                
                analytics_logout_btn = gr.Button("🚪 Logout", size="sm")


    # Callbacks
    detect_btn.click(
        fn=process_single_image,
        inputs=[input_image, conf_threshold, iou_threshold, final_classes_box],
        outputs=[input_image, output_image, single_stats_table, single_stats_graph],
    )
    clear_btn.click(
        fn=lambda: (None, None, pd.DataFrame(columns=["Class", "Count"]), None),
        inputs=None,
        outputs=[input_image, output_image, single_stats_table, single_stats_graph],
    )
    select_all_btn.click(fn=lambda: FINAL_CLASSES, outputs=[final_classes_box])
    unselect_all_btn.click(fn=lambda: [], outputs=[final_classes_box])

    process_batch_btn.click(
        fn=process_batch_images,
        inputs=[
            zip_file,
            batch_conf_threshold,
            batch_iou_threshold,
            batch_final_classes_box,
        ],
        outputs=[
            batch_gallery,
            batch_status,
            batch_stats_table,
            batch_stats_summary_table,
            batch_stats_graph,
        ],
    )
    clear_batch_btn.click(
        fn=lambda: (None, [], "Ready to process ZIP file..."),
        inputs=None,
        outputs=[zip_file, batch_gallery, batch_status],
    )
    batch_select_all_btn.click(
        fn=lambda: FINAL_CLASSES, outputs=[batch_final_classes_box]
    )
    batch_unselect_all_btn.click(fn=lambda: [], outputs=[batch_final_classes_box])

    single_download_json_btn.click(
        fn=download_single_annotations, inputs=None, outputs=[single_json_output]
    )
    single_download_image_btn.click(
        fn=lambda img: download_single_image(img),
        inputs=[output_image],
        outputs=[single_image_output],
    )

    download_json_btn.click(
        fn=download_batch_annotations, inputs=None, outputs=[json_file_output]
    )
    download_zip_btn.click(
        fn=lambda gallery: download_batch_zip(gallery),
        inputs=[batch_gallery],
        outputs=[zip_file_output],
    )
    
    # Analytics tab callbacks
    analytics_login_btn.click(
        fn=analytics_login,
        inputs=[analytics_username, analytics_password],
        outputs=[
            analytics_login_panel,
            analytics_content,
            analytics_summary,
            analytics_map,
            analytics_daily_chart,
            analytics_country_table,
            analytics_daily_table,
            analytics_recent_table
        ]
    )
    
    analytics_refresh_btn.click(
        fn=refresh_analytics,
        inputs=[analytics_days_filter],
        outputs=[
            analytics_summary,
            analytics_map,
            analytics_daily_chart,
            analytics_country_table,
            analytics_daily_table,
            analytics_recent_table
        ]
    )
    
    analytics_logout_btn.click(
        fn=lambda: (
            gr.update(visible=True),   # Show login panel
            gr.update(visible=False),  # Hide analytics content
            "",                         # Clear username
            ""                          # Clear password
        ),
        inputs=None,
        outputs=[
            analytics_login_panel,
            analytics_content,
            analytics_username,
            analytics_password
        ]
    )



def _cleanup_pool():
    """Cleanup worker pool on shutdown."""
    global _model_pool
    if _model_pool is not None:
        _model_pool.shutdown(wait=False)
        _model_pool = None


if __name__ == "__main__":
    import atexit
    atexit.register(_cleanup_pool)
    
    # Pre-initialize the pool before Gradio starts (avoids spawn from threads)
    print("Initializing model worker pool...", flush=True)
    _get_model_pool()
    print("Worker pool ready.", flush=True)
    
    demo.queue()
    demo.launch(
        debug=False,
        show_error=True,
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        max_threads=4,
        inbrowser=False,
        ssl_verify=True,
        quiet=False,
    )


