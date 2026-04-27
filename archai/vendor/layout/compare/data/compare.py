"""
Main comparison script: Compare old models vs new models vs ground truth.
Calculates mAP@50, mAP@[.50:.95], Precision, Recall.
Creates side-by-side visualization.
"""
import os
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not available. Metrics calculation will be limited.")
    COCO = None
    COCOeval = None

import tempfile

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from original_annotations import load_ground_truth
from old_models import process_dataset as process_old_models
from new_models import process_dataset as process_new_models


def draw_coco_annotations_simple(image_path, coco_json, title="", ax=None):
    """
    Draw COCO annotations on image (simpler version for comparison).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    
    img = Image.open(image_path).convert("RGB")
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis("off")
    
    if not coco_json.get("images"):
        return ax
    
    img_info = coco_json["images"][0]
    img_id = img_info["id"]
    anns = [a for a in coco_json["annotations"] if a["image_id"] == img_id]
    
    id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    
    # Color map
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_map = {}
    
    # Track label positions to avoid overlap
    placed_labels = []
    
    def find_label_position(bbox, text_width, text_height, image_width, image_height):
        """Find a good position for label to avoid overlap."""
        x, y, w, h = bbox
        candidates = [
            (x, y - text_height - 5),  # Above top-left
            (x, y),  # Top-left corner
            (x + w - text_width, y),  # Top-right corner
            (x, y + h + 5),  # Below bottom-left
        ]
        
        for pos_x, pos_y in candidates:
            # Check if position is within image bounds
            if pos_x < 0 or pos_y < 0 or pos_x + text_width > image_width or pos_y + text_height > image_height:
                continue
            
            # Check overlap with existing labels
            overlap = False
            for placed_x, placed_y, placed_w, placed_h in placed_labels:
                if not (pos_x + text_width < placed_x or pos_x > placed_x + placed_w or
                        pos_y + text_height < placed_y or pos_y > placed_y + placed_h):
                    overlap = True
                    break
            
            if not overlap:
                return pos_x, pos_y
        
        # If all positions overlap, use top-left anyway
        return x, y
    
    img_width, img_height = img.size
    
    for ann in anns:
        name = id_to_name.get(ann["category_id"], f"cls_{ann['category_id']}")
        
        # Get or assign color
        if name not in color_map:
            color_idx = len(color_map) % len(colors)
            color_map[name] = colors[color_idx]
        
        color = color_map[name]
        
        # Get bbox for label positioning
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if not bbox or len(bbox) < 4:
            # Try to get bbox from segmentation
            segs = ann.get("segmentation", [])
            if segs and isinstance(segs, list) and len(segs) > 0:
                if isinstance(segs[0], list) and len(segs[0]) >= 6:
                    coords = segs[0]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                else:
                    continue
            else:
                continue
        
        x, y, w, h = bbox
        
        # Draw segmentation or bbox
        segs = ann.get("segmentation", [])
        if segs and isinstance(segs, list) and len(segs) > 0:
            if isinstance(segs[0], list) and len(segs[0]) >= 6:
                # Polygon
                coords = segs[0]
                xs = coords[0::2]
                ys = coords[1::2]
                poly = patches.Polygon(
                    list(zip(xs, ys)),
                    closed=True,
                    edgecolor=color,
                    facecolor=color,
                    linewidth=2,
                    alpha=0.3,
                )
                ax.add_patch(poly)
                # Edge
                poly_edge = patches.Polygon(
                    list(zip(xs, ys)),
                    closed=True,
                    edgecolor=color,
                    facecolor="none",
                    linewidth=2,
                    alpha=0.8,
                )
                ax.add_patch(poly_edge)
        else:
            # Bbox
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                edgecolor=color,
                facecolor=color,
                linewidth=2,
                alpha=0.3,
            )
            ax.add_patch(rect)
            # Edge
            rect_edge = patches.Rectangle(
                (x, y),
                w,
                h,
                edgecolor=color,
                facecolor="none",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(rect_edge)
        
        # Add label
        # Estimate text size (approximate)
        text_width = len(name) * 7  # Approximate character width
        text_height = 12  # Approximate text height
        
        label_x, label_y = find_label_position(bbox, text_width, text_height, img_width, img_height)
        placed_labels.append((label_x, label_y, text_width, text_height))
        
        # Draw label with background
        # Convert color to RGB tuple if it's an array
        if isinstance(color, np.ndarray):
            edge_color = tuple(color[:3])
        elif isinstance(color, (list, tuple)) and len(color) >= 3:
            edge_color = tuple(color[:3])
        else:
            edge_color = color
        
        ax.text(
            label_x,
            label_y,
            name,
            color='black',
            fontsize=9,
            fontweight='bold',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=edge_color,
                linewidth=2,
                alpha=0.9,
            ),
            zorder=10,  # Ensure labels are on top
        )
    
    return ax


def validate_and_fix_annotation(ann, img_width, img_height):
    """
    Validate and fix annotation segmentation/bbox.
    Converts bbox to polygon if segmentation is missing or invalid.
    """
    segs = ann.get("segmentation", [])
    bbox = ann.get("bbox", [0, 0, 0, 0])
    
    # Check if segmentation is valid
    has_valid_seg = False
    if segs and isinstance(segs, list) and len(segs) > 0:
        # Check if it's a polygon (list of coordinates)
        if isinstance(segs[0], list) and len(segs[0]) >= 6:
            # Valid polygon
            has_valid_seg = True
        # Check if it's RLE (dict)
        elif isinstance(segs, dict) or (isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], dict)):
            # RLE format - assume valid
            has_valid_seg = True
    
    # If no valid segmentation, create polygon from bbox
    if not has_valid_seg and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = bbox
        # Create polygon from bbox: [x, y, x+w, y, x+w, y+h, x, y+h]
        polygon = [x, y, x + w, y, x + w, y + h, x, y + h]
        ann["segmentation"] = [polygon]
        # Update area if needed
        if ann.get("area", 0) == 0:
            ann["area"] = w * h
        has_valid_seg = True
    
    return has_valid_seg


def filter_valid_annotations(coco_dict):
    """
    Filter out annotations with invalid segmentation/bbox.
    Convert bbox-only annotations to polygon format.
    """
    # Get image dimensions
    img_id_to_size = {}
    for img in coco_dict["images"]:
        img_id_to_size[img["id"]] = (img["width"], img["height"])
    
    valid_annotations = []
    for ann in coco_dict["annotations"]:
        img_id = ann["image_id"]
        if img_id in img_id_to_size:
            img_width, img_height = img_id_to_size[img_id]
            if validate_and_fix_annotation(ann, img_width, img_height):
                valid_annotations.append(ann)
    
    coco_dict["annotations"] = valid_annotations
    return coco_dict


def calculate_metrics(gt_coco, pred_coco, output_dir):
    """
    Calculate mAP@50, mAP@[.50:.95], Precision, Recall using pycocotools.
    
    Args:
        gt_coco: Ground truth COCO format dict
        pred_coco: Predictions COCO format dict
        output_dir: Directory to save results
    
    Returns:
        Dictionary with metrics
    """
    if not HAS_PYCOCOTOOLS:
        return {
            'mAP@50': 0.0,
            'mAP@[.50:.95]': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'error': 'pycocotools not available'
        }
    
    # Filter and fix invalid annotations
    gt_coco_clean = filter_valid_annotations(gt_coco.copy())
    pred_coco_clean = filter_valid_annotations(pred_coco.copy())
    
    if len(gt_coco_clean["annotations"]) == 0:
        print("Warning: No valid ground truth annotations after filtering")
        return {
            'mAP@50': 0.0,
            'mAP@[.50:.95]': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'error': 'No valid GT annotations'
        }
    
    if len(pred_coco_clean["annotations"]) == 0:
        print("Warning: No valid prediction annotations after filtering")
        return {
            'mAP@50': 0.0,
            'mAP@[.50:.95]': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'error': 'No valid prediction annotations'
        }
    
    # Save to temporary JSON files for pycocotools
    gt_file = os.path.join(output_dir, "gt_temp.json")
    pred_file = os.path.join(output_dir, "pred_temp.json")
    
    with open(gt_file, 'w') as f:
        json.dump(gt_coco_clean, f)
    
    with open(pred_file, 'w') as f:
        json.dump(pred_coco_clean, f)
    
    # Load with pycocotools
    try:
        gt_coco_obj = COCO(gt_file)
        pred_coco_obj = COCO(pred_file)
    except Exception as e:
        print(f"Error loading COCO files: {e}")
        return {
            'mAP@50': 0.0,
            'mAP@[.50:.95]': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'error': f'COCO load error: {str(e)}'
        }
    
    # Get all image IDs
    img_ids = sorted(gt_coco_obj.getImgIds())
    
    if len(img_ids) == 0:
        return {
            'mAP@50': 0.0,
            'mAP@[.50:.95]': 0.0,
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'error': 'No images in GT'
        }
    
    # Get all category IDs from ground truth
    cat_ids = sorted(gt_coco_obj.getCatIds())
    
    # Try segmentation evaluation first, fall back to bbox if it fails
    eval_type = 'segm'
    try:
        coco_eval = COCOeval(gt_coco_obj, pred_coco_obj, eval_type)
        coco_eval.params.imgIds = img_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract metrics
        metrics = {
            'mAP@50': float(coco_eval.stats[1]),  # mAP@0.50
            'mAP@[.50:.95]': float(coco_eval.stats[0]),  # mAP@[.50:.95]
            'mAP@75': float(coco_eval.stats[2]),  # mAP@0.75
            'mAP_small': float(coco_eval.stats[3]),
            'mAP_medium': float(coco_eval.stats[4]),
            'mAP_large': float(coco_eval.stats[5]),
            'mAR_1': float(coco_eval.stats[6]),
            'mAR_10': float(coco_eval.stats[7]),
            'mAR_100': float(coco_eval.stats[8]),
            'mAR_small': float(coco_eval.stats[9]),
            'mAR_medium': float(coco_eval.stats[10]),
            'mAR_large': float(coco_eval.stats[11]),
        }
        
        # Calculate Precision and Recall
        precision = metrics['mAP@50']  # Approximate
        recall = metrics['mAR_100']  # Maximum recall with 100 detections
        
        metrics['Precision'] = precision
        metrics['Recall'] = recall
        metrics['F1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    except Exception as e:
        print(f"Error during {eval_type} evaluation: {e}")
        # Try bbox evaluation as fallback
        try:
            print("Trying bbox evaluation as fallback...")
            coco_eval = COCOeval(gt_coco_obj, pred_coco_obj, 'bbox')
            coco_eval.params.imgIds = img_ids
            coco_eval.params.catIds = cat_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            metrics = {
                'mAP@50': float(coco_eval.stats[1]),
                'mAP@[.50:.95]': float(coco_eval.stats[0]),
                'mAP@75': float(coco_eval.stats[2]),
                'mAP_small': float(coco_eval.stats[3]),
                'mAP_medium': float(coco_eval.stats[4]),
                'mAP_large': float(coco_eval.stats[5]),
                'mAR_1': float(coco_eval.stats[6]),
                'mAR_10': float(coco_eval.stats[7]),
                'mAR_100': float(coco_eval.stats[8]),
                'mAR_small': float(coco_eval.stats[9]),
                'mAR_medium': float(coco_eval.stats[10]),
                'mAR_large': float(coco_eval.stats[11]),
            }
            
            precision = metrics['mAP@50']
            recall = metrics['mAR_100']
            
            metrics['Precision'] = precision
            metrics['Recall'] = recall
            metrics['F1'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics['eval_type'] = 'bbox'  # Note that we used bbox evaluation
            
        except Exception as e2:
            print(f"Error during bbox evaluation: {e2}")
            import traceback
            traceback.print_exc()
            metrics = {
                'mAP@50': 0.0,
                'mAP@[.50:.95]': 0.0,
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'error': f'{eval_type} error: {str(e)}, bbox error: {str(e2)}'
            }
    
    return metrics


def create_comparison_visualization(image_path, gt_coco, old_coco, new_coco, output_path):
    """
    Create side-by-side comparison: Original + GT | Old Models | New Models
    """
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    # Left: Original image with ground truth
    draw_coco_annotations_simple(image_path, gt_coco, "Ground Truth", axes[0])
    
    # Middle: Old models
    draw_coco_annotations_simple(image_path, old_coco, "Old Models", axes[1])
    
    # Right: New models
    draw_coco_annotations_simple(image_path, new_coco, "New Models", axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison visualization to {output_path}")


def align_categories(gt_coco, pred_coco):
    """
    Align category IDs between GT and predictions.
    Maps prediction categories to GT categories by name.
    """
    # Create name to ID maps
    gt_name_to_id = {c["name"]: c["id"] for c in gt_coco["categories"]}
    pred_name_to_id = {c["name"]: c["id"] for c in pred_coco["categories"]}
    
    # Create mapping from pred category ID to GT category ID
    pred_to_gt_map = {}
    for pred_name, pred_id in pred_name_to_id.items():
        if pred_name in gt_name_to_id:
            pred_to_gt_map[pred_id] = gt_name_to_id[pred_name]
        else:
            # If category doesn't exist in GT, skip it
            print(f"Warning: Category '{pred_name}' not in ground truth, skipping...")
    
    # Update prediction annotations
    new_anns = []
    for ann in pred_coco["annotations"]:
        old_cat_id = ann["category_id"]
        if old_cat_id in pred_to_gt_map:
            new_ann = ann.copy()
            new_ann["category_id"] = pred_to_gt_map[old_cat_id]
            new_anns.append(new_ann)
    
    pred_coco["annotations"] = new_anns
    
    # Update categories to match GT
    pred_coco["categories"] = [
        c for c in gt_coco["categories"]
        if c["name"] in pred_name_to_id
    ]
    
    return pred_coco


def main():
    """
    Main comparison function.
    """
    # Paths
    data_dir = os.path.join(SCRIPT_DIR, "Aleyna 1 (2024)")
    xml_path = os.path.join(data_dir, "Annotations", "annotations.xml")
    images_dir = os.path.join(data_dir, "Images")
    output_dir = os.path.join(SCRIPT_DIR, "results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("COMPARISON: Old Models vs New Models vs Ground Truth")
    print("=" * 60)
    
    # 1. Load ground truth
    print("\n[1/4] Loading ground truth annotations...")
    gt_coco = load_ground_truth(xml_path, images_dir)
    print(f"  ✓ Loaded {len(gt_coco['images'])} images")
    print(f"  ✓ Loaded {len(gt_coco['annotations'])} annotations")
    print(f"  ✓ Categories: {[c['name'] for c in gt_coco['categories']]}")
    
    # Save GT
    gt_output = os.path.join(output_dir, "ground_truth.json")
    with open(gt_output, 'w') as f:
        json.dump(gt_coco, f, indent=2)
    print(f"  ✓ Saved to {gt_output}")
    
    # 2. Run old models
    print("\n[2/4] Running old models...")
    old_output_dir = os.path.join(output_dir, "old_models")
    os.makedirs(old_output_dir, exist_ok=True)
    old_coco = process_old_models(images_dir, old_output_dir)
    print(f"  ✓ Processed {len(old_coco['images'])} images")
    print(f"  ✓ Generated {len(old_coco['annotations'])} annotations")
    
    old_output = os.path.join(output_dir, "old_models_merged.json")
    with open(old_output, 'w') as f:
        json.dump(old_coco, f, indent=2)
    print(f"  ✓ Saved to {old_output}")
    
    # 3. Run new models
    print("\n[3/4] Running new models...")
    new_output_dir = os.path.join(output_dir, "new_models")
    os.makedirs(new_output_dir, exist_ok=True)
    new_coco = process_new_models(images_dir, new_output_dir)
    print(f"  ✓ Processed {len(new_coco['images'])} images")
    print(f"  ✓ Generated {len(new_coco['annotations'])} annotations")
    
    new_output = os.path.join(output_dir, "new_models_merged.json")
    with open(new_output, 'w') as f:
        json.dump(new_coco, f, indent=2)
    print(f"  ✓ Saved to {new_output}")
    
    # 4. Calculate metrics
    print("\n[4/4] Calculating metrics...")
    
    # Align categories
    old_coco_aligned = align_categories(gt_coco.copy(), old_coco.copy())
    new_coco_aligned = align_categories(gt_coco.copy(), new_coco.copy())
    
    # Calculate metrics for old models
    print("\n  Calculating metrics for OLD MODELS...")
    old_metrics = calculate_metrics(gt_coco, old_coco_aligned, output_dir)
    print(f"    mAP@50: {old_metrics['mAP@50']:.4f}")
    print(f"    mAP@[.50:.95]: {old_metrics['mAP@[.50:.95]']:.4f}")
    print(f"    Precision: {old_metrics['Precision']:.4f}")
    print(f"    Recall: {old_metrics['Recall']:.4f}")
    
    # Calculate metrics for new models
    print("\n  Calculating metrics for NEW MODELS...")
    new_metrics = calculate_metrics(gt_coco, new_coco_aligned, output_dir)
    print(f"    mAP@50: {new_metrics['mAP@50']:.4f}")
    print(f"    mAP@[.50:.95]: {new_metrics['mAP@[.50:.95]']:.4f}")
    print(f"    Precision: {new_metrics['Precision']:.4f}")
    print(f"    Recall: {new_metrics['Recall']:.4f}")
    
    # Save metrics
    metrics_output = os.path.join(output_dir, "metrics.json")
    with open(metrics_output, 'w') as f:
        json.dump({
            'old_models': old_metrics,
            'new_models': new_metrics
        }, f, indent=2)
    print(f"\n  ✓ Saved metrics to {metrics_output}")
    
    # 5. Create visualizations for each image
    print("\n[5/5] Creating comparison visualizations...")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    for img_info in gt_coco["images"]:
        image_name = img_info["file_name"]
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            continue
        
        # Get COCO for this image
        img_id = img_info["id"]
        
        # Filter annotations for this image
        gt_img_coco = {
            "images": [img_info],
            "annotations": [a for a in gt_coco["annotations"] if a["image_id"] == img_id],
            "categories": gt_coco["categories"]
        }
        
        old_img_coco = {
            "images": [img_info],
            "annotations": [a for a in old_coco["annotations"] if a["image_id"] == img_id],
            "categories": old_coco["categories"]
        }
        
        new_img_coco = {
            "images": [img_info],
            "annotations": [a for a in new_coco["annotations"] if a["image_id"] == img_id],
            "categories": new_coco["categories"]
        }
        
        # Create visualization
        output_path = os.path.join(vis_dir, f"{Path(image_name).stem}_comparison.png")
        create_comparison_visualization(
            image_path,
            gt_img_coco,
            old_img_coco,
            new_img_coco,
            output_path
        )
    
    print(f"\n  ✓ Saved visualizations to {vis_dir}")
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

