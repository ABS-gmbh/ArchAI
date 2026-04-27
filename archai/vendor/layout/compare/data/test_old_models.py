"""
Test script for old_models.py to verify it works like app.py
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pycocotools.mask as mask_util
import cv2

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from old_models import load_old_models, run_old_models_on_image


def visualize_annotations(image_path, coco_json, output_path):
    """
    Visualize COCO annotations on the image.
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(img)
    ax.set_title("Old Models Predictions", fontsize=16, fontweight='bold')
    ax.axis("off")
    
    if not coco_json.get("images") or not coco_json.get("annotations"):
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
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
        
        # Get bbox
        bbox = ann.get("bbox", [0, 0, 0, 0])
        if not bbox or len(bbox) < 4:
            # Try to get bbox from segmentation
            segs = ann.get("segmentation", {})
            if isinstance(segs, dict) and 'counts' in segs:
                # RLE mask
                try:
                    rle = segs
                    if isinstance(rle['counts'], str):
                        rle['counts'] = rle['counts'].encode('utf-8')
                    mask = mask_util.decode(rle)
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        bbox = [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]
                    else:
                        continue
                except Exception as e:
                    continue
            elif isinstance(segs, list) and len(segs) > 0:
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
        segs = ann.get("segmentation", {})
        if isinstance(segs, dict) and 'counts' in segs:
            # RLE mask - draw as filled polygon using contour
            try:
                rle = segs
                if isinstance(rle['counts'], str):
                    rle['counts'] = rle['counts'].encode('utf-8')
                mask = mask_util.decode(rle)
                
                # Use cv2 to find contours (memory efficient)
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if len(contour) > 2:
                        # Convert contour to list of (x, y) tuples
                        poly_coords = [(pt[0][0], pt[0][1]) for pt in contour]
                        poly = patches.Polygon(
                            poly_coords, closed=True,
                            edgecolor=color, facecolor=color,
                            linewidth=2, alpha=0.3
                        )
                        ax.add_patch(poly)
                        poly_edge = patches.Polygon(
                            poly_coords, closed=True,
                            edgecolor=color, facecolor="none",
                            linewidth=2, alpha=0.8
                        )
                        ax.add_patch(poly_edge)
            except Exception as e:
                # Fall back to bbox
                rect = patches.Rectangle(
                    (x, y), w, h,
                    edgecolor=color, facecolor=color,
                    linewidth=2, alpha=0.3
                )
                ax.add_patch(rect)
                rect_edge = patches.Rectangle(
                    (x, y), w, h,
                    edgecolor=color, facecolor="none",
                    linewidth=2, alpha=0.8
                )
                ax.add_patch(rect_edge)
        elif isinstance(segs, list) and len(segs) > 0:
            if isinstance(segs[0], list) and len(segs[0]) >= 6:
                # Polygon
                coords = segs[0]
                xs = coords[0::2]
                ys = coords[1::2]
                poly = patches.Polygon(
                    list(zip(xs, ys)), closed=True,
                    edgecolor=color, facecolor=color,
                    linewidth=2, alpha=0.3
                )
                ax.add_patch(poly)
                poly_edge = patches.Polygon(
                    list(zip(xs, ys)), closed=True,
                    edgecolor=color, facecolor="none",
                    linewidth=2, alpha=0.8
                )
                ax.add_patch(poly_edge)
            else:
                # Fall back to bbox
                rect = patches.Rectangle(
                    (x, y), w, h,
                    edgecolor=color, facecolor=color,
                    linewidth=2, alpha=0.3
                )
                ax.add_patch(rect)
                rect_edge = patches.Rectangle(
                    (x, y), w, h,
                    edgecolor=color, facecolor="none",
                    linewidth=2, alpha=0.8
                )
                ax.add_patch(rect_edge)
        else:
            # Bbox only
            rect = patches.Rectangle(
                (x, y), w, h,
                edgecolor=color, facecolor=color,
                linewidth=2, alpha=0.3
            )
            ax.add_patch(rect)
            rect_edge = patches.Rectangle(
                (x, y), w, h,
                edgecolor=color, facecolor="none",
                linewidth=2, alpha=0.8
            )
            ax.add_patch(rect_edge)
        
        # Add label
        text_width = len(name) * 7
        text_height = 12
        label_x, label_y = find_label_position(bbox, text_width, text_height, img_width, img_height)
        placed_labels.append((label_x, label_y, text_width, text_height))
        
        edge_color = tuple(color[:3]) if isinstance(color, np.ndarray) else color
        ax.text(
            label_x, label_y, name,
            color='black', fontsize=9, fontweight='bold',
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=edge_color,
                linewidth=2,
                alpha=0.9,
            ),
            zorder=10,
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved visualization to: {output_path}")


def test_single_image():
    """Test old models on a single image."""
    print("=" * 70)
    print("TESTING OLD MODELS ON SINGLE IMAGE")
    print("=" * 70)
    
    # Find a test image (use first available image from SampleBatch2)
    test_image_dir = Path(SCRIPT_DIR) / "SampleBatch2" / "Images"
    if not test_image_dir.exists():
        print(f"⚠️  Test image directory not found: {test_image_dir}")
        print("Please provide a test image path.")
        return
    
    # Get first image
    image_files = list(test_image_dir.glob("*.jpg")) + list(test_image_dir.glob("*.png"))
    if not image_files:
        print(f"⚠️  No images found in {test_image_dir}")
        return
    
    test_image_path = image_files[0]
    print(f"\n📸 Testing with image: {test_image_path.name}")
    
    # Load models
    print("\n[1/3] Loading models...")
    models = load_old_models()
    
    # Check if all models loaded
    failed_models = [name for name, model in models.items() if model is None]
    if failed_models:
        print(f"⚠️  Warning: Some models failed to load: {failed_models}")
    
    # Run predictions
    print(f"\n[2/3] Running predictions...")
    try:
        coco = run_old_models_on_image(
            str(test_image_path),
            models,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        print(f"  ✓ Generated {len(coco['annotations'])} annotations")
        print(f"  ✓ Categories: {[c['name'] for c in coco['categories']]}")
        
        # Count annotations per model/category
        category_counts = {}
        for ann in coco['annotations']:
            cat_id = ann['category_id']
            cat_name = next((c['name'] for c in coco['categories'] if c['id'] == cat_id), f"cat_{cat_id}")
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        print(f"\n  Annotation counts by category:")
        for cat_name, count in sorted(category_counts.items()):
            print(f"    - {cat_name}: {count}")
        
        # Check for masks (Line Detection should have masks)
        masks_count = sum(1 for ann in coco['annotations'] if 'segmentation' in ann)
        print(f"\n  ✓ Annotations with masks: {masks_count}")
        
    except Exception as e:
        print(f"  ❌ Error running predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save results
    print(f"\n[3/4] Saving results...")
    output_path = Path(SCRIPT_DIR) / "test_old_models_output.json"
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"  ✓ Saved to: {output_path}")
    
    # Visualize annotations
    print(f"\n[4/4] Creating visualization...")
    vis_output_path = Path(SCRIPT_DIR) / "test_old_models_visualization.png"
    try:
        visualize_annotations(str(test_image_path), coco, str(vis_output_path))
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to create visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Results saved:")
    print(f"  - JSON: {output_path}")
    print(f"  - Visualization: {vis_output_path}")


if __name__ == "__main__":
    test_single_image()

