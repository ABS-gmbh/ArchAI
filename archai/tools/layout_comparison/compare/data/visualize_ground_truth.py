"""
Visualize ground truth annotations from COCO format on images.
This helps verify the accuracy of XML to COCO conversion.
"""
import os
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from original_annotations import load_ground_truth

try:
    import pycocotools.mask as mask_util
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not available. Mask visualization may be limited.")


def decode_rle(rle, width, height):
    """Decode COCO RLE to binary mask."""
    if not HAS_PYCOCOTOOLS:
        return None
    try:
        rle_decoded = rle.copy()
        rle_decoded['counts'] = rle_decoded['counts'].encode('utf-8')
        mask = mask_util.decode(rle_decoded)
        return mask
    except Exception as e:
        print(f"Warning: Failed to decode RLE: {e}")
        return None


def draw_coco_annotations(image_path, coco_json, output_path=None, show_labels=True):
    """
    Draw COCO annotations on an image.
    
    Args:
        image_path: Path to image file
        coco_json: COCO format dictionary
        output_path: Path to save visualized image (if None, returns numpy array)
        show_labels: Whether to show class labels
    
    Returns:
        numpy array of visualized image (if output_path is None)
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Get image info from COCO
    image_name = os.path.basename(image_path)
    img_info = None
    for img_data in coco_json["images"]:
        if img_data["file_name"] == image_name:
            img_info = img_data
            break
    
    if img_info is None:
        print(f"Warning: Image {image_name} not found in COCO data")
        return img_array
    
    img_id = img_info["id"]
    
    # Get annotations for this image
    annotations = [a for a in coco_json["annotations"] if a["image_id"] == img_id]
    
    if len(annotations) == 0:
        print(f"No annotations found for {image_name}")
        return img_array
    
    # Create category name map
    id_to_name = {c["id"]: c["name"] for c in coco_json["categories"]}
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 20))
    ax.imshow(img_array)
    ax.axis("off")
    ax.set_title(f"{image_name}\n({len(annotations)} annotations)", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Generate distinct colors for each category
    num_categories = len(coco_json["categories"])
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_categories)))
    if num_categories > 20:
        # Use additional colormap for more categories
        colors2 = plt.cm.Set3(np.linspace(0, 1, num_categories - 20))
        colors = np.vstack([colors, colors2])
    
    category_colors = {}
    for idx, cat in enumerate(coco_json["categories"]):
        category_colors[cat["id"]] = colors[idx % len(colors)]
    
    # Draw each annotation
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = id_to_name.get(cat_id, f"category_{cat_id}")
        color = category_colors.get(cat_id, [1, 0, 0, 0.5])  # Red fallback
        
        # Get segmentation
        segs = ann.get("segmentation", [])
        bbox = ann.get("bbox", [0, 0, 0, 0])
        
        # Draw segmentation (polygon or mask)
        if segs:
            if isinstance(segs, list) and len(segs) > 0:
                # Check if it's RLE (dict) or polygon (list of coordinates)
                if isinstance(segs, dict) or (isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], dict)):
                    # RLE mask
                    if isinstance(segs, list):
                        rle = segs[0]
                    else:
                        rle = segs
                    
                    if HAS_PYCOCOTOOLS:
                        mask = decode_rle(rle, img_info["width"], img_info["height"])
                        if mask is not None:
                            # Draw mask with transparency
                            mask_colored = np.zeros((*mask.shape, 4))
                            mask_colored[mask > 0] = [*color[:3], 0.3]  # Semi-transparent fill
                            ax.imshow(mask_colored, alpha=0.5)
                            
                            # Draw mask outline
                            try:
                                from scipy import ndimage
                                contours = ndimage.binary_erosion(mask) ^ mask
                                ax.contour(contours, colors=[color[:3]], linewidths=2, alpha=0.8)
                            except ImportError:
                                # Fallback: just draw the mask without contour
                                pass
                
                elif isinstance(segs[0], list) and len(segs[0]) >= 6:
                    # Polygon: flat list [x1, y1, x2, y2, ...]
                    coords = segs[0]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    
                    # Draw polygon with fill
                    poly = patches.Polygon(
                        list(zip(xs, ys)),
                        closed=True,
                        edgecolor=color[:3],
                        facecolor=color[:3],
                        linewidth=2.5,
                        alpha=0.3,  # Semi-transparent fill
                    )
                    ax.add_patch(poly)
                    
                    # Draw polygon outline
                    poly_edge = patches.Polygon(
                        list(zip(xs, ys)),
                        closed=True,
                        edgecolor=color[:3],
                        facecolor="none",
                        linewidth=2.5,
                        alpha=0.8,  # More opaque edge
                    )
                    ax.add_patch(poly_edge)
        
        # Draw bounding box if no segmentation or as fallback
        if not segs or (isinstance(segs, list) and len(segs) == 0):
            x, y, w, h = bbox
            if w > 0 and h > 0:
                rect = patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor=color[:3],
                    facecolor=color[:3],
                    linewidth=2.5,
                    alpha=0.3,
                )
                ax.add_patch(rect)
                
                # Draw bbox outline
                rect_edge = patches.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor=color[:3],
                    facecolor="none",
                    linewidth=2.5,
                    alpha=0.8,
                )
                ax.add_patch(rect_edge)
        
        # Add label
        if show_labels:
            # Get position for label (use bbox or polygon center)
            if segs and isinstance(segs, list) and len(segs) > 0:
                if isinstance(segs[0], list) and len(segs[0]) >= 6:
                    # Polygon
                    coords = segs[0]
                    xs = coords[0::2]
                    ys = coords[1::2]
                    label_x = min(xs)
                    label_y = min(ys) - 10
                else:
                    # Use bbox
                    x, y, w, h = bbox
                    label_x = x
                    label_y = y - 10
            else:
                x, y, w, h = bbox
                label_x = x
                label_y = y - 10
            
            # Draw label with background
            ax.text(
                label_x,
                label_y,
                cat_name,
                color='black',
                fontsize=10,
                fontweight='bold',
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor=color[:3],
                    linewidth=2,
                    alpha=0.9,
                ),
                zorder=10,
            )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved visualization to: {output_path}")
        return None
    else:
        # Return as numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return buf


def visualize_all_images(coco_json, images_dir, output_dir):
    """
    Visualize annotations for all images in the COCO dataset.
    
    Args:
        coco_json: COCO format dictionary
        images_dir: Directory containing images
        output_dir: Directory to save visualized images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing {len(coco_json['images'])} images...")
    
    for img_info in coco_json["images"]:
        image_name = img_info["file_name"]
        image_path = Path(images_dir) / image_name
        
        if not image_path.exists():
            print(f"Warning: Image {image_name} not found, skipping...")
            continue
        
        output_path = Path(output_dir) / f"{Path(image_name).stem}_annotated.png"
        
        print(f"Processing {image_name}...")
        draw_coco_annotations(
            str(image_path),
            coco_json,
            output_path=str(output_path),
            show_labels=True
        )
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    """Main function to visualize ground truth annotations."""
    # Paths
    data_dir = os.path.join(SCRIPT_DIR, "Aleyna 1 (2024)")
    xml_path = os.path.join(data_dir, "Annotations", "annotations.xml")
    images_dir = os.path.join(data_dir, "Images")
    output_dir = os.path.join(SCRIPT_DIR, "visualizations_gt")
    
    # Option 1: Load from existing COCO JSON
    coco_json_path = os.path.join(SCRIPT_DIR, "ground_truth_coco.json")
    if os.path.exists(coco_json_path):
        print(f"Loading COCO JSON from: {coco_json_path}")
        with open(coco_json_path, 'r') as f:
            coco_json = json.load(f)
    else:
        # Option 2: Generate from XML
        print(f"Loading from XML: {xml_path}")
        coco_json = load_ground_truth(xml_path, images_dir)
        
        if coco_json:
            # Save for future use
            with open(coco_json_path, 'w') as f:
                json.dump(coco_json, f, indent=4)
            print(f"Saved COCO JSON to: {coco_json_path}")
    
    if not coco_json:
        print("Error: Failed to load annotations")
        return
    
    print(f"\nLoaded {len(coco_json['images'])} images")
    print(f"Loaded {len(coco_json['annotations'])} annotations")
    print(f"Categories: {[c['name'] for c in coco_json['categories']]}")
    
    # Visualize all images
    visualize_all_images(coco_json, images_dir, output_dir)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nCheck the visualizations in: {output_dir}")
    print("Compare them with the original images to verify conversion accuracy.")


if __name__ == "__main__":
    main()

