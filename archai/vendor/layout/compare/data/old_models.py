"""
Run old models (Line, Border, Zones) and convert predictions to COCO format.
"""
import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO, YOLOE
import tempfile
from typing import Dict, List
import pycocotools.mask as mask_util
import cv2


# Model files (same as app_original_app_with_three_models.py)
MODEL_FILES = {
    "Line Detection": "best_line_detection_yoloe (1).pt",
    "Border Detection": "border_model_weights.pt",
    "Zones Detection": "zones_model_weights.pt"
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def load_old_models():
    """Load the three old models."""
    models = {}
    for name, model_file in MODEL_FILES.items():
        model_path = os.path.join(PROJECT_ROOT, model_file)
        if os.path.exists(model_path):
            try:
                if name == "Line Detection":
                    models[name] = YOLOE(model_path)
                else:
                    models[name] = YOLO(model_path)
                print(f"✓ Loaded {name} model")
            except Exception as e:
                print(f"✗ Error loading {name} model: {e}")
                models[name] = None
        else:
            print(f"✗ Model file not found: {model_path}")
            models[name] = None
    return models


def results_to_coco(result, model_name, image_id, image_width, image_height, category_map):
    """
    Convert YOLO result to COCO format annotations.
    Handles masks properly for YOLOE Line Detection model (like app.py).
    
    Args:
        result: YOLO Results object (single result, not list)
        model_name: Name of the model (for special handling)
        image_id: COCO image ID
        image_width: Image width
        image_height: Image height
        category_map: Dict mapping class names to COCO category IDs
    
    Returns:
        List of COCO annotation dictionaries
    """
    annotations = []
    ann_id = 1
    
    if result is None:
        return annotations
    
    # Get boxes and masks
    boxes = result.boxes
    if boxes is None:
        return annotations
    
    # Get masks if available
    masks = result.masks
    has_masks = masks is not None and len(masks) > 0
    
    num_detections = len(boxes)
    
    for i in range(num_detections):
        # Get box coordinates
        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = box
        
        # Get class
        cls_id = int(boxes.cls[i].cpu().numpy())
        cls_name = result.names[cls_id]
        
        # Map "object" to "line" for Line Detection model (like app.py)
        if model_name == "Line Detection" and cls_name == "object":
            cls_name = "line"
        
        # Skip if class not in category map
        if cls_name not in category_map:
            continue
        
        # Get confidence
        conf = float(boxes.conf[i].cpu().numpy())
        
        # Convert bbox to COCO format [x, y, width, height]
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        
        # Get segmentation
        segmentation = None
        area = bbox[2] * bbox[3]  # Default to bbox area
        
        if has_masks and i < len(masks.data):
            try:
                # Get mask (like app.py handles YOLOE masks)
                mask = masks.data[i].cpu().numpy()
                
                # Handle mask resizing similar to app.py
                if mask.shape != (image_height, image_width):
                    # Resize mask to image size using cv2 (like app.py)
                    mask_np = (mask > 0).astype(np.uint8)
                    resized_mask = cv2.resize(
                        mask_np,
                        (image_width, image_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask = resized_mask.astype(np.uint8)
                else:
                    mask = (mask > 0.5).astype(np.uint8)
                
                # Convert to COCO RLE format
                rle = mask_util.encode(np.asfortranarray(mask))
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = rle['counts'].decode('utf-8')
                segmentation = rle
                area = float(mask_util.area(rle))
            except Exception as e:
                print(f"Warning: Failed to convert mask to RLE for detection {i}: {e}")
                # Fall back to bbox
                pass
        
        # Create COCO annotation
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_map[cls_name],
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "score": conf
        }
        
        if segmentation is not None:
            ann["segmentation"] = segmentation
        
        annotations.append(ann)
        ann_id += 1
    
    return annotations


def run_old_models_on_image(image_path, models, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run old models on a single image and return COCO format predictions.
    Matches the behavior of app.py for consistent results.
    
    Args:
        image_path: Path to image file
        models: Dict of loaded models
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        COCO format dictionary with predictions
    """
    # Load image as numpy array (like app.py does)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
    
    # Create category map (map all detected classes to sequential IDs)
    all_classes = set()
    results_dict = {}
    
    # Run each model
    for model_name, model in models.items():
        if model is None:
            continue
        
        try:
            # Use numpy array for prediction (like app.py)
            # Access result as [0] immediately (like app.py)
            result = model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold
            )[0]
            
            # Collect class names and map "object" to "line" for Line Detection
            if result.names:
                for cls_id, cls_name in result.names.items():
                    # Map "object" to "line" for Line Detection model (like app.py)
                    if model_name == "Line Detection" and cls_name == "object":
                        all_classes.add("line")
                    else:
                        all_classes.add(cls_name)
            
            results_dict[model_name] = result
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results_dict[model_name] = None
    
    # Create category mapping
    category_map = {cls_name: idx + 1 for idx, cls_name in enumerate(sorted(all_classes))}
    
    # Convert all results to COCO format
    all_annotations = []
    ann_id = 1
    
    for model_name, result in results_dict.items():
        if result is None:
            continue
        
        annotations = results_to_coco(
            result,
            model_name,
            image_id=1,  # Will be set later
            image_width=image_width,
            image_height=image_height,
            category_map=category_map
        )
        
        # Update annotation IDs
        for ann in annotations:
            ann["id"] = ann_id
            ann_id += 1
        
        all_annotations.extend(annotations)
    
    # Create COCO format
    coco = {
        "info": {"description": "Old models predictions"},
        "licenses": [],
        "images": [{
            "id": 1,
            "width": image_width,
            "height": image_height,
            "file_name": os.path.basename(image_path)
        }],
        "annotations": all_annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": ""}
            for name, cid in sorted(category_map.items(), key=lambda x: x[1])
        ]
    }
    
    return coco


def process_dataset(images_dir, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process all images in a directory with old models.
    
    Args:
        images_dir: Directory containing images
        output_dir: Directory to save COCO JSON files
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        Merged COCO format dictionary for all images
    """
    # Load models
    models = load_old_models()
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    all_coco_dicts = []
    image_id = 1
    
    for image_file in sorted(image_files):
        image_path = os.path.join(images_dir, image_file)
        print(f"Processing {image_file}...")
        
        try:
            coco = run_old_models_on_image(
                image_path,
                models,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # Update image ID
            coco["images"][0]["id"] = image_id
            
            # Update annotation image_ids
            for ann in coco["annotations"]:
                ann["image_id"] = image_id
            
            all_coco_dicts.append(coco)
            image_id += 1
            
            # Save individual file
            output_path = os.path.join(output_dir, f"{Path(image_file).stem}_old.json")
            with open(output_path, 'w') as f:
                json.dump(coco, f, indent=2)
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    # Merge all COCO dicts
    merged = {
        "info": {"description": "Old models predictions - merged"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Collect all categories
    all_categories = {}
    for coco in all_coco_dicts:
        for cat in coco["categories"]:
            if cat["name"] not in all_categories:
                all_categories[cat["name"]] = cat["id"]
    
    # Update category IDs to be sequential
    category_map = {name: idx + 1 for idx, name in enumerate(sorted(all_categories.keys()))}
    reverse_map = {old_id: category_map[name] for name, old_id in all_categories.items()}
    
    merged["categories"] = [
        {"id": cid, "name": name, "supercategory": ""}
        for name, cid in sorted(category_map.items(), key=lambda x: x[1])
    ]
    
    # Merge images and annotations
    ann_id = 1
    for coco in all_coco_dicts:
        merged["images"].extend(coco["images"])
        
        for ann in coco["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = ann_id
            # Update category_id using reverse_map
            old_cat_id = ann["category_id"]
            # Find category name
            cat_name = next((c["name"] for c in coco["categories"] if c["id"] == old_cat_id), None)
            if cat_name and cat_name in category_map:
                new_ann["category_id"] = category_map[cat_name]
            merged["annotations"].append(new_ann)
            ann_id += 1
    
    return merged


if __name__ == "__main__":
    # Test on single image
    test_image = "../../e-codices_bbb-0219_044r_max.jpg"
    models = load_old_models()
    coco = run_old_models_on_image(test_image, models)
    
    print(f"Predictions: {len(coco['annotations'])} annotations")
    print(f"Categories: {[c['name'] for c in coco['categories']]}")
    
    with open("test_old_models.json", "w") as f:
        json.dump(coco, f, indent=2)

