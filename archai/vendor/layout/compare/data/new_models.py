"""
Run new models (emanuskript, catmus, zone) and convert predictions to COCO format.
Uses the same logic as app.py and test_combined_models.py.
"""
import os
import sys
import json
from pathlib import Path
import tempfile

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Import from project root
from test_combined_models import run_model_predictions, combine_and_filter_predictions


def run_new_models_on_image(image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run new models on a single image and return COCO format predictions.
    
    Args:
        image_path: Path to image file
        conf_threshold: Confidence threshold (not directly used, but kept for consistency)
        iou_threshold: IoU threshold (not directly used, but kept for consistency)
    
    Returns:
        COCO format dictionary with predictions
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run 3 models and save predictions JSON
        labels_folders = run_model_predictions(image_path, tmp_dir)
        
        # Combine & filter to coco_class_mapping
        coco_json = combine_and_filter_predictions(
            image_path, labels_folders, output_json_path=None
        )
    
    return coco_json


def process_dataset(images_dir, output_dir, conf_threshold=0.25, iou_threshold=0.45):
    """
    Process all images in a directory with new models.
    
    Args:
        images_dir: Directory containing images
        output_dir: Directory to save COCO JSON files
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        Merged COCO format dictionary for all images
    """
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
            coco = run_new_models_on_image(
                image_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # Update image ID if needed
            if len(coco["images"]) > 0:
                coco["images"][0]["id"] = image_id
                
                # Update annotation image_ids
                for ann in coco["annotations"]:
                    ann["image_id"] = image_id
            
            all_coco_dicts.append(coco)
            image_id += 1
            
            # Save individual file
            output_path = os.path.join(output_dir, f"{Path(image_file).stem}_new.json")
            with open(output_path, 'w') as f:
                json.dump(coco, f, indent=2)
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Merge all COCO dicts
    merged = {
        "info": {"description": "New models predictions - merged"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Use categories from first COCO (they should all be the same)
    if len(all_coco_dicts) > 0:
        merged["categories"] = all_coco_dicts[0]["categories"]
    
    # Merge images and annotations
    ann_id = 1
    for coco in all_coco_dicts:
        merged["images"].extend(coco["images"])
        
        for ann in coco["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = ann_id
            merged["annotations"].append(new_ann)
            ann_id += 1
    
    return merged


if __name__ == "__main__":
    # Test on single image
    test_image = "../../e-codices_bbb-0219_044r_max.jpg"
    coco = run_new_models_on_image(test_image)
    
    print(f"Predictions: {len(coco['annotations'])} annotations")
    print(f"Categories: {[c['name'] for c in coco['categories']]}")
    
    with open("test_new_models.json", "w") as f:
        json.dump(coco, f, indent=2)

