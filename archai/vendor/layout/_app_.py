from typing import Tuple, Dict, List, Union
import gradio as gr
import supervision as sv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO, YOLOE
import zipfile
import os
import tempfile
import cv2
import json
from datetime import datetime
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Define custom models
MODEL_FILES = {
    "Line Detection": "best_line_detection_yoloe (1).pt",  # Use YOLOE for this
    "Border Detection": "border_model_weights.pt",         # Still YOLO
    "Zones Detection": "zones_model_weights.pt"            # Still YOLO
}

# Dictionary to store loaded models
models: Dict[str, Union[YOLO, YOLOE]] = {}

# Model class definitions - Expected/desired classes
EXPECTED_MODEL_CLASSES = {
    "Line Detection": [
        "line"
    ],
    "Border Detection": [
        "border",
        "decorated_initial",
        "historiated_initial", 
        "illustration",
        "page",
        "simple_initial"
    ],
    "Zones Detection": [
        "CustomZone-PageHeight",
        "CustomZone-PageWidth",
        "DamageZone",
        "DigitizationArtefactZone",
        "DropCapitalZone",
        "GraphicZone",
        "MainZone",
        "MarginTextZone",
        "MusicZone",
        "NumberingZone",
        "PageZone",
        "QuireMarksZone",
        "RunningTitleZone",
        "StampZone",
        "TitlePageZone"
    ]
}

# Model class definitions - will be populated dynamically from actual models
MODEL_CLASSES = {}

# Global variables to store results for download
current_results = []
current_images = []

# Load all custom models
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

for name, model_file in MODEL_FILES.items():
    model_path = os.path.join(script_dir, model_file)
    if os.path.exists(model_path):
        try:
            if name == "Line Detection":
                # Load YOLOE for line detection
                models[name] = YOLOE(model_path)
            else:
                # Load YOLO for other tasks
                models[name] = YOLO(model_path)
            
            # Read actual classes from the model
            if models[name] is not None:
                # Read classes from model
                actual_classes = list(models[name].names.values())
                
                # Map "object" to "line" for Line Detection model in MODEL_CLASSES
                if name == "Line Detection" and "object" in actual_classes:
                    actual_classes = ["line" if c == "object" else c for c in actual_classes]
                    print(f"   Mapped class 'object' to 'line' in Line Detection model for UI")
                
                MODEL_CLASSES[name] = actual_classes
                
                # Check for mismatch with expected classes
                if name in EXPECTED_MODEL_CLASSES:
                    expected = set(EXPECTED_MODEL_CLASSES[name])
                    actual = set(actual_classes)
                    if expected != actual:
                        print(f"⚠️  WARNING: {name} model class mismatch!")
                        print(f"   Expected: {sorted(expected)}")
                        print(f"   Actual: {sorted(actual)}")
                        print(f"   Missing in model: {sorted(expected - actual)}")
                        print(f"   Extra in model: {sorted(actual - expected)}")
                        print(f"   ⚠️  Using ACTUAL classes from model: {sorted(actual)}")
            
            print(f"✓ Loaded {name} model from {model_path}")
            print(f"  Classes available: {MODEL_CLASSES.get(name, 'Unknown')}")
        except Exception as e:
            print(f"✗ Error loading {name} model: {e}")
            models[name] = None
            # Fallback to expected classes if model fails to load
            MODEL_CLASSES[name] = EXPECTED_MODEL_CLASSES.get(name, [])
    else:
        print(f"✗ Warning: Model file {model_path} not found")
        models[name] = None
        # Fallback to expected classes if model file not found
        MODEL_CLASSES[name] = EXPECTED_MODEL_CLASSES.get(name, [])


# Create annotators
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)
BOX_ANNOTATOR = sv.BoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()

def detect_and_annotate_combined(
    image: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    return_annotations: bool = False,
    selected_classes: Dict[str, List[str]] = None
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """Run all three models and combine their outputs in a single annotated image"""
    print(f"🔍 Starting detection on image shape: {image.shape}")
    
    # Colors for different models - more distinct colors
    colors = {
        "Line Detection": sv.Color.from_hex("#FF0000"),      # Bright Red
        "Border Detection": sv.Color.from_hex("#00FF00"),   # Bright Green  
        "Zones Detection": sv.Color.from_hex("#0080FF")     # Bright Blue
    }
    
    # Model prefixes for clear labeling
    model_prefixes = {
        "Line Detection": "[LINE]",
        "Border Detection": "[BORDER]", 
        "Zones Detection": "[ZONE]"
    }
    
    annotated_image = image.copy()
    total_detections = 0
    detections_data = {}
    
    # Run each model and annotate with different colors
    for model_name, model in models.items():
        if model is None:
            print(f"⏭️  Skipping {model_name} (model not loaded)")
            detections_data[model_name] = []
            continue
            
        # Check if any classes are selected for this model BEFORE running inference
        if selected_classes and model_name in selected_classes:
            selected_class_names = selected_classes[model_name]
            # If no classes selected for this model, skip it entirely (don't run inference)
            if not selected_class_names:
                print(f"⏭️  Skipping {model_name} (no classes selected)")
                detections_data[model_name] = []
                continue
        elif selected_classes is not None:
            # If selected_classes is provided but this model not in it, skip it
            print(f"⏭️  Skipping {model_name} (model not in selected classes)")
            detections_data[model_name] = []
            continue
        
        print(f"🤖 Running {model_name} model...")
        
        # Perform inference (guard against per-model failures)
        try:
            results = model.predict(
                image,
                conf=conf_threshold,
                iou=iou_threshold
            )[0]
        except Exception as e:
            print(f"✗ {model_name} inference failed: {e}")
            detections_data[model_name] = []
            continue

        model_detections = []
        
        if len(results.boxes) > 0:
            # Convert results to supervision Detections
            boxes = results.boxes.xyxy.cpu().numpy()
            confidence = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            # Filter by selected classes - only show selected classes
            if selected_classes and model_name in selected_classes:
                selected_class_names = selected_classes[model_name]
                
                # Get class names for this model
                model_class_names = results.names
                # Find class IDs that match selected class names
                selected_class_ids = []
                for class_id, class_name in model_class_names.items():
                    # For Line Detection: also match "object" when user selects "line"
                    if model_name == "Line Detection" and class_name == "object" and "line" in selected_class_names:
                        selected_class_ids.append(class_id)
                    elif class_name in selected_class_names:
                        selected_class_ids.append(class_id)
                
                # Filter detections to only show selected classes
                mask = np.isin(class_ids, selected_class_ids)
                if not np.any(mask):
                    print(f"   No detections match selected classes for {model_name}")
                    detections_data[model_name] = []
                    continue
                
                boxes = boxes[mask]
                confidence = confidence[mask]
                class_ids = class_ids[mask]
                print(f"   Filtered to {len(boxes)} detections matching selected classes: {selected_class_names}")
            
            total_detections += len(boxes)
            
            # Store detection data for COCO format
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidence, class_ids)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                class_name = results.names[class_id]
                # Map "object" to "line" for Line Detection model
                if model_name == "Line Detection" and class_name == "object":
                    class_name = "line"
                
                model_detections.append({
                    "bbox": [float(x1), float(y1), float(width), float(height)],  # COCO format: [x, y, width, height]
                    "class_name": class_name,
                    "confidence": float(conf)
                })
            
            
            # Create Detections object for visualization
            detections = sv.Detections(
                xyxy=boxes,
                confidence=confidence,
                mask=results.masks.data.cpu().numpy() if results.masks is not None else None,
                class_id=class_ids
            )
            
            # Create labels with clear model prefixes and confidence scores
            model_prefix = model_prefixes[model_name]
            labels = []
            for class_id, conf in zip(class_ids, confidence):
                class_name = results.names[class_id]
                # Map "object" to "line" for Line Detection model
                if model_name == "Line Detection" and class_name == "object":
                    class_name = "line"
                labels.append(f"{model_prefix} {class_name} ({conf:.2f})")

            # Create annotators with specific colors and improved styling
            box_annotator = sv.BoxAnnotator(
                color=colors[model_name],
                thickness=3  # Thicker boxes for better visibility
            )
            label_annotator = sv.LabelAnnotator(
                text_color=sv.Color.WHITE,
                color=colors[model_name],
                text_thickness=2,
                text_scale=0.6,
                text_padding=8
            )
            
        # Replace the "annotate image" block inside detect_and_annotate_combined with this

            # Annotate image depending on model type
            if model_name == "Line Detection" and results.masks is not None:

                original_h, original_w = annotated_image.shape[:2]

                if detections.mask is not None:
                    all_resized_masks = []
                    for i, mask in enumerate(detections.mask):
                        # ensure binary mask
                        mask_np = (mask > 0).astype(np.uint8)
                        resized_mask = cv2.resize(
                            mask_np,
                            (original_w, original_h),
                            interpolation=cv2.INTER_NEAREST
                        )
                        resized_mask = resized_mask.astype(bool)  # <- important
                        all_resized_masks.append(resized_mask)

                    all_resized_masks = np.stack(all_resized_masks, axis=0)  # (N, H, W)
                    detections.mask = all_resized_masks  # overwrite with clean boolean masks
                    print("Resized masks:", detections.mask.shape, detections.mask.dtype)
                else:
                    detections.mask = None


                # Use MaskAnnotator for line detection
                mask_annotator = sv.MaskAnnotator(
                    color=colors[model_name],
                    opacity=0.6
                )
                annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
                
                # Add labels on top of masks
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=labels
                )
            else:
                # Use BoxAnnotator for Border and Zones
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
                annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        else:
            print(f"   No detections found for {model_name}")
        
        detections_data[model_name] = model_detections
    
    print(f"🎯 Detection completed. Total detections: {total_detections}")
    
    if return_annotations:
        return annotated_image, detections_data
    else:
        return annotated_image

def process_zip_file(zip_file_path: str, conf_threshold: float, iou_threshold: float, selected_classes: Dict[str, List[str]] = None) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, Dict]], Dict]:
    """Process all images in a zip file and return annotated images, detection data, and image info"""
    print(f"📁 Opening ZIP file: {zip_file_path}")
    results = []
    annotations_data = []
    image_info = {}
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print(f"📋 ZIP file contents: {zip_ref.namelist()}")
            
            # Create temporary directory to extract files
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"📂 Extracting to temporary directory: {temp_dir}")
                zip_ref.extractall(temp_dir)
                
                # List all files in temp directory
                all_files = os.listdir(temp_dir)
                print(f"📄 Files extracted: {all_files}")
                
                # Process each image file (recursively search through folders)
                image_count = 0
                
                # Walk through all directories and subdirectories
                for root, dirs, files in os.walk(temp_dir):
                    print(f"📂 Searching in directory: {root}")
                    
                    for filename in files:
                        # Skip macOS hidden files
                        if filename.startswith('._') or filename.startswith('.DS_Store'):
                            print(f"⏭️  Skipping system file: {filename}")
                            continue
                            
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            image_count += 1
                            image_path = os.path.join(root, filename)
                            print(f"🖼️  Processing image {image_count}: {filename} (from {os.path.relpath(root, temp_dir)})")
                            
                            # Load image
                            image = cv2.imread(image_path)
                            if image is not None:
                                print(f"✅ Image loaded successfully: {image.shape}")
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                
                                # Store image info
                                height, width = image.shape[:2]
                                image_info[filename] = (height, width)
                                
                                # Process with all models and get annotation data
                                print(f"🔍 Running detection models on {filename}...")
                                annotated_image, detections_data = detect_and_annotate_combined(
                                    image, conf_threshold, iou_threshold, return_annotations=True, selected_classes=selected_classes
                                )
                                print(f"✅ Detection completed for {filename}")
                                
                                results.append((filename, annotated_image))
                                annotations_data.append((filename, detections_data))
                            else:
                                print(f"❌ Failed to load image: {filename}")
                        else:
                            print(f"⏭️  Skipping non-image file: {filename}")
                
                print(f"📊 Total images processed: {len(results)} out of {image_count} image files found")
                print(f"📁 Searched through all subdirectories recursively")
        
        print(f"🎉 ZIP processing completed successfully! Processed {len(results)} images")
        return results, annotations_data, image_info
        
    except Exception as e:
        print(f"💥 ERROR in process_zip_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], [], {}

def create_coco_annotations(results_data: List, image_info: Dict) -> Dict:
    """Convert detection results to COCO JSON format"""
    coco_data = {
        "info": {
            "description": "Medieval Manuscript Detection Results",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Medieval YOLO Models",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Create categories from all models
    category_id = 1
    category_map = {}
    
    # Add categories for each model type
    for model_name in ["Line Detection", "Border Detection", "Zones Detection"]:
        if model_name in models and models[model_name] is not None:
            model = models[model_name]
            for class_id, class_name in model.names.items():
                full_name = f"{model_name}_{class_name}"
                if full_name not in category_map:
                    category_map[full_name] = category_id
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": full_name,
                        "supercategory": model_name
                    })
                    category_id += 1
    
    annotation_id = 1
    
    for image_idx, (filename, detections_by_model) in enumerate(results_data):
        # Add image info
        image_id = image_idx + 1
        img_height, img_width = image_info.get(filename, (0, 0))
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": img_width,
            "height": img_height,
            "license": 1
        })
        
        # Add annotations for each model
        for model_name, detections in detections_by_model.items():
            if detections:
                for detection in detections:
                    bbox = detection["bbox"]  # [x, y, width, height]
                    class_name = detection["class_name"]
                    confidence = detection["confidence"]
                    
                    full_category_name = f"{model_name}_{class_name}"
                    category_id = category_map.get(full_category_name, 1)
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                        "score": confidence
                    })
                    annotation_id += 1
    
    return coco_data

def create_download_zip(images: List[Tuple[str, np.ndarray]], annotations: Dict) -> str:
    """Create a ZIP file with images and annotations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"medieval_detection_results_{timestamp}.zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add images
        for filename, image_array in images:
            # Convert numpy array to PIL Image and save as bytes
            pil_image = Image.fromarray(image_array.astype('uint8'))
            img_bytes = io.BytesIO()
            
            # Determine format from filename
            if filename.lower().endswith('.png'):
                pil_image.save(img_bytes, format='PNG')
            else:
                pil_image.save(img_bytes, format='JPEG')
            
            # Add to ZIP
            zipf.writestr(f"images/{filename}", img_bytes.getvalue())
        
        # Add annotations
        annotations_json = json.dumps(annotations, indent=2)
        zipf.writestr("annotations.json", annotations_json)
        
        # Add README
        readme_content = f"""Medieval Manuscript Detection Results
=============================================

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Contents:
- images/: Annotated images with detection results
- annotations.json: COCO format annotations

Models and Color Coding:
- Line Detection (Red boxes with [LINE] prefix)
- Border Detection (Green boxes with [BORDER] prefix) 
- Zones Detection (Blue boxes with [ZONE] prefix)

Label format: [MODEL] class_name (confidence_score)
Annotation format: COCO JSON
For more info: https://cocodataset.org/#format-data
"""
        zipf.writestr("README.txt", readme_content)
    
    return zip_path

def calculate_statistics(detections_data: Dict, selected_classes: Dict[str, List[str]] = None) -> Dict[str, int]:
    """Calculate statistics (count per class) from detections_data"""
    stats = {}
    
    for model_name, detections in detections_data.items():
        if not detections:
            continue
            
        # Filter by selected classes if provided
        for detection in detections:
            class_name = detection["class_name"]
            
            # Only count if class is in selected classes (if selected_classes is provided)
            if selected_classes:
                if model_name not in selected_classes:
                    continue
                if class_name not in selected_classes[model_name]:
                    continue
            
            # Create full class identifier (model_name + class_name)
            full_class_name = f"{model_name} - {class_name}"
            
            if full_class_name not in stats:
                stats[full_class_name] = 0
            stats[full_class_name] += 1
    
    return stats

def create_statistics_table(stats: Dict[str, int], image_name: str = None) -> pd.DataFrame:
    """Create a pandas DataFrame table from statistics"""
    if not stats:
        return pd.DataFrame(columns=["Class", "Count"])
    
    data = []
    for class_name, count in sorted(stats.items()):
        data.append({"Class": class_name, "Count": count})
    
    df = pd.DataFrame(data)
    if image_name:
        df.insert(0, "Image", image_name)
    
    return df

def create_statistics_graph(stats: Dict[str, int], image_name: str = None) -> str:
    """Create a bar chart from statistics and return as image path"""
    if not stats:
        # Return empty graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No detections found", ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        classes = sorted(stats.keys())
        counts = [stats[c] for c in classes]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(classes)), counts, color='steelblue')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Detection Statistics{(" - " + image_name) if image_name else ""}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
    
    # Save to temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(tempfile.gettempdir(), f"statistics_graph_{timestamp}.png")
    fig.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return graph_path

def create_statistics_csv(stats: Dict[str, int], image_name: str = None) -> str:
    """Create CSV file from statistics"""
    df = create_statistics_table(stats, image_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(tempfile.gettempdir(), f"statistics_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path

def create_statistics_json(stats: Dict[str, int], image_name: str = None) -> str:
    """Create JSON file from statistics"""
    data = {
        "image": image_name,
        "timestamp": datetime.now().isoformat(),
        "statistics": stats
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(tempfile.gettempdir(), f"statistics_{timestamp}.json")
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return json_path

def calculate_batch_statistics(results_data: List[Tuple[str, Dict]], selected_classes: Dict[str, List[str]] = None) -> pd.DataFrame:
    """Calculate statistics for all images in batch processing - per image"""
    all_stats = []
    
    for filename, detections_by_model in results_data:
        stats = calculate_statistics(detections_by_model, selected_classes)
        df = create_statistics_table(stats, filename)
        if not df.empty:
            all_stats.append(df)
    
    if all_stats:
        combined_df = pd.concat(all_stats, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame(columns=["Image", "Class", "Count"])

def calculate_batch_statistics_summary(results_data: List[Tuple[str, Dict]], selected_classes: Dict[str, List[str]] = None) -> pd.DataFrame:
    """Calculate overall aggregated statistics for all images in batch"""
    # Aggregate statistics across all images
    all_stats = {}
    
    for filename, detections_by_model in results_data:
        stats = calculate_statistics(detections_by_model, selected_classes)
        for class_name, count in stats.items():
            if class_name not in all_stats:
                all_stats[class_name] = 0
            all_stats[class_name] += count
    
    # Create summary table
    if not all_stats:
        return pd.DataFrame(columns=["Class", "Total Count"])
    
    data = []
    for class_name, count in sorted(all_stats.items()):
        data.append({"Class": class_name, "Total Count": count})
    
    return pd.DataFrame(data)

def create_batch_statistics_graph(results_data: List[Tuple[str, Dict]], selected_classes: Dict[str, List[str]] = None) -> str:
    """Create a graph showing statistics across all images in batch"""
    # Aggregate statistics across all images
    all_stats = {}
    
    for filename, detections_by_model in results_data:
        stats = calculate_statistics(detections_by_model, selected_classes)
        for class_name, count in stats.items():
            if class_name not in all_stats:
                all_stats[class_name] = 0
            all_stats[class_name] += count
    
    return create_statistics_graph(all_stats, "Batch Processing")

def create_batch_statistics_csv(results_data: List[Tuple[str, Dict]], selected_classes: Dict[str, List[str]] = None) -> str:
    """Create CSV file from batch statistics - includes both per-image and summary"""
    # Get per-image statistics
    per_image_df = calculate_batch_statistics(results_data, selected_classes)
    # Get summary statistics
    summary_df = calculate_batch_statistics_summary(results_data, selected_classes)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(tempfile.gettempdir(), f"batch_statistics_{timestamp}.csv")
    
    # Write both to CSV with separator
    with open(csv_path, 'w') as f:
        # Write per-image statistics
        f.write("=== PER IMAGE STATISTICS ===\n")
        per_image_df.to_csv(f, index=False)
        f.write("\n\n=== OVERALL SUMMARY STATISTICS ===\n")
        summary_df.to_csv(f, index=False)
    
    return csv_path

def create_batch_statistics_json(results_data: List[Tuple[str, Dict]], selected_classes: Dict[str, List[str]] = None) -> str:
    """Create JSON file from batch statistics - includes both per-image and summary"""
    # Calculate summary statistics
    summary_stats = {}
    for filename, detections_by_model in results_data:
        stats = calculate_statistics(detections_by_model, selected_classes)
        for class_name, count in stats.items():
            if class_name not in summary_stats:
                summary_stats[class_name] = 0
            summary_stats[class_name] += count
    
    data = {
        "batch_processing": True,
        "timestamp": datetime.now().isoformat(),
        "total_images": len(results_data),
        "per_image_statistics": [],
        "overall_summary": summary_stats
    }
    
    for filename, detections_by_model in results_data:
        stats = calculate_statistics(detections_by_model, selected_classes)
        data["per_image_statistics"].append({
            "filename": filename,
            "statistics": stats
        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(tempfile.gettempdir(), f"batch_statistics_{timestamp}.json")
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return json_path

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Medieval Manuscript Detection with Custom YOLO Models")
    gr.Markdown("""
    **Models and Color Coding:**
    - 🔵**Line Detection** - Red boxes with [LINE] prefix
    - 🟢 **Border Detection** - Green boxes with [BORDER] prefix  
    - 🟠 **Zones Detection** - Blue boxes with [ZONE] prefix
    
    Each detection shows: **[MODEL] class_name (confidence_score)**
    """)
    
    with gr.Tabs():
        # Single Image Tab
        with gr.TabItem("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Input Image",
                        type='numpy'
                    )
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
                                value=0.45,
                                info="Decrease for stricter detection, increase for more overlapping boxes"
                            )
                    
                    with gr.Accordion("Class Selection", open=False):
                        gr.Markdown("**Select which classes to detect for each model:**")
                        with gr.Row():
                            with gr.Column():
                                line_classes = gr.CheckboxGroup(
                                    label="Line Detection Classes",
                                    choices=MODEL_CLASSES["Line Detection"],
                                    value=MODEL_CLASSES["Line Detection"],  # All selected by default
                                    info="Select at least one class for detection"
                                )
                                with gr.Row():
                                    line_select_all = gr.Button("Select All", size="sm")
                                    line_unselect_all = gr.Button("Unselect All", size="sm")
                            with gr.Column():
                                border_classes = gr.CheckboxGroup(
                                    label="Border Detection Classes", 
                                    choices=MODEL_CLASSES["Border Detection"],
                                    value=MODEL_CLASSES["Border Detection"],  # All selected by default
                                    info="Select at least one class for detection"
                                )
                                with gr.Row():
                                    border_select_all = gr.Button("Select All", size="sm")
                                    border_unselect_all = gr.Button("Unselect All", size="sm")
                        with gr.Row():
                            with gr.Column():
                                zones_classes = gr.CheckboxGroup(
                                    label="Zones Detection Classes",
                                    choices=MODEL_CLASSES["Zones Detection"],
                                    value=MODEL_CLASSES["Zones Detection"],  # All selected by default
                                    info="Select at least one class for detection"
                                )
                                with gr.Row():
                                    zones_select_all = gr.Button("Select All", size="sm")
                                    zones_unselect_all = gr.Button("Unselect All", size="sm")
                    with gr.Row():
                        clear_btn = gr.Button("Clear")
                        detect_btn = gr.Button("Detect with All Models", variant="primary")
                        
                with gr.Column():
                    output_image = gr.Image(
                        label="Combined Detection Result",
                        type='numpy'
                    )
                    
                    # Single image download buttons
                    with gr.Row():
                        single_download_json_btn = gr.Button(
                            "📄 Download Annotations (JSON)",
                            variant="secondary",
                            size="sm"
                        )
                        single_download_image_btn = gr.Button(
                            "🖼️ Download Image",
                            variant="secondary",
                            size="sm"
                        )
                    
                    # Single image file outputs
                    single_json_output = gr.File(
                        label="📄 JSON Download",
                        visible=True,
                        height=50
                    )
                    single_image_output = gr.File(
                        label="🖼️ Image Download",
                        visible=True,
                        height=50
                    )
                    
                    # Statistics section for single image
                    with gr.Accordion("📊 Statistics", open=False):
                        with gr.Tabs():
                            with gr.TabItem("Table"):
                                single_stats_table = gr.Dataframe(
                                    label="Detection Statistics",
                                    headers=["Class", "Count"],
                                    wrap=True
                                )
                            with gr.TabItem("Graph"):
                                single_stats_graph = gr.Image(
                                    label="Detection Statistics Graph",
                                    type='filepath'
                                )
                        
                        # Statistics download buttons
                        with gr.Row():
                            single_download_stats_csv_btn = gr.Button(
                                "📊 Download Statistics (CSV)",
                                variant="secondary",
                                size="sm"
                            )
                            single_download_stats_json_btn = gr.Button(
                                "📊 Download Statistics (JSON)",
                                variant="secondary",
                                size="sm"
                            )
                        
                        single_stats_csv_output = gr.File(
                            label="📊 Statistics CSV Download",
                            visible=False,
                            height=50
                        )
                        single_stats_json_output = gr.File(
                            label="📊 Statistics JSON Download",
                            visible=False,
                            height=50
                        )
        
        # Batch Processing Tab
        with gr.TabItem("Batch Processing (ZIP)"):
            with gr.Row():
                with gr.Column():
                    zip_file = gr.File(
                        label="Upload ZIP file with images",
                        file_types=[".zip"]
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
                                value=0.45,
                            )
                    
                    with gr.Accordion("Class Selection", open=False):
                        gr.Markdown("**Select which classes to detect for each model:**")
                        with gr.Row():
                            with gr.Column():
                                batch_line_classes = gr.CheckboxGroup(
                                    label="Line Detection Classes",
                                    choices=MODEL_CLASSES["Line Detection"],
                                    value=MODEL_CLASSES["Line Detection"],  # All selected by default
                                    info="Select at least one class for detection"
                                )
                                with gr.Row():
                                    batch_line_select_all = gr.Button("Select All", size="sm")
                                    batch_line_unselect_all = gr.Button("Unselect All", size="sm")
                            with gr.Column():
                                batch_border_classes = gr.CheckboxGroup(
                                    label="Border Detection Classes", 
                                    choices=MODEL_CLASSES["Border Detection"],
                                    value=MODEL_CLASSES["Border Detection"],  # All selected by default
                                    info="Select at least one class for detection"
                                )
                                with gr.Row():
                                    batch_border_select_all = gr.Button("Select All", size="sm")
                                    batch_border_unselect_all = gr.Button("Unselect All", size="sm")
                        with gr.Row():
                            with gr.Column():
                                batch_zones_classes = gr.CheckboxGroup(
                                    label="Zones Detection Classes",
                                    choices=MODEL_CLASSES["Zones Detection"],
                                    value=MODEL_CLASSES["Zones Detection"],  # All selected by default
                                    info="Select at least one class for detection"
                                )
                                with gr.Row():
                                    batch_zones_select_all = gr.Button("Select All", size="sm")
                                    batch_zones_unselect_all = gr.Button("Unselect All", size="sm")
                    
                    # Add status message box
                    batch_status = gr.Textbox(
                        label="Processing Status",
                        value="Ready to process ZIP file...",
                        interactive=False,
                        max_lines=3
                    )
                    
                    with gr.Row():
                        clear_batch_btn = gr.Button("Clear")
                        process_batch_btn = gr.Button("Process ZIP", variant="primary")
                        
                with gr.Column():
                    batch_gallery = gr.Gallery(
                        label="Batch Processing Results",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto",
                        type="numpy"  # Explicitly handle numpy arrays
                    )
                    
                    # Download buttons
                    with gr.Row():
                        download_json_btn = gr.Button(
                            "📄 Download COCO Annotations (JSON)",
                            variant="secondary"
                        )
                        download_zip_btn = gr.Button(
                            "📦 Download Results (ZIP)",
                            variant="secondary"
                        )
                    
                    # File outputs for downloads
                    json_file_output = gr.File(
                        label="📄 JSON Download",
                        visible=True,
                        height=50
                    )
                    zip_file_output = gr.File(
                        label="📦 ZIP Download",
                        visible=True,
                        height=50
                    )
                    
                    # Statistics section for batch processing
                    with gr.Accordion("📊 Statistics", open=False):
                        with gr.Tabs():
                            with gr.TabItem("Per Image"):
                                batch_stats_table = gr.Dataframe(
                                    label="Detection Statistics Per Image",
                                    wrap=True
                                )
                            with gr.TabItem("Overall Summary"):
                                batch_stats_summary_table = gr.Dataframe(
                                    label="Overall Statistics Summary (All Images Combined)",
                                    wrap=True
                                )
                            with gr.TabItem("Graph"):
                                batch_stats_graph = gr.Image(
                                    label="Detection Statistics Graph (Aggregated)",
                                    type='filepath'
                                )
                        
                        # Statistics download buttons
                        with gr.Row():
                            batch_download_stats_csv_btn = gr.Button(
                                "📊 Download Statistics (CSV)",
                                variant="secondary",
                                size="sm"
                            )
                            batch_download_stats_json_btn = gr.Button(
                                "📊 Download Statistics (JSON)",
                                variant="secondary",
                                size="sm"
                            )
                        
                        batch_stats_csv_output = gr.File(
                            label="📊 Statistics CSV Download",
                            visible=False,
                            height=50
                        )
                        batch_stats_json_output = gr.File(
                            label="📊 Statistics JSON Download",
                            visible=False,
                            height=50
                        )

    # Global variables for single image results
    single_image_result = None
    single_image_annotations = None
    single_image_filename = None
    single_image_selected_classes = None
    
    def process_single_image(
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        line_classes: List[str],
        border_classes: List[str],
        zones_classes: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
        global single_image_result, single_image_annotations, single_image_filename, single_image_selected_classes
        
        if image is None:
            single_image_result = None
            single_image_annotations = None
            single_image_filename = None
            single_image_selected_classes = None
            return None, None, pd.DataFrame(columns=["Class", "Count"]), None
            
        # Validate that at least one class is selected
        all_selected = (line_classes or []) + (border_classes or []) + (zones_classes or [])
        if not all_selected:
            raise gr.Error("⚠️ Please select at least one class for detection!")
        
        # Prepare selected classes dictionary
        selected_classes = {
            "Line Detection": line_classes or [],
            "Border Detection": border_classes or [],
            "Zones Detection": zones_classes or []
        }
        
        # Process with annotations
        try:
            annotated_image, detections_data = detect_and_annotate_combined(
                image, conf_threshold, iou_threshold, return_annotations=True, selected_classes=selected_classes
            )
        except Exception as e:
            # Surface a nice error to the UI without crashing the app
            raise gr.Error(f"Detection failed: {str(e)}")
        
        # Calculate statistics
        stats = calculate_statistics(detections_data, selected_classes)
        stats_table = create_statistics_table(stats, single_image_filename)
        stats_graph_path = create_statistics_graph(stats, single_image_filename)
        
        # Store results globally for download
        single_image_result = annotated_image
        single_image_annotations = detections_data
        single_image_selected_classes = selected_classes
        single_image_filename = f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        return image, annotated_image, stats_table, stats_graph_path

    # Global variables for batch results
    current_batch_results = []
    current_batch_selected_classes = None
    
    def process_batch_images_with_status(
        zip_file,
        conf_threshold: float,
        iou_threshold: float,
        line_classes: List[str],
        border_classes: List[str],
        zones_classes: List[str]
    ):
        global current_batch_results, current_batch_selected_classes
        
        print("🚀 ========== BATCH PROCESSING STARTED ==========")
        
        if zip_file is None:
            print("❌ No ZIP file provided")
            return [], "Please upload a ZIP file first.", pd.DataFrame(columns=["Image", "Class", "Count"]), pd.DataFrame(columns=["Class", "Total Count"]), None
        
        print(f"📁 ZIP file received: {zip_file.name}")
        print(f"⚙️  Settings: conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")
        
        try:
            # Validate that at least one class is selected
            all_selected = (line_classes or []) + (border_classes or []) + (zones_classes or [])
            if not all_selected:
                raise gr.Error("⚠️ Please select at least one class for detection!")
            
            # Prepare selected classes dictionary
            selected_classes = {
                "Line Detection": line_classes or [],
                "Border Detection": border_classes or [],
                "Zones Detection": zones_classes or []
            }
            current_batch_selected_classes = selected_classes
            
            # Process zip file
            print("🔄 Starting ZIP file processing...")
            results, annotations_data, image_info = process_zip_file(zip_file.name, conf_threshold, iou_threshold, selected_classes)
            
            # Store batch results globally
            current_batch_results = results
            
            if not results:
                error_msg = "No valid images found in ZIP file."
                print(f"❌ {error_msg}")
                return [], error_msg
            
            # Store data globally for download
            global current_results, current_images
            current_images = results
            current_results = annotations_data
            
            print(f"📊 ZIP processing returned {len(results)} results")
            
            # Convert results to format expected by Gallery
            print("🔄 Converting results for Gradio Gallery...")
            gallery_images = []
            
            for i, (filename, annotated_image) in enumerate(results):
                print(f"🖼️  Converting image {i+1}/{len(results)}: {filename}")
                print(f"   Image shape: {annotated_image.shape}, dtype: {annotated_image.dtype}")
                
                # Ensure the image is in the right format and range
                if annotated_image.dtype != 'uint8':
                    print(f"   Converting dtype from {annotated_image.dtype} to uint8")
                    # Normalize if needed
                    if annotated_image.max() <= 1.0:
                        annotated_image = (annotated_image * 255).astype('uint8')
                        print(f"   Normalized from [0,1] to [0,255]")
                    else:
                        annotated_image = annotated_image.astype('uint8')
                        print(f"   Cast to uint8")
                
                print(f"   Final image shape: {annotated_image.shape}, dtype: {annotated_image.dtype}")
                
                # For Gradio gallery, we can pass numpy arrays directly
                # Format: (image_data, caption)
                gallery_images.append((annotated_image, filename))
                print(f"   ✅ Added {filename} to gallery")
            
            # Calculate statistics (use annotations_data, not results)
            stats_table = calculate_batch_statistics(annotations_data, selected_classes)
            stats_summary_table = calculate_batch_statistics_summary(annotations_data, selected_classes)
            stats_graph_path = create_batch_statistics_graph(annotations_data, selected_classes)
            
            success_msg = f"✅ Successfully processed {len(gallery_images)} images!"
            print(f"🎉 {success_msg}")
            print(f"📋 Gallery contains {len(gallery_images)} items")
            print("🏁 ========== BATCH PROCESSING COMPLETED ==========\n")
            
            return gallery_images, success_msg, stats_table, stats_summary_table, stats_graph_path
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"💥 EXCEPTION in process_batch_images_with_status: {error_msg}")
            import traceback
            traceback.print_exc()
            print("💀 ========== BATCH PROCESSING FAILED ==========\n")
            return [], error_msg, pd.DataFrame(columns=["Image", "Class", "Count"]), pd.DataFrame(columns=["Class", "Total Count"]), None

    def clear_single():
        global single_image_result, single_image_annotations, single_image_filename, single_image_selected_classes
        single_image_result = None
        single_image_annotations = None
        single_image_filename = None
        single_image_selected_classes = None
        return None, None, pd.DataFrame(columns=["Class", "Count"]), None
    
    def clear_batch():
        global current_results, current_images
        current_results = []
        current_images = []
        return None, [], "Ready to process ZIP file..."
    
    def download_annotations():
        """Create and return COCO JSON annotations file"""
        global current_results, current_images
        
        if not current_results:
            print("❌ No annotation data available for download")
            return None
        
        try:
            # Create image info dictionary
            image_info = {}
            for filename, image_array in current_images:
                height, width = image_array.shape[:2]
                image_info[filename] = (height, width)
            
            # Create COCO annotations
            coco_data = create_coco_annotations(current_results, image_info)
            
            # Save to temporary file with proper name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"medieval_annotations_{timestamp}.json"
            json_path = os.path.join(tempfile.gettempdir(), json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"💾 Created annotations file: {json_path}")
            print(f"📁 File size: {os.path.getsize(json_path)} bytes")
            
            # Verify file exists and is readable
            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                return json_path
            else:
                print(f"❌ File verification failed: {json_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating annotations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_results_zip():
        """Create and return ZIP file with images and annotations"""
        global current_results, current_images
        
        if not current_results or not current_images:
            print("❌ No results data available for ZIP download")
            return None
        
        try:
            # Create image info dictionary
            image_info = {}
            for filename, image_array in current_images:
                height, width = image_array.shape[:2]
                image_info[filename] = (height, width)
            
            # Create COCO annotations
            coco_data = create_coco_annotations(current_results, image_info)
            
            # Create ZIP file
            zip_path = create_download_zip(current_images, coco_data)
            
            print(f"💾 Created results ZIP: {zip_path}")
            print(f"📁 ZIP file size: {os.path.getsize(zip_path)} bytes")
            
            # Verify file exists and is readable
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                return zip_path
            else:
                print(f"❌ ZIP file verification failed: {zip_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating ZIP file: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_single_annotations():
        """Download COCO annotations for single image"""
        global single_image_annotations, single_image_result, single_image_filename
        
        if single_image_annotations is None or single_image_result is None:
            print("❌ No single image annotation data available")
            return None
        
        try:
            # Create image info
            height, width = single_image_result.shape[:2]
            image_info = {single_image_filename: (height, width)}
            
            # Create annotations data in the expected format
            annotations_data = [(single_image_filename, single_image_annotations)]
            
            # Create COCO annotations
            coco_data = create_coco_annotations(annotations_data, image_info)
            
            # Save to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"single_image_annotations_{timestamp}.json"
            json_path = os.path.join(tempfile.gettempdir(), json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"💾 Created single image annotations: {json_path}")
            print(f"📁 File size: {os.path.getsize(json_path)} bytes")
            
            # Verify file exists
            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                return json_path
            else:
                print(f"❌ Single image file verification failed: {json_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating single image annotations: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def download_single_image():
        """Download processed single image"""
        global single_image_result, single_image_filename
        
        if single_image_result is None:
            print("❌ No single image result available")
            return None
        
        try:
            # Convert to PIL and save
            pil_image = Image.fromarray(single_image_result.astype('uint8'))
            
            # Save to temporary file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"processed_image_{timestamp}.jpg"
            img_path = os.path.join(tempfile.gettempdir(), img_filename)
            
            pil_image.save(img_path, 'JPEG', quality=95)
            
            print(f"💾 Created single image file: {img_path}")
            print(f"📁 Image file size: {os.path.getsize(img_path)} bytes")
            
            # Verify file exists
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                return img_path
            else:
                print(f"❌ Single image file verification failed: {img_path}")
                return None
            
        except Exception as e:
            print(f"❌ Error creating single image file: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Connect buttons to functions for single image
    detect_btn.click(
        process_single_image,
        inputs=[input_image, conf_threshold, iou_threshold, line_classes, border_classes, zones_classes],
        outputs=[input_image, output_image, single_stats_table, single_stats_graph]
    )
    clear_btn.click(
        clear_single,
        inputs=None,
        outputs=[input_image, output_image, single_stats_table, single_stats_graph]
    )
    
    # Select All/Unselect All handlers for single image
    line_select_all.click(
        fn=lambda: MODEL_CLASSES["Line Detection"],
        outputs=[line_classes]
    )
    line_unselect_all.click(
        fn=lambda: [],
        outputs=[line_classes]
    )
    border_select_all.click(
        fn=lambda: MODEL_CLASSES["Border Detection"],
        outputs=[border_classes]
    )
    border_unselect_all.click(
        fn=lambda: [],
        outputs=[border_classes]
    )
    zones_select_all.click(
        fn=lambda: MODEL_CLASSES["Zones Detection"],
        outputs=[zones_classes]
    )
    zones_unselect_all.click(
        fn=lambda: [],
        outputs=[zones_classes]
    )
    
    # Connect buttons to functions for batch processing
    process_batch_btn.click(
        process_batch_images_with_status,
        inputs=[zip_file, batch_conf_threshold, batch_iou_threshold, batch_line_classes, batch_border_classes, batch_zones_classes],
        outputs=[batch_gallery, batch_status, batch_stats_table, batch_stats_summary_table, batch_stats_graph]
    )
    clear_batch_btn.click(
        clear_batch,
        inputs=None,
        outputs=[zip_file, batch_gallery, batch_status]
    )
    
    # Select All/Unselect All handlers for batch processing
    batch_line_select_all.click(
        fn=lambda: MODEL_CLASSES["Line Detection"],
        outputs=[batch_line_classes]
    )
    batch_line_unselect_all.click(
        fn=lambda: [],
        outputs=[batch_line_classes]
    )
    batch_border_select_all.click(
        fn=lambda: MODEL_CLASSES["Border Detection"],
        outputs=[batch_border_classes]
    )
    batch_border_unselect_all.click(
        fn=lambda: [],
        outputs=[batch_border_classes]
    )
    batch_zones_select_all.click(
        fn=lambda: MODEL_CLASSES["Zones Detection"],
        outputs=[batch_zones_classes]
    )
    batch_zones_unselect_all.click(
        fn=lambda: [],
        outputs=[batch_zones_classes]
    )
    
    # Connect download buttons
    download_json_btn.click(
        fn=download_annotations,
        inputs=[],
        outputs=[json_file_output]
    )
    download_zip_btn.click(
        fn=download_results_zip,
        inputs=[],
        outputs=[zip_file_output]
    )
    
    # Connect single image download buttons
    single_download_json_btn.click(
        fn=download_single_annotations,
        inputs=[],
        outputs=[single_json_output]
    )
    single_download_image_btn.click(
        fn=download_single_image,
        inputs=[],
        outputs=[single_image_output]
    )
    
    # Statistics download handlers for single image
    def download_single_stats_csv():
        global single_image_annotations, single_image_filename, single_image_selected_classes
        if single_image_annotations is None:
            return None
        stats = calculate_statistics(single_image_annotations, single_image_selected_classes)
        csv_path = create_statistics_csv(stats, single_image_filename)
        return csv_path
    
    def download_single_stats_json():
        global single_image_annotations, single_image_filename, single_image_selected_classes
        if single_image_annotations is None:
            return None
        stats = calculate_statistics(single_image_annotations, single_image_selected_classes)
        json_path = create_statistics_json(stats, single_image_filename)
        return json_path
    
    single_download_stats_csv_btn.click(
        fn=download_single_stats_csv,
        inputs=[],
        outputs=[single_stats_csv_output]
    )
    single_download_stats_json_btn.click(
        fn=download_single_stats_json,
        inputs=[],
        outputs=[single_stats_json_output]
    )
    
    # Statistics download handlers for batch processing
    def download_batch_stats_csv():
        global current_results, current_batch_selected_classes
        if not current_results:
            return None
        csv_path = create_batch_statistics_csv(current_results, current_batch_selected_classes)
        return csv_path
    
    def download_batch_stats_json():
        global current_results, current_batch_selected_classes
        if not current_results:
            return None
        json_path = create_batch_statistics_json(current_results, current_batch_selected_classes)
        return json_path
    
    batch_download_stats_csv_btn.click(
        fn=download_batch_stats_csv,
        inputs=[],
        outputs=[batch_stats_csv_output]
    )
    batch_download_stats_json_btn.click(
        fn=download_batch_stats_json,
        inputs=[],
        outputs=[batch_stats_json_output]
    )

if __name__ == "__main__":
    # Configure launch settings for better stability
    # Enable Gradio queue for more robust concurrency and error isolation
    demo.queue()
    demo.launch(
        debug=False,  # Disable debug mode for production
        show_error=True,
        server_name="0.0.0.0",
        server_port=8000,
        share=False,
        max_threads=4,  # Limit concurrent requests
        auth=None,
        inbrowser=False,
        favicon_path=None,
        ssl_verify=True,
        quiet=False
    )
