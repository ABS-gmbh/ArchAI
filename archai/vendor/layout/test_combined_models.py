#!/usr/bin/env python3
"""
Script to combine predictions from three YOLO models:
- best_emanuskript_segmentation.pt (segmentation model for manuscript elements)
- best_catmus.pt (segmentation model for lines and zones)
- best_zone_detection.pt (detection model for zones)

WORKFLOW SUMMARY:
================

1. MODEL PREDICTIONS (run_model_predictions):
   - Runs each of the 3 models on the input image
   - Saves predictions to JSON files in separate folders
   - Emanuskript: detects manuscript elements (Main script, Plain initial, etc.)
   - Catmus: detects lines (DefaultLine, InterlinearLine)
   - Zone: detects zones (MainZone, DropCapitalZone, etc.)

2. COMBINING & FILTERING (combine_and_filter_predictions):
   - Uses ImageBatch class to:
     a) Load all predictions from the 3 JSON files
     b) Unify class names (maps catmus/zone names to coco_class_mapping)
     c) Filter overlapping/conflicting annotations using spatial indexing
     d) Convert to COCO format
   - Only keeps classes defined in coco_class_mapping (25 classes total)

3. OUTPUT:
   - COCO format JSON file with filtered annotations
   - Only contains classes from coco_class_mapping

KEY CLASSES IN coco_class_mapping:
- Main script black/coloured
- Variant script black/coloured  
- Plain initial (coloured/highlighted/black)
- Historiated, Inhabited, Embellished
- Page Number, Quire Mark, Running header
- Gloss, Illustrations, Column
- Music, MusicZone, MusicLine
- Border, Table, Diagram
- GraphicZone

The ImageBatch class handles:
- Spatial overlap detection (removes duplicates)
- Class name unification (catmus_zones_mapping)
- Annotation filtering based on overlap thresholds
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from ultralytics import YOLO
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add current directory to path to import ImageBatch
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = current_dir  # This file is in the project root
sys.path.insert(0, current_dir)

try:
    from utils.image_batch_classes import ImageBatch, coco_class_mapping
except ImportError as e:
    print(f"Warning: Could not import ImageBatch: {e}")
    print("Make sure all dependencies are installed (rtree, shapely, etc.)")
    ImageBatch = None

def _run_emanuskript_model(image_path, output_dir, project_root):
    """Run emanuskript segmentation model (for parallel execution)."""
    import sys
    emanuskript_dir = os.path.join(output_dir, 'emanuskript')
    os.makedirs(emanuskript_dir, exist_ok=True)
    image_id = Path(image_path).stem
    
    print("[emanuskript] Running segmentation model...", flush=True)
    sys.stdout.flush()
    emanuskript_model_path = os.path.join(project_root, "best_emanuskript_segmentation.pt")
    emanuskript_model = YOLO(emanuskript_model_path)
    emanuskript_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20]
    emanuskript_results = emanuskript_model.predict(
        image_path,
        classes=emanuskript_classes,
        iou=0.3,
        device='cpu',
        augment=False,
        stream=False
    )
    emanuskript_path = f'{emanuskript_dir}/{image_id}.json'
    with open(emanuskript_path, 'w') as f:
        f.write(emanuskript_results[0].to_json())
    print(f"[emanuskript] ✓ Saved to: {emanuskript_path}", flush=True)
    del emanuskript_model
    del emanuskript_results
    return ('emanuskript', emanuskript_dir)


def _run_catmus_model(image_path, output_dir, project_root):
    """Run catmus segmentation model (for parallel execution)."""
    import sys
    catmus_dir = os.path.join(output_dir, 'catmus')
    os.makedirs(catmus_dir, exist_ok=True)
    image_id = Path(image_path).stem
    
    print("[catmus] Running segmentation model...", flush=True)
    sys.stdout.flush()
    catmus_model_path = os.path.join(project_root, "best_catmus.pt")
    catmus_model = YOLO(catmus_model_path)
    catmus_classes = [1, 7]  # DefaultLine and InterlinearLine
    catmus_results = catmus_model.predict(
        image_path,
        classes=catmus_classes,
        iou=0.3,
        device='cpu',
        augment=False,
        stream=False
    )
    catmus_path = f'{catmus_dir}/{image_id}.json'
    with open(catmus_path, 'w') as f:
        f.write(catmus_results[0].to_json())
    print(f"[catmus] ✓ Saved to: {catmus_path}", flush=True)
    del catmus_model
    del catmus_results
    return ('catmus', catmus_dir)


def _run_zone_model(image_path, output_dir, project_root):
    """Run zone detection model (for parallel execution)."""
    import sys
    zone_dir = os.path.join(output_dir, 'zone')
    os.makedirs(zone_dir, exist_ok=True)
    image_id = Path(image_path).stem
    
    print("[zone] Running detection model...", flush=True)
    sys.stdout.flush()
    zone_model_path = os.path.join(project_root, "best_zone_detection.pt")
    zone_model = YOLO(zone_model_path)
    zone_results = zone_model.predict(
        image_path,
        device='cpu',
        iou=0.3,
        augment=False,
        stream=False
    )
    zone_path = f'{zone_dir}/{image_id}.json'
    with open(zone_path, 'w') as f:
        f.write(zone_results[0].to_json())
    print(f"[zone] ✓ Saved to: {zone_path}", flush=True)
    del zone_model
    del zone_results
    return ('zone', zone_dir)


def _run_model_worker(args):
    """Worker function for multiprocessing - unpacks args and calls appropriate model."""
    model_name, image_path, output_dir, project_root = args
    if model_name == 'emanuskript':
        return _run_emanuskript_model(image_path, output_dir, project_root)
    elif model_name == 'catmus':
        return _run_catmus_model(image_path, output_dir, project_root)
    elif model_name == 'zone':
        return _run_zone_model(image_path, output_dir, project_root)


def run_model_predictions(image_path, output_dir, force_sequential=False):
    """Run all three models on the image and save predictions.
    
    Uses parallel execution with 3 processes if 3+ CPU cores are available
    and not running in a daemon process, otherwise runs models sequentially.
    
    Args:
        image_path: Path to image file
        output_dir: Directory to save predictions
        force_sequential: Force sequential mode (useful when called from worker processes)
    """
    import sys
    
    # Detect available CPU cores
    num_cores = multiprocessing.cpu_count()
    
    # Check if we're in a daemon process (can't spawn children)
    current_process = multiprocessing.current_process()
    is_daemon = current_process.daemon if hasattr(current_process, 'daemon') else False
    
    # Use parallel only if: enough cores, not forced sequential, not in daemon
    use_parallel = num_cores >= 3 and not force_sequential and not is_daemon
    
    print("=" * 60, flush=True)
    print("Running Model Predictions", flush=True)
    if is_daemon:
        print(f"  Running in worker process - using SEQUENTIAL mode", flush=True)
    else:
        print(f"  CPU cores detected: {num_cores}", flush=True)
        print(f"  Execution mode: {'PARALLEL (3 workers)' if use_parallel else 'SEQUENTIAL'}", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    
    if use_parallel:
        # Run all 3 models in parallel using ProcessPoolExecutor
        model_args = [
            ('emanuskript', image_path, output_dir, PROJECT_ROOT),
            ('catmus', image_path, output_dir, PROJECT_ROOT),
            ('zone', image_path, output_dir, PROJECT_ROOT),
        ]
        
        results = {}
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_run_model_worker, args): args[0] for args in model_args}
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    name, dir_path = future.result()
                    results[name] = dir_path
                    print(f"  ✓ {model_name} model completed", flush=True)
                except Exception as e:
                    print(f"  ✗ {model_name} model failed: {e}", flush=True)
                    raise
        
        return results
    else:
        # Sequential execution (1 core fallback)
        print("\n[1/3] Running emanuskript model...", flush=True)
        sys.stdout.flush()
        _, emanuskript_dir = _run_emanuskript_model(image_path, output_dir, PROJECT_ROOT)
        
        print("\n[2/3] Running catmus model...", flush=True)
        sys.stdout.flush()
        _, catmus_dir = _run_catmus_model(image_path, output_dir, PROJECT_ROOT)
        
        print("\n[3/3] Running zone model...", flush=True)
        sys.stdout.flush()
        _, zone_dir = _run_zone_model(image_path, output_dir, PROJECT_ROOT)
        
        return {
            'catmus': catmus_dir,
            'emanuskript': emanuskript_dir,
            'zone': zone_dir
        }


def combine_and_filter_predictions(image_path, labels_folders, output_json_path=None):
    """Combine predictions from all models and filter to coco_class_mapping classes."""
    import sys
    
    print("\n" + "=" * 60, flush=True)
    print("Combining and Filtering Predictions", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    
    if ImageBatch is None:
        print("\nERROR: ImageBatch class not available.", flush=True)
        print("Please install missing dependencies:", flush=True)
        print("  pip install rtree shapely", flush=True)
        return None
    
    try:
        # Validate inputs
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        for name, folder in labels_folders.items():
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Labels folder not found: {folder} ({name})")
        
        # Create a temporary folder with just the image file
        # ImageBatch.load_images() loads all files in the folder, so we need only images
        temp_image_dir = tempfile.mkdtemp()
        image_filename = os.path.basename(image_path)
        temp_image_path = os.path.join(temp_image_dir, image_filename)
        shutil.copy2(image_path, temp_image_path)
        
        # Create ImageBatch instance
        image_folder = temp_image_dir
        
        image_batch = ImageBatch(
            image_folder=image_folder,
            catmus_labels_folder=labels_folders['catmus'],
            emanuskript_labels_folder=labels_folders['emanuskript'],
            zone_labels_folder=labels_folders['zone']
        )
        
        # Load images
        print("\n[Step 1] Loading images...", flush=True)
        image_batch.load_images()
        print(f"  ✓ Loaded {len(image_batch.images)} image(s)", flush=True)
        
        # Load annotations from all three models
        print("\n[Step 2] Loading annotations from all models...", flush=True)
        sys.stdout.flush()
        image_batch.load_annotations()
        
        total_annotations = sum(len(img.annotations) for img in image_batch.images)
        print(f"  ✓ Loaded {total_annotations} total annotations", flush=True)
        # Unify names (maps catmus/zone names to coco_class_mapping names)
        print("\n[Step 3] Unifying class names...", flush=True)
        image_batch.unify_names()
        
        # Filter annotations (removes overlapping/conflicting annotations)
        print("\n[Step 4] Filtering annotations...", flush=True)
        for img in image_batch.images:
            filtered = img.filter_annotations()
            print(f"  Image {img.filename}: {len(img.annotations)} -> {len(filtered)} annotations", flush=True)
        
        # Get COCO format JSON
        print("\n[Step 5] Generating COCO format...", flush=True)
        sys.stdout.flush()
        coco_json = image_batch.return_coco_file()
        
        # Filter to only classes in coco_class_mapping
        valid_category_ids = set(coco_class_mapping.values())
        
        filtered_annotations = [
            ann for ann in coco_json['annotations']
            if ann['category_id'] in valid_category_ids
        ]
        
        coco_json['annotations'] = filtered_annotations
        
        # Update categories to only include valid ones
        coco_json['categories'] = [
            cat for cat in coco_json['categories']
            if cat['id'] in valid_category_ids
        ]
        
        print(f"  ✓ Final annotations: {len(filtered_annotations)}", flush=True)
        print(f"  ✓ Final categories: {len(coco_json['categories'])}", flush=True)
        # Save to file if path provided
        if output_json_path:
            with open(output_json_path, 'w') as f:
                json.dump(coco_json, f, indent=2)
            print(f"\n  ✓ Saved COCO JSON to: {output_json_path}", flush=True)
        
        return coco_json
    except Exception as e:
        print(f"\nERROR in combine_and_filter_predictions: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        # Cleanup temporary image directory
        if 'temp_image_dir' in locals():
            shutil.rmtree(temp_image_dir, ignore_errors=True)


def print_summary(coco_json):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    # Category counts
    category_counts = {}
    for ann in coco_json['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    # Map category IDs to names
    id_to_name = {cat['id']: cat['name'] for cat in coco_json['categories']}
    
    print(f"\nTotal Annotations: {len(coco_json['annotations'])}")
    print(f"Total Categories: {len(coco_json['categories'])}")
    print(f"\nAnnotations per Category:")
    for cat_id in sorted(category_counts.keys()):
        name = id_to_name.get(cat_id, f"Unknown({cat_id})")
        count = category_counts[cat_id]
        print(f"  {name:30s}: {count:4d}")


def visualize_results(image_path, coco_json):
    """Visualize the combined results on the image."""
    print("\n" + "=" * 60)
    print("Visualizing Results")
    print("=" * 60)
    
    try:
        from utils.image_batch_classes import ImageBatch
        import tempfile
        
        # Create temporary labels folders for visualization
        with tempfile.TemporaryDirectory() as temp_dir:
            # We need to recreate the ImageBatch with the combined results
            # For now, just show the COCO JSON structure
            print("\nTo visualize, you can:")
            print("1. Use the COCO JSON file with any COCO visualization tool")
            print("2. Load the JSON in your annotation tool")
            print("3. Use the ImageBatch.plot_annotations() method")
            
    except Exception as e:
        print(f"Visualization not available: {e}")


def main():
    """Main function to run the complete pipeline."""
    
    # Configuration
    image_path = "bnf-naf-10039__page-001-of-004.jpg"
    output_json = "combined_predictions.json"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Create temporary directory for predictions
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Step 1: Run all three models
        labels_folders = run_model_predictions(image_path, temp_dir)
        
        # Step 2: Combine and filter predictions
        coco_json = combine_and_filter_predictions(
            image_path,
            labels_folders,
            output_json_path=output_json
        )
        
        # Step 3: Print summary
        print_summary(coco_json)
        
        # Step 4: Visualize (optional)
        visualize_results(image_path, coco_json)
        
        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nOutput saved to: {output_json}")


if __name__ == "__main__":
    main()

