"""
Visualize annotations for SampleBatch2, SampleBatch3, and SampleBatch4.
These folders already have COCO format JSON files, so we just need to visualize them.
"""
import os
import json
import sys
from pathlib import Path

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from visualize_ground_truth import visualize_all_images, draw_coco_annotations


# List of sample batch folders
SAMPLE_BATCH_FOLDERS = [
    "SampleBatch2",
    "SampleBatch3",
    "SampleBatch4",
]


def visualize_sample_batch(folder_name, base_dir=None):
    """
    Visualize annotations for a sample batch folder.
    
    Args:
        folder_name: Name of the sample batch folder
        base_dir: Base directory containing the folders (default: SCRIPT_DIR)
    
    Returns:
        dict with processing results
    """
    if base_dir is None:
        base_dir = SCRIPT_DIR
    
    folder_path = Path(base_dir) / folder_name
    
    if not folder_path.exists():
        print(f"⚠️  Warning: Folder not found: {folder_path}")
        return {
            "folder": folder_name,
            "status": "not_found",
            "images": 0,
            "annotations": 0
        }
    
    print("\n" + "=" * 70)
    print(f"Processing: {folder_name}")
    print("=" * 70)
    
    # Paths
    json_path = folder_path / "Annotations" / "instances_default.json"
    images_dir = folder_path / "Images"
    
    # Check if required files/directories exist
    if not json_path.exists():
        print(f"⚠️  Warning: JSON file not found: {json_path}")
        return {
            "folder": folder_name,
            "status": "no_json",
            "images": 0,
            "annotations": 0
        }
    
    if not images_dir.exists():
        print(f"⚠️  Warning: Images directory not found: {images_dir}")
        return {
            "folder": folder_name,
            "status": "no_images",
            "images": 0,
            "annotations": 0
        }
    
    # Load COCO JSON
    print(f"\n[Loading COCO JSON]")
    print(f"  JSON: {json_path}")
    print(f"  Images: {images_dir}")
    
    try:
        with open(json_path, 'r') as f:
            coco_json = json.load(f)
        
        # Verify it's COCO format
        if not all(key in coco_json for key in ['images', 'annotations', 'categories']):
            print(f"⚠️  Warning: JSON file doesn't appear to be in COCO format")
            print(f"  Keys found: {list(coco_json.keys())}")
            return {
                "folder": folder_name,
                "status": "invalid_format",
                "images": 0,
                "annotations": 0
            }
        
        num_images = len(coco_json["images"])
        num_annotations = len(coco_json["annotations"])
        num_categories = len(coco_json["categories"])
        
        print(f"  ✓ Loaded {num_images} images")
        print(f"  ✓ Loaded {num_annotations} annotations")
        print(f"  ✓ Loaded {num_categories} categories")
        
        # Create visualizations directory inside the folder
        vis_output_dir = folder_path / "visualizations"
        
        print(f"\n[Creating visualizations]")
        visualize_all_images(coco_json, str(images_dir), str(vis_output_dir))
        
        print(f"  ✓ Visualizations saved to: {vis_output_dir}")
        
        return {
            "folder": folder_name,
            "status": "success",
            "images": num_images,
            "annotations": num_annotations,
            "categories": num_categories,
            "visualizations_path": str(vis_output_dir)
        }
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file: {e}")
        return {
            "folder": folder_name,
            "status": "json_error",
            "error": str(e),
            "images": 0,
            "annotations": 0
        }
    except Exception as e:
        print(f"❌ Error processing {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "folder": folder_name,
            "status": "error",
            "error": str(e),
            "images": 0,
            "annotations": 0
        }


def main():
    """Main function to visualize all sample batches."""
    print("=" * 70)
    print("VISUALIZING SAMPLE BATCHES")
    print("=" * 70)
    print(f"\nProcessing {len(SAMPLE_BATCH_FOLDERS)} sample batch folders:")
    for folder in SAMPLE_BATCH_FOLDERS:
        print(f"  - {folder}")
    
    results = []
    
    for folder_name in SAMPLE_BATCH_FOLDERS:
        result = visualize_sample_batch(folder_name)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"\n✓ Successfully processed: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  - {r['folder']}: {r['images']} images, {r['annotations']} annotations, {r['categories']} categories")
    
    if failed:
        print(f"\n⚠️  Failed/Skipped: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  - {r['folder']}: {r['status']}")
            if 'error' in r:
                print(f"    Error: {r['error']}")
    
    # Save summary to JSON
    summary_path = Path(SCRIPT_DIR) / "sample_batches_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_batches": len(SAMPLE_BATCH_FOLDERS),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }, f, indent=4)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nEach sample batch folder now contains:")
    print("  - visualizations/ (annotated images)")


if __name__ == "__main__":
    main()

