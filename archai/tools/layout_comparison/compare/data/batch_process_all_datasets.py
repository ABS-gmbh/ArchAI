"""
Batch process all datasets: Convert XML to COCO and create visualizations.

Processes all folders:
- Aleyna 1 (2024)
- Annika 2 (2024)
- Luise 1 (2024)
- Luise 2 (2024)
- Nuray 1 (2024)
- Nuray 2 (2024)

For each folder:
1. Converts XML annotations to COCO format
2. Creates visualizations of annotations on images
3. Saves outputs inside each folder
"""
import os
import sys
import json
from pathlib import Path

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from original_annotations import load_ground_truth
from visualize_ground_truth import visualize_all_images


# List of all dataset folders to process
DATASET_FOLDERS = [
    "Aleyna 1 (2024)",
    "Annika 2 (2024)",
    "Luise 1 (2024)",
    "Luise 2 (2024)",
    "Nuray 1 (2024)",
    "Nuray 2 (2024)",
]


def process_dataset(folder_name, base_dir=None):
    """
    Process a single dataset folder.
    
    Args:
        folder_name: Name of the dataset folder
        base_dir: Base directory containing the dataset folders (default: SCRIPT_DIR)
    
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
    xml_path = folder_path / "Annotations" / "annotations.xml"
    images_dir = folder_path / "Images"
    
    # Check if required files/directories exist
    if not xml_path.exists():
        print(f"⚠️  Warning: XML file not found: {xml_path}")
        return {
            "folder": folder_name,
            "status": "no_xml",
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
    
    # Step 1: Convert XML to COCO
    print(f"\n[Step 1/2] Converting XML to COCO format...")
    print(f"  XML: {xml_path}")
    print(f"  Images: {images_dir}")
    
    try:
        coco_json = load_ground_truth(str(xml_path), str(images_dir))
        
        if not coco_json:
            print(f"❌ Error: Failed to parse XML")
            return {
                "folder": folder_name,
                "status": "parse_error",
                "images": 0,
                "annotations": 0
            }
        
        num_images = len(coco_json["images"])
        num_annotations = len(coco_json["annotations"])
        
        print(f"  ✓ Loaded {num_images} images")
        print(f"  ✓ Loaded {num_annotations} annotations")
        print(f"  ✓ Categories: {len(coco_json['categories'])}")
        
        # Save COCO JSON inside the dataset folder
        coco_output_path = folder_path / "ground_truth_coco.json"
        with open(coco_output_path, 'w') as f:
            json.dump(coco_json, f, indent=4)
        print(f"  ✓ Saved COCO JSON to: {coco_output_path}")
        
    except Exception as e:
        print(f"❌ Error converting XML to COCO: {e}")
        import traceback
        traceback.print_exc()
        return {
            "folder": folder_name,
            "status": "conversion_error",
            "error": str(e),
            "images": 0,
            "annotations": 0
        }
    
    # Step 2: Create visualizations
    print(f"\n[Step 2/2] Creating visualizations...")
    
    try:
        # Create visualizations directory inside the dataset folder
        vis_output_dir = folder_path / "visualizations"
        
        visualize_all_images(coco_json, str(images_dir), str(vis_output_dir))
        
        print(f"  ✓ Visualizations saved to: {vis_output_dir}")
        
    except Exception as e:
        print(f"⚠️  Warning: Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail the whole process if visualization fails
    
    return {
        "folder": folder_name,
        "status": "success",
        "images": num_images,
        "annotations": num_annotations,
        "categories": len(coco_json["categories"]),
        "coco_json_path": str(coco_output_path),
        "visualizations_path": str(vis_output_dir)
    }


def main():
    """Main function to process all datasets."""
    print("=" * 70)
    print("BATCH PROCESSING: XML to COCO Conversion & Visualization")
    print("=" * 70)
    print(f"\nProcessing {len(DATASET_FOLDERS)} datasets:")
    for folder in DATASET_FOLDERS:
        print(f"  - {folder}")
    
    results = []
    
    for folder_name in DATASET_FOLDERS:
        result = process_dataset(folder_name)
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    
    print(f"\n✓ Successfully processed: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  - {r['folder']}: {r['images']} images, {r['annotations']} annotations")
    
    if failed:
        print(f"\n⚠️  Failed/Skipped: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  - {r['folder']}: {r['status']}")
    
    # Save summary to JSON
    summary_path = Path(SCRIPT_DIR) / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "total_datasets": len(DATASET_FOLDERS),
            "successful": len(successful),
            "failed": len(failed),
            "results": results
        }, f, indent=4)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE!")
    print("=" * 70)
    print("\nEach dataset folder now contains:")
    print("  - ground_truth_coco.json (COCO format annotations)")
    print("  - visualizations/ (annotated images)")


if __name__ == "__main__":
    main()

