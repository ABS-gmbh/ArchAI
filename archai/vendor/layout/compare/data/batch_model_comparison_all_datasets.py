"""
Batch model comparison for all expert-annotated datasets.

Datasets:
- Aleyna 1 (2024)
- Annika 2 (2024)
- Luise 1 (2024)
- Luise 2 (2024)
- Nuray 1 (2024)
- Nuray 2 (2024)

For each folder (e.g. "Aleyna 1 (2024)"):
- Uses the existing `ground_truth_coco.json`
- Runs OLD models and NEW models on all images in `Images/`
- Calculates detection/segmentation metrics vs ground truth
- Creates side‑by‑side visualizations:
    Ground Truth | Old Models | New Models
- Saves everything under `<folder>/model_comparison/`
"""
import os
import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from old_models import process_dataset as process_old_models
from new_models import process_dataset as process_new_models
from compare import calculate_metrics, align_categories, draw_coco_annotations_simple

# Re‑use the same dataset list as in batch_process_all_datasets.py
DATASET_FOLDERS = [
    "Aleyna 1 (2024)",
    "Annika 2 (2024)",
    "Luise 1 (2024)",
    "Luise 2 (2024)",
    "Nuray 1 (2024)",
    "Nuray 2 (2024)",
]


def create_side_by_side_visualization(image_path, gt_coco, old_coco, new_coco, output_path):
    """
    Create side‑by‑side visualization: GT | Old Models | New Models
    """
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Left: Ground Truth
    draw_coco_annotations_simple(image_path, gt_coco, "Ground Truth", axes[0])

    # Middle: Old Models
    draw_coco_annotations_simple(image_path, old_coco, "Old Models", axes[1])

    # Right: New Models
    draw_coco_annotations_simple(image_path, new_coco, "New Models", axes[2])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved comparison to: {output_path}")


def process_expert_dataset(folder_name, base_dir=None):
    """
    Process a single expert dataset:
    - Load ground_truth_coco.json
    - Run old & new models
    - Compute metrics
    - Create GT | Old | New visualizations
    """
    if base_dir is None:
        base_dir = SCRIPT_DIR

    folder_path = Path(base_dir) / folder_name

    if not folder_path.exists():
        print(f"⚠️  Warning: Folder not found: {folder_path}")
        return None

    print("\n" + "=" * 70)
    print(f"Processing expert dataset: {folder_name}")
    print("=" * 70)

    # Paths
    gt_json_path = folder_path / "ground_truth_coco.json"
    images_dir = folder_path / "Images"
    output_dir = folder_path / "model_comparison"
    os.makedirs(output_dir, exist_ok=True)

    if not gt_json_path.exists():
        print(f"❌ Error: ground_truth_coco.json not found at {gt_json_path}")
        return None

    if not images_dir.exists():
        print(f"❌ Error: Images directory not found at {images_dir}")
        return None

    # Load ground truth
    print(f"\n[1/5] Loading ground truth...")
    with open(gt_json_path, "r") as f:
        gt_coco = json.load(f)

    print(f"  ✓ Loaded {len(gt_coco['images'])} images")
    print(f"  ✓ Loaded {len(gt_coco['annotations'])} annotations")

    # Run old models
    print(f"\n[2/5] Running old models...")
    old_output_dir = output_dir / "old_models"
    os.makedirs(old_output_dir, exist_ok=True)

    try:
        old_coco = process_old_models(str(images_dir), str(old_output_dir))
        print(f"  ✓ Generated {len(old_coco['annotations'])} annotations")
        print(f"  ✓ Categories: {[c['name'] for c in old_coco['categories']]}")
    except Exception as e:
        print(f"  ❌ Error running old models: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Run new models
    print(f"\n[3/5] Running new models...")
    new_output_dir = output_dir / "new_models"
    os.makedirs(new_output_dir, exist_ok=True)

    try:
        new_coco = process_new_models(str(images_dir), str(new_output_dir))
        print(f"  ✓ Generated {len(new_coco['annotations'])} annotations")
    except Exception as e:
        print(f"  ❌ Error running new models: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Calculate metrics
    print(f"\n[4/5] Calculating metrics...")

    # Align categories with ground truth (by name matching)
    old_coco_aligned = align_categories(gt_coco.copy(), old_coco.copy())
    new_coco_aligned = align_categories(gt_coco.copy(), new_coco.copy())

    # Metrics for old models
    print(f"\n  Old Models Metrics:")
    old_metrics = calculate_metrics(gt_coco, old_coco_aligned, str(output_dir))
    print(f"    mAP@50: {old_metrics.get('mAP@50', 0):.4f}")
    print(f"    mAP@[.50:.95]: {old_metrics.get('mAP@[.50:.95]', 0):.4f}")
    print(f"    Precision: {old_metrics.get('Precision', 0):.4f}")
    print(f"    Recall: {old_metrics.get('Recall', 0):.4f}")

    # Metrics for new models
    print(f"\n  New Models Metrics:")
    new_metrics = calculate_metrics(gt_coco, new_coco_aligned, str(output_dir))
    print(f"    mAP@50: {new_metrics.get('mAP@50', 0):.4f}")
    print(f"    mAP@[.50:.95]: {new_metrics.get('mAP@[.50:.95]', 0):.4f}")
    print(f"    Precision: {new_metrics.get('Precision', 0):.4f}")
    print(f"    Recall: {new_metrics.get('Recall', 0):.4f}")

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"old_models": old_metrics, "new_models": new_metrics}, f, indent=4)
    print(f"  ✓ Saved metrics to: {metrics_path}")

    # Create visualizations
    print(f"\n[5/5] Creating side-by-side visualizations...")
    vis_dir = output_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    for img_info in gt_coco["images"]:
        image_name = img_info["file_name"]
        image_path = images_dir / image_name

        if not image_path.exists():
            continue

        img_id = img_info["id"]

        # Filter annotations for this image
        gt_img_coco = {
            "images": [img_info],
            "annotations": [a for a in gt_coco["annotations"] if a["image_id"] == img_id],
            "categories": gt_coco["categories"],
        }

        old_img_coco = {
            "images": [img_info],
            "annotations": [a for a in old_coco["annotations"] if a["image_id"] == img_id],
            "categories": old_coco["categories"],
        }

        new_img_coco = {
            "images": [img_info],
            "annotations": [a for a in new_coco["annotations"] if a["image_id"] == img_id],
            "categories": new_coco["categories"],
        }

        out_path = vis_dir / f"{Path(image_name).stem}_comparison.png"
        create_side_by_side_visualization(
            str(image_path),
            gt_img_coco,
            old_img_coco,
            new_img_coco,
            str(out_path),
        )

    print(f"  ✓ Visualizations saved to: {vis_dir}")

    return {
        "folder": folder_name,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "old_annotations": len(old_coco["annotations"]),
        "new_annotations": len(new_coco["annotations"]),
        "gt_annotations": len(gt_coco["annotations"]),
    }


def main():
    """Run model comparison for all expert datasets."""
    print("=" * 70)
    print("MODEL COMPARISON ON EXPERT DATASETS")
    print("=" * 70)
    print(f"\nProcessing {len(DATASET_FOLDERS)} folders:")
    for folder in DATASET_FOLDERS:
        print(f"  - {folder}")

    results = []

    for folder_name in DATASET_FOLDERS:
        result = process_expert_dataset(folder_name)
        if result:
            results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\n{r['folder']}:")
        print(f"  Ground Truth: {r['gt_annotations']} annotations")
        print(f"  Old Models: {r['old_annotations']} annotations")
        print(f"    mAP@50: {r['old_metrics'].get('mAP@50', 0):.4f}")
        print(f"    Precision: {r['old_metrics'].get('Precision', 0):.4f}")
        print(f"    Recall: {r['old_metrics'].get('Recall', 0):.4f}")
        print(f"  New Models: {r['new_annotations']} annotations")
        print(f"    mAP@50: {r['new_metrics'].get('mAP@50', 0):.4f}")
        print(f"    Precision: {r['new_metrics'].get('Precision', 0):.4f}")
        print(f"    Recall: {r['new_metrics'].get('Recall', 0):.4f}")

    # Save summary
    summary_path = Path(SCRIPT_DIR) / "expert_datasets_model_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✓ Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()


