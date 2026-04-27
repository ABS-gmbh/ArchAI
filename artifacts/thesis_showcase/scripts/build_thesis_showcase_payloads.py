#!/usr/bin/env python3
"""
build_thesis_showcase_payloads.py
=================================
Reproducible builder for the ArchAI thesis showcase bundle (Figures 24-30).

This script documents the exact steps used to assemble the thesis showcase
payloads from current ArchAI pipeline outputs.  It can be run in two modes:

  --verify   Validate that all expected files exist and schemas are correct.
  --rebuild  Re-fetch pipeline outputs from the ArchAI backend and regenerate
             all JSON/CSV payloads (requires a running backend).

Design principles:
  - Reuse existing pipeline outputs when already present on disk.
  - Never fabricate ground truth, CER/WER, gold entity links, or unsupported
    citations.
  - Prefer the strongest honest outputs from the current system.
  - Mark unavailable subsystems explicitly.
  - Keep all outputs reproducible.

Usage:
  python scripts/build_thesis_showcase_payloads.py --verify
  python scripts/build_thesis_showcase_payloads.py --rebuild --backend http://127.0.0.1:8000
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
FIGURES = ["figure24", "figure25", "figure26", "figure29", "figure30"]

CANONICAL_PAGES = [
    {
        "page_id": "old-english",
        "label": "Old English page",
        "language": "Old English",
        "description": "Grammatical / glossing text in insular script",
    },
    {
        "page_id": "latin-abaton",
        "label": "Latin page",
        "language": "Latin",
        "description": "Abaton prognostic page",
    },
    {
        "page_id": "old-french",
        "label": "Old French page",
        "language": "Old French",
        "description": "Moral / doctrinal page",
    },
]

EXPECTED_FILES = {
    "figure24": [
        "examples.json",
        "examples.csv",
        "README.md",
        "crops/ex01_old-english_r21.png",
        "crops/ex02_latin-abaton_r17.png",
        "crops/ex03_old-french_r19.png",
        "outputs/ex01_old-english_r21_pipeline.txt",
        "outputs/ex01_old-english_r21_saia.txt",
        "outputs/ex02_latin-abaton_r17_pipeline.txt",
        "outputs/ex02_latin-abaton_r17_saia.txt",
        "outputs/ex03_old-french_r19_pipeline.txt",
        "outputs/ex03_old-french_r19_saia.txt",
    ],
    "figure25": [
        "comparison.json",
        "comparison.md",
        "README.md",
        "screenshots/answer_panel.png",
        "screenshots/evidence_panel.png",
    ],
    "figure26": [
        "limitations.csv",
        "limitations_summary.json",
        "README.md",
        "screenshots/latin_abaton_panel.png",
        "screenshots/old_french_layout_panel.png",
    ],
    "figure29": [
        "by_language.json",
        "by_language.csv",
        "README.md",
    ],
    "figure30": [
        "examples.json",
        "examples.csv",
        "README.md",
        "pivot_to_semantic_extraction.md",
    ],
    "root": [
        "manifest.json",
        "README.md",
    ],
}

# ---------------------------------------------------------------------------
# Schema validators
# ---------------------------------------------------------------------------

FIGURE24_SCHEMA_KEYS = {
    "example_id", "page_id", "language", "condition_label", "crop_path",
    "pipeline_text", "glm_text", "saia_text", "transkribus_text",
    "best_system_label", "readability_label", "selection_rationale", "notes",
}

FIGURE25_SCHEMA_KEYS = {
    "page_id", "language", "question", "plain_chat_answer", "archai_answer",
    "plain_chat_generic_phrases", "archai_supported_phrases",
    "evidence_spans", "evidence_ids", "selection_rationale", "notes",
}

FIGURE26_SCHEMA_KEYS = {
    "case_id", "page_id", "language", "stage", "limitation_type",
    "severity_label", "short_note", "screenshot_path",
}

FIGURE29_SCHEMA_KEYS = {
    "language", "pages_processed", "ocr_output_generated",
    "grounded_answer_generated", "evidence_trace_available",
    "entity_output_behavior", "overall_showcase_success_label", "notes",
}

FIGURE30_SCHEMA_KEYS = {
    "example_id", "page_id", "language", "mention_text", "page_context",
    "candidates", "selected_link", "outcome_class", "explanation",
    "screenshot_path",
}


def validate_json_schema(filepath: Path, required_keys: set, label: str):
    """Validate that a JSON array file has the expected keys."""
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict) and "cases" in data:
        records = data["cases"]
    elif isinstance(data, list):
        records = data
    else:
        records = [data]

    errors = []
    for i, rec in enumerate(records):
        missing = required_keys - set(rec.keys())
        if missing:
            errors.append(f"  {label}[{i}]: missing keys {missing}")
    return errors


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def verify(bundle_root: Path):
    """Check that all expected files exist and schemas are valid."""
    print(f"Verifying bundle at: {bundle_root}\n")
    all_ok = True

    # Check file existence
    for figure, files in EXPECTED_FILES.items():
        base = bundle_root if figure == "root" else bundle_root / figure
        for fname in files:
            fpath = base / fname
            if not fpath.exists():
                print(f"  MISSING: {fpath.relative_to(bundle_root)}")
                all_ok = False

    # Check screenshot directory population for figure30
    fig30_ss = bundle_root / "figure30" / "screenshots"
    if fig30_ss.exists():
        pngs = list(fig30_ss.glob("*.png"))
        if len(pngs) < 1:
            print(f"  WARNING: figure30/screenshots/ has no PNG files")

    # Schema validation
    schema_checks = [
        ("figure24/examples.json", FIGURE24_SCHEMA_KEYS, "Figure24"),
        ("figure26/limitations_summary.json", FIGURE26_SCHEMA_KEYS, "Figure26"),
        ("figure29/by_language.json", FIGURE29_SCHEMA_KEYS, "Figure29"),
        ("figure30/examples.json", FIGURE30_SCHEMA_KEYS, "Figure30"),
    ]
    for rel, keys, label in schema_checks:
        fpath = bundle_root / rel
        if fpath.exists():
            errs = validate_json_schema(fpath, keys, label)
            if errs:
                all_ok = False
                for e in errs:
                    print(e)

    # Figure25 is a single object, validate separately
    fig25 = bundle_root / "figure25" / "comparison.json"
    if fig25.exists():
        with open(fig25) as f:
            data = json.load(f)
        missing = FIGURE25_SCHEMA_KEYS - set(data.keys())
        if missing:
            print(f"  Figure25: missing keys {missing}")
            all_ok = False

    if all_ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed. See above.")
    return all_ok


# ---------------------------------------------------------------------------
# Rebuild mode (stub -- requires running backend)
# ---------------------------------------------------------------------------

def rebuild(bundle_root: Path, backend_url: str):
    """Re-fetch pipeline outputs and regenerate payloads."""
    print(f"Rebuild mode is a stub in this version.")
    print(f"Backend URL: {backend_url}")
    print(f"Bundle root: {bundle_root}")
    print()
    print("To rebuild from scratch, ensure the ArchAI backend is running at")
    print(f"  {backend_url}")
    print("and the three canonical pages are loaded as fixtures.")
    print()
    print("Steps that would be executed:")
    print("  1. POST each page to /api/ocr/page_with_trace")
    print("  2. POST each page to /api/predict/single for layout segmentation")
    print("  3. Extract region crops and POST to /api/ocr/extract?backend=saia")
    print("  4. Select strongest regions by coherence heuristic")
    print("  5. Generate plain-chat and grounded-chat answers")
    print("  6. Collect semantic extraction mentions")
    print("  7. Assemble JSON/CSV payloads")
    print("  8. Update manifest.json")
    print()
    print("For now, run with --verify to validate existing outputs.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build or verify the ArchAI thesis showcase bundle."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Validate that all expected files exist and schemas are correct.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-fetch pipeline outputs and regenerate all payloads.",
    )
    parser.add_argument(
        "--backend", default="http://127.0.0.1:8000",
        help="ArchAI backend URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--bundle-root", default=None,
        help="Override bundle root directory.",
    )
    args = parser.parse_args()

    root = Path(args.bundle_root) if args.bundle_root else BUNDLE_ROOT

    if args.verify:
        ok = verify(root)
        sys.exit(0 if ok else 1)
    elif args.rebuild:
        ok = rebuild(root, args.backend)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
