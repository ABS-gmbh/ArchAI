# Figure 26 -- Error / Limitation Analysis (Showcase Version)

This folder contains a bounded limitation analysis for the ArchAI pipeline, observed on the three canonical multilingual showcase pages.

## Design principle

This analysis increases thesis credibility by acknowledging real limitations without making the system look worse than necessary. It is a compact, representative set of 6 cases (the spec allows 3-8), not an exhaustive catalogue of failures.

## Categories covered

1. **OCR ambiguity** -- language misidentification on Old English and Latin pages
2. **Abbreviation handling difficulty** -- unexpanded insular abbreviations in region-level OCR
3. **Layout/segmentation challenge** -- noisy OCR on narrow heading regions
4. **Weak entity extraction on non-entity pages** -- common nouns surfaced as mentions but correctly not linked
5. **Unsupported inference risk avoided or reduced by grounding** -- conservative suppression of authority links on doctrinal content

## Severity labels

All cases are rated medium or low. No high-severity failures were observed on these showcase pages. This is consistent with the selection principle of choosing pages where the pipeline behaves convincingly.

## File inventory

- `limitations_summary.json` -- structured data for all 6 cases (full schema)
- `limitations.csv` -- same data in CSV format
- `screenshots/*.png` -- supporting screenshots for selected cases

## Schema

Each case record contains: case_id, page_id, language, stage, limitation_type, severity_label, short_note, screenshot_path.
