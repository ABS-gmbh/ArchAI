# Figure 24 -- OCR Example Comparison (Showcase Version)

This folder contains a qualitative side-by-side OCR showcase using the strongest real manuscript regions from the three canonical multilingual thesis pages.

## Important framing

No authoritative ground-truth transcriptions are available for these pages. Therefore no CER, WER, or character-level accuracy is reported. All assessment uses qualitative readability labels only: high, medium, or low.

## Selected examples

| Example ID | Page | Language | Condition | Readability | Why selected |
|---|---|---|---|---|---|
| ex01_old-english_r21 | Old English | Old English | Main text band | medium | Cleanest insular-script region; coherent multi-line output with macrons and Tironian nota |
| ex02_latin-abaton_r17 | Latin Abaton | Latin | Dense prognostic line | low | Harder example showing abbreviation difficulty; key Latin words partially recoverable |
| ex03_old-french_r19 | Old French | Old French | Doctrinal body text | medium | Recognisable Old French vocabulary; functional word segmentation in Gothic textura |

## Systems compared

Only the SAIA/Qwen backend was available for region-level OCR. Kraken and Calamari were not installed in this environment. Transkribus and GLM outputs are not available for these pages.

The pipeline and SAIA produce identical output on all three regions because the current pipeline routes region OCR through the SAIA backend. This confirms reproducibility but does not provide a multi-system comparison.

## File inventory

- `examples.json` -- structured data for all three examples (full schema)
- `examples.csv` -- same data in CSV format
- `crops/*.png` -- manuscript region crops (one per example)
- `outputs/*_pipeline.txt` -- pipeline OCR output (plain text)
- `outputs/*_saia.txt` -- SAIA backend OCR output (plain text)

## Schema

Each record contains: example_id, page_id, language, condition_label, crop_path, pipeline_text, glm_text, saia_text, transkribus_text, best_system_label, readability_label, selection_rationale, notes.
