# Figure 29 -- Performance by Language (Showcase Version)

This table reports multilingual pipeline coverage indicators for the three canonical showcase pages. It does not claim benchmark accuracy, CER, WER, or gold entity-link precision.

## Design principle

Since no authoritative ground truth is available, this figure uses honest binary and categorical indicators rather than numeric accuracy scores. The goal is to support a thesis figure showing that the pipeline behaved convincingly across all three languages.

## Language summary

| Language | Pages | OCR | Grounded answer | Evidence trace | Entity behaviour | Success |
|---|---|---|---|---|---|---|
| Old English | 1 | Yes | Yes | Yes | Grammatical/lexical mentions; no authority links | medium |
| Latin | 1 | Yes | Yes | Yes | Semantic concept mentions; no authority links | high |
| Old French | 1 | Yes | Yes | Yes | Semantic concept mentions; no authority links | high |

## Notes on the "medium" label for Old English

The Old English page receives a medium label (rather than high) because language detection drifts and abbreviation handling is incomplete. The pipeline still produces usable OCR and grounded answers, but the output requires more human interpretation than the Latin or Old French pages.

## Indicators computed

All indicators are derived from current pipeline artifacts without fabrication:
- **pages_processed**: count of canonical pages run through the pipeline
- **ocr_output_generated**: whether the pipeline produced a non-empty OCR transcript
- **grounded_answer_generated**: whether the grounding pipeline produced a cited answer
- **evidence_trace_available**: whether chunk_id / offset evidence metadata is present
- **entity_output_behavior**: qualitative description of mention/entity extraction results
- **overall_showcase_success_label**: qualitative assessment (high / medium / low)

## File inventory

- `by_language.json` -- structured per-language data
- `by_language.csv` -- same data in CSV format
