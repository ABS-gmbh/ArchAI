# Figure 25 -- Plain Chat vs ArchAI Grounded Answer

This folder contains one showcase comparison demonstrating that the ArchAI grounding pipeline produces more evidence-anchored answers than a plain OCR-plus-LLM baseline.

## Selected case

- **Page:** latin-abaton (Latin Abaton prognostic page)
- **Question:** "What kind of text is this page, and which recurring concepts or warnings are visible on it?"
- **Why selected:** The Latin Abaton page produced the clearest contrast. The plain-chat answer is topically correct but includes generic scholarly framing and interpretive claims not traceable to specific text spans. The ArchAI answer anchors every claim to a specific chunk_id and character offset range, making the evidence trail auditable.

## Key differences observed

1. **Plain chat** uses generic scholarly language ("typical of prognostic literature") and provides translations that may be speculative.
2. **ArchAI grounded answer** cites a specific chunk (bc0c5595..., offsets 0-1108) for every quoted passage and avoids interpretive claims beyond what the retrieved evidence supports.
3. **Evidence trace:** The ArchAI answer includes asset_ref, run_id, chunk_id, and offset metadata inline, enabling a reader to verify each claim against the pipeline's stored evidence.

## File inventory

- `comparison.json` -- structured comparison data (full schema)
- `comparison.md` -- human-readable narrative comparison
- `screenshots/answer_panel.png` -- screenshot of answer presentation
- `screenshots/evidence_panel.png` -- screenshot of evidence panel

## Schema

The comparison record contains: page_id, language, question, plain_chat_answer, archai_answer, plain_chat_generic_phrases, archai_supported_phrases, evidence_spans, evidence_ids, answer_screenshot_path, evidence_screenshot_path, selection_rationale, notes.
