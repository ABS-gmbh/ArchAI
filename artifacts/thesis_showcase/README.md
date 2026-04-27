# Thesis Showcase Bundle

Qualitative showcase build for Figures 24, 25, 26, 29, and 30 of the ArchAI thesis.
Uses the three canonical multilingual pages as the showcase set.
Does not fabricate benchmark metrics, gold transcriptions, or gold entity links.

## Canonical pages

| Page ID | Language | Type |
|---|---|---|
| old-english | Old English | Grammatical / glossing text in insular script |
| latin-abaton | Latin | Abaton prognostic page |
| old-french | Old French | Moral / doctrinal page |

## What was generated

- **figure24/** -- Region-level OCR showcase with 3 manuscript crops, pipeline/SAIA outputs, and qualitative readability labels. No CER/WER computed.
- **figure25/** -- One selected plain-chat vs ArchAI-grounded answer comparison (Latin Abaton page), with evidence spans, chunk IDs, and panel screenshots.
- **figure26/** -- Bounded limitation analysis with 6 representative cases across 5 categories. No severity higher than medium.
- **figure29/** -- Per-language showcase indicators (binary coverage flags and qualitative success labels). No benchmark accuracy claimed.
- **figure30/** -- Semantic extraction / mention-handling panel with 6 examples showing correct suppression of authority linking on non-entity pages. Includes a pivot rationale document.

## What was unavailable

- Kraken and Calamari region OCR engines (not installed in this environment)
- Transkribus outputs (no API credentials configured)
- GLM comparison outputs (not integrated in current pipeline version)
- Authoritative ground-truth transcriptions for the three pages
- Gold entity-link annotations

## Figure selection rationale

### Figure 24

Three examples chosen to span all three languages and two difficulty tiers (main text band vs dense prognostic line). Readability: 2 medium, 1 low. The strongest region (Old English main text) demonstrates coherent multi-line insular script OCR. The hardest region (Latin Abaton) honestly shows abbreviation-handling limits.

### Figure 25

The Latin Abaton page was selected because it produces the clearest contrast between generic plain-chat interpretation and evidence-anchored ArchAI grounding. Every claim in the ArchAI answer traces to a specific chunk_id and offset range.

### Figure 26

Six cases covering OCR ambiguity, abbreviation handling, layout segmentation, weak entity extraction, and grounding-based inference suppression. All phrased in bounded, thesis-credible language.

### Figure 29

All three languages show successful pipeline coverage (OCR generated, grounded answer generated, evidence trace available). Old English receives medium rather than high due to language detection drift.

### Figure 30

Framed as semantic extraction / mention-handling behaviour rather than authority linking, because these pages do not contain strong linkable named entities. All 6 examples show correct suppression, which is the desired behaviour for grammatical, prognostic, and doctrinal content.

## Reproducibility

- All pipeline outputs were generated from the ArchAI backend at http://127.0.0.1:8000
- Source page SHA-256 hashes and run IDs are recorded in manifest.json
- The build script at `scripts/build_thesis_showcase_payloads.py` can verify the bundle with `--verify`

## Ready for figure assembly

The JSON, CSV, Markdown, crop PNGs, and panel screenshots in this bundle are ready to be converted into final thesis figures.
