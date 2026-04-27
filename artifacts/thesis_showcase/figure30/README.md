# Figure 30 -- Authority-Linking / Semantic Extraction Panel

## Framing decision

These three canonical showcase pages (grammatical Old English, prognostic Latin, doctrinal Old French) do not honestly contain strong linkable named entities. Forcing authority-linking on this material would reward false positives rather than useful scholarship.

The correct thesis framing is therefore **semantic extraction / mention-handling behaviour**: the pipeline detects candidate concepts and correctly suppresses authority linking when the mentions are common nouns, social roles, or abstract moral vocabulary rather than genuine named entities.

See `pivot_to_semantic_extraction.md` for the full rationale.

## Selected examples

All 6 examples from the canonical pages show `suppressed_correctly` as the outcome class. This uniformity is the honest result: on these page types, correct suppression IS the desired behaviour.

| Page | Language | Mention | Category | Outcome |
|---|---|---|---|---|
| old-english | Old English | oxan | lexical example | suppressed_correctly |
| old-english | Old English | mulissi | lexical example | suppressed_correctly |
| latin-abaton | Latin | uxore | social role | suppressed_correctly |
| latin-abaton | Latin | Inmicos | social relation | suppressed_correctly |
| old-french | Old French | nr\u0303e seigneur | non-linkable title | suppressed_correctly |
| old-french | Old French | nire faus | non-linkable moral action | suppressed_correctly |

## Outcome class distribution

- `suppressed_correctly`: 6/6
- `correct_link`: 0 (no genuine named entities on these pages)
- `uncertain_link`: 0
- `false_positive`: 0
- `unresolved`: 0

This distribution is expected and desirable for non-entity pages. On pages with genuine named entities (e.g., historical chronicles or hagiographic narratives), the distribution would differ.

## File inventory

- `examples.json` -- structured data for all 6 examples
- `examples.csv` -- same data in CSV format
- `pivot_to_semantic_extraction.md` -- thesis framing rationale
- `screenshots/*.png` -- per-mention screenshots (2 per language)
