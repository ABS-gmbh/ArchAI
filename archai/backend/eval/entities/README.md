# Entity extraction benchmark

`gold.json` marks the named entities - people, places, works, dates - in the
five reference transcriptions of `eval/quality_gate/readings.json`.
`scripts/benchmark_entities.py` runs each page through the full-page
pipeline's mention extractor and scores every mention of those types: a
mention is found when it overlaps a gold span of the same type, and anything
else is a false positive. Titles and roles are not scored.

```bash
python scripts/benchmark_entities.py
```

| Page | Entities |
|---|---|
| `old-french` | none |
| `latin-abaton` | none |
| `latin-apocalypse` | *iohannis*; *hyerusalem* |
| `middle-french-antitus` | *La satyre megere*; *Ame de monfalcon*; *Antitus*; *lausanne*; *lan de grace Mille cinq cens* |
| `old-french-verse` | *Caldꝰ*; *Marcadigas*, four times; *Ynabele*; *despaigne* (d'Espaigne) |

That is 14 entities. On the two pages with none, every mention is wrong.

## Caveats

- **The annotation is not expert.** It was made for this benchmark and has
  not been checked by a specialist. *nr̃e seigneur* (God) is not annotated.
- **One page was tuned on.** The extractor's anchors were written for
  `middle-french-antitus` (Lausanne, *pere*, *prince*, *euesque*), so the
  script also reports totals without it.
- **The pages were also the development set.** The extraction rules were
  developed while looking at these pages. Precision was therefore also
  checked on 72 transcriptions held out from a pipeline database, by reading
  every mention extracted from them (see the PR that added this benchmark).
- **Five pages, 14 entities.** That is enough to see the extractor fail on
  diplomatic text and to check the fix. It is not a corpus-level score.
