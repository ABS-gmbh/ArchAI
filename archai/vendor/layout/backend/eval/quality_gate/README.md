# Quality gate calibration set

`readings.json` holds five reference transcriptions and every recogniser's
reading of the same pages. `scripts/calibrate_quality_gate.py` runs the
post-OCR quality gate on each of them, and on synthetic corruptions of each
reference, and prints the CER, the detected language, the lexical contrast, the
label and the verdict. `tests/test_quality_gate_calibration.py` pins the
verdicts.

```bash
python scripts/calibrate_quality_gate.py
python scripts/calibrate_quality_gate.py --db app/archai.sqlite
```

With `--db`, every distinct transcription of 300 characters or more in a
pipeline database is checked as well, with corrupted copies of each. There is
no reference for those, so only the verdicts and the contrast are reported.

## Pages

| page_id | Page | Lines |
|---|---|---|
| `old-french` | Old French treatise on the commandments, two columns (the `french.JPEG` demo page) | 52 |
| `latin-abaton` | Latin sortes verses, one column (the `Latin.png` demo page) | 28 |
| `latin-apocalypse` | Revelation 21:9-11 as a lection (e-codices `sbe-0027`, f. 1v) | 18 |
| `middle-french-antitus` | Antitus's dedication of *La Satyre Megere* (e-codices `acv-P-Antitus`, f. 1r) | 7 |
| `old-french-verse` | Old French verse in octosyllabic couplets (e-codices `fmb-cb-0001`, f. 1r) | 70 |

The references follow the CATMuS Medieval conventions: graphematic, with
abbreviations unexpanded (`⁊`, `ꝑ`, `nr̃e`) and `u`/`v`, `i`/`j` as written.
`old-french` and `latin-abaton` were read off the images alone. The three
e-codices pages were transcribed with a CATMuS Medieval draft beside the image,
each line corrected against it, which pulls those references towards CATMuS
wherever a reading is uncertain: the CATMuS readings of those pages look
somewhat better than they are. No palaeographer has checked the references.

## Readings

- `kraken_catmus`, `kraken_cremma_medieval`, `kraken_mccatmus`: the segmented
  full-page route with each Kraken model, read with
  `scripts/benchmark_recognizers.py` from ABS-gmbh/ArchAI#13 (577cabf).
- `glmocr`: GLM-OCR through Ollama 0.35.0, which loops or aborts on these
  pages. Three readings repeat a block of lines, or the model's own
  instructions, until the token limit, at 3 to 15 times the page's length;
  `latin-apocalypse` came back as 13 characters, and `old-french-verse` failed
  with a 502 and has no reading. They are kept as real decoding loops, not as a
  measure of GLM-OCR.

## What the gate decides

| Sample | CER | Allowed |
|---|---|---|
| References | 0% | 5 of 5 |
| CATMuS readings | 4-18% | 5 of 5 |
| CREMMA readings | 10-47% | 4 of 5; the one at 47% is blocked |
| McCATMuS readings | 34-41% | 3 of 5 (see below) |
| GLM-OCR decoding loops | | 0 of 3 |
| Synthetic noise | 7-16% | 10 of 10 |
| Synthetic noise | 17-23% | 3 of 5; the two Latin pages are blocked |
| Noise at 35-55%, reversed words, random letters | | 0 of 20 |

Of 72 transcriptions held out from a pipeline database, the gate allows 60.
The 12 it blocks are nine garbled readings of the apocalypse page and three
readings full of uncertainty marks. It blocks all 216 corrupted copies (noise
at 50%, reversed words, random letters). The gate this replaced allowed 1 of
the 72. The database holds users' transcriptions and is not committed.

## Limits

- The gate tells language from what is not language, not right words from
  wrong ones. McCATMuS errs in French-looking letter strings ("oy urrudigus aus
  arost" for "Marcadigas anõ a voit"), and its readings of the three French
  pages pass at 34-40% CER. Telling those apart needs a word list.
- The contrast is measured from 150 trigrams, eight to fifteen manuscript
  lines; below that the trigram hit rate alone decides. Shorter text is judged
  less precisely: `LEXICAL_PLAUSIBILITY_FLOOR` in
  `app/services/ocr_quality_config.py` gives the share of page windows blocked
  at each length.
- The gate judges what was read, not how much of the page: GLM-OCR's
  13-character reading of `latin-apocalypse` is graded HIGH.
- Trigram profiles exist for Latin and French only. German, English and other
  languages are not judged lexically.
- Five pages in two languages, transcribed by non-specialists, are enough to
  show that the old gate was inverted and to place the thresholds. They do not
  give a corpus-level error rate.
