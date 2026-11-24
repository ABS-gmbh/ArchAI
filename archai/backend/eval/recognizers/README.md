# Recogniser benchmark references

Reference transcriptions for `scripts/benchmark_recognizers.py`, which compares
the engines behind `/ocr/extract_full_page` on the same pages.

| page_id | Page | Script | Lines |
|---|---|---|---|
| `old-french` | Old French treatise on the commandments, two columns (the `french.JPEG` demo page) | Gothic textualis | 52 |
| `latin-abaton` | Latin sortes verses, one column (the `Latin.png` demo page) | Gothic textualis | 28 |
| `latin-apocalypse` | Revelation 21:9-11 as a lection, one column (e-codices `sbe-0027`, f. 1v) | Romanesque minuscule | 18 |
| `middle-french-antitus` | Antitus's dedication of *La Satyre Megere* to Aymon de Montfalcon, bishop of Lausanne (e-codices `acv-P-Antitus`, f. 1r) | Bastarda | 7 |
| `old-french-verse` | Old French verse in octosyllabic couplets, two columns (e-codices `fmb-cb-0001`, f. 1r) | Gothic textualis | 70 |

The images are not committed. `manifest.jsonl` names each one by SHA-256, so
keep them in any directory, under any file name, and pass that directory as
`--images`. The e-codices pages are the full-size downloads (about 6130 x 8175
px); the first two are the thesis demo pages.

## How the references were made

Each line follows the CATMuS Medieval conventions the Kraken models are trained
on: graphematic, with abbreviations left unexpanded (`⁊`, `ꝑ`, `q̃`, `nr̃e`),
`u`/`v` and `i`/`j` as written, and the scribe's punctuation. An initial set
apart from its line is joined to its word. The references cover the text
columns only. Marginal numbers, shelfmarks and stamps are left out, because the
route returns them in `secondary_lines`, apart from the text.

`old-french` and `latin-abaton` were read off the images alone. The three
e-codices pages were transcribed with a CATMuS Medieval draft beside the image,
each line corrected against the image. A draft pulls a reference towards the
model that produced it wherever a reading is uncertain, so when comparing
engines, rather than versions of one engine, weight the first two pages most.

None of the references has been checked by a palaeographer, so some readings
will be wrong. Five pages are enough to rank engines whose error rates differ
by a factor of two or more. They do not give a corpus-level accuracy figure.

## Results

CER (diplomatic policy) through the full-page route, on the evaluation VM
(2026-11-24, `scripts/eval_vm/`), with versal initials kept apart as Kraken
reads them (`OCR_JOIN_DETACHED_INITIALS=false`) and joined (the default):

| Engine | old-french | latin-abaton | latin-apocalypse | middle-french-antitus | old-french-verse | micro |
|---|---|---|---|---|---|---|
| CATMuS Medieval (`segmented`) | 6.5 / 6.5 | 6.6 / **5.5** | 18.0 / 18.0 | 4.1 / 4.1 | 7.2 / **5.4** | 7.6 / **6.7** |
| CREMMA Medieval | 9.9 / 9.9 | 14.6 / 13.0 | 46.9 / 46.9 | 19.6 / 19.6 | 11.8 / 10.1 | 15.1 / 14.1 |
| McCATMuS | 34.0 / 34.0 | 38.8 / 38.3 | 41.0 / 41.0 | 40.0 / 40.0 | 35.4 / 34.9 | 36.5 / 36.2 |

Only the two pages that set their initials in a column of their own change
(`app/services/detached_initials.py`). latin-apocalypse keeps its two misreads
"R Rian" and "R habe bat", since only 2 of its 21 lines start with a
detached glyph. The join's three thresholds were set on latin-abaton and
old-french-verse, the two development pages with versals, with one
annotator. On those two pages it also joins six one-letter words set as
versals (*Q sperat*, *A superis*, *H uestis*, *A point*, and two versal ⁊
misread as L and I).

`eval/quality_gate/readings.json` was read before the join, so its CATMuS
readings of latin-abaton and old-french-verse keep the detached initials. The
gate gives the same verdicts on the joined readings, so the file was not
regenerated.

