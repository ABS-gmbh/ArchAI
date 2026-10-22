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
