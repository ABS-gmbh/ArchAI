# Recogniser benchmark references

Reference transcriptions for `scripts/benchmark_recognizers.py`, which compares
the engines behind `/ocr/extract_full_page` on the same pages.

| page_id | Page | Lines |
|---|---|---|
| `old-french` | Old French treatise on the commandments, two columns (the `french.JPEG` demo page) | 52 |
| `latin-abaton` | Latin sortes verses, one column (the `Latin.png` demo page) | 28 |

The images are the thesis demo pages and are not committed. `manifest.jsonl`
names each by SHA-256, so keep them in any directory, under any file name, and
pass that directory as `--images`.

## How the references were made

Each line was read off the page image for this benchmark, following the CATMuS
Medieval conventions the Kraken models are trained on: graphematic, with
abbreviations left unexpanded (`⁊`, `ꝑ`, `q̃`, `nr̃e`), `u`/`v` and `i`/`j` as
written, and the scribe's punctuation. They cover the text columns only.
Marginal numbers, shelfmarks and stamps are left out, because the route returns
them in `secondary_lines`, apart from the text.

The references have not been checked by a palaeographer, so some readings may be
wrong. Two pages are enough to rank engines whose error rates differ by a factor
of two or more, which is how far apart they are. They do not give a
corpus-level accuracy figure.
