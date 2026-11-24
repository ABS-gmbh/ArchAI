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
  `scripts/benchmark_recognizers.py` from ABS-gmbh/ArchAI#16 (577cabf).
- `glmocr`: GLM-OCR through Ollama 0.35.0, which loops or aborts on these
  pages. Three readings repeat a block of lines, or the model's own
  instructions, until the token limit, at 3 to 15 times the page's length;
  `latin-apocalypse` came back as 13 characters, and `old-french-verse` failed
  with a 502 and has no reading. They are kept as real decoding loops, not as a
  measure of GLM-OCR. Every GLM-OCR figure in this file is from these Ollama
  0.35.0 readings. With Ollama 0.40.6 (dev machine, 2026-11-24) all four
  readings repeat the page to 2,961-4,135 characters, latin-apocalypse
  included, and the gate blocks all four as UNRELIABLE
  (`eval/recognizers/README.md`, "GLM-OCR").

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
  13-character reading of `latin-apocalypse` (Ollama 0.35.0) is graded HIGH.
- Trigram profiles (`app/services/lexicon_trust.py`). The lexical contrast
  is always the best of Latin, Old French, Middle French and French, plus the
  detected language's own profile; the hit-rate floor takes the best of the
  same. What each detected language is scored against:

  | detected as | profile |
  |---|---|
  | latin, old_french, middle_french, french | its own, hand-picked (73-102 trigrams) |
  | spanish | its own, generated from the CATMuS train split (since 7879792c) |
  | dutch | its own, generated from the CATMuS train split (since 7879792c) |
  | anglo_norman | Old French |
  | italian, portuguese | Latin |
  | catalan, occitan | Middle French |
  | german, english, any other language | none: no lexical check at all |

  A page detected as a language with no profile skips both lexical gates and
  is judged on structure alone.
- **The figures below do not hold on manuscripts outside this set.** On 93
  pseudo-pages from 48 held-out manuscripts (`eval/heldout/README.md`,
  2026-11-24), the gate before the Spanish and Dutch profiles blocked 6 of
  76 CATMuS Medieval readings at CER <= 10%, five of them medieval Spanish
  (half of the well-read Castilian and Navarrese pages), and allowed 77 of
  128 readings at CER >= 25% from the two recognisers pages are not routed
  to. With the profiles (shipped, next item) it blocks 1 of 76 and allows 81
  of 128. This set has no Spanish page and only 59 rows, so it could show
  neither.
- **Medieval Spanish and Dutch profiles** (7879792c,
  `scripts/build_trigram_profiles.py`). The profiles were built from the
  CATMuS Medieval train split, text only, with the 100 most frequent trigrams
  through the gate's own tokenisation and no threshold touched. They were
  measured once on the held-out pseudo-pages:

  | | CER <= 10% blocked | CER >= 25% allowed |
  |---|---|---|
  | before | 6/80 (CATMuS 6/76) | 77/128 (CREMMA 31/47, McCATMuS 46/81) |
  | after | 1/80 (CATMuS 1/76) | 81/128 (CREMMA 31/47, McCATMuS 50/81) |

  - **What changed.** Spanish rescues all five good CATMuS readings it
    blocked (Castilian 4/8 -> 0/8, Navarrese 1/2 -> 0/2).
  - **The cost.** Four bad McCATMuS readings now pass (Castilian 1/6 -> 4/6
    allowed, Navarrese 0/2 -> 1/2); McCATMuS reads a line only when CATMuS
    Medieval reads nothing.
  - **Independent check.** 93 Castilian pages of the dev split: clean
    allowed 88 -> 93, reversed-word and random-letter 0/93 -> 0/93.
  - **Dutch.** No held-out reading's verdict changes; of the two held-out
    Middle Dutch references reversed word by word, one that was allowed is
    now blocked. It replaces no lexical check with one: reversed-word Dutch
    allowed 21/40 -> 3/40 (still 3, not 0), clean 40/40, random letters
    0/40. Those 40 pages come from the first 4 train files of Middle Dutch,
    3 manuscripts, which the profile was built from.
  - **Still let through.** 1 of the 43 held-out Latin references reversed
    word by word is allowed by the shipped gate.
  - **Not shipped.** Italian, Catalan and Occitan profiles built the same way
    rescued no good reading and let 5, 1 and 0 more bad ones through.
  - **What it rests on.** 4 Castilian manuscripts and 1 Navarrese in the
    held-out sample, 93 dev-split pages, and one sampling seed.
  - **The held-out sample is now spent for gate decisions.** It has been used
    once to decide whether to ship a gate change, so the next gate change must
    be judged on a fresh sample (another seed, other blocks), not on this one.
- Measured on the calibration set with langdetect seeded (59 rows,
  2026-11-23). The committed readings keep versal initials detached, as
  Kraken wrote them before the join became the default (75181719); on fresh
  readings with the join (55 rows, no GLM-OCR; `scripts/eval_vm/README.md`,
  3c and 3d) the verdicts are the same and 3 readings at CER >= 25% are
  allowed. On the committed rows no reading at CER <= 10% is blocked, and 4
  readings at CER >= 25% are allowed:

  | Label | n | allowed | CER range |
  |---|---|---|---|
  | HIGH | 24 | 24 | 0.0% - 97.9% |
  | OK | 7 | 7 | 6.7% - 34.3% |
  | RISKY | 10 | 0 | 17.3% - 82.0% |
  | UNRELIABLE | 18 | 0 | 35.4% - 1376% |

  The four allowed at CER >= 25%: old-french kraken_mccatmus (34.3%, OK),
  middle-french-antitus kraken_mccatmus (40.0%, HIGH), old-french-verse
  kraken_mccatmus (35.4%, HIGH), and latin-apocalypse glmocr (97.9%, HIGH,
  the 13-character reading above). On the verse page McCATMuS scores *cleaner*
  than the correct readings on gibberish (0.0009 vs 0.006-0.009) and
  non-wordlike tokens (0.004 vs 0.04-0.06). Its lexical contrast (3.80 vs
  4.25-4.73) and plausibility (0.44 vs 0.57-0.58) are lower, but still above
  the HIGH and floor limits. No single signal separates the four from the 27
  correctly allowed readings without blocking a low-CER one, except at
  thresholds fitted to these rows (see "Tried, not shipped"). Production
  routing puts CATMuS Medieval first under every language hint
  (`select_backend_plan`; backend election is off). McCATMuS reads a line
  there only when every earlier backend fails, or when it is asked for
  explicitly.
- Five pages in two languages, transcribed by non-specialists, were enough to
  show that the old gate was inverted and to place the thresholds on them.
  They do not give a corpus-level error rate, and the thresholds placed on
  them do not transfer: on held-out manuscripts the shipped gate allows 81 of
  128 readings at CER >= 25% (`eval/heldout/README.md`, section 2).

## Tried, not shipped

- **A per-recogniser confidence floor against bad readings** (seen
  2026-11-24, not built). On the held-out pages, 81 of 128 readings at
  CER >= 25% pass the shipped gate, all from the two recognisers pages are not routed
  to. Within each recogniser, page confidence follows page CER closely
  (Spearman -0.86 for CREMMA Medieval, -0.70 for McCATMuS, -0.67 for CATMuS
  Medieval; `eval/heldout/README.md` section 3), so a floor per recogniser
  could separate them. It cannot be calibrated here: those two recognisers
  produced only 4 pages at CER <= 10% in the sample, and the sample is now
  spent for gate decisions (see above). Confidence is not comparable across
  recognisers, so one floor for all would not do.

- **Lexical contrast for languages without a profile** (2026-11-23).
  `ocr_text_metrics.lexical_contrast` returns nothing for a detected language
  with no trigram profile (German, Italian, ...), so such a page skips the
  lexical gates. The idea was to measure it as the best fit over the four
  profiles instead. On this set it changes 0 of 59 verdicts: with langdetect
  seeded, no reading here is detected as a language without a profile. The
  one that once was, latin-apocalypse at 51.8% CER detected as Dutch and
  graded HIGH, came from unseeded randomness, which seeding removed. Outside
  the set, a Middle High German passage typed from memory, so only
  indicative, scored a best-fit contrast of 2.93. That would keep the page
  allowed but cost it HIGH (which needs 3.0). No gain was measured, and real
  German pages would be downgraded, so it was not shipped. Germanic pages
  need their own profiles. Separately, an Old English passage, also typed
  from memory, was detected as Latin and scored 1.54, under the UNRELIABLE
  limit. That needs a real Old English reading to confirm.
- **A threshold on a single existing signal against the McCATMuS readings**
  (2026-11-23). Over the 31 allowed rows, two signals catch 3 of the 4 false
  allows without blocking a reading at CER <= 10%, and both are fitted:
  - `char_entropy` < 4.153: 30 of the 31 allowed rows lie within 10% of the
    threshold.
  - `trailing_fragment_ratio` > 0: it is non-zero only on readings at
    CER >= 22.9% here. But "any trailing fragment at all" is the most fitted
    threshold possible. The signal is computed on whole-page text, where the
    code deliberately ignores line-edge fragments, since medieval lines break
    words without hyphens. And it carries the Latin-vowel bias of audit
    SVC-16.
  Neither was shipped. Telling McCATMuS's French-looking strings from French
  needs a word list, or held-out pages to test the fragment signal on.
- **The uncertainty-density double count (SVC-17), shipped.** For the record,
  scripts/calibrate_uncertainty_markers.py builds a synthetic marked set from
  these readings. Across 29 marked pairs the fix changes no verdict and
  lowers every density by about a quarter. None of the 24 real readings among
  the 59 rows (5 references, 19 recogniser readings) contains a marker.

