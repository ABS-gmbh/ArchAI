# Held-out measurement on CATMuS Medieval test manuscripts

Every other accuracy figure in this repository rests on five development
pages read by one annotator. This one does not: it reads lines from 48
manuscripts that none of this project's tuning ever saw, with ground truth
made by other people.

```bash
python scripts/benchmark_heldout.py sample --data DIR --manifest eval/heldout/manifest.jsonl
python scripts/benchmark_heldout.py run    --data DIR --manifest eval/heldout/manifest.jsonl --out lines.jsonl
python scripts/benchmark_heldout.py report --lines lines.jsonl --json report.json
```

`manifest.jsonl` names the 2,764 lines used (file and row, with a hash of the
reference text), not the images. `sample` with the default seed (20261124)
writes the same manifest from the same data, and `run` checks the hashes.
`report` is plain arithmetic over what `run` wrote; it is what
`tests/test_benchmark_heldout.py` covers.

## What "held out" means here

- **Data.** The test split of the CATMuS Medieval dataset on Hugging Face
  (`CATMuS/medieval`, CC-BY 4.0, revision e119659, release 1.6.0): 51 parquet
  files, 2.09 GB, 48 manuscripts. Its `test/` manuscripts are disjoint from
  `train/` and `dev/`.
- **The routed model.** The backend routes every page to the CATMuS Medieval
  1.5.0 Kraken model. Its metadata gives 113,228 training lines and 94.3%
  accuracy. None of the 48 test manuscripts is in release 1.5.1's train or
  dev split (29 were already in its test split, 19 are new in 1.6). So the
  model did not train on these manuscripts, **if** it was trained on that
  release's train split; its training configuration is not published in the
  metadata.
- **Not independent in every sense.** Some test manuscripts are siblings of
  training ones (for example Archives nationales LL 108B in test and LL 108A
  in dev), so hands and collections can overlap. The transcription conventions
  are the dataset's own, which is also what the model was trained to produce.
- **The other two models.** Whether CREMMA Medieval and McCATMuS saw any of
  these manuscripts could not be established. CREMMA's training data includes
  BnF French manuscripts that may overlap. Their figures are an upper bound on
  how well they do, not a clean held-out number.
- **Sample.** Per manuscript, two blocks of 30 consecutive main-text lines at
  random offsets: 2,764 lines, 93 blocks. Each line image is read as its own
  crop through `KrakenBackend.recognize`, the code the full-page route uses.
  No line failed for any model.
- **Not covered.** The page layout step (line images are given), the
  detached-initial join (it needs the layout), GLM-OCR, and entity extraction
  and linking (the dataset has no entity annotation).

Run on the evaluation VM on 2026-11-24 (CPU, torch 2.9.0, Kraken 6.0.3).

## 1. Which recogniser

Micro CER, diplomatic policy, with 95% intervals resampled over manuscripts
(not lines).

| | lines / manuscripts | CATMuS Medieval | CREMMA Medieval | McCATMuS |
|---|---|---|---|---|
| all | 2,764 / 48 | 6.9% [5.8, 8.2] | 28.8% [25.4, 32.5] | 37.8% [34.6, 41.2] |
| Latin | 1,264 / 23 | 7.6% [5.9, 9.9] | 32.5% | 41.5% |
| French | 720 / 12 | 5.2% [3.7, 7.0] | 22.2% | 34.9% |
| Castilian | 240 / 4 | 6.3% [2.6, 8.3] | 29.8% | 32.5% |
| Italian | 240 / 4 | 9.5% [5.6, 12.7] | 33.0% | 31.1% |
| 12th century and earlier | 415 / 8 | 6.5% to 7.7% | 29.9% to 38.8% | 40.5% to 42.1% |
| 13th to 16th century | 2,349 / 40 | 6.5% to 7.5% | 22.0% to 31.7% | 27.4% to 43.8% |

CATMuS Medieval is the best of the three on every language, century and
script stratum, and the paired difference of each other model against it has
its whole interval above zero in every stratum with more than one manuscript.
The closest is Humanistica script, where McCATMuS reads 20.9% against 6.3%
(difference +14.6 points [+9.4, +22.2]). Catalan, Middle Dutch, Navarrese,
Occitan, Venetian, print and uncial have one manuscript each and no real
interval.

**Routing every page to CATMuS Medieval holds on this data.** The model's own
figure (94.3% accuracy, about 5.7% CER) and this sample's 6.9% [5.8, 8.2] are
close.

## 2. The quality gate

Each block's lines, joined, are one pseudo-page of about 30 lines; the gate is
run on it as the pipeline runs it. "Blocked" means names are not extracted and
the text is not indexed for search.

| reading by | pages at CER ≤ 10% | of them blocked | pages at CER ≥ 25% | of them allowed |
|---|---|---|---|---|
| CATMuS Medieval (routed) | 76 | **6** | 0 | – |
| CREMMA Medieval | 3 | 0 | 47 | **31** |
| McCATMuS | 1 | 0 | 81 | **46** |

The claim made from the five development pages, "no reading at CER ≤ 10% is
blocked, and 4 of 59 at CER ≥ 25% are allowed", **does not survive**:

- **Good readings blocked.** Six of 76 good CATMuS readings are blocked, and
  five of the six are medieval Spanish: both blocks of Madrid, BNE, MSS. 3995
  (CER 2.2% and 2.0%), both of Paris, BnF, esp. 458 (8.7%, 9.3%) and one of
  Paris, BnF, esp. 65 (Navarrese, 4.0%). All are detected as Spanish, which
  was scored against the Latin profile. They are labelled RISKY mainly on the
  hit-rate floor: plausibility 0.18-0.22, under 0.25, while their contrasts
  are 2.23-2.94, above the 2.2 RISKY limit. The sixth is a Latin
  page detected as Catalan (2.6%). Of the Castilian and Navarrese pages the
  routed model read well, half are blocked (5 of 10). The development set has
  no Spanish page, so this could not show there.
- **Bad readings allowed.** Of readings at CER ≥ 25%, 77 of 128 are allowed
  (31 of 47 for CREMMA, 46 of 81 for McCATMuS; all 30 French ones, 32 of 69
  Latin ones). The gate does not catch a fluent-looking wrong reading from
  another recogniser. The routed model produced no page at or above 25% in
  this sample (its worst is 24.2%), so in the product this matters where a
  fallback recogniser reads a page or some of its lines.

The two findings need different work: the first a lexical profile that knows
medieval Castilian, the second a signal that the gate does not have at all.

**After 016a998** (Spanish and Dutch profiles built from the dataset's train
split, `scripts/build_trigram_profiles.py`; see
`eval/quality_gate/README.md`), on the same pseudo-pages:

| reading by | CER ≤ 10% blocked, before → after | CER ≥ 25% allowed, before → after |
|---|---|---|
| CATMuS Medieval (routed) | 6/76 → 1/76 | – |
| CREMMA Medieval | 0/3 → 0/3 | 31/47 → 31/47 |
| McCATMuS | 0/1 → 0/1 | 46/81 → 50/81 |

- **What changed.** The five Spanish pages pass; the remaining block is the
  Latin page detected as Catalan.
- **The cost.** Four more McCATMuS readings at CER ≥ 25% pass (Castilian
  1/6 → 4/6, Navarrese 0/2 → 1/2).
- **What it rests on.** 4 Castilian manuscripts and 1 Navarrese here, 93
  dev-split Castilian pages, and one sampling seed.
- **This sample is spent for gate decisions.** It has now been used once to
  decide whether to ship a gate change, so the next gate change must be
  judged on a fresh sample (another seed, other blocks), not on this one.

## 3. Kraken's confidence

Spearman correlation of confidence with CER:

| model | lines | line level | pseudo-pages | page level | confidence, 5th to 95th percentile |
|---|---|---|---|---|---|
| CATMuS Medieval | 2,763 | −0.39 | 93 | −0.67 | 0.936 to 0.996 |
| CREMMA Medieval | 2,757 | −0.56 | 93 | −0.86 | 0.893 to 0.994 |
| McCATMuS | 2,764 | −0.39 | 93 | −0.70 | 0.907 to 0.993 |

This is stronger than on the five development pages (−0.23 at line level
over 549 lines): within one model, lower confidence does go with more errors.
The ranges overlap almost entirely across models, though: McCATMuS at 37.8%
CER is nearly as confident as CATMuS Medieval at 6.9%. Confidence is not
comparable between models and was not tested here as a gate.

Raw rows (`lines.jsonl`, with the reference and the text read for each line)
and `report.json` (every pseudo-page's CER, verdict and detected language)
are kept outside the repository with the evaluation VM's results.
