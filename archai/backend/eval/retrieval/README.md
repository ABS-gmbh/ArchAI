# Retrieval question benchmark

`questions.json` holds questions a reader might ask about the reference pages
of `eval/quality_gate/readings.json`, in English. Each answerable question
names the span of the page's reference transcription that answers it.
`scripts/benchmark_retrieval.py --questions` projects that span into every
recogniser's reading of the page and scores how highly each ranker places the
chunks holding it.

```bash
python scripts/benchmark_retrieval.py --questions eval/retrieval/questions.json
python scripts/benchmark_retrieval.py --questions eval/retrieval/questions.json --question-set 2026-10
python scripts/benchmark_retrieval.py --questions eval/retrieval/questions.json --separation
```

## How it was built

- Every question and answer span was written by hand from the reference text.
  No retriever had been run on a set before it was written.
- `set` records when a question was written:
  - `2026-10`: 35 questions, the set behind the earlier results.
  - `2026-11`: 27 more.
- Answers are exact substrings of the reference. Each occurs exactly once,
  which `tests/test_benchmark_retrieval.py` checks.
- `not_in_sources` holds 17 questions on the same pages and topics whose
  answer the page does not contain, so they have no span. Examples: the fourth
  commandment on a page that stops at the third, and the gates of the
  heavenly city, described only in verses after the page's lection. They are
  for measuring what retrieval returns when there is nothing to find (a
  relevance floor) and whether an answer says "not in the sources". The
  passage benchmark does not use them.

## Pages and counts

| page_id | 2026-10 | 2026-11 | not in sources |
|---|---|---|---|
| `old-french` | 8 | 7 | 5 |
| `latin-abaton` | 10 | 9 | 4 |
| `latin-apocalypse` | 7 | 4 | 4 |
| `old-french-verse` | 10 | 7 | 4 |

`middle-french-antitus` has 7 lines, i.e. two chunks, and the benchmark skips
targets with fewer than four, so it has no questions.

A question becomes one lookup per reading of its page in which the answer
could be aligned. The benchmark prints the lookups per page. With the local
embedder, the 2026-10 set gives 151 lookups, the 2026-11 set 119, and both
together 270.

## Relevance floor

`--separation` asks whether a floor could work. For each lookup it takes the
score of the top hit on three signals: dense cosine, that cosine as a z-score
over the page's chunks, and BM25. It then reports the AUC of answerable
lookups against not-in-sources ones (0.5 is chance), with a 95% interval
resampled over readings. With the local embedder (ChromaDB's MiniLM):

| Signal | AUC (270 answerable vs 77 not-in-sources lookups) |
|---|---|
| top dense cosine | 0.439 [0.368, 0.515] |
| top dense z-score | 0.471 [0.403, 0.536] |
| top BM25 score | 0.506 [0.447, 0.566] |

None of the three separates them, so no floor is set: a floor that removed
the hits of an unanswerable question would remove the answer to an
answerable one just as often. A cosine floor of 0.20 empties 80% of
not-in-sources lookups but keeps the relevant chunk in only 3% of answerable
ones. With the production embedder (multilingual-e5-large-instruct, run
locally with `--embedder e5-local`) the top cosine does a little better than
chance, but still gives no usable floor:

| Signal | AUC, e5 (270 vs 77 lookups) |
|---|---|
| top dense cosine | 0.606 [0.568, 0.659] |
| top dense z-score | 0.526 [0.461, 0.588] |
| top BM25 score | 0.507 [0.447, 0.566] |

The median top cosine is 0.787 for answerable lookups and 0.782 for
not-in-sources ones, so any cosine floor removes the two almost equally. No
floor is set for either embedder. Not answering from irrelevant evidence is
left to the chat instruction instead.

## Results: MiniLM against multilingual-e5

Measured on the evaluation VM on 2026-11-24 (`scripts/eval_vm/`): CPU,
torch 2.9.0, sentence-transformers 6.3.0, e5 loaded as float32. Both sets
together: 270 question lookups over 18 readings of 4 pages (latin-abaton 87,
latin-apocalypse 44, old-french 71, old-french-verse 68). BM25 weight 1.
Intervals are 95%, resampled over readings.

```bash
python scripts/benchmark_retrieval.py --readings eval/quality_gate/readings.json --questions eval/retrieval/questions.json --embedder e5-local
```

| Ranker | MiniLM MRR | e5 MRR | e5 without the query instruction, MRR | MiniLM Hit@1 / Hit@5 | e5 Hit@1 / Hit@5 |
|---|---|---|---|---|---|
| dense | 0.398 [0.333, 0.464] | 0.590 [0.524, 0.653] | 0.582 [0.514, 0.647] | 0.207 / 0.681 | 0.426 [0.343, 0.504] / 0.822 [0.766, 0.874] |
| BM25 4-grams | 0.407 [0.338, 0.467] | (same) | (same) | 0.330 / 0.519 | (same) |
| hybrid (RRF) | 0.475 [0.415, 0.533] | 0.611 [0.564, 0.651] | 0.594 [0.545, 0.636] | 0.278 / 0.770 | 0.452 [0.396, 0.500] / 0.811 [0.756, 0.867] |

- e5 ranks English questions over Old French and Latin far better than
  MiniLM: dense MRR +0.19, with intervals apart.
- With e5, fusing BM25 adds nothing measurable on questions (paired change
  against dense +0.021 [-0.016, +0.062]). With MiniLM it adds +0.077
  [+0.049, +0.104].
- The e5-instruct query instruction is worth +0.008 (dense) and +0.017
  (hybrid) MRR, inside the noise. It does no harm, so the pipeline keeps it.

BM25's weight in the hybrid (`--lexical-weight`), MRR on the same 270
lookups (unpaired intervals):

| Weight | MiniLM | e5 |
|---|---|---|
| 0.5 | 0.465 [0.404, 0.523] | 0.601 [0.548, 0.642] |
| 1 | 0.475 [0.415, 0.533] | 0.611 [0.564, 0.651] |
| 2 | 0.513 [0.461, 0.557] | 0.598 [0.541, 0.644] |
| 3 | 0.521 [0.475, 0.560] | 0.591 [0.533, 0.636] |

With e5 the curve is flat, and 1, the default in the e5 space
(`rag_lexical_weight`), is its best point. MiniLM does better at 2 to 3,
which is why the local space uses 2 (`rag_lexical_weight_local`).

On the passage benchmark (`--readings` alone, 222 lookups over 16 readings),
e5 is no better than MiniLM. Dense MRR is 0.680 [0.604, 0.756] against 0.707
[0.643, 0.760], and hybrid 0.817 [0.745, 0.881] against 0.863 [0.814, 0.905].
BM25 4-grams alone is best there, at 0.886 [0.843, 0.924]. Matching OCR noise
is a lexical task, while answering a question across languages is a semantic
one.

These numbers rest on 4 pages, 2 languages and 1 annotator. That is enough to
say e5 beats MiniLM on cross-lingual questions. It is not enough to tune
top_k or the weight within e5, where every difference above is inside the
intervals.

## Caveats

- One person wrote every question, in English, against Old French and Latin
  pages. No second annotator checked the questions or the spans.
- Four pages is a small sample. Lookups on one page are not independent, which
  is why the benchmark's intervals are resampled over readings.
- Some answers overlap: "How is the mountain described?" falls inside "Where
  was he carried in the spirit?". The relevant chunk is the same, but the
  question is different.
