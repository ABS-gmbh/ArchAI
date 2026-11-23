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
ones. The production embedder (multilingual-e5-large-instruct through the
provider) has not been measured. Rerun with `--embedder provider` or
`--embedder e5-local` before considering a floor for it. Not answering from
irrelevant evidence is left to the chat instruction instead.

## Caveats

- One person wrote every question, in English, against Old French and Latin
  pages. No second annotator checked the questions or the spans.
- Four pages is a small sample. Lookups on one page are not independent, which
  is why the benchmark's intervals are resampled over readings.
- Some answers overlap: "How is the mountain described?" falls inside "Where
  was he carried in the spirit?". The relevant chunk is the same, but the
  question is different.
