# Chat answer evaluation

`answers.json` holds 79 hand-labelled chat questions about the reference
pages of `eval/quality_gate/readings.json`. `scripts/evaluate_chat_answers.py`
asks them through the chat path, with one gate-approved run per page as
context, and scores the result.

```bash
python scripts/evaluate_chat_answers.py                                  # retrieval only, no chat model
python scripts/evaluate_chat_answers.py --embedder e5-local              # the same, with multilingual-e5-large-instruct run locally
python scripts/evaluate_chat_answers.py --live --out /tmp/answers.jsonl  # ask the configured chat model
python scripts/evaluate_chat_answers.py --score /tmp/answers.jsonl       # re-score saved answers
```

## Items

- **62 answerable** (`a00`-`a61`): the questions of
  `eval/retrieval/questions.json`, in the same order. Each item carries:
  - `answer_span`: the reference text that answers it;
  - `expected`: a short English answer written by hand from that span;
  - `accept`: groups of lower-case terms. A correct answer contains at least
    one term of every group (e.g. `[["sabbath", "saturday"], ["holy",
    "sanctif", "hallow"]]` for the third commandment).
- **17 not in the sources** (`n00`-`n16`): the `not_in_sources` questions of
  the same file. The page does not answer them, so the right answer says so.

| page_id | answerable | not in sources |
|---|---|---|
| `old-french` | 15 | 5 |
| `latin-abaton` | 19 | 4 |
| `latin-apocalypse` | 11 | 4 |
| `old-french-verse` | 17 | 4 |

## How it was built

Every `expected` answer and `accept` list was written by hand from the
reference, before any chat model had answered the question. The readings of
the Latin oracle lines (*sortes*) and the Old French follow the literal sense
of the reference. No specialist has checked them.

## What is scored

Each page is loaded into a throwaway pipeline DB and vector store in a
temporary directory (never `app/archai.sqlite`), chunked as the pipeline
chunks. Questions go through `chat_ai._prepare_messages`, so retrieval, the
evidence blocks and the grounding instructions are the production ones. The
context carries the run only, not a transcript, so the answer must come from
search.

- **Without `--live`** no chat model is called. Per item, the report says
  whether the evidence put in the prompt holds a chunk containing the answer
  span (`evidence_holds_answer`), the retrieval status, and whether the
  "not in the sources" instruction is present.
- **With `--live`** the configured chat model answers. Verification is off,
  so the draft answer itself is scored. Every answer is saved to `--out`.
  - Answerable items are `correct` (every accept group matches), `abstained`
    (says the sources do not contain it, and the terms do not match) or
    `wrong`. They are `cited` if the answer names the chunk_id of a chunk
    holding the span.
  - Not-in-sources items are `abstained` or `answered`.

The report gives counts as k/n per kind and per page, with no headline rate.

## Results

Retrieval only, local embedder (ChromaDB's MiniLM), BM25 weight 2. The
evidence is the block put into the prompt; overlapping hits are merged.

| top_k | answer in evidence (n=62) | mean evidence chars | note |
|---|---|---|---|
| 5, before merging | 53 | 1,739 | the former default |
| 5, merged | 53 | 1,295 | |
| 8, merged | 56 | 1,620 | the default (`RAG_TOP_K`) |
| 10, merged | 59 | 1,768 | |

The "not in the sources" instruction is present for all 79 items. The 9
answerable questions whose answer stays out at top_k 5 are English questions
that share no word with the Old French or Latin passage. Their answer chunk
ranks 6th to 15th of the page's 7 to 17 windows. Three of them come in at
top_k 8.

With multilingual-e5-large-instruct (`--embedder e5-local`, on the evaluation
VM, 2026-11-24, BM25 weight 1 as in the e5 space):

| top_k | MiniLM (weight 2) | e5 | e5, weight 2 | e5, weight 0.5 |
|---|---|---|---|---|
| 5 | 53/62 | 55/62 | 54/62 | 56/62 |
| 8 (default) | 56/62 | 62/62 | 61/62 | 62/62 |
| 10 | 59/62 | 62/62 | | |

At top_k 5, MiniLM misses a00, a06, a09, a39, a41, a42, a56, a58 and a61, and
e5 misses a06, a08, a31, a39, a41, a49 and a58. e5 recovers five of MiniLM's
nine misses and loses three others. Each lookup searches a single page of 4
to 17 windows, so top_k 8 is a large share of the page, and 62/62 comes
easily there. The question benchmark (`eval/retrieval/README.md`) also
searches one reading of one page, but over 270 lookups on OCR readings, and
scores the whole ranking (MRR); it is the stronger evidence: e5 dense MRR
0.590 against MiniLM's 0.398.

No live answers have been scored yet: the chat provider key returns 401.

## Caveats

- Term matching is a heuristic. A polarity answer ("no") can match by
  accident, and a correct paraphrase can miss every term. Read the saved
  answers before quoting a number.
- The abstention pattern (`ABSTAIN_RE`) recognises common English phrasings
  ("does not mention", "not in the sources", "no information"). Others are
  scored as `answered` or `wrong`.
- Seventeen not-in-sources items, over four pages, is a small sample.
- One annotator, no second check.
