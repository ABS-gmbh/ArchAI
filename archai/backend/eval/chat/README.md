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
# a model, thinking off, the first 20 streamed, embeddings kept between runs, a stopped run resumed:
python scripts/evaluate_chat_answers.py --live --embedder provider --embed-cache /tmp/embed.json \
    --model qwen3.5-397b-a17b --no-thinking --stream-first 20 --resume --out /tmp/answers.jsonl
# a verifier on saved answers, nothing answered again:
python scripts/evaluate_chat_answers.py --embedder provider --embed-cache /tmp/embed.json \
    --verify-saved /tmp/answers.jsonl --verifier qwen3.6-35b-a3b --no-thinking --out /tmp/checks.jsonl
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
  - Answerable items are `correct` (every accept group matches, and the
    answer does not open by saying the sources lack the answer), `hedged`
    (it opens that way, but the terms match: some go on to quote the line
    and answer, some only refuse, so read them), `abstained` (says the
    sources do not contain it, and the terms do not match) or `wrong`.
  - They are `cited` if the answer cites a label ([1], [2], ...) whose
    evidence block covers a chunk holding the span; for answers saved
    before labels, if it names that chunk's id.
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

### Live answers (2026-11-25, the provider, on a Mac)

The 79 questions through the chat path with the provider's
multilingual-e5-large-instruct for retrieval (one embedding cache, so every
model saw the same evidence), top_k 8, temperature 0.2 (the route's
default), verification off. The first 20 questions are streamed to time the
first token; tokens per second is completion tokens over total seconds on
the others. Raw answers: `~/Developer/.archai-tools/vm/results/chat_live/`
(`answers_<model>_<mode>.jsonl`, `report_*.txt`), outside the repository.

| model | thinking | correct / hedged / wrong / abstained (62) | not in sources abstained / answered | cited | first token, median / p90 | answer seconds, median / p90 | tokens/s |
|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b-instruct-2507, before citation labels | n/a | 45 / 12 / 4 / 1 | 16 / 1 | 27 | – | – | – |
| qwen3-30b-a3b-instruct-2507, with labels | n/a | 47 / 10 / 4 / 1 | 16 / 1 | 56 | – | 3.4 / 18.8 | 4.0 |
| **qwen3.5-397b-a17b** | off | **61 / 1 / 0 / 0** | **17 / 0** | **62** | **0.43 / 1.45 s** | 3.8 / 20.5 | 17.9 |
| qwen3.8-27b | off | 55 / 3 / 2 / 2 | 17 / 0 | 62 | 6.4 / 7.5 s | 8.7 / 12.4 | 13.3 |
| qwen3.6-35b-a3b, 72 of 79 | off | 53 / 3 / 3 / 3 | 10 / 0 (of 10) | 59 | 4.4 / 9.4 s | 8.5 / 9.9 | 22.9 |
| qwen3.5-397b-a17b, 10 latin-abaton questions | on | 10 / 0 / 0 / 0 | – | 10 | 20.1 / 43.6 s | 22.0 / 45.0 | – |

Read, not only counted:

- **qwen3-30b**: the four wrong answers (a20, a25, a34, a59) are wrong; a25
  names Marcadigas, not Caldus, and calls the name "likely a variant of
  Margaret". It answers n00 ("the fourth commandment") with the third. Of
  its ten hedged answers five quote the line and give the answer (a13,
  a15, a38, a44, a50) and five only refuse (a16, a43, a45, a49, a56).
- **qwen3.5-397b**: its one hedged answer (a45) refuses; nothing is wrong.
  Ten of its 61 correct answers read by eye give the expected answer from
  the quoted line and cite it; some add unasked material (a28, a55) and
  a12 mistranslates a word before the right conclusion. It names the third
  commandment as the third and says the page does not give the fourth.
- **qwen3.8-27b**: a19 and a34 are wrong, a25 and a26 abstain; of its
  hedged answers a52 gives "great and high", a14 and a45 refuse.
- **qwen3.6-35b**: a20 and a25 are wrong; a04 is right (it quotes the third
  commandment and explains the Sabbath) but lacks the English "holy" the
  terms ask for. Its run stopped at 72 of 79 (n10-n16 missing) when the
  provider's daily cap was reached.
- **Citations**: before the evidence carried labels, the model was asked to
  cite "(asset_ref=..., run_id=..., chunk_id=..., offsets=...)" and 35 of 62
  answers did not name the chunk that holds the answer. With labels, 56 of 62
  cite the block that holds it, and the two larger models 62 of 62.
- Not measured yet: deepseek-v4.1-flash, gemma-4-31b-it, mistral-large-4
  (samples), the verifiers, and the vision models.

### Thinking

The provider's Qwen 3.5, 3.6 and 3.8 models, DeepSeek V4/V4.1 Flash,
mistral-large-4 and GLM 5.3 reason before they answer. The reasoning arrives
in `reasoning_content`, not in the answer, so with a token budget the answer
can come back empty (a 200-token budget: no answer from any of them except
deepseek-v4-flash), and without one the first answer token waits for the
reasoning (qwen3.5-397b: 20 s median). `extra_body={"chat_template_kwargs":
{"enable_thinking": false}}` stops it on the Qwen and DeepSeek models
(first token 0.1-1.1 s), is accepted and changes nothing on the models that
do not think (qwen3-30b, qwen3-coder, qwen3-omni, gemma-4, llama-3.1-8b,
eurollm), is accepted and ignored by GLM 5.3 and GLM 5.3 Flash (as are
`thinking.type=disabled` and `chat_template_kwargs.thinking=false`), and is
refused by mistral-large-4 with 400 "chat_template is not supported". Probe
scripts and outputs: `thinking_probe.py`, `glm_switch_probe.py` in the
results folder.

### What a question costs the provider

Counted on the requests the client sends:

- **A question**: 2 embedding requests (the same query text, once for the
  chunk collection and once for the entity collection), 1 chat request, and
  1 more when the answer is checked (the verifier). 3 without the check, 4
  with it.
- **A page read** with the provider embedder: 1 embedding request per 64
  chunks (one for a page of this size) and 1 per 64 entity documents when
  the page has linked names. The recogniser and the authority lookups do
  not use the provider.
- **The provider's limits, per user (the key)**: 10 requests in any 10
  seconds, 400 an hour, 800 a day ("user daily request limit reached (800
  of 800)", met at 13:00 on 2026-11-25 with Retry-After 39,597 s, that is
  until midnight).
- **In the app's units**: 800 a day is about 200 checked questions, or 266
  unchecked ones, for a whole installation on one key, fewer for every page
  indexed; 400 an hour is about 100 checked questions an hour. The OpenAI
  client's own retries spent more: before the eval set max_retries=0, two
  79-question runs made 38 and 60 extra chat requests, most likely 429s
  from the 10-in-10-seconds limit, retried without a word in the log.

## Caveats

- Term matching is a heuristic. A polarity answer ("no") can match by
  accident, and a correct paraphrase can miss every term. Read the saved
  answers before quoting a number.
- The abstention pattern (`ABSTAIN_RE`) recognises common English phrasings
  ("does not mention", "not in the sources", "no information"). Others are
  scored as `answered` or `wrong`.
- Seventeen not-in-sources items, over four pages, is a small sample.
- One annotator, no second check.
- The live table rests on 79 questions about four pages, written by one
  annotator, scored by term matching and read by eye where the terms and
  the reading disagree. It ranks models on these pages; it is not an
  accuracy rate for other manuscripts.
- Temperature was 0.2 for every model, the chat route's default, not each
  model's recommended sampling (0.6 or 1.0 with top_p 0.95 in the
  provider's documentation).
