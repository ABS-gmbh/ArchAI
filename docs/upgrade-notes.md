# Upgrade notes

From the code of 2026-11-09 (commit `26c7611`, before pull request 90) to this
version. Derived from the two versions' code: the `Settings` class, both
`.env.example` files, the API schema, the frontend's environment, the READMEs,
the dependency pins and CI. `tests/test_upgrade_notes.py` fails if a setting
named here stops existing, or a default below stops matching the code.

## 1. Before you upgrade

**Back up** as [deployment.md](deployment.md#backing-up) says. Nothing in the
pipeline database's schema changes, and the older version still starts on
data the new one has used, but a rollback drops some settings (section 5).

These behave differently from the first start:

- **Startup refuses a data path it cannot write.** If `ARCHAI_DB_PATH`,
  `CHROMA_PERSIST_DIR`, `ARCHAI_WIKIDATA_CACHE`, `OCR_EVIDENCE_PATH` or
  `TASK_BASE_DIR` points somewhere this process cannot write (or create), the
  server stops with an error naming the setting. A good start logs one line,
  `Data paths: ...`, ending with `ANALYTICS_DB_PATH='…' (read only)`.
- **`.env` now sets the environment-only variables too.** `ARCHAI_DB_PATH`,
  `ARCHAI_WIKIDATA_CACHE`, `ARCHAI_DEBUG_RAG` and `ARCHAI_DISABLE_PROCESS_POOL`
  written in `.env` were ignored before; they now take effect. Check your
  `.env` for them before starting.
- **Settings that refuse a bad value** stop startup with a message naming
  them: `LOG_LEVEL`, `LOG_FORMAT`, `ANALYTICS_LOGIN_MAX_FAILURES`,
  `ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME`, `ANALYTICS_LOGIN_WINDOW_SECONDS`,
  `API_TOKEN` (at least 24 characters, and not only spaces), `CORS_ORIGINS`
  (`*` is refused), `RUN_RETENTION_DAYS`, `RUN_RETENTION_MAX_PER_SWEEP`,
  `CHAT_AI_TIMEOUT_SECONDS`, and below 1: `MAX_JSON_MEGABYTES`,
  `MAX_UPLOAD_MEGABYTES`, `MAX_EXTRACTED_MEGABYTES`, `MAX_ZIP_MEMBERS`,
  `MAX_CONCURRENT_BATCHES`, `MAX_ACTIVE_BATCHES`, `TASK_TTL_MINUTES`,
  `OCR_MAX_DECODE_PIXELS`, `OCR_RUN_RESERVATION_TTL_SECONDS` (also above
  86400), `OCR_RUN_RESERVATION_MAX_LIVE`, `OCR_TRACE_STREAM_MAX_OPEN`, which
  were quietly treated as 1. A `JWT_SECRET` shorter than 32 characters is
  only logged as an ERROR.
- **A frontend opened under a name needs `FRONTEND_ORIGINS`.** The `/api`
  proxy answers only requests sent to loopback, an IP address or a host named
  in `FRONTEND_ORIGINS` (DNS rebinding); under any other name every `/api`
  request is refused with 403. A listed name accepts only the listed origins,
  scheme included. Opening it at `localhost` or an IP address needs nothing.
- **The chat model is `qwen3.5-397b-a17b`.** `CHAT_RAG_MODEL` (and its
  alias `ARCHAI_CHAT_AI_MODEL`) defaulted to `qwen3-30b-a3b-instruct-2507`.
  On the same 79 questions and the same retrieved passages the new default
  answered 61 of 62 correctly with none wrong, where the old one answered 47
  correctly and 4 wrongly, and it starts answering as fast (measured against
  the provider, 2026-11-25, thinking off). It needs `CHAT_DISABLE_THINKING`
  left on: with thinking it starts after about 20 s. A `.env` that names the
  old model keeps it; remove the line to get the new one.
- **Translations use the chat model.** `TRANSLATION_MODEL` now defaults to
  empty, which means `CHAT_RAG_MODEL` translates. Its former default,
  `llama-3.3-70b-instruct`, is no longer offered by the provider, and with it
  every translation failed. If your `.env` still names that model, remove the
  line or name one the provider offers.
- **The port is 8010.** The READMEs and `make backend` used 8000 while the
  frontend expected 8010; all now say 8010. If you start the backend
  elsewhere, set the frontend's `BACKEND_URL`.
- **`/api/health`** answers 503 ("unavailable") when the pipeline database
  cannot be used, and 200 with "degraded" when the OCR engine, the chat key,
  the layout models or the line recogniser is missing, when the provider
  refused the chat key on its last call (health asks it nothing), or when pages cannot
  be indexed for search (`RAG_EMBEDDING_MODEL` needs the chat key and there
  is none; `RAG_EMBEDDING_MODEL=` indexes locally, see deployment.md,
  "Health"). It used to say "ok" regardless. The local embedder downloads its
  model (83 MB, from chroma-onnx-models.s3.amazonaws.com) the first time a
  page is indexed, into `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/`. On
  a server without internet access, fill that directory in advance
  (deployment.md, "Health", has the commands), or that first index fails. A monitor that expects "ok" will now alert on a deployment
  without models.
- **With `API_TOKEN` set**, the API docs are off unless `ENABLE_API_DOCS` is
  true, and every route but `/api/health` answers 401 without the token,
  before reading the request body.
- **Limits that did not exist:** JSON bodies over `MAX_JSON_MEGABYTES` (413),
  chat messages over `CHAT_MAX_MESSAGE_CHARS` (422), batch ZIPs with more than
  `MAX_ZIP_MEMBERS` members (the task ends in error), more than
  `OCR_TRACE_STREAM_MAX_OPEN` open trace streams (429), and at most
  `MAX_CONCURRENT_BATCHES` batches running at once (the rest wait).
- **Analytics login:** after `ANALYTICS_LOGIN_MAX_FAILURES` failures from one
  address, or `ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME` for one user name
  from anywhere, within `ANALYTICS_LOGIN_WINDOW_SECONDS`, logins answer 429.
  Anyone who can reach the login can therefore lock the analytics user out
  for up to 15 minutes ([deployment.md](deployment.md)).
- **Reading changes what new runs contain:**
  - Detached initials are joined to their word (`OCR_JOIN_DETACHED_INITIALS`).
  - Chat passes the top 8 passages, not 5 (`RAG_TOP_K`).
  - The quality gate scores Spanish and Dutch against their own profiles.
    Well-read Castilian and Navarrese pages it used to block become
    searchable. Garbled Dutch, which got no lexical check before, is now
    blocked.
  - Runs already stored keep their verdicts until they are read again.
- **Run reservations** are new (`POST /api/ocr/runs`); an unused one expires
  after `OCR_RUN_RESERVATION_TTL_SECONDS`.
- **Retention stays off** (`RUN_RETENTION_DAYS` is 0): nothing is deleted
  unless you set it.

## 2. New settings

Backend (`archai/backend/.env`). None was removed or renamed; one default
changed: `RAG_TOP_K` from 5 to 8. `.env.example` no longer lists
`SAIA_MODEL_PROBE`, which no version read.

| Setting | Default | Change it when |
|---|---|---|
| `API_TOKEN` | `""` | the backend is reachable from anywhere but localhost |
| `ENABLE_API_DOCS` | `None` | unset: docs on without a token, off with one |
| `LOG_LEVEL` | `"INFO"` | debugging (DEBUG) |
| `LOG_FORMAT` | `"text"` | a log collector reads it (json) |
| `TASK_BASE_DIR` | archai/backend/.tasks | data is kept outside the checkout |
| `MAX_JSON_MEGABYTES` | `48` | pages larger than an A4 PNG at 600 dpi are sent |
| `MAX_ZIP_MEMBERS` | `2000` | batches have more pages |
| `MAX_CONCURRENT_BATCHES` | `2` | the machine has memory for more |
| `MAX_ACTIVE_BATCHES` | `8` | more batches may wait in the queue at once |
| `CHAT_AI_TIMEOUT_SECONDS` | `180` | the chat provider is slower |
| `CHAT_STREAM_CUMULATIVE` | `False` | the provider re-sends the whole answer with every stream chunk (an OpenAI-compatible one does not) |
| `CHAT_DISABLE_THINKING` | `True` | rarely: off, models that think by default (qwen3.5, qwen3.6, qwen3.8) answer slowly and sometimes with empty text |
| `CHAT_DISABLE_THINKING_EXCEPT` | `"mistral-large"` | the provider refuses the thinking switch for another model (name prefixes, comma-separated) |
| `CHAT_MAX_MESSAGE_CHARS` | `200000` | rarely |
| `ANALYTICS_LOGIN_MAX_FAILURES` | `5` | many people share one address |
| `ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME` | `10` | rarely |
| `ANALYTICS_LOGIN_WINDOW_SECONDS` | `900` | rarely |
| `RUN_RETENTION_DAYS` | `0` | old runs should be deleted (preview with `scripts/prune_runs.py`) |
| `RUN_RETENTION_MAX_PER_SWEEP` | `50` | rarely |
| `OCR_RUN_RESERVATION_TTL_SECONDS` | `600` | rarely |
| `OCR_RUN_RESERVATION_MAX_LIVE` | `100` | many workspaces read at once |
| `OCR_TRACE_STREAM_MAX_OPEN` | `32` | many workspaces watch reads at once |
| `OCR_MAX_DECODE_PIXELS` | `89478485` | pages are larger than 89 megapixels (the largest seen are 50) |
| `OCR_TILE_CONCURRENCY` | `2` | the OCR provider allows more parallel requests |
| `OCR_JOIN_DETACHED_INITIALS` | `True` | initials should stay separate lines |
| `OCR_DETACHED_INITIAL_MIN_COLUMN_LINES` | `5` | rarely |
| `OCR_DETACHED_INITIAL_MIN_SHARE` | `0.5` | rarely |
| `OCR_DETACHED_INITIAL_MIN_GAP_RATIO` | `1.0` | rarely |
| `RAG_LEXICAL_WEIGHT_LOCAL` | `2.0` | retrieval uses ChromaDB's default model and BM25 should weigh differently |

Frontend (environment of `npm start`), all new except `BACKEND_URL`:

| Setting | Meaning |
|---|---|
| `BACKEND_URL` | Was read by a rewrite in `next.config.ts`; now by the `/api` route handler at run time. Default unchanged. |
| `BACKEND_API_TOKEN` | The backend's `API_TOKEN`, added by the proxy; never sent to the browser. `NEXT_PUBLIC_BACKEND_API_TOKEN` makes the build refuse to run. |
| `TRUSTED_PROXY_HOPS` | Reverse proxies in front that append to X-Forwarded-For. Default 0. |
| `TRUST_FORWARDED_HOST` | `true` when a proxy rewrites Host. |
| `FRONTEND_ORIGINS` | The origins the frontend is opened at by name; required for any name. |

See [deployment.md, "Behind a reverse proxy"](deployment.md#behind-a-reverse-proxy).

## 3. For scripts that call the API

- New: `POST /api/ocr/runs` (201) reserves a run. Pass its id as `run_id` to
  `POST /api/ocr/extract_full_page` or `/api/ocr/page_with_trace` to follow
  the read. Errors: 409 if it is unknown or used, 410 if it expired, 429 if
  too many are waiting.
- `POST /api/predict/batch` answers 429, with `Retry-After`, while
  `MAX_ACTIVE_BATCHES` batches are queued or running; it used to queue every
  batch without bound.
- `GET /api/health` answers `limits` (`json_bytes`, `decode_pixels`) and a
  `search_index` check besides the others.
- `GET /api/evidence/spans/{page_id}` and `GET /api/ocr/trace/{run_id}/tables`
  take a `limit`. `extract_full_page` answers `line_regions` too.
- The token goes in `X-API-Token` or `Authorization: Bearer`. As `?token=`
  it is accepted only on `GET /api/ocr/trace/stream/{run_id}` and
  `GET /api/predict/batch/{task_id}/progress`.
- Error `detail`s are fixed sentences. Exception text, paths and upstream
  bodies go to the log. Provider failures answer 502 (key rejected), 504
  (timeout) or 429 (rate limit), not 500. A 429 says how long to wait when
  the provider did (its `Retry-After`, also passed on as that header), in
  chat, translation, label analysis and the answer check's note.
- An analytics session token is accepted only for `ANALYTICS_USERNAME`.
  `/static/trace.html` is gone.
- `POST /api/chat/completions` answers `sources`, in the response and in the
  stream's `answer` and `done` events: one entry per evidence block shown to the model,
  with its `label` ("1", "2", ...), `run_id`, `chunk_ids`, `start_offset` and
  `end_offset` in the run's text, each chunk's span, `line_start` and
  `line_end` on the page where they can be counted, and `cited` when the
  answer names the label. Answers cite as [1] or [2][3] instead of
  "(asset_ref=..., run_id=..., chunk_id=..., offsets=...)". The workspace
  shows each cited label with lines on the page as a button that lights them
  up, lists the cited sources under the answer, and keeps them with the
  conversation.

## 4. What users of the workspace will see

- A new layout: the page on a grey well, its transcription beside it with
  each line lit up on the page, and Ask, Names, Steps and Document panels.
  On narrower screens they become tabs; below 768 px the page is a tab too,
  and the first one.
- The open document survives a reload, a closed tab or a browser crash:
  its pages, readings, names, steps and conversation are kept in the
  browser's IndexedDB and restored without reading again. "Close document"
  in the Document tab removes them.
- A banner when the backend is unreachable or missing an engine. Optional
  notices can be hidden.
- Layout regions drawn on the page with a legend of their kinds.
- Run steps in words instead of log lines, a confidence per line, and names
  with their candidates.
- Typed requests ("translate", "crop the initial", "segment this page") are
  matched as words, so questions containing them go to chat.
- Keyboard and screen-reader work: focus no longer falls to the page top
  when a button becomes unavailable or disappears, page changes are
  announced, and the page image and line confidence have names.
- Medieval characters drawn in Gentium Plus and Junicode.

## 5. If you roll back

[deployment.md, "Upgrading and rolling back"](deployment.md#upgrading-and-rolling-back)
has the drill. The 2026-11-09 version starts on upgraded data and answers the
same runs. It ignores `API_TOKEN` (the API is open again), `ENABLE_API_DOCS`
(docs always on) and `TASK_BASE_DIR` (tasks back in `archai/backend/.tasks`).
An unused run reservation stays reserved, and its stream stays open for up to
an hour.

## 6. Known limits

- **One worker process.** A second one on the same database is logged as an
  error, not refused.
- **The gate's lexical check has profiles for Latin, French (old, middle,
  modern), Spanish and Dutch only.** Pages in other languages (Italian,
  Catalan, Occitan, German, English…) get no lexical plausibility check.
- **Exercised live only in part.** Chat answers were run against the provider
  on 2026-11-25 (`archai/backend/eval/chat/README.md`). Label analysis and the
  provider's OCR path were not: their configured models are no longer
  offered by the provider.
- **GLM-OCR is not a usable fallback with Ollama 0.40.6.** Measured on the
  five reference pages (`archai/backend/eval/recognizers/README.md`): it
  repeats each page until the token limit (CER 252% to 1,350%, about 40% on
  the text before the repetition) or aborts with a 502. The quality gate
  blocks every such reading, so the page is not searched or linked.
- **Only the last open document survives a reload**, and only in the browser
  that opened it (IndexedDB): another browser or device starts empty, and so
  does one that refuses storage (the Document tab says so). Its annotated
  layout image is not kept (the regions are), and a read or question still
  running when the page goes is lost.
- **Not backed up:** layout tasks, which expire after `TASK_TTL_MINUTES`.
- **Development tooling** has one `npm audit` finding (in eslint's
  dependencies). The production dependencies have none.
