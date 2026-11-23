# Deploying the ArchAI backend

How to run `archai/backend` on a server, written from what the code does
today. The repository ships no container or service files; the commands below
are the ones the code supports. For first-time setup and the settings in
general, see [`archai/backend/README.md`](../archai/backend/README.md).

## Run one worker

Run a single uvicorn worker process:

```bash
cd archai/backend
uvicorn app.main:app --host 127.0.0.1 --port 8010 --workers 1
```

Several workers (`--workers N`, or several uvicorn processes behind one
proxy) are not supported. The backend keeps state in process memory that the
other workers cannot see:

| What | With several workers |
|---|---|
| Layout tasks (`/api/predict/*`, `/api/download/*`) | Tasks live in a dictionary in the worker that created them. Their state is written to `.tasks/<id>/state.json` only when the task ends. A batch's progress stream or results requested from another worker answer 404 until the batch has finished. |
| Chroma vector store | Every worker opens its own embedded `chromadb.PersistentClient` on the same directory. The embedded mode is meant for one process: each worker keeps its own view of the collections, so one does not see another's writes, and concurrent writers to one directory are not something it supports. |
| Per-run locks | Authority linking and RAG indexing serialise work on one run with `threading` locks, which only hold inside one process. Two workers linking or indexing the same run can again write duplicate candidates or index the same chunks twice. |
| Analytics tokens | With `JWT_SECRET` unset, each worker makes its own random signing secret, so a token issued by one worker is rejected by the others. |
| Timers | Each worker runs the task cleanup and, if enabled, the retention sweep every 15 minutes. Both are idempotent, so this is wasteful rather than harmful, but each worker deletes up to `RUN_RETENTION_MAX_PER_SWEEP` runs per sweep. |
| Rate limits and models | The Wikidata rate limit is per process, so N workers send N times the request rate. Each worker loads its own layout models (and, on Linux, its own pool of `MAX_POOL_WORKERS` model processes), multiplying memory. |

At startup the backend takes an advisory lock on `<database>.lock` beside the
pipeline database. A second process using the same database logs an ERROR
("A second backend process is using the data directory ...") and keeps
running: it is a warning, so check the log after changing how the server is
started.

The pipeline database itself would cope: SQLite runs in WAL mode with a 10 s
busy timeout, which lets several processes on one host read while one writes.
Keep it on a local disk, not a network file system, where WAL locking does not
work.

Concurrency inside the one worker comes from FastAPI's threadpool: blocking
route handlers, OCR, linking and indexing run in worker threads, so one slow
page does not hold up other requests.

## Settings with no safe default for a server

Put these in `archai/backend/.env`, or in the environment. Settings are read
from `.env` and `.env.local` **in the working directory**, so start the server
in `archai/backend` or pass them as environment variables.

| Variable | Why it must be set |
|---|---|
| `API_TOKEN` | Without it the API has no authentication (startup logs a warning). Set at least 24 random characters, e.g. `python -c 'import secrets; print(secrets.token_urlsafe(32))'`. |
| `CORS_ORIGINS` | The default allows only `localhost`/`127.0.0.1` on ports 3000 and 3001. Set the frontend's real origin, or leave it unset only if the browser reaches the backend through the frontend's `/api` proxy (same origin). |
| `CHAT_AI_API_KEY` | Without it, chat, translation, SAIA OCR and provider embeddings fail ("Chat API key not configured"). |
| `GLMOCR_OLLAMA_HOST` | The Ollama server running `glm-ocr`, if GLM-OCR or the segmented engine's fallback is to work. |
| `ANALYTICS_USERNAME`, `ANALYTICS_PASSWORD`, `JWT_SECRET` | Analytics login refuses with 503 until the first two are set. Set `JWT_SECRET` too: without it the tokens are signed with a random per-process secret, so sessions end at every restart, and startup logs an ERROR saying so. |
| `ARCHAI_DB_PATH`, `CHROMA_PERSIST_DIR`, `ARCHAI_WIKIDATA_CACHE`, `OCR_EVIDENCE_PATH`, `TASK_BASE_DIR` | Their defaults are inside the code checkout. Point them at a data directory (the first four are the ones to back up). |
| `MAX_UPLOAD_MEGABYTES` | The default is 1024 MB per upload, and request bodies may be a third larger for base64. Set what your pages need. |
| `LOG_FORMAT=json` | One JSON object per log line, for a log collector. |

Leave `ENABLE_TEST_OCR_FIXTURES=false` (canned OCR answers for four test
images) and leave `ARCHAI_DEBUG_RAG` unset (it exposes `/api/rag/debug/*`).
Both are off by default. Set `ENABLE_API_DOCS=false`: it is on by default for
development, and on a server it publishes `/docs`, `/redoc` and
`/openapi.json`, a map of every route, without authentication. A setting of the wrong type, or out of range, stops
startup with an error that names the variable.

Layout tasks keep their uploads and outputs in `TASK_BASE_DIR` (default
`archai/backend/.tasks`). It must be a writable directory: the server creates
it if missing and refuses to start, naming the variable, if it cannot write
there. It needs no backup: task files expire after `TASK_TTL_MINUTES` (60).

## What to put in front

The backend speaks plain HTTP. Unless `API_TOKEN` is set it has **no
authentication** except on the analytics routes: anyone who can reach it can
run OCR (spending SAIA credits and GPU or CPU time), read any run's text by
its id, write evidence spans and delete a run's vectors. Do not expose it
without a token or a proxy that authenticates.

With `API_TOKEN` set, every route except `/api/health` requires the token,
compared in constant time, and answers 401 otherwise. Send it as
`X-API-Token: <token>` or `Authorization: Bearer <token>`. Analytics clients
use `X-API-Token`, because their session token travels as `Authorization:
Bearer`. The two event streams a browser opens with EventSource
(`GET /api/ocr/trace/stream/{run_id}`, `GET /api/predict/batch/{id}/progress`)
also accept it as `?token=`, and uvicorn's access log shows that value as
`[redacted]`. A browser should not hold the token: set `BACKEND_API_TOKEN` on
the frontend's server and its `/api` proxy adds it (see
[the frontend](#the-frontend)).

- **Bind to localhost** (`--host 127.0.0.1`, as above) and put a reverse proxy
  in front that terminates TLS.
- **Authenticate**: set `API_TOKEN`, or authenticate at the proxy (basic
  auth, single sign-on, an IP allow-list) for every route except, if a monitor
  needs it, `/api/health`. Both together do no harm.
- **Body size:** the backend refuses bodies over `MAX_UPLOAD_MEGABYTES × 4/3 +
  1 MB` with 413. Set the proxy's limit to the same value or lower.
- **Timeouts:** a full-page OCR request is synchronous and can take minutes
  (GLM-OCR waits up to `GLMOCR_OLLAMA_TIMEOUT_SECONDS`, 300 s by default).
  Give the proxy a read timeout of at least 10 minutes on `/api/ocr/`.
- **Streams:** `/api/chat/completions`, `/api/ocr/trace/stream/{id}` and
  `/api/predict/batch/{id}/progress` are server-sent events. They send
  `X-Accel-Buffering: no`, which nginx honours. For another proxy, turn off
  response buffering on those paths.
- **Client addresses:** uvicorn trusts `X-Forwarded-For` only from
  127.0.0.1. If the proxy runs on another address, add
  `--forwarded-allow-ips=<proxy address>` so the access log shows the client's
  address.
- **Rate limiting:** the backend refuses analytics logins from an address
  with 429 after `ANALYTICS_LOGIN_MAX_FAILURES` (5) failures within
  `ANALYTICS_LOGIN_WINDOW_SECONDS` (900). It keys on the client address it
  sees, so behind a proxy uvicorn must trust the proxy's `X-Forwarded-For`
  (see client addresses above); otherwise every client shares the proxy's
  address, and one guesser locks everyone out. If that cannot be arranged,
  limit at the proxy instead. The OCR and chat routes are expensive and have
  no limit of their own; limit them at the proxy.

### The frontend

`archai/frontend` is a Next.js app: `npm ci`, `npm run build`, `npm start`
(Node 20.9 or newer). On a server hostname the browser sends its requests to
the frontend's own `/api/...`. A route handler in the Next server
(`src/app/api/[...path]/route.ts`) forwards them to the backend, streaming
uploads and server-sent events through without buffering them. So the browser
needs no direct access to the backend, and `CORS_ORIGINS` does not have to
list the frontend. Two variables of the Next server configure it, read when
it starts (`npm start`), not at build time:

| Variable | Meaning |
|---|---|
| `BACKEND_URL` | Where the backend listens. Default `http://127.0.0.1:8010`. |
| `BACKEND_API_TOKEN` | The backend's `API_TOKEN`. The proxy adds it to every request as `X-API-Token`; it is never sent to the browser. Leave unset if the backend has no token. |

Never name it `NEXT_PUBLIC_BACKEND_API_TOKEN`: a `NEXT_PUBLIC_` variable is
copied into the JavaScript every browser downloads. The build and `npm start`
refuse to run with that variable set.

With `BACKEND_API_TOKEN` set, **the frontend must itself sit behind the
deployment's own login** (single sign-on, basic auth at the reverse proxy, a
VPN), since anyone who can reach the frontend can use the backend through it.
The token then only keeps the backend from being used without the frontend.

The proxy passes `X-Forwarded-For` on as it receives it. If the frontend is
reachable without a reverse proxy in front, a client can set that header and
the backend logs and rate-limits that address instead of its own. Put a
reverse proxy in front that replaces the header (nginx:
`proxy_set_header X-Forwarded-For $remote_addr;`).

`NEXT_PUBLIC_BACKEND_URL` instead makes the browser call the backend
directly; it is fixed at build time. Against a backend with `API_TOKEN` set,
the browser's direct requests are refused with 401, and it falls back to the
`/api` proxy.

## Data to back up

| Data | Default location | Notes |
|---|---|---|
| Pipeline database | `archai/backend/app/archai.sqlite` (`ARCHAI_DB_PATH`), plus its `-wal` and `-shm` files | Runs, transcriptions, mentions, links, evidence spans. Back up with `sqlite3 archai.sqlite ".backup /backup/archai.sqlite"` while running, or copy all three files with the server stopped. |
| Vector store | `archai/backend/.data/chroma` (`CHROMA_PERSIST_DIR`) | Can be rebuilt from the database (`POST /api/index/run/{id}` per run), but that costs an embedding call per chunk. Copy it with the server stopped. |
| OCR evidence log | `archai/backend/data/evidence/ocr_evidence.jsonl` (`OCR_EVIDENCE_PATH`) | Append-only record of every OCR call's raw and final text. |
| Authority cache | `archai/backend/app/.data/wikidata_cache.sqlite` (`ARCHAI_WIKIDATA_CACHE`) | A cache; losing it only costs requests. |
| Analytics database | `archai/backend/analytics.db` (`ANALYTICS_DB_PATH`) | Read only by the backend; written by whatever records visits. |
| Model weights | `archai/backend/models`, `archai/backend/weights` | Not data, but needed to start; see [`model_setup.md`](model_setup.md). |
| Layout tasks | `archai/backend/.tasks` (`TASK_BASE_DIR`) | Temporary: uploads and outputs, removed after `TASK_TTL_MINUTES`. No backup needed; give it room for the largest upload batch. |

## Health

```bash
curl -s http://127.0.0.1:8010/api/health
```

`status` is `ok`, `degraded` (HTTP 200: Ollama unreachable or no chat key) or
`unavailable` (HTTP 503: the pipeline database cannot be used). Point a load
balancer or monitor at it and alert on 503. The response carries no secrets,
paths or hosts. A dependency going down is logged once, not on every poll.

## Logs

The application logs to stderr through one root handler, at `LOG_LEVEL`
(default `INFO`), as text or, with `LOG_FORMAT=json`, one JSON object per line
with `ts`, `level`, `logger`, `message` and any extra fields. uvicorn's own
startup and access lines keep uvicorn's format. An unexpected error is
logged with its traceback and answered with a fixed sentence, never with the
exception's text.

## Retention

Runs accumulate: every analysis of a page adds one, with its chunks,
mentions and vectors. Nothing is deleted unless you ask.

1. Preview what a window would delete; this deletes nothing:

   ```bash
   cd archai/backend
   python scripts/prune_runs.py --older-than-days 180
   ```

   A run is kept if it was updated within the window or has not finished. It
   is also kept if it has no page identity, if it is its page's newest run,
   newest completed run or searched run, if it has a benchmark reference or an
   annotation, or if the thesis showcase names it. The report counts each
   reason.

2. Delete once, by hand: add `--apply` (and `--vacuum` to compact the
   database; that locks it while it runs).

3. Or let the server do it: set `RUN_RETENTION_DAYS` (0, the default, is off)
   and optionally `RUN_RETENTION_MAX_PER_SWEEP` (50). The first sweep after
   startup only logs what the next one, 15 minutes later, will delete. Each
   sweep deletes at most that many runs, oldest first, never an annotated
   one, and logs each deletion.

Take a backup before the first `--apply` or the first sweep.
