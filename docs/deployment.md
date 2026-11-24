# Deploying the ArchAI backend

How to run `archai/backend` on a server, written from what the code does
today. The repository ships no container or service files; the commands below
are the ones the code supports. For first-time setup and the settings in
general, see [`archai/backend/README.md`](../archai/backend/README.md).

## Run one worker

Run a single uvicorn worker process:

```bash
cd archai/backend
uvicorn app.main:app --host 127.0.0.1 --port 8010 --workers 1 --no-server-header
```

`--no-server-header` drops uvicorn's `server: uvicorn` header, which tells a
client what software answers. Every response carries
`X-Content-Type-Options: nosniff`; the frontend's own headers are described
under [the frontend](#the-frontend).

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

## Memory

Measured on the evaluation VM (Ubuntu 24.04, CPU, 2026-11-24):

- With the layout process pool, the default on Linux (`MAX_POOL_WORKERS=3`), a
  running backend held about 5.3 GB: the main process about 1.0 GB and three
  pool workers of 1.4 to 1.5 GB each.
- With `ARCHAI_DISABLE_PROCESS_POOL=1`, the layout models run in the request's
  process. A benchmark reading the five reference pages that way stayed at 3
  to 4 GB, peaking at 5.5 GB on a full-size e-codices page (6130 x 8175 px).
  The full pipeline with indexing peaked at 6.7 GB.
- The two together, a backend with its pool and a benchmark run with its own
  pool, exhausted 15 GB without swap, and the kernel killed the benchmark.
  Plan for about 6 GB per backend process, plus headroom for the largest page
  you accept. Either give the host swap or keep to one process, since each
  process loads its own models.

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
| `ANALYTICS_USERNAME`, `ANALYTICS_PASSWORD`, `JWT_SECRET` | Analytics login refuses with 503 until the first two are set. Set `JWT_SECRET` too, at least 32 characters (with a shorter one the server starts, and logs an ERROR at startup naming the setting and the minimum): without it the tokens are signed with a random per-process secret, so sessions end at every restart, and startup logs an ERROR saying so. A session token is accepted only for `ANALYTICS_USERNAME`. |
| `ARCHAI_DB_PATH`, `CHROMA_PERSIST_DIR`, `ARCHAI_WIKIDATA_CACHE`, `OCR_EVIDENCE_PATH`, `TASK_BASE_DIR` | Their defaults are inside the code checkout. Point them at a data directory (the first four are the ones to back up). |
| `MAX_UPLOAD_MEGABYTES` | The default is 1024 MB per upload, and request bodies may be a third larger for base64. Set what your pages need. |
| `MAX_JSON_MEGABYTES` | Every request body but an upload to `/api/predict/single` or `/api/predict/batch`, 48 MB by default: enough for one A4 page at 600 dpi as a PNG data URL (36 MB). |
| `LOG_FORMAT=json` | One JSON object per log line, for a log collector. |

Leave `ENABLE_TEST_OCR_FIXTURES=false` (canned OCR answers for four test
images) and leave `ARCHAI_DEBUG_RAG` unset (it exposes `/api/rag/debug/*`).
Both are off by default. `/docs`, `/redoc` and `/openapi.json`, a map of
every route, are on while `API_TOKEN` is unset (development) and off once it is
set. `ENABLE_API_DOCS=true` switches them on with a token too, and then they
answer without it, since a browser opening `/docs` cannot send one; `false`
switches them off either way. A setting of the wrong type, or out of range, stops
startup with an error that names the variable.

Layout tasks keep their uploads and outputs in `TASK_BASE_DIR` (default
`archai/backend/.tasks`). It must be a writable directory: the server creates
it if missing and refuses to start, naming the variable, if it cannot write
there. It needs no backup: task files expire after `TASK_TTL_MINUTES` (60).

The other four data paths are checked at startup too: if the process could
not write `ARCHAI_DB_PATH`, `CHROMA_PERSIST_DIR`, `ARCHAI_WIKIDATA_CACHE` or
`OCR_EVIDENCE_PATH` (or create it, when it does not exist yet), the server
refuses to start with an error naming the variable. Startup then logs one
INFO line, `Data paths: ARCHAI_DB_PATH='…', CHROMA_PERSIST_DIR='…', …`, with
every path in use, ending with `ANALYTICS_DB_PATH='…' (read only)`, which the
server only reads and does not check; check the line after changing the
settings.

## What to put in front

The backend speaks plain HTTP. Unless `API_TOKEN` is set it has **no
authentication** except on the analytics routes: anyone who can reach it can
run OCR (spending SAIA credits and GPU or CPU time), read any run's text by
its id, write evidence spans and delete a run's vectors. Do not expose it
without a token or a proxy that authenticates.

With `API_TOKEN` set, every route except `/api/health` requires the token,
compared in constant time, and answers 401 otherwise. The token is checked
before any of the request body is read, so a client without it cannot make the
server receive or parse an upload. Send it as
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
- **Body size:** the backend refuses uploads to `/api/predict/single` and
  `/api/predict/batch` over `MAX_UPLOAD_MEGABYTES × 4/3 + 1 MB`, and every
  other body over `MAX_JSON_MEGABYTES` (48 MB), with 413, whatever its
  `Content-Type`. Set the proxy's limit for the two upload routes to the
  first, and for everything else to the second or lower. The largest JSON request is a page
  image sent as a base64 data URL: an A4 page scanned at 600 dpi makes a 36 MB
  body as PNG, 10.5 MB as JPEG.
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
- **Rate limiting:** the backend counts failed analytics logins in two ways
  within `ANALYTICS_LOGIN_WINDOW_SECONDS` (900): per client address, up to
  `ANALYTICS_LOGIN_MAX_FAILURES` (5), and per user name from any address, up
  to `ANALYTICS_LOGIN_MAX_FAILURES_PER_USERNAME` (10). Past either it answers
  429 until the oldest failure leaves the window, even to the right password;
  a successful login clears both counts. **What this means for the operator:**
  anyone who can reach the login can lock the analytics user out for up to 15
  minutes, by sending ten wrong passwords for its name, from any address, and
  can keep it locked by repeating that. Only the analytics page is affected;
  a restart of the backend clears the counts. That is the price of a limit a
  client cannot escape by changing its address. If it matters, put the
  analytics login behind the deployment's own login or an IP allow-list at
  the proxy. The per-address count needs the real client address (see client
  addresses above, and `TRUSTED_PROXY_HOPS` for the frontend below); otherwise
  every client shares the proxy's address and five wrong guesses by anyone
  lock that address for everyone. The OCR and chat routes are expensive and
  have no limit of their own; limit them at the proxy.

### The frontend

`archai/frontend` is a Next.js app: `npm ci`, `npm run build`, `npm start`
(Node 20.9 or newer). `npm start` listens on every interface, port 3000; with
the reverse proxy on the same host, bind it to localhost:
`npm start -- --hostname 127.0.0.1`. On a server hostname the browser sends its requests to
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
| `TRUST_FORWARDED_HOST` | `true` if a reverse proxy in front rewrites `Host` (nginx's default `proxy_set_header Host $proxy_host`) and sets `X-Forwarded-Host` to the public host. Set it only if the frontend cannot be reached except through that proxy. |
| `FRONTEND_ORIGINS` | Further origins the frontend is opened at, comma-separated (`https://archai.example`), when neither `Host` nor a trusted `X-Forwarded-Host` names them. |
| `TRUSTED_PROXY_HOPS` | How many reverse proxies in front of the frontend append the client's address to `X-Forwarded-For` (nginx: `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`). Default 0. Set it only if the frontend cannot be reached except through them. |

Never name it `NEXT_PUBLIC_BACKEND_API_TOKEN`: a `NEXT_PUBLIC_` variable is
copied into the JavaScript every browser downloads. The build and `npm start`
refuse to run with that variable set.

If the backend has `API_TOKEN` set and `BACKEND_API_TOKEN` is missing or
different, the backend refuses the proxy's requests. The proxy then answers
401 with "The frontend server could not authenticate to the backend. The
frontend server's log says which setting to check.", which the workspace shows
where the request failed; the browser is not told whether a token is set. The
frontend server's log gets one line naming the setting: `BACKEND_API_TOKEN is
not set`, or that the backend refused it. `/api/health` needs no token, so the
status banner stays green in that case.

The proxy does not pass the browser's `Origin` to the backend, nor the
backend's `Access-Control-*` headers to the browser: the browser talks to the
frontend, same-origin, and needs no CORS answer.

With `BACKEND_API_TOKEN` set, **the frontend must itself sit behind the
deployment's own login** (single sign-on, basic auth at the reverse proxy, a
VPN), since anyone who can reach the frontend can use the backend through it.
The token then only keeps the backend from being used without the frontend.

`NEXT_PUBLIC_BACKEND_URL` instead makes the browser call the backend
directly; it is fixed at build time. Against a backend with `API_TOKEN` set,
the browser's direct requests are refused with 401, and it falls back to the
`/api` proxy.

The frontend's server sends these headers on every response
(`next.config.ts`):
- `X-Content-Type-Options: nosniff`;
- `Referrer-Policy: same-origin`;
- `X-Frame-Options: DENY`;
- a `Permissions-Policy` that turns off camera, microphone, geolocation,
  payment and the other device features the workspace does not use;
- in a production build, a `Content-Security-Policy`.

The policy allows scripts, styles, fonts and requests only from the page's
own origin, plus the backend's direct URL (`NEXT_PUBLIC_BACKEND_URL`, or
`127.0.0.1:8010`). It allows images from the page's origin and from
`data:`/`blob:` URLs, and forbids plugins and framing. Inline scripts and
styles are still allowed, because Next writes its bootstrap scripts inline.

`NEXT_PUBLIC_BACKEND_URL` is read when the frontend is built, so set it
before `npm run build`. Set `Strict-Transport-Security` at the reverse proxy
that terminates TLS.

#### Behind a reverse proxy

A reverse proxy in front of the frontend (TLS, a login) changes two things the
frontend's `/api` proxy looks at: the `Host` header and the client's address.
Tell it about them with these settings of the Next server:

- `TRUST_FORWARDED_HOST=true` when the proxy rewrites `Host` (nginx does by
  default: `proxy_set_header Host $proxy_host`) and sets `X-Forwarded-Host`
  to the public host; or `FRONTEND_ORIGINS=https://archai.example`, the
  public origin itself. Current browsers do not need either: they mark the
  workspace's own requests `Sec-Fetch-Site: same-origin`, which the frontend
  accepts whatever `Host` says. They are still needed for browsers that send
  no `Sec-Fetch-Site` (Chrome before 76, Firefox before 90, Safari before
  16.4), and when the proxy serves the workspace's pages and its `/api` from
  different origins, so the browser does not see them as one. Without them, in those cases, every
  change the workspace makes (reading a page, a layout run) is refused with
  403.
- `TRUSTED_PROXY_HOPS=1` when the proxy appends the client's address to
  `X-Forwarded-For`, so the backend logs and rate-limits real addresses.

Set the first and last only if the frontend cannot be reached except through
the proxy. An nginx location that needs `TRUST_FORWARDED_HOST=true` and
`TRUSTED_PROXY_HOPS=1`:

```nginx
location / {
    proxy_pass http://127.0.0.1:3000;
    proxy_set_header X-Forwarded-Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

(With `proxy_set_header Host $host;` as well, `Host` is the public host and
`TRUST_FORWARDED_HOST` is not needed.)

The proxy refuses with 403 a `POST`, `PUT`, `PATCH` or `DELETE` that comes
from another site: it adds the API token to whatever reaches it, so without
the check any site a user visits could make their browser send uploads and run
reservations through it. The browser says where a request comes from in
`Sec-Fetch-Site`, a header no page can set: `same-origin` is accepted,
`cross-site` is refused. With `same-site`, `none` or no `Sec-Fetch-Site` at
all, the request's `Origin` must be the frontend's own: the host in its `Host`
header, which is right when the browser talks to `npm start` directly, by name
or by IP address, or a trusted `X-Forwarded-Host`, or one of
`FRONTEND_ORIGINS`. A refused request is answered "Refused: this request came
from another site … see docs/deployment.md, 'Behind a reverse proxy'", and the
frontend server's log says once which `Origin` and `Host` it compared.
Requests without `Origin` (scripts, server-side clients) are not affected.

The proxy never passes on what a client says about its own address
(`X-Forwarded-For`, `Forwarded`, `X-Real-IP`, `X-Forwarded-Host`/`-Proto`/`-Port`).
Next fills `X-Forwarded-For` from the connection only when the client sent
none, and the proxy cannot see the connection, so by default the backend gets
no client address from it, and every browser counts as the frontend server's
address (127.0.0.1) in the backend's log and login limit. With
`TRUSTED_PROXY_HOPS=1` and one reverse proxy in front that appends the address
it saw, the proxy sends the backend that last entry, which a client cannot
choose; with two proxies in a chain, `2`, and so on.

### Checking a deployment

`archai/backend/scripts/smoke_api.py` drives every route over HTTP and exits 1
if one answers differently from what it expects. Run it against a backend on
**scratch data**, not a live one: it writes runs, tasks and evidence spans,
and expects `MAX_UPLOAD_MEGABYTES=2`, `MAX_JSON_MEGABYTES=2`, no chat key, no Ollama and no layout
models. Its docstring has the command for such a backend. Then run it
directly, with the token, and through the frontend, without it:

```bash
cd archai/backend
python scripts/smoke_api.py --base http://127.0.0.1:8010 --db /tmp/smoke/archai.sqlite --token "$API_TOKEN"
python scripts/smoke_api.py --base http://127.0.0.1:3000 --db /tmp/smoke/archai.sqlite
```

`--token` adds the authentication checks: no token, a wrong one, the token
as `?token=` on the two event streams.

## Data to back up

| Data | Default location | Notes |
|---|---|---|
| Pipeline database | `archai/backend/app/archai.sqlite` (`ARCHAI_DB_PATH`), plus its `-wal` and `-shm` files | Runs, transcriptions, mentions, links, evidence spans. Not `archai.sqlite.lock`, the one-worker lock: it is recreated at startup. |
| Vector store | `archai/backend/.data/chroma` (`CHROMA_PERSIST_DIR`) | Can be rebuilt from the database (`POST /api/index/run/{id}` per run), but that costs an embedding call per chunk. The server rewrites its files whenever it opens it, even to read, so copy it only with the server stopped. |
| OCR evidence log | `archai/backend/data/evidence/ocr_evidence.jsonl` (`OCR_EVIDENCE_PATH`) | Append-only record of every OCR call's raw and final text. |
| Authority cache | `archai/backend/app/.data/wikidata_cache.sqlite` (`ARCHAI_WIKIDATA_CACHE`) | A cache; losing it only costs requests. |
| Analytics database | `archai/backend/analytics.db` (`ANALYTICS_DB_PATH`) | Read only by the backend; written by whatever records visits. |
| Model weights | `archai/backend/models` (`MODEL_DIR`), `weights/kraken_models` at the repository root | Not data, and the server starts without them, but pages are then read with GLM-OCR only; see [`model_setup.md`](model_setup.md). |
| Layout tasks | `archai/backend/.tasks` (`TASK_BASE_DIR`) | Temporary: uploads and outputs, removed after `TASK_TTL_MINUTES`. No backup needed; give it room for the largest upload batch. |

### Backing up

The vector store can only be copied with the server stopped, so stop it for
the backup:

1. Stop the backend (and with it every process using the data; see
   [Run one worker](#run-one-worker)).
2. Copy, keeping the names: the pipeline database with its `-wal` and `-shm`
   files if they exist, the whole `CHROMA_PERSIST_DIR` directory, the OCR
   evidence log, the authority cache and the analytics database, from the
   paths the `Data paths:` startup line names.
3. Start the backend.

Without a stop, `sqlite3 <file> ".backup <target>"` takes a consistent copy of
each SQLite file while the server runs: the pipeline database, the authority
cache (`wikidata_cache.sqlite`) and the analytics database. A plain copy of a
SQLite file in use can be torn. The evidence log is appended to: a copy taken
while a page is being read can end in half a line. The vector store still
needs the stop, or a rebuild after restoring.

### Restoring

1. Stop the backend.
2. At each target path, remove what is there, and for the pipeline database
   also `archai.sqlite-wal` and `archai.sqlite-shm`. A `-wal` left by a server
   that stopped without cleaning up is replayed onto whatever database is put
   next to it: in a test, a backup of 5 runs restored next to a crashed
   server's `-wal` opened as 52 runs, and `PRAGMA integrity_check` still said
   `ok`.
3. Put the backed-up files there (a database taken with `.backup` is one
   file), and point the settings at them if the paths changed.
4. Start the backend and check the `Data paths:` line names the restored
   files. Open a few runs (`GET /api/ocr/trace/{run_id}`) and run a search.

Layout tasks are not backed up: a task finished before the backup answers 404
afterwards. If you do move `TASK_BASE_DIR` with its tasks, the backend finds
their files in the new place.

This procedure was checked on 2026-11-24: an installation's data made with the
code of 2026-11-09 (before pull request 90), upgraded, then backed up both
ways above and restored into a new, empty directory,
answered every run, transcription, name, authority link, evidence span,
search and analytics login as before. The only differences were the expected
ones: its layout task was gone.

## Upgrading and rolling back

[upgrade-notes.md](upgrade-notes.md) lists what changes from the version of
2026-11-09: what to check before the first start, the new settings and their
defaults, and the changes users and API scripts will see.

Back up first. The upgrade to this version needs no migration: the pipeline
database's schema is unchanged, and starting the new version rewrites none of
it. Starting either version rewrites the vector store's index files (the
same vectors). New files appear next to the database: `archai.sqlite.lock`,
which the older version ignores.

Rolling back to the version of 2026-11-09 works on data the new version has
used: it starts, and answers the same runs, names, links, spans and searches.
What it does not have, it drops without a word:
- **`API_TOKEN`**: the older version has no token. Rolled back, the API answers
  anyone who can reach it; keep it on localhost or behind a proxy that
  authenticates.
- **`ENABLE_API_DOCS`**: `/docs` is always on.
- **`TASK_BASE_DIR`**: the older version keeps layout tasks in
  `archai/backend/.tasks`, whatever the setting says.
- **Run reservations** (`POST /api/ocr/runs`): a reserved run the new version
  left behind stays `RESERVED`, and the older version's event stream for it
  stays open until its one-hour limit.

## Health

```bash
curl -s http://127.0.0.1:8010/api/health
```

`status` is `ok`, `degraded` (HTTP 200: Ollama unreachable, no chat key, or
layout weights or line recogniser not installed) or `unavailable` (HTTP 503:
the pipeline database cannot be used). The workspace shows degraded checks in
a banner a user can hide; it cannot hide an unavailable backend. Point a load
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
