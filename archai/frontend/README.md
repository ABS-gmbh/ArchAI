# ArchAI Frontend

The Next.js frontend of the ArchAI application (`archai/`).

It provides the document workspace UI:

- document/page navigation
- chat interaction panel
- OCR extraction flows
- evidence and provenance-oriented interactions

## Stack

- Next.js App Router
- React 19
- TypeScript
- Tailwind CSS

## Local Development

Node 20.9 or newer (CI uses 20.9.0).

```bash
cd archai/frontend
npm ci
npm run dev -- --hostname 127.0.0.1
```

Then open `http://127.0.0.1:3000/workspace`. `npm ci` installs exactly what
`package-lock.json` records (`npm install` may rewrite it). Without
`--hostname`, the dev server listens on every network interface: anyone on the
network could use the workspace, and through its `/api` proxy the backend.
`make frontend` in `archai/` does the same on port 3001.

## Checks

What CI runs (`.github/workflows/ci.yml`):

```bash
npx eslint --max-warnings=0
npx tsc --noEmit
npm run check:fonts      # the transcription fonts draw every character the recognisers output
npx next build --webpack
```

## Build

```bash
cd archai/frontend
npm run build
npm run start
```

## Backend address and token

The browser calls the backend directly on `localhost` (`http://127.0.0.1:8010`,
or `NEXT_PUBLIC_BACKEND_URL`), and otherwise through this server's `/api`
proxy (`src/app/api/[...path]/route.ts`). The proxy reads two variables when
the server starts:

- `BACKEND_URL`: where the backend listens (default `http://127.0.0.1:8010`).
- `BACKEND_API_TOKEN`: the backend's `API_TOKEN`, if it has one. The proxy adds
  it to each request; the browser never sees it. Never use a `NEXT_PUBLIC_`
  name for it: those are bundled into the browser's JavaScript, and the build
  refuses to run with `NEXT_PUBLIC_BACKEND_API_TOKEN` set.

With `BACKEND_API_TOKEN` set, the frontend itself must sit behind the
deployment's own login, since anyone who can reach the frontend can use the
backend through it. See [`docs/deployment.md`](../../docs/deployment.md#the-frontend).

## What the browser keeps

The open document is kept in the browser's IndexedDB (database
`archai-workspace`), so a reload, a closed tab reopened or a browser crash
does not lose it: its page images (as Blobs), each page's finished reading
(lines with their boxes, quality badges, recogniser, run id, steps, names
once loaded), the layout regions, the conversation, the zoom and the current
page. It is restored without any request beyond `/health` and
`/chat/models`.

- One document is kept: the one last open. Opening another replaces it, and
  "Close document" in the Document tab removes it.
- It is saved when a read or answer has finished and when the page or zoom
  changes, never while a read or question is under way.
- Where storage is unavailable, refused or full, the workspace works as
  before and the Document tab says the document will not survive a reload.
- The record has a schema version (`WORKSPACE_SCHEMA_VERSION` in
  `src/components/workspace/workspacePersistence.ts`). A record that fails
  its check, or comes from another version, is discarded with one console
  line. Raise the version when the record's shape changes, and add the step
  from the old one in `migrateRecord`.
- A run kept here may have been removed on the server (`RUN_RETENTION_DAYS`).
  The reading is still shown, and what needs the run (its names, if they
  were never loaded) says so.
- Not kept: the annotated layout image (a link to the server's task files,
  which expire), a failed read's message, and anything still running.

## Related Backend

Run the paired API service from:

- `archai/backend`

See [`../README.md`](../README.md) for full workspace run instructions.
