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

## Related Backend

Run the paired API service from:

- `archai/backend`

See [`../README.md`](../README.md) for full workspace run instructions.
