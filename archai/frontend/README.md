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

```bash
cd archai/frontend
npm install
npm run dev
```

Default URL: `http://127.0.0.1:3000`

## Build

```bash
cd archai/frontend
npm run build
npm run start
```

## Related Backend

Run the paired API service from:

- `archai/backend`

See [`../README.md`](../README.md) for full workspace run instructions.
