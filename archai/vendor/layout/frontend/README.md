# ArchAI Vendor Layout Frontend

This is the Next.js frontend for the `archai/vendor/layout` workspace.

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
cd archai/vendor/layout/frontend
npm install
npm run dev
```

Default URL: `http://127.0.0.1:3000`

## Build

```bash
cd archai/vendor/layout/frontend
npm run build
npm run start
```

## Related Backend

Run the paired API service from:

- `archai/vendor/layout/backend`

See [`../README.md`](../README.md) for full workspace run instructions.
