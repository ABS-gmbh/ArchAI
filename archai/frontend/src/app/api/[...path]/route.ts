/**
 * Same-origin proxy from the browser to the backend: /api/... is sent to
 * `${BACKEND_URL}/api/...`.
 *
 * With BACKEND_API_TOKEN set (server-side only, never NEXT_PUBLIC_), the proxy
 * adds it as X-API-Token, so the browser works against a backend that has
 * API_TOKEN set without ever holding the token. The client's own
 * Authorization header (an analytics session) is passed on unchanged.
 *
 * Bodies are streamed both ways: an upload is not read into memory first, and
 * a server-sent event reaches the browser as soon as the backend sends it.
 * node:http is used rather than fetch(), whose 300 s headers timeout would cut
 * off a full-page OCR request, which answers only when it has finished.
 */
import http from "node:http";
import https from "node:https";
import { Readable } from "node:stream";

import type { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Read by lib/api/client.ts to tell "the proxy could not reach the backend" from a backend error. */
const PROXY_STATUS_HEADER = "x-archai-proxy";

// Hop-by-hop headers (RFC 9110 7.6.1) belong to one connection, not the request.
const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

const TOKEN_HEADER = "x-api-token";

function backendTarget(req: NextRequest): URL {
  const base = (process.env.BACKEND_URL || "http://127.0.0.1:8010").replace(/\/+$/, "");
  return new URL(`${base}${req.nextUrl.pathname}${req.nextUrl.search}`);
}

function upstreamHeaders(req: NextRequest, target: URL): http.OutgoingHttpHeaders {
  const headers: http.OutgoingHttpHeaders = {};
  req.headers.forEach((value, name) => {
    if (!HOP_BY_HOP.has(name) && name !== "host") {
      headers[name] = value;
    }
  });
  headers.host = target.host;
  const token = process.env.BACKEND_API_TOKEN;
  if (token) {
    headers[TOKEN_HEADER] = token;
  }
  return headers;
}

function downstreamHeaders(res: http.IncomingMessage): Headers {
  const headers = new Headers();
  for (const [name, value] of Object.entries(res.headers)) {
    if (value === undefined || HOP_BY_HOP.has(name) || name === TOKEN_HEADER) {
      continue;
    }
    for (const item of Array.isArray(value) ? value : [value]) {
      headers.append(name, item);
    }
  }
  if ((headers.get("content-type") || "").includes("text/event-stream")) {
    // no-transform keeps Next's gzip from holding events back until a buffer
    // fills; X-Accel-Buffering does the same for nginx in front of Next.
    headers.set("cache-control", "no-cache, no-transform");
    headers.set("x-accel-buffering", "no");
  }
  return headers;
}

function unreachable(): Response {
  return Response.json(
    { detail: "The frontend could not reach the backend. Check that it is running and that BACKEND_URL points at it." },
    { status: 503, headers: { [PROXY_STATUS_HEADER]: "backend-unreachable", "cache-control": "no-store" } },
  );
}

async function proxy(req: NextRequest): Promise<Response> {
  const target = backendTarget(req);
  const transport = target.protocol === "https:" ? https : http;
  const method = req.method.toUpperCase();

  return new Promise<Response>((resolve) => {
    let settled = false;
    const settle = (response: Response) => {
      if (!settled) {
        settled = true;
        resolve(response);
      }
    };

    const upstream = transport.request(target, { method, headers: upstreamHeaders(req, target) }, (res) => {
      const status = res.statusCode ?? 502;
      const noBody = method === "HEAD" || status === 204 || status === 304;
      if (noBody) {
        res.resume();
      }
      settle(
        new Response(noBody ? null : (Readable.toWeb(res) as ReadableStream<Uint8Array>), {
          status,
          statusText: res.statusMessage,
          headers: downstreamHeaders(res),
        }),
      );
    });

    upstream.on("error", (err: NodeJS.ErrnoException) => {
      if (!settled) {
        console.error(`[api proxy] ${method} ${req.nextUrl.pathname}: backend unreachable (${err.code ?? err.message})`);
        settle(unreachable());
      }
    });

    // The browser went away (closed tab, aborted upload, closed stream): stop the backend request too.
    req.signal.addEventListener("abort", () => upstream.destroy(), { once: true });

    if (req.body && method !== "GET" && method !== "HEAD") {
      const body = Readable.fromWeb(req.body as Parameters<typeof Readable.fromWeb>[0]);
      body.on("error", (err) => upstream.destroy(err));
      body.pipe(upstream);
    } else {
      upstream.end();
    }
  });
}

export const GET = proxy;
export const HEAD = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
