/**
 * Same-origin proxy from the browser to the backend: /api/... is sent to
 * `${BACKEND_URL}/api/...`.
 *
 * With BACKEND_API_TOKEN set (server-side only, never NEXT_PUBLIC_), the proxy
 * adds it as X-API-Token, so the browser works against a backend that has
 * API_TOKEN set without ever holding the token. The client's own
 * Authorization header (an analytics session) is passed on unchanged. What the
 * client says about its address (X-Forwarded-For, Forwarded, X-Real-IP) is
 * not: the backend sees the address a trusted reverse proxy recorded, or none
 * (TRUSTED_PROXY_HOPS, proxyRules.ts).
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

import {
  backendTarget,
  crossSiteRefusal,
  downstreamHeaders,
  originOptions,
  tokenRefusal,
  trustedProxyHops,
  upstreamHeaders,
} from "./proxyRules";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** Read by lib/api/client.ts to tell "the proxy could not reach the backend" from a backend error. */
const PROXY_STATUS_HEADER = "x-archai-proxy";

// One connection per request. Node's default agent keeps idle connections to
// reuse them, and uvicorn closes an idle connection after 5 s: a request sent
// on a connection the backend is closing at that moment is reset, and a POST
// (a page upload) cannot be sent again. A live read was lost this way. A new
// connection to a backend on the same host or network costs a handshake.
const NO_REUSE = { http: new http.Agent({ keepAlive: false }), https: new https.Agent({ keepAlive: false }) };

function unreachable(): Response {
  return Response.json(
    { detail: "The frontend could not reach the backend. Check that it is running and that BACKEND_URL points at it." },
    { status: 503, headers: { [PROXY_STATUS_HEADER]: "backend-unreachable", "cache-control": "no-store" } },
  );
}

/**
 * The backend's own answer to a missing or wrong API token (app/auth.py), as
 * opposed to an analytics 401, which carries no such challenge.
 */
function isApiTokenRefusal(res: http.IncomingMessage): boolean {
  return res.statusCode === 401 && /realm="archai"/i.test(String(res.headers["www-authenticate"] ?? ""));
}

let tokenRefusalLogged = false;

function tokenRefused(method: string, path: string): Response {
  const refusal = tokenRefusal(Boolean(process.env.BACKEND_API_TOKEN));
  if (!tokenRefusalLogged) {
    tokenRefusalLogged = true;
    console.error(`[api proxy] ${method} ${path}: ${refusal.log}`);
  }
  return Response.json(
    { detail: refusal.detail },
    { status: refusal.status, headers: { [PROXY_STATUS_HEADER]: "token-refused", "cache-control": "no-store" } },
  );
}

async function proxy(req: NextRequest): Promise<Response> {
  const target = backendTarget(process.env.BACKEND_URL || "http://127.0.0.1:8010", req.nextUrl.pathname, req.nextUrl.search);
  if (!target) {
    return Response.json(
      { detail: "Not found." },
      { status: 404, headers: { [PROXY_STATUS_HEADER]: "path-refused", "cache-control": "no-store" } },
    );
  }
  const transport = target.protocol === "https:" ? https : http;
  const method = req.method.toUpperCase();

  const crossSite = crossSiteRefusal(method, req.headers, originOptions());
  if (crossSite) {
    return Response.json(
      { detail: crossSite },
      { status: 403, headers: { [PROXY_STATUS_HEADER]: "cross-site-refused", "cache-control": "no-store" } },
    );
  }

  return new Promise<Response>((resolve) => {
    let settled = false;
    const settle = (response: Response) => {
      if (!settled) {
        settled = true;
        resolve(response);
      }
    };

    const agent = target.protocol === "https:" ? NO_REUSE.https : NO_REUSE.http;
    const headers = upstreamHeaders(req.headers, target.host, {
      trustedProxyHops: trustedProxyHops(),
      token: process.env.BACKEND_API_TOKEN || "",
    });
    const upstream = transport.request(target, { method, agent, headers }, (res) => {
      if (isApiTokenRefusal(res)) {
        res.resume();
        settle(tokenRefused(method, req.nextUrl.pathname));
        return;
      }
      const status = res.statusCode ?? 502;
      const noBody = method === "HEAD" || status === 204 || status === 304;
      if (noBody) {
        res.resume();
      }
      settle(
        new Response(noBody ? null : (Readable.toWeb(res) as ReadableStream<Uint8Array>), {
          status,
          statusText: res.statusMessage,
          headers: downstreamHeaders(res.headers),
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
