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
  DRAIN_MAX_MS,
  earlyAnswerPlan,
  hostRefusal,
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

// How long the proxy waits for the backend's "100 Continue" before sending a
// body anyway (RFC 9110 15.2.1: a client need not wait indefinitely). uvicorn
// answers at once on the same machine or network.
const CONTINUE_WAIT_MS = 1000;

let tokenRefusalLogged = false;
// The first cross-site refusal only: one line tells the operator, and a site
// that keeps trying does not fill the log.
let crossSiteLogged = false;
let hostLogged = false;

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
  const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8010";
  const target = backendTarget(backendUrl, req.nextUrl.pathname, req.nextUrl.search);
  if (!target) {
    return Response.json(
      { detail: "Not found." },
      { status: 404, headers: { [PROXY_STATUS_HEADER]: "path-refused", "cache-control": "no-store" } },
    );
  }
  const transport = target.protocol === "https:" ? https : http;
  const method = req.method.toUpperCase();

  // Before anything reaches the backend with the token: a host the frontend
  // does not know (DNS rebinding), then a page of another site.
  const options = originOptions();
  const wrongHost = hostRefusal(req.headers, options);
  if (wrongHost) {
    if (!hostLogged) {
      hostLogged = true;
      console.error(`[api proxy] ${req.nextUrl.pathname}: ${wrongHost.log}`);
    }
    return Response.json(
      { detail: wrongHost.detail },
      { status: 403, headers: { [PROXY_STATUS_HEADER]: "host-refused", "cache-control": "no-store" } },
    );
  }
  const crossSite = crossSiteRefusal(method, req.headers, options);
  if (crossSite) {
    if (!crossSiteLogged) {
      crossSiteLogged = true;
      console.error(`[api proxy] ${req.nextUrl.pathname}: ${crossSite.log}`);
    }
    return Response.json(
      { detail: crossSite.detail },
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
    // The request body as it is passed on, and how much of it has been read.
    let incoming: Readable | null = null;
    let incomingRead = 0;
    let incomingDone = true;

    /**
     * The backend answered before the whole body was passed on (its body
     * limit, its token gate). The proxy stopped reading it then, and the
     * client's next request on that connection waited behind the unread bytes
     * until the connection was reset. A small rest is read and discarded; for
     * a large one the answer closes the connection (earlyAnswerPlan).
     */
    const afterEarlyAnswer = (): "drain" | "close" | null => {
      if (!incoming || incomingDone) {
        return null;
      }
      const rest = incoming;
      rest.unpipe(upstream);
      const plan = earlyAnswerPlan(Number(req.headers.get("content-length") ?? Number.NaN), incomingRead);
      if (plan === "drain") {
        const giveUp = setTimeout(() => rest.destroy(), DRAIN_MAX_MS);
        rest.once("close", () => clearTimeout(giveUp));
        rest.resume();
      }
      return plan;
    };
    const withClose = (response: Response, plan: "drain" | "close" | null): Response => {
      if (plan === "close") {
        response.headers.set("connection", "close");
      }
      return response;
    };

    // A body is sent only once the backend asks for it (Expect: 100-continue).
    // Its body limit and token gate answer before they read anything; a body
    // already on its way then met a closed connection, and the proxy reported
    // the backend unreachable (503) for 14 of 30 refused 4 MB requests, where
    // the backend itself always answered 413 (2026-11-24). A backend that
    // never sends 100 Continue gets the body after CONTINUE_WAIT_MS anyway.
    const sendsBody = Boolean(req.body) && method !== "GET" && method !== "HEAD";
    if (sendsBody) {
      headers.expect = "100-continue";
    }
    const upstream = transport.request(target, { method, agent, headers }, (res) => {
      const plan = afterEarlyAnswer();
      if (isApiTokenRefusal(res)) {
        res.resume();
        settle(withClose(tokenRefused(method, req.nextUrl.pathname), plan));
        return;
      }
      const status = res.statusCode ?? 502;
      const noBody = method === "HEAD" || status === 204 || status === 304;
      if (noBody) {
        res.resume();
      }
      settle(
        withClose(
          new Response(noBody ? null : (Readable.toWeb(res) as ReadableStream<Uint8Array>), {
            status,
            statusText: res.statusMessage,
            headers: downstreamHeaders(res.headers, new URL(backendUrl)),
          }),
          plan,
        ),
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

    if (sendsBody && req.body) {
      const body = Readable.fromWeb(req.body as Parameters<typeof Readable.fromWeb>[0]);
      incoming = body;
      incomingDone = false;
      // Paused first: a "data" listener would otherwise start the flow before
      // the pipe exists, and a small body would be read and lost while the
      // proxy waits for 100 Continue. pipe() (or the drain) resumes it.
      body.pause();
      body.on("data", (chunk: Buffer) => {
        incomingRead += chunk.length;
      });
      body.on("end", () => {
        incomingDone = true;
      });
      body.on("error", (err) => upstream.destroy(err));
      let sending = false;
      const send = () => {
        clearTimeout(fallback);
        if (!sending && !settled) {
          sending = true;
          body.pipe(upstream);
        }
      };
      const fallback = setTimeout(send, CONTINUE_WAIT_MS);
      upstream.once("continue", send);
      upstream.once("response", () => clearTimeout(fallback));
      upstream.once("close", () => clearTimeout(fallback));
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
