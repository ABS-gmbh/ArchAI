// Rules of the /api proxy (route.ts) that do not need a request or a socket,
// so the unit tests can check them (proxyRules.test.ts).
import type http from "node:http";

// Hop-by-hop headers (RFC 9110 7.6.1) belong to one connection, not the request.
export const HOP_BY_HOP = new Set([
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

export const TOKEN_HEADER = "x-api-token";

// What a client says about where it is. The backend's uvicorn trusts
// X-Forwarded-For from 127.0.0.1, where this proxy runs, and keys the analytics
// login limit on it: passed on as the client sent them, these let a client
// choose its own address for every guess.
const CLIENT_FORWARDING = new Set([
  // Not forwarding, but the browser's: the backend's CORS answer to it was
  // passed back to the browser from the frontend's own origin.
  "origin",
  "forwarded",
  "x-forwarded-for",
  "x-forwarded-host",
  "x-forwarded-port",
  "x-forwarded-proto",
  "x-real-ip",
]);

/** TRUSTED_PROXY_HOPS: how many reverse proxies in front of the frontend append to X-Forwarded-For. */
export function trustedProxyHops(value: string | undefined = process.env.TRUSTED_PROXY_HOPS): number {
  const hops = Number.parseInt(String(value ?? "").trim(), 10);
  return Number.isFinite(hops) && hops > 0 ? hops : 0;
}

/**
 * The client's address as the nearest trusted reverse proxy saw it, or null.
 *
 * Each trusted proxy appends the address it received the request from, so
 * the entry `hops` places from the end is the client's; anything before it is
 * the client's own say. With no trusted proxy there is no such entry: Next
 * fills in X-Forwarded-For from the connection only when the client sent none,
 * and a route handler cannot see the connection, so the entries cannot be told
 * from the client's own.
 */
export function clientAddress(forwardedFor: string | null, hops: number): string | null {
  if (hops <= 0) {
    return null;
  }
  const entries = String(forwardedFor ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  return entries.length >= hops ? entries[entries.length - hops] : null;
}

export interface UpstreamOptions {
  trustedProxyHops: number;
  token: string;
}

export function upstreamHeaders(request: Headers, backendHost: string, options: UpstreamOptions): http.OutgoingHttpHeaders {
  const headers: http.OutgoingHttpHeaders = {};
  request.forEach((value, name) => {
    if (!HOP_BY_HOP.has(name) && !CLIENT_FORWARDING.has(name) && name !== "host") {
      headers[name] = value;
    }
  });
  headers.host = backendHost;
  const address = clientAddress(request.get("x-forwarded-for"), options.trustedProxyHops);
  if (address) {
    headers["x-forwarded-for"] = address;
  }
  if (options.token) {
    headers[TOKEN_HEADER] = options.token;
  }
  return headers;
}

export interface OriginOptions {
  /** TRUST_FORWARDED_HOST=true: the reverse proxy in front sets X-Forwarded-Host, and a client cannot. */
  trustForwardedHost: boolean;
  /** FRONTEND_ORIGINS: further origins the frontend is opened at, e.g. "https://archai.example". */
  allowedOrigins: string[];
}

export function originOptions(env: NodeJS.ProcessEnv = process.env): OriginOptions {
  return {
    trustForwardedHost: String(env.TRUST_FORWARDED_HOST ?? "").trim().toLowerCase() === "true",
    allowedOrigins: String(env.FRONTEND_ORIGINS ?? "")
      .split(",")
      .map((origin) => origin.trim().replace(/\/+$/, "").toLowerCase())
      .filter(Boolean),
  };
}

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

function withoutDefaultPort(host: string): string {
  return host.trim().toLowerCase().replace(/:(80|443)$/, "");
}

/**
 * Why a request must be refused as coming from another site, or null.
 *
 * The proxy adds the API token to whatever reaches it, so a page on another
 * site that makes the user's browser POST here (a form, a bodiless fetch) would
 * be acting with the frontend's rights. A browser says where a request comes
 * from in Origin (and Sec-Fetch-Site): a state-changing request whose origin is
 * not the frontend's own is refused. A request with no Origin is not a
 * browser's cross-site request, and is let through.
 */
export function crossSiteRefusal(method: string, headers: Headers, options: OriginOptions): string | null {
  if (SAFE_METHODS.has(method.toUpperCase())) {
    return null;
  }
  const refusal = "Refused: this request came from another site. The frontend accepts changes only from its own pages.";
  if ((headers.get("sec-fetch-site") || "").toLowerCase() === "cross-site") {
    return refusal;
  }
  const origin = headers.get("origin");
  if (origin === null) {
    return null;
  }
  let parsed: URL;
  try {
    parsed = new URL(origin);
  } catch {
    return refusal; // "null" (a sandboxed frame, a file) or garbage
  }
  if (options.allowedOrigins.includes(parsed.origin.toLowerCase())) {
    return null;
  }
  const hosts = [headers.get("host") || ""];
  if (options.trustForwardedHost) {
    hosts.push((headers.get("x-forwarded-host") || "").split(",")[0]);
  }
  const originHost = withoutDefaultPort(parsed.host);
  return hosts.some((host) => host && withoutDefaultPort(host) === originHost) ? null : refusal;
}

/**
 * The backend URL for a request to the proxy, or null when the path would
 * leave the backend's /api: new URL() resolves dot segments, encoded ones
 * included ("/api/%2e%2e/docs" is "/docs").
 */
export function backendTarget(backendUrl: string, pathname: string, search: string): URL | null {
  const base = backendUrl.replace(/\/+$/, "");
  const target = new URL(`${base}${pathname}${search}`);
  const apiRoot = `${new URL(base).pathname.replace(/\/+$/, "")}/api/`;
  return target.pathname.startsWith(apiRoot) ? target : null;
}

export function downstreamHeaders(res: http.IncomingHttpHeaders): Headers {
  const headers = new Headers();
  for (const [name, value] of Object.entries(res)) {
    // "server" names the backend's software (uvicorn): nothing a browser needs.
    // The backend's CORS headers answer an origin the browser never sees: the
    // browser talks to the frontend, same-origin.
    if (
      value === undefined ||
      HOP_BY_HOP.has(name) ||
      name === TOKEN_HEADER ||
      name === "server" ||
      name.startsWith("access-control-")
    ) {
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

/**
 * The backend refused the token this server sends (or the absence of one).
 * The browser is told only that the frontend could not authenticate, the same
 * whether a token is configured or not; which setting is missing or wrong is
 * for the operator, in the server log.
 */
export function tokenRefusal(configured: boolean): { status: number; detail: string; log: string } {
  return {
    status: 401,
    detail:
      "The frontend server could not authenticate to the backend. The frontend server's log says which setting to check.",
    log: configured
      ? "the backend answered 401 to BACKEND_API_TOKEN. Set BACKEND_API_TOKEN on the frontend server to the backend's API_TOKEN, and restart it."
      : "the backend answered 401 to a request without a token (BACKEND_API_TOKEN is not set). Set BACKEND_API_TOKEN on the frontend server to the backend's API_TOKEN, and restart it.",
  };
}
