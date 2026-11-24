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
