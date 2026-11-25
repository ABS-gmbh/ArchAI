export class ApiError extends Error {
  status: number;
  detail: string;
  /** The wait a 429 asked for (the provider's Retry-After), in seconds; null without one. */
  retryAfterSeconds: number | null;

  // Fields, not constructor parameter properties: Node's type stripping (npm test) does not take those.
  constructor(status: number, detail: string, retryAfterSeconds: number | null = null) {
    super(detail);
    this.status = status;
    this.detail = detail;
    this.retryAfterSeconds = retryAfterSeconds;
    this.name = "ApiError";
  }
}

/** A response's Retry-After in seconds (a number or an HTTP date), or null. */
export function retryAfterSeconds(res: { headers: { get(name: string): string | null } }, now = Date.now()): number | null {
  const raw = (res.headers.get("retry-after") ?? "").trim();
  if (!raw) {
    return null;
  }
  const seconds = /^\d+(\.\d+)?$/.test(raw) ? Number(raw) : (Date.parse(raw) - now) / 1000;
  return Number.isFinite(seconds) && seconds > 0 ? Math.ceil(seconds) : null;
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

/** Names the address that was tried, which NEXT_PUBLIC_BACKEND_URL may have set, not a fixed one. */
function unreachableMessage(direct: string): string {
  let origin = "";
  try {
    origin = /^https?:\/\//i.test(direct) ? new URL(direct).origin : "";
  } catch {
    origin = "";
  }
  return origin
    ? `The backend did not answer at ${origin}. Check that it is running there, then try again.`
    : "The backend did not answer. Check that it is running, then try again.";
}

type Transport = "direct" | "proxy";

// Which way reached the backend last time. Requests that change something are
// sent once, down this route only.
let knownTransport: Transport | null = null;

function isJsonResponse(res: Response): boolean {
  return (res.headers.get("content-type") || "").toLowerCase().includes("application/json");
}

function isReadOnly(init?: RequestInit): boolean {
  const method = (init?.method || "GET").toUpperCase();
  return method === "GET" || method === "HEAD";
}

/**
 * The direct route reached a backend that requires API_TOKEN, which the
 * browser does not hold. The backend refused before doing anything, so the
 * request may be sent again through the proxy, which adds the token.
 */
class DirectRouteNeedsToken extends Error {
  constructor() {
    super("the backend requires an API token; use the proxy");
    this.name = "DirectRouteNeedsToken";
  }
}

/** The backend's answer to a missing API token (app/auth.py), as opposed to an analytics 401. */
async function isApiTokenRefusal(res: Response): Promise<boolean> {
  if (res.status !== 401) {
    return false;
  }
  if (/realm="archai"/i.test(res.headers.get("www-authenticate") || "")) {
    return true;
  }
  // Cross-origin the browser hides WWW-Authenticate (the backend's CORS
  // exposes only Content-Disposition), so fall back to the detail it sends.
  const detail = await readErrorDetail(res.clone()).catch(() => "");
  return detail.startsWith("Authentication required");
}

/**
 * A fetch through one route. A proxy that cannot reach the backend counts as a
 * network failure; a direct route refused for want of the API token throws
 * DirectRouteNeedsToken.
 */
async function fetchVia(transport: Transport, path: string, init?: RequestInit): Promise<Response> {
  const direct = apiUrl(path);
  const proxied = `/api${path}`;
  if (transport === "direct") {
    const res = await fetch(direct, init);
    if (direct !== proxied && (await isApiTokenRefusal(res))) {
      throw new DirectRouteNeedsToken();
    }
    return res;
  }
  const res = await fetch(proxied, init);
  if (
    direct !== proxied &&
    (res.headers.get("x-archai-proxy") === "backend-unreachable" || (res.status >= 500 && !isJsonResponse(res)))
  ) {
    // src/app/api/[...path]/route.ts marks its own 503; a plain-text 5xx comes from Next itself.
    throw new TypeError("proxy could not reach the backend");
  }
  return res;
}

/**
 * fetch() against the backend: the direct URL, or the same-origin /api proxy
 * when the direct one cannot be reached (backend down, CORS).
 *
 * Only GET and HEAD are retried through the other route. A POST whose direct
 * request failed may already have reached the backend - the connection can
 * drop after the server started work - and sending it again through the proxy
 * would start a second OCR run or a second paid completion. So a request that
 * changes something goes once, down the route that last worked; when no route
 * is known yet, /health is probed first to find one.
 *
 * A backend with API_TOKEN set answers the direct route 401: the browser does
 * not hold the token. That request is sent again through the proxy, which
 * adds the token - a POST too, since a 401 means the backend did nothing with
 * it - and the proxy is used from then on. (Request bodies here are strings or
 * FormData, which can be sent twice; a ReadableStream body could not.)
 *
 * When the backend cannot be reached, this throws ApiError(503) saying so,
 * instead of "Failed to fetch". Aborts are re-thrown unchanged.
 */
export async function fetchBackend(path: string, init?: RequestInit): Promise<Response> {
  const unreachable = () => new ApiError(503, unreachableMessage(apiUrl(path)));

  if (isReadOnly(init)) {
    const order: Transport[] = knownTransport === "proxy" ? ["proxy", "direct"] : ["direct", "proxy"];
    for (const transport of order) {
      try {
        const res = await fetchVia(transport, path, init);
        knownTransport = transport;
        return res;
      } catch (err) {
        if (isAbortError(err)) {
          throw err;
        }
      }
    }
    knownTransport = null;
    throw unreachable();
  }

  if (knownTransport === null) {
    try {
      // Finds the route (and records it) without sending the request itself.
      await fetchBackend("/health", { signal: init?.signal });
    } catch (err) {
      if (isAbortError(err)) {
        throw err;
      }
      throw unreachable();
    }
  }
  const transport: Transport = knownTransport ?? "direct";
  try {
    return await fetchVia(transport, path, init);
  } catch (err) {
    if (isAbortError(err)) {
      throw err;
    }
    if (err instanceof DirectRouteNeedsToken) {
      // Refused before any work was done, so sending it once more is safe.
      knownTransport = "proxy";
      return fetchBackend(path, init);
    }
    // Not retried: the request may have been received. The next request probes again.
    knownTransport = null;
    throw new ApiError(
      503,
      `The connection to the backend was lost during the request. It was not sent again, because the backend may already be working on it. Check the result before retrying.`,
    );
  }
}

/**
 * The message of a failed response: FastAPI's `detail` when it is a string, its
 * `error` when it is an object carrying one, the messages of a validation
 * error list, else the body text, else the status text. The body is read once.
 */
export async function readErrorDetail(res: Response): Promise<string> {
  const raw = await res.text().catch(() => "");
  let body: unknown = null;
  try {
    body = raw ? JSON.parse(raw) : null;
  } catch {
    body = null;
  }
  if (body && typeof body === "object") {
    const { detail, message } = body as { detail?: unknown; message?: unknown };
    if (typeof detail === "string" && detail.trim()) {
      return detail.trim();
    }
    if (Array.isArray(detail) && detail.length) {
      // Request validation: [{loc: ["body", "temperature"], msg: "..."}]
      return detail
        .map((item) => {
          const entry = (item ?? {}) as { loc?: unknown; msg?: unknown };
          const field = Array.isArray(entry.loc) ? entry.loc.filter((part) => part !== "body").join(".") : "";
          const text = typeof entry.msg === "string" ? entry.msg : JSON.stringify(item);
          return field ? `${field}: ${text}` : text;
        })
        .join("; ");
    }
    if (detail && typeof detail === "object") {
      const { error } = detail as { error?: unknown };
      if (typeof error === "string" && error.trim()) {
        return error.trim();
      }
    }
    if (typeof message === "string" && message.trim()) {
      return message.trim();
    }
  }
  // A JSON body with nothing to say ("{}", {"detail": null}) is not a reason to show.
  const text = body !== null && typeof body === "object" ? "" : raw.trim();
  return text || `The server answered ${res.status}${res.statusText ? ` ${res.statusText}` : ""} without giving a reason.`;
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetchBackend(path, init);
  if (!res.ok) {
    throw new ApiError(res.status, await readErrorDetail(res));
  }
  return res.json();
}

export function apiUrl(path: string): string {
  const envBase = process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "");
  if (envBase) {
    return `${envBase}/api${path}`;
  }
  if (typeof window !== "undefined") {
    const host = window.location.hostname;
    if (host === "127.0.0.1" || host === "localhost") {
      return `http://127.0.0.1:8010/api${path}`;
    }
  }
  return `/api${path}`;
}
