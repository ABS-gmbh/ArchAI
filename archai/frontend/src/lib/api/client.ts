export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function unreachableMessage(direct: string): string {
  return `Backend API unreachable at ${direct}. Start the backend on 127.0.0.1:8010 and retry.`;
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

/** A fetch through one route; a proxy that cannot reach the backend counts as a network failure. */
async function fetchVia(transport: Transport, path: string, init?: RequestInit): Promise<Response> {
  const direct = apiUrl(path);
  const proxied = `/api${path}`;
  if (transport === "direct") {
    return fetch(direct, init);
  }
  const res = await fetch(proxied, init);
  if (direct !== proxied && res.status >= 500 && !isJsonResponse(res)) {
    // Next answers a plain-text 5xx when its rewrite cannot reach the backend.
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
  try {
    return await fetchVia(knownTransport ?? "direct", path, init);
  } catch (err) {
    if (isAbortError(err)) {
      throw err;
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
  return raw.trim() || res.statusText || `HTTP ${res.status}`;
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
