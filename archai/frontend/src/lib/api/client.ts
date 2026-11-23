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

/**
 * fetch() against the backend: the direct URL first, then the same-origin
 * /api proxy when the direct request fails at the network level (backend down,
 * CORS). When both fail, or the proxy can only answer with a bare 5xx because
 * it cannot reach the backend either, this throws an ApiError(503) that says
 * the backend is unreachable, instead of "Failed to fetch" or "Internal Server
 * Error". Aborts are re-thrown unchanged.
 */
export async function fetchBackend(path: string, init?: RequestInit): Promise<Response> {
  const direct = apiUrl(path);
  try {
    return await fetch(direct, init);
  } catch (err) {
    if (isAbortError(err)) {
      throw err;
    }
  }
  const proxied = `/api${path}`;
  let res: Response;
  try {
    res = await fetch(proxied, init);
  } catch (proxyErr) {
    if (isAbortError(proxyErr)) {
      throw proxyErr;
    }
    throw new ApiError(503, unreachableMessage(direct));
  }
  if (direct !== proxied && res.status >= 500 && !isJsonResponse(res)) {
    // The direct request never reached the backend, and the proxy answered
    // with a plain-text 5xx: it could not reach the backend either.
    throw new ApiError(503, unreachableMessage(direct));
  }
  return res;
}

function isJsonResponse(res: Response): boolean {
  return (res.headers.get("content-type") || "").toLowerCase().includes("application/json");
}

export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetchBackend(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string" && body.detail.trim()) {
        detail = body.detail;
      } else if (body && typeof body.message === "string" && body.message.trim()) {
        detail = body.message;
      }
    } catch {
      const text = await res.text().catch(() => "");
      if (text.trim()) {
        detail = text.trim();
      }
    }
    throw new ApiError(res.status, detail);
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
