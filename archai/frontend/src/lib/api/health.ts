import { ApiError, fetchBackend } from "./client";

export interface BackendHealthCheck {
  ok: boolean;
  detail?: string;
}

export interface BackendHealth {
  /** "unreachable" is the frontend's own state: the backend did not answer at all. */
  status: "ok" | "degraded" | "unavailable" | "unreachable";
  /** One sentence per failing check, ready to show. */
  problems: string[];
}

const CHECK_LABELS: Record<string, string> = {
  database: "Pipeline database",
  ollama: "OCR engine (Ollama)",
  chat_api_key: "Chat provider key",
};

/**
 * GET /health, reduced to what the workspace shows. Never throws: a backend
 * that cannot be reached is reported as status "unreachable".
 */
export async function getBackendHealth(signal?: AbortSignal): Promise<BackendHealth> {
  let res: Response;
  try {
    res = await fetchBackend("/health", { signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw err;
    }
    const detail = err instanceof ApiError ? err.detail : "The backend did not answer.";
    return { status: "unreachable", problems: [detail] };
  }
  let body: { status?: string; checks?: Record<string, BackendHealthCheck> } | null = null;
  try {
    body = await res.json();
  } catch (err) {
    // An abort can also land while the body is being read: it must not turn
    // into a result, or a superseded check overwrites the newer one.
    if (signal?.aborted || (err instanceof DOMException && err.name === "AbortError")) {
      throw err instanceof DOMException ? err : new DOMException("Aborted", "AbortError");
    }
    body = null;
  }
  if (!body || typeof body.status !== "string") {
    // An older backend, or a proxy error page: only the HTTP status is known.
    return res.ok
      ? { status: "ok", problems: [] }
      : { status: "unavailable", problems: [`The backend answered HTTP ${res.status}.`] };
  }
  const problems = Object.entries(body.checks ?? {})
    .filter(([, check]) => check && check.ok === false)
    .map(([name, check]) => `${CHECK_LABELS[name] ?? name}: ${check.detail || "not available"}`);
  const status = body.status === "ok" || body.status === "degraded" || body.status === "unavailable"
    ? body.status
    : res.ok ? "ok" : "unavailable";
  return { status, problems };
}
