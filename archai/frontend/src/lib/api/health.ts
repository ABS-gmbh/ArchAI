import { ApiError, fetchBackend } from "./client";

export interface BackendHealthCheck {
  ok: boolean;
  detail?: string;
}

export interface BackendHealth {
  /** "unreachable" is the frontend's own state: the backend did not answer at all. */
  status: "ok" | "degraded" | "unavailable" | "unreachable";
  /** What a reader cannot do right now, one sentence each, with no setting or file named. */
  problems: string[];
  /** One sentence per failing check as the backend words it, for whoever runs the server. */
  details: string[];
}

const CHECK_LABELS: Record<string, string> = {
  database: "Pipeline database",
  ollama: "OCR engine (Ollama)",
  chat_api_key: "Chat provider key",
};

// Checks whose detail is already a sentence that says what fails and what to do.
const SENTENCE_CHECKS = new Set(["layout_models", "line_recognizer", "search_index", "task_models"]);

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
    return { status: "unreachable", problems: ["The server is not answering. Try again in a few minutes."], details: [detail] };
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
      ? { status: "ok", problems: [], details: [] }
      : { status: "unavailable", problems: ["The server cannot answer right now. Try again in a few minutes."], details: [`The backend answered HTTP ${res.status}.`] };
  }
  const status = body.status === "ok" || body.status === "degraded" || body.status === "unavailable"
    ? body.status
    : res.ok ? "ok" : "unavailable";
  return { status, ...describeChecks(body.checks ?? {}) };
}

// What each failing check means to someone reading a manuscript. The backend's
// own sentences name settings, files and services ("GLMOCR_OLLAMA_HOST",
// "set RAG_EMBEDDING_MODEL=", "see docs/deployment.md"): right for whoever
// runs the server, and all a visitor of a public site saw (2026-11-25).
const READER_SENTENCES: Record<string, string> = {
  database: "Readings cannot be saved or opened right now.",
  ollama: "One of the two ways of reading a page (GLM-OCR) is not available here.",
  chat_api_key: "Questions and translations are not available here right now.",
  layout_models: "Pages cannot be read line by line here.",
  line_recognizer: "Lines cannot be transcribed here.",
  search_index: "Pages are read, but answers cannot yet draw on their text.",
  task_models: "A model this server was set to use is no longer offered; another is used where there is one.",
};
const UNKNOWN_CHECK = "Part of the server is not available.";
// One cause, one sentence: without the chat key nothing can be asked, and the
// pages are not indexed for the same reason.
const NO_CHAT_NO_SEARCH = "Questions, translations and search are not available here right now.";

/**
 * The failing checks of a /health body, said twice: for a reader (`problems`)
 * and as the backend words them for the server's operator (`details`).
 */
export function describeChecks(checks: Record<string, BackendHealthCheck | null | undefined>): { problems: string[]; details: string[] } {
  const failing = Object.entries(checks).filter(([, check]) => check && check.ok === false) as [string, BackendHealthCheck][];
  const details = failing.map(([name, check]) =>
    SENTENCE_CHECKS.has(name) && check.detail
      ? check.detail
      : `${CHECK_LABELS[name] ?? name}: ${check.detail || "not available"}`,
  );
  const names = new Set(failing.map(([name]) => name));
  const merged = names.has("chat_api_key") && names.has("search_index");
  const problems: string[] = [];
  for (const [name] of failing) {
    const sentence = merged && name === "chat_api_key"
      ? NO_CHAT_NO_SEARCH
      : merged && name === "search_index"
        ? null
        : READER_SENTENCES[name] ?? UNKNOWN_CHECK;
    if (sentence && !problems.includes(sentence)) {
      problems.push(sentence);
    }
  }
  return { problems, details };
}

/** The request limits in a /health body, or null when it has none (an older backend) or they are not numbers. */
export function requestLimitsFrom(body: unknown): { jsonBytes: number; decodePixels: number } | null {
  const limits = (body as { limits?: { json_bytes?: unknown; decode_pixels?: unknown } } | null)?.limits;
  const jsonBytes = Number(limits?.json_bytes);
  const decodePixels = Number(limits?.decode_pixels);
  if (!Number.isFinite(jsonBytes) || jsonBytes <= 0 || !Number.isFinite(decodePixels) || decodePixels <= 0) {
    return null;
  }
  return { jsonBytes, decodePixels };
}

/**
 * The server's request limits from GET /health, for checking a page before it
 * is sent; null when they cannot be had, and then only the server checks.
 */
export async function getRequestLimits(signal?: AbortSignal): Promise<{ jsonBytes: number; decodePixels: number } | null> {
  try {
    const res = await fetchBackend("/health", { signal });
    return requestLimitsFrom(await res.json());
  } catch {
    return null;
  }
}
