import { applyStreamEvent, type ChatStreamEvent, type ChatStreamState, checkOutcome, INITIAL_STREAM } from "./chatStream";
import { ApiError, fetchBackend, readErrorDetail } from "./client";

export interface ChatModelListResponse {
  models: string[];
  /** Whether the list is the provider's ("ok"), or configured names because it refused the key or did not answer. */
  provider?: "ok" | "refused" | "unreachable";
  default_model: string;
  vision_models: string[];
  task_models?: {
    ocr_model: string;
    chat_rag_model: string;
    translation_model: string;
    label_visual_model: string;
    label_visual_fallback_model: string;
    verifier_model: string;
    embedding_model: string;
  };
}

export interface ChatMessagePayload {
  role: "user" | "assistant" | "system";
  content:
    | string
    | Array<
        | { type: "text"; text: string }
        | { type: "image_url"; image_url: { url: string } }
      >;
}

export interface ChatCompletionPayload {
  messages: ChatMessagePayload[];
  model?: string | null;
  temperature?: number;
  stream?: boolean;
  context?: Record<string, unknown>;
}

/** How the answer's evidence was retrieved; field names are a stable backend contract. */
export interface ChatRetrievalStatus {
  status: "ok" | "degraded" | "unavailable" | "no_searchable_run" | "not_used";
  chunk_mode: "hybrid" | "dense" | "lexical" | "none" | "error";
  entity_mode: "dense" | "none" | "error";
  chunk_hits: number;
  entity_hits: number;
  run_scoped: boolean;
  /** The prompt carried an excerpt of the page's transcript even though search found nothing. */
  transcript_in_context?: boolean;
  notes: string[];
  verifier: "ran" | "unavailable" | "skipped" | "disabled";
}

export interface ChatCompletionResult {
  text: string;
  /** The stream closed before its final event: `text` is only the part that arrived. */
  incomplete?: boolean;
  /** How the answer's check ended (chatStream.ts checkOutcome); streamed answers only. */
  check?: "checked" | "checking" | "not_checked" | "none";
  retrieval?: ChatRetrievalStatus | null;
  model?: string;
  stage_metadata?: {
    stage_name: string;
    model_used: string;
    mode_used?: string | null;
    duration_ms?: number | null;
  } | null;
  inspection?: Record<string, unknown> | null;
  verification?: {
    assessment: string;
    corrected_answer: string;
    notes: string[];
    citations_checked: string[];
    model_used: string;
    /** The answering model checked its own answer (the verifier's model is not offered). */
    self_check?: boolean;
    stage_metadata?: {
      stage_name: string;
      model_used: string;
      mode_used?: string | null;
      duration_ms?: number | null;
    } | null;
    inspection?: Record<string, unknown> | null;
  } | null;
}

export interface LabelRegionPayload {
  region_id: string;
  bbox_xyxy: [number, number, number, number];
  polygons?: number[][];
}

export interface LabelAnalysisPayload {
  question: string;
  label_name: string;
  image_b64: string;
  regions: LabelRegionPayload[];
  filename?: string | null;
  page_id?: string | null;
  document_id?: string | null;
}

export interface LabelAnalysisResult {
  status: string;
  text: string;
  label_name: string;
  analysis_mode?: string | null;
  model_used: string;
  warnings: string[];
  region_count: number;
  crop_image_b64: string;
  crop_bounds_xyxy: number[];
  ocr_text?: string | null;
  stage_metadata?: {
    stage_name: string;
    model_used: string;
    mode_used?: string | null;
    duration_ms?: number | null;
  } | null;
  inspection?: Record<string, unknown> | null;
}

export async function getChatModels(): Promise<ChatModelListResponse> {
  const res = await fetchBackend("/chat/models");
  if (!res.ok) {
    throw new ApiError(res.status, await readErrorDetail(res));
  }
  return (await res.json()) as ChatModelListResponse;
}

/**
 * A chat answer. Streamed, *onAnswer* is called once the answer is whole
 * (before the verifier's verdict), and the promise resolves when the stream
 * ends, with the verdict if it came.
 */
export async function createChatCompletion(
  payload: ChatCompletionPayload,
  onDelta?: (delta: string) => void,
  onAnswer?: (answer: ChatCompletionResult) => void,
): Promise<ChatCompletionResult> {
  const useStream = payload.stream !== false;
  const res = await fetchBackend("/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: useStream }),
  });

  if (!res.ok) {
    throw new ApiError(res.status, await readErrorDetail(res));
  }

  if (!useStream) {
    const data = (await res.json()) as ChatCompletionResult;
    if (typeof data.text !== "string") {
      throw new Error("Invalid completion response.");
    }
    return data;
  }

  if (!res.body) {
    throw new Error("Streaming response body is missing.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let state: ChatStreamState = INITIAL_STREAM;
  const result = (): ChatCompletionResult => ({
    text: state.answer ?? state.text,
    model: state.model,
    stage_metadata: state.stageMetadata as ChatCompletionResult["stage_metadata"],
    inspection: state.inspection as ChatCompletionResult["inspection"],
    verification: state.verification as ChatCompletionResult["verification"],
    retrieval: state.retrieval as ChatCompletionResult["retrieval"],
    // A stream that ends without "done" was cut off (proxy timeout, dropped
    // connection): what arrived may not be the whole answer.
    incomplete: !state.finished && state.answer === null,
    check: checkOutcome(state),
  });

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let splitIndex = buffer.indexOf("\n\n");
    while (splitIndex !== -1) {
      const chunk = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 2);

      for (const rawLine of chunk.split("\n")) {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) {
          continue;
        }
        const payloadText = line.slice(5).trim();
        if (!payloadText) {
          continue;
        }

        const event = JSON.parse(payloadText) as ChatStreamEvent;
        if (event.type === "error") {
          throw new Error(event.error || "Streaming completion failed.");
        }
        const next = applyStreamEvent(state, event);
        state = next.state;
        if (event.type === "delta") {
          onDelta?.(event.delta);
        }
        // The answer is whole: the reader may ask again while it is checked.
        if (next.answered) {
          onAnswer?.(result());
        }
      }

      splitIndex = buffer.indexOf("\n\n");
    }
  }

  // The stream is over: a check still awaited will not come, and the answer is "not checked".
  state = { ...state, checking: false };
  return result();
}

export async function analyzeSegmentLabel(
  payload: LabelAnalysisPayload,
): Promise<LabelAnalysisResult> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 85_000);
  let res: Response;
  try {
    res = await fetchBackend("/chat/label-analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Label analysis timed out. Please retry.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!res.ok) {
    throw new ApiError(res.status, await readErrorDetail(res));
  }

  return (await res.json()) as LabelAnalysisResult;
}
