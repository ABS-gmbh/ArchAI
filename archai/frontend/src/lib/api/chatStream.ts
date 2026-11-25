// The events of a streamed chat answer, folded into what the workspace shows.
// Plain TypeScript, so chatStream.test.ts can import it.
//
// The backend sends the answer's deltas, then "answer" (the answer is whole:
// text, model, retrieval, whether a check follows), then "verification" (the
// verdict) when the answer is checked, then "done" with everything again. The
// workspace frees the composer at "answer" and shows the verdict when it comes
// (2026-11-25: a slow verifier held the composer). An older backend sends no
// "answer" and no "verification": its "done" completes the answer and carries
// the verdict.

export interface StreamVerification {
  assessment: string;
  corrected_answer: string;
  notes: string[];
  citations_checked?: string[];
  model_used?: string;
  self_check?: boolean;
}

export type ChatStreamEvent =
  | { type: "delta"; delta: string }
  | { type: "answer"; text: string; model?: string; retrieval?: unknown; checking?: boolean }
  | { type: "verification"; verification: StreamVerification | null; retrieval?: unknown }
  | { type: "done"; text?: string; model?: string; stage_metadata?: unknown; inspection?: unknown; verification?: StreamVerification | null; retrieval?: unknown }
  | { type: "error"; error?: string };

export interface ChatStreamState {
  /** Every delta, the backend's appended blocks included. */
  text: string;
  /** The answer as the "answer" event (or "done") closed it. */
  answer: string | null;
  model?: string;
  retrieval?: unknown;
  stageMetadata?: unknown;
  inspection?: unknown;
  /** The answer event announced a check. */
  checkAnnounced: boolean;
  /** A check was announced and neither its verdict nor the end of the stream has come yet. */
  checking: boolean;
  verification: StreamVerification | null;
  /** "done" arrived: the stream was not cut off. */
  finished: boolean;
}

export const INITIAL_STREAM: ChatStreamState = {
  text: "",
  answer: null,
  checkAnnounced: false,
  checking: false,
  verification: null,
  finished: false,
};

/**
 * The state after *event*, and whether this event made the answer whole (the
 * moment the reader may ask again). "error" is the caller's to throw.
 */
export function applyStreamEvent(state: ChatStreamState, event: ChatStreamEvent): { state: ChatStreamState; answered: boolean } {
  switch (event.type) {
    case "delta":
      return { state: { ...state, text: state.text + event.delta }, answered: false };
    case "answer":
      return {
        state: {
          ...state,
          answer: event.text,
          model: event.model ?? state.model,
          retrieval: event.retrieval ?? state.retrieval,
          checkAnnounced: Boolean(event.checking),
          checking: Boolean(event.checking),
        },
        answered: state.answer === null,
      };
    case "verification":
      return {
        state: { ...state, verification: event.verification ?? null, retrieval: event.retrieval ?? state.retrieval, checking: false },
        answered: false,
      };
    case "done": {
      const answeredNow = state.answer === null;
      return {
        state: {
          ...state,
          text: event.text ?? state.text,
          answer: state.answer ?? event.text ?? state.text,
          model: event.model ?? state.model,
          stageMetadata: event.stage_metadata,
          inspection: event.inspection,
          verification: event.verification ?? state.verification,
          retrieval: event.retrieval ?? state.retrieval,
          checking: false,
          finished: true,
        },
        answered: answeredNow,
      };
    }
    default:
      return { state, answered: false };
  }
}

/** How a message's check ended: checked, still checking, or not checked (no verdict came, or none was announced). */
export function checkOutcome(state: ChatStreamState): "checked" | "checking" | "not_checked" | "none" {
  if (state.verification && state.verification.assessment && state.verification.assessment !== "unavailable") {
    return "checked";
  }
  if (!state.checkAnnounced) {
    return "none";
  }
  return state.checking ? "checking" : "not_checked";
}
