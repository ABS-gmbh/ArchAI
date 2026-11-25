// Whether the Ask panel can expect an answer, from GET /chat/models when the
// workspace opens. A key the provider refused used to show nothing until a
// question failed (2026-11-25): the model list fell back to configured names
// and looked healthy. The backend now marks the list (`provider`), and the
// panel says it before anything is asked. Plain TypeScript, so
// chatAvailability.test.ts can import it.

export type ChatProviderState = "ok" | "refused" | "unreachable" | "unknown";

/** The list's `provider` field; "unknown" for an older backend or anything else. */
export function providerState(payload: unknown): ChatProviderState {
  const value = (payload as { provider?: unknown } | null)?.provider;
  return value === "ok" || value === "refused" || value === "unreachable" ? value : "unknown";
}

export interface ChatNotice {
  /** Shown at the top of the Ask panel. */
  sentence: string;
  /** Whether asking can work: the composer stays usable only then. */
  canAsk: boolean;
  /** The composer's placeholder when it cannot be used. */
  disabledReason: string | null;
}

/** What the Ask panel says before a question, or null when nothing is wrong (or nothing is known). */
export function chatNotice(state: ChatProviderState): ChatNotice | null {
  if (state === "refused") {
    return {
      sentence:
        "The chat provider refused this server's key, so questions and translations cannot be answered until it is replaced. Tell whoever runs the server.",
      canAsk: false,
      disabledReason: "Chat is unavailable: the provider refused this server's key",
    };
  }
  if (state === "unreachable") {
    return {
      sentence: "The chat provider did not answer when the workspace opened. Questions may fail until it does; you can still try.",
      canAsk: true,
      disabledReason: null,
    };
  }
  return null;
}

/**
 * The model a question goes to: the backend's `default_model` when the model
 * list answered, then its chat task model, and *offline* (the frontend's own
 * default) only when the backend said neither or did not answer.
 */
export function defaultChatModel(payload: unknown, offline: string): string {
  const list = payload as { default_model?: unknown; task_models?: { chat_rag_model?: unknown } | null } | null;
  const named = [list?.default_model, list?.task_models?.chat_rag_model].find(
    (value): value is string => typeof value === "string" && value.trim() !== "",
  );
  return named ? named.trim() : offline;
}
