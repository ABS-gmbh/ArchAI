"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type BackendHealth, getBackendHealth } from "@/lib/api/health";

const RECHECK_MS = 60_000;

const HEADLINES: Record<Exclude<BackendHealth["status"], "ok">, string> = {
  degraded: "Some features are unavailable",
  unavailable: "The backend cannot serve requests",
  unreachable: "The backend is not reachable",
};

const BUTTON =
  "shrink-0 rounded border border-current px-2 py-0.5 text-xs font-medium outline-none hover:bg-black/5 focus-visible:ring-2 focus-visible:ring-current focus-visible:ring-offset-1 aria-disabled:opacity-50 dark:hover:bg-white/10";

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex="-1"])';

/** The first control after *root* in the document that can take focus now. */
function nextFocusableAfter(root: HTMLElement): HTMLElement | null {
  for (const element of document.querySelectorAll<HTMLElement>(FOCUSABLE)) {
    const after = root.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_FOLLOWING;
    if (after && !root.contains(element) && element.getClientRects().length > 0) {
      return element;
    }
  }
  return null;
}

/**
 * A banner that says what is wrong with the backend before a request fails:
 * unreachable, unavailable (database down) or degraded (an optional engine,
 * the chat key or the models missing). Renders nothing while the backend is
 * healthy or not yet checked.
 *
 * Degraded notices can be hidden: a deployment that never uses Ollama should
 * not carry "OCR engine (Ollama)" all session. They stay hidden until a
 * problem appears that was not on screen when Hide was pressed, or the page
 * reloads. A failing backend cannot be hidden.
 */
export function BackendStatusBanner() {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  // True from the start: the first check begins as the banner mounts.
  const [checking, setChecking] = useState(true);
  // The problems on screen when Hide was pressed; in memory only, so a reload shows them again.
  const [hidden, setHidden] = useState<ReadonlySet<string>>(() => new Set());
  const controllerRef = useRef<AbortController | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Fetches and stores the health. It sets no state before its first await, so
  // the effect that starts the first check sets none synchronously; `checking`
  // starts true for that one, and `check` below sets it for the others.
  const runCheck = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      const result = await getBackendHealth(controller.signal);
      if (!controller.signal.aborted) {
        setHealth(result);
      }
    } catch {
      // Aborted by a newer check or by unmount.
    } finally {
      if (controllerRef.current === controller) {
        setChecking(false);
      }
    }
  }, []);

  const check = useCallback(() => {
    setChecking(true);
    void runCheck();
  }, [runCheck]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- runCheck sets state only after its first await, asynchronously; the rule does not follow the await.
    void runCheck();
    const timer = window.setInterval(check, RECHECK_MS);
    return () => {
      window.clearInterval(timer);
      controllerRef.current?.abort();
    };
  }, [runCheck, check]);

  if (!health || health.status === "ok") {
    return null;
  }
  const severe = health.status !== "degraded";
  if (!severe && health.problems.length > 0 && health.problems.every((problem) => hidden.has(problem))) {
    return null;
  }
  // Hide removes the banner, and with it the focused button: without a target,
  // a keyboard user is left on the document body. Focus the next control instead.
  const hide = () => {
    const next = rootRef.current ? nextFocusableAfter(rootRef.current) : null;
    setHidden(new Set(health.problems));
    requestAnimationFrame(() => next?.focus());
  };
  return (
    <div
      ref={rootRef}
      className={`flex flex-wrap items-start gap-x-3 gap-y-1 border-b px-4 py-2 text-sm ${
        severe
          ? "border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100"
          : "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100"
      }`}
    >
      {/* Only the message is a live region: the button's label changes on every check. */}
      <div role="status" className="min-w-0 flex-1">
        <p className="font-medium">{HEADLINES[health.status]}</p>
        {health.problems.length ? (
          <ul className="mt-0.5 list-inside list-disc text-xs">
            {health.problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        ) : null}
      </div>
      <div className="flex shrink-0 gap-2">
        {/* aria-disabled, not disabled: a disabled button loses focus, and the
            keyboard user who pressed it would be left on the document body. */}
        <button
          type="button"
          onClick={() => {
            if (!checking) {
              check();
            }
          }}
          aria-disabled={checking || undefined}
          className={BUTTON}
        >
          {checking ? "Checking…" : "Check again"}
        </button>
        {severe ? null : (
          <button
            type="button"
            onClick={hide}
            aria-label="Hide these notices until another problem appears"
            className={BUTTON}
          >
            Hide
          </button>
        )}
      </div>
    </div>
  );
}
