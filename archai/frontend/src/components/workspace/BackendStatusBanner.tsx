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
  "shrink-0 rounded border border-current px-2 py-0.5 text-xs font-medium outline-none hover:bg-black/5 focus-visible:ring-2 focus-visible:ring-current focus-visible:ring-offset-1 disabled:opacity-50 dark:hover:bg-white/10";

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
  const [checking, setChecking] = useState(false);
  // The problems on screen when Hide was pressed; in memory only, so a reload shows them again.
  const [hidden, setHidden] = useState<ReadonlySet<string>>(() => new Set());
  const controllerRef = useRef<AbortController | null>(null);

  const check = useCallback(async () => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setChecking(true);
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

  useEffect(() => {
    void check();
    const timer = window.setInterval(() => void check(), RECHECK_MS);
    return () => {
      window.clearInterval(timer);
      controllerRef.current?.abort();
    };
  }, [check]);

  if (!health || health.status === "ok") {
    return null;
  }
  const severe = health.status !== "degraded";
  if (!severe && health.problems.length > 0 && health.problems.every((problem) => hidden.has(problem))) {
    return null;
  }
  return (
    <div
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
        <button type="button" onClick={() => void check()} disabled={checking} className={BUTTON}>
          {checking ? "Checking…" : "Check again"}
        </button>
        {severe ? null : (
          <button
            type="button"
            onClick={() => setHidden(new Set(health.problems))}
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
