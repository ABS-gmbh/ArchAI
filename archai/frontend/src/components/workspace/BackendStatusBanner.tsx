"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { type BackendHealth, getBackendHealth } from "@/lib/api/health";

const RECHECK_MS = 60_000;

const HEADLINES: Record<Exclude<BackendHealth["status"], "ok">, string> = {
  degraded: "Some features are unavailable",
  unavailable: "The backend cannot serve requests",
  unreachable: "The backend is not reachable",
};

/**
 * A banner that says what is wrong with the backend before a request fails:
 * unreachable, unavailable (database down) or degraded (OCR engine or chat key
 * missing). Renders nothing while the backend is healthy or not yet checked.
 */
export function BackendStatusBanner() {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [checking, setChecking] = useState(false);
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
      <button
        type="button"
        onClick={() => void check()}
        disabled={checking}
        className="shrink-0 rounded border border-current px-2 py-0.5 text-xs font-medium hover:bg-black/5 disabled:opacity-50 dark:hover:bg-white/10"
      >
        {checking ? "Checking…" : "Check again"}
      </button>
    </div>
  );
}
