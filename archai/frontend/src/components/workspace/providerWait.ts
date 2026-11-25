// How long the reader waits after the provider limited requests (a 429), and
// what the Ask panel says meanwhile. Plain TypeScript, so
// providerWait.test.ts can import it.
//
// The provider limits each user to 10 requests in 10 s and 400 an hour. The
// panel said "Try again in 7 seconds." and left Ask enabled, so a reader who
// asked again at once met the limit again (2026-11-25). Now a short wait
// (the per-10-s limit) holds Ask until it has passed, counting down beside
// it; a long one (the hourly cap) is not a reason to lock the panel for an
// hour: it says the clock time the reader can ask again, and Ask stays on.
// The wait comes from the provider's Retry-After, never from the sentence.
// It lives in memory only, so a reload ends it.

/** The longest wait Ask is held for, in seconds; a longer one is said as a clock time. */
export const HOLD_UP_TO_SECONDS = 60;

export interface ProviderWait {
  /** When the wait ends (ms since the epoch). */
  until: number;
  /** Ask is held until then (a short wait), or only told the time (a long one). */
  holds: boolean;
}

/** The wait a 429 asked for, from *now*; null without one. */
export function waitFrom(retryAfterSeconds: number | null | undefined, now: number): ProviderWait | null {
  if (typeof retryAfterSeconds !== "number" || !Number.isFinite(retryAfterSeconds) || retryAfterSeconds <= 0) {
    return null;
  }
  return { until: now + Math.ceil(retryAfterSeconds) * 1000, holds: retryAfterSeconds <= HOLD_UP_TO_SECONDS };
}

/** "14:32", the local clock time of *ms*. */
export function clockTime(ms: number): string {
  const at = new Date(ms);
  return `${String(at.getHours()).padStart(2, "0")}:${String(at.getMinutes()).padStart(2, "0")}`;
}

export interface WaitShown {
  /** Ask is held now. */
  holdsAsk: boolean;
  /** Said beside Ask, or null when there is nothing to say (no wait, or it is over). */
  note: string | null;
}

/** What the panel shows of *wait* at *now*. */
export function waitShown(wait: ProviderWait | null, now: number): WaitShown {
  if (!wait || now >= wait.until) {
    return { holdsAsk: false, note: null };
  }
  if (wait.holds) {
    const seconds = Math.max(1, Math.ceil((wait.until - now) / 1000));
    return { holdsAsk: true, note: `Ask again in ${seconds} s` };
  }
  return { holdsAsk: false, note: `You can ask again at ${clockTime(wait.until)}` };
}
