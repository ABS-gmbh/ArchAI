// Whether the conversation follows a change by scrolling to its end. Plain
// TypeScript, so conversationScroll.test.ts can import it.
//
// The panel scrolled to the end on every change: each streamed piece, and the
// verdict that comes seconds after the answer (2026-11-25). A reader who had
// scrolled up to read an earlier answer was taken away from it mid-sentence
// (measured: from the top to 494 px down when a verdict arrived). Now the
// panel follows only a reader who is at the end already, or who has just
// asked something.

/** How close to the end, in pixels, still counts as at the end. */
export const AT_END_PX = 48;

/** Whether the scroller is at (or within AT_END_PX of) its end. */
export function isAtEnd(scroller: { scrollHeight: number; scrollTop: number; clientHeight: number }): boolean {
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= AT_END_PX;
}

/**
 * Whether to scroll to the end after the messages changed: when the reader
 * was at the end, or when a message of theirs was added (they asked, and
 * expect to see the answer come).
 */
export function followsTheChange(change: { countBefore: number; messages: readonly { role: string }[]; wasAtEnd: boolean }): boolean {
  if (change.wasAtEnd) {
    return true;
  }
  return change.messages.slice(change.countBefore).some((message) => message.role === "user");
}
