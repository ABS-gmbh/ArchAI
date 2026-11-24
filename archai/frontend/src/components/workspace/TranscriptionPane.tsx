"use client";

import { useEffect, useRef } from "react";

import { ExtractionBadges } from "./ExtractionBadges";
import { Button, Icon, Pill } from "./ui";
import { type ExtractionBadge, recogniserName } from "./workspaceExtraction";

export interface TranscriptionLine {
  id: string;
  text: string;
  /** The recogniser's confidence in this line, 0..1, when it reports one. */
  confidence: number | null;
  /** Set when a recogniser other than the page's main one read this line. */
  fallbackBackend: string | null;
  /** Whether the line has a box on the page that the viewer can show. */
  onPage: boolean;
}

export type ReadPhase = "idle" | "reading" | "done" | "failed";

export interface ReadState {
  phase: ReadPhase;
  /** Lines read so far, from the run's progress events. */
  progress: { done: number; total: number } | null;
  /** The pipeline stage in words, while reading. */
  stage: string | null;
  error: string | null;
  readBy: string | null;
}

export const IDLE_READ_STATE: ReadState = { phase: "idle", progress: null, stage: null, error: null, readBy: null };

interface TranscriptionPaneProps {
  pageName: string | null;
  state: ReadState;
  lines: TranscriptionLine[];
  /** Text found outside the text columns: marginalia, folio numbers, stamps. */
  marginLines: string[];
  badges: ExtractionBadge[];
  activeLineId: string | null;
  /** The line to bring into view, set when a box on the page is clicked. */
  scrollToLineId: string | null;
  onActiveLine: (id: string | null) => void;
  onRead: () => void;
  onTranslate: () => void;
  busy: boolean;
}

/**
 * The page's text, one manuscript line per row, numbered as on the page. A row
 * and its box on the page light up together. The header carries the reading's
 * verdict, so how far to trust the text is read before the text itself.
 */
export function TranscriptionPane({
  pageName,
  state,
  lines,
  marginLines,
  badges,
  activeLineId,
  scrollToLineId,
  onActiveLine,
  onRead,
  onTranslate,
  busy,
}: TranscriptionPaneProps) {
  const listRef = useRef<HTMLOListElement | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  // The read buttons in the pane are replaced by the progress line as soon as
  // they are pressed, which would leave a keyboard user on the document body.
  // Focus the pane's heading instead: it stays through the read.
  const readHere = () => {
    onRead();
    requestAnimationFrame(() => headingRef.current?.focus());
  };

  useEffect(() => {
    if (!scrollToLineId || !listRef.current) {
      return;
    }
    const row = listRef.current.querySelector<HTMLElement>(`[data-line-id="${CSS.escape(scrollToLineId)}"]`);
    // Someone who asked for less motion gets the row at once, not a glide to it.
    const still = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    row?.scrollIntoView({ block: "nearest", behavior: still ? "auto" : "smooth" });
  }, [scrollToLineId]);

  const hasText = lines.length > 0;
  const activeInList = lines.some((line) => line.id === activeLineId);
  const ratio = state.progress && state.progress.total > 0 ? Math.min(1, state.progress.done / state.progress.total) : null;

  return (
    <section aria-labelledby="transcription-heading" className="flex h-full min-h-0 flex-col">
      <header className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2 border-b border-rule px-6 pb-4 pt-5">
        <div className="min-w-0">
          <h2
            id="transcription-heading"
            ref={headingRef}
            tabIndex={-1}
            className="rounded-sm text-base font-semibold outline-none focus-visible:ring-2 focus-visible:ring-brand"
          >
            Transcription
          </h2>
          <p className="mt-0.5 truncate text-sm text-ink-muted">
            {state.phase === "done" && hasText
              ? `${lines.length} ${lines.length === 1 ? "line" : "lines"}${state.readBy ? `, read by ${recogniserName(state.readBy)}` : ""}`
              : pageName ?? "No page open"}
          </p>
          {state.phase === "done" ? <ExtractionBadges badges={badges} /> : null}
        </div>
        {hasText ? (
          <Button variant="secondary" onClick={onTranslate} disabled={busy}>
            Translate
          </Button>
        ) : null}
      </header>

      {/* Said once when a read ends; the progress line below announces itself while it runs. */}
      <p role="status" className="sr-only">
        {state.phase === "done" ? (hasText ? `Page read: ${lines.length} ${lines.length === 1 ? "line" : "lines"}.` : "Page read: no text found.") : ""}
      </p>

      <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
        {state.phase === "reading" ? (
          <div role="status">
            <p className="text-sm">
              {state.progress
                ? `Reading line ${state.progress.done} of ${state.progress.total}`
                : state.stage ?? "Reading the page"}
            </p>
            <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-well" aria-hidden="true">
              {ratio !== null ? (
                <div className="h-full rounded-full bg-brand transition-[width] duration-300 motion-reduce:transition-none" style={{ width: `${ratio * 100}%` }} />
              ) : null}
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              {state.stage && state.progress ? state.stage : "The text appears here when the page has been read."}
            </p>
            <div className="mt-5 space-y-3" aria-hidden="true">
              {[92, 84, 96, 71, 88, 90, 64].map((width, index) => (
                <div key={index} className="flex items-center gap-3">
                  <span className="h-3 w-5 flex-none rounded-sm bg-well" />
                  <span className="h-4 rounded-sm bg-well" style={{ width: `${width}%` }} />
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {state.phase === "failed" ? (
          <div role="alert" className="rounded-lg border border-line p-4">
            <p className="text-sm font-medium">The page could not be read.</p>
            <p className="mt-1 text-sm text-ink-muted">{state.error ?? "The request failed."}</p>
            <Button variant="secondary" className="mt-3" onClick={readHere} disabled={busy}>
              <Icon name="retry" /> Try again
            </Button>
          </div>
        ) : null}

        {state.phase === "idle" && !hasText ? (
          <div className="rounded-lg border border-dashed border-control p-5">
            <p className="text-sm">{pageName ? "This page has not been read yet." : "Open a page to read it."}</p>
            <p className="mt-1 text-sm text-ink-muted">
              Reading finds the lines of text on the page, transcribes each one, and checks how far the result can be trusted.
            </p>
            {pageName ? (
              <Button variant="secondary" className="mt-3" onClick={readHere} disabled={busy}>
                Read this page
              </Button>
            ) : null}
          </div>
        ) : null}

        {state.phase === "done" && !hasText ? (
          <div className="rounded-lg border border-dashed border-control p-5">
            <p className="text-sm">No readable text was found on this page.</p>
            <p className="mt-1 text-sm text-ink-muted">
              A blank leaf, a full-page image or a very faint hand can leave nothing to transcribe.
            </p>
          </div>
        ) : null}

        {state.phase !== "reading" && hasText ? (
          <>
            <ol
              ref={listRef}
              aria-label="Lines of the page"
              className="font-text text-[17px] leading-7"
              onMouseLeave={() => onActiveLine(null)}
              onKeyDown={(event) => {
                // One tab stop for the list; the arrow keys walk its lines.
                if (event.key !== "ArrowDown" && event.key !== "ArrowUp" && event.key !== "Home" && event.key !== "End") {
                  return;
                }
                const rows = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>("button[data-row]"));
                const at = rows.indexOf(document.activeElement as HTMLButtonElement);
                if (at === -1) {
                  return;
                }
                event.preventDefault();
                const next =
                  event.key === "Home" ? 0 : event.key === "End" ? rows.length - 1 : Math.min(rows.length - 1, Math.max(0, at + (event.key === "ArrowDown" ? 1 : -1)));
                rows[next].focus();
              }}
            >
              {lines.map((line, index) => {
                const active = line.id === activeLineId;
                return (
                  <li key={line.id} data-line-id={line.id}>
                    <button
                      type="button"
                      data-row
                      aria-current={active ? "true" : undefined}
                      tabIndex={active || (!activeInList && index === 0) ? 0 : -1}
                      onMouseEnter={() => onActiveLine(line.id)}
                      onFocus={() => onActiveLine(line.id)}
                      onClick={() => onActiveLine(line.id)}
                      className={`group grid w-full grid-cols-[2rem_minmax(0,1fr)_auto] items-baseline gap-x-3 rounded-md px-2 py-0.5 text-left outline-none focus-visible:ring-2 focus-visible:ring-brand ${
                        active ? "bg-brand-wash" : "hover:bg-well"
                      }`}
                    >
                      <span className={`text-right font-sans text-xs tabular-nums ${active ? "text-brand" : "text-ink-muted"}`}>
                        {index + 1}
                      </span>
                      <span className="min-w-0 break-words">
                        {line.text || <span className="text-ink-muted">(nothing read)</span>}
                        {line.fallbackBackend ? (
                          <span className="ml-2 inline-block align-middle font-sans">
                            <Pill tone="warning" title="The page's main recogniser read nothing here, so this line comes from a fallback recogniser.">
                              {recogniserName(line.fallbackBackend)}
                            </Pill>
                          </span>
                        ) : null}
                      </span>
                      <span className="font-sans">
                        {active && typeof line.confidence === "number" ? (
                          <span className="text-xs tabular-nums text-ink-muted" title="The recogniser's own confidence in this line. Not an accuracy measurement.">
                            <span className="sr-only">Recogniser confidence </span>
                            {Math.round(line.confidence * 100)}%
                          </span>
                        ) : null}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
            {marginLines.length ? (
              <div className="mt-6 border-t border-rule pt-4">
                <h3 className="text-sm font-medium">Outside the text columns</h3>
                <p className="mt-0.5 text-xs text-ink-muted">Marginalia, folio numbers and stamps. Shown here, kept out of search and chat.</p>
                <ul className="mt-2 font-text text-[17px] leading-7">
                  {marginLines.map((text, index) => (
                    <li key={`${index}-${text}`} className="px-2 text-ink-muted">
                      {text}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </>
        ) : null}
      </div>
    </section>
  );
}
