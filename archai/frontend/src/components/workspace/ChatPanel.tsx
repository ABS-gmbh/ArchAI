"use client";

import { Fragment, type ReactNode, useEffect, useId, useRef } from "react";
import Image from "next/image";

import type { WorkspaceChatMessage } from "@/lib/workspace/types";

import { describeVerification } from "./chatAnswer";
import { followsTheChange, isAtEnd } from "./conversationScroll";
import { type ChatSource, citedSourceList, describeLines, sourceLineIds, splitCitations } from "./citations";
import { ExtractionBadges } from "./ExtractionBadges";
import { RetrievalBadges } from "./RetrievalBadges";
import { Button, Icon } from "./ui";

interface ChatPanelProps {
  messages: WorkspaceChatMessage[];
  /** Any request is running (a question, a page read, a region search): the composer is off. */
  sending: boolean;
  /** A question is being answered. */
  asking: boolean;
  loadingLabel: string;
  error: string | null;
  input: string;
  onInput: (value: string) => void;
  onSend: () => void;
  onClear: () => void;
  /** Why the composer is off, in words; null when a question can be asked. */
  disabledReason: string | null;
  /** Said at the top of the panel before anything is asked: the chat provider refused the key, or did not answer. */
  notice?: string | null;
  /** Said once to screen readers when a reply ends. */
  announcement: string;
  hasTranscription: boolean;
  /** The page's lines, numbered as in the transcription, and the run they were read in: what an answer's [n] can point at. */
  lines?: readonly { id: string }[];
  runId?: string | null;
  /** The lines kept lit by a click on a [n], or null. */
  pinnedLineIds?: readonly string[] | null;
  /** Light up these lines while a [n] has focus or the pointer; null when it loses them. */
  onPreviewLines?: (ids: string[] | null) => void;
  /** A click on a [n]: keep its lines lit (a second click lets them go). */
  onPinLines?: (ids: string[]) => void;
}

interface CitationProps {
  lines: readonly { id: string }[];
  runId: string | null;
  pinnedKey: string | null;
  onPreviewLines: (ids: string[] | null) => void;
  onPinLines: (ids: string[]) => void;
}

/** One [n] that stands for lines of the page: a button that lights them up. */
function CitationButton({
  source,
  lineIds,
  children,
  pinnedKey,
  onPreviewLines,
  onPinLines,
  className,
}: { source: ChatSource; lineIds: string[]; children: ReactNode; className: string } & Pick<
  CitationProps,
  "pinnedKey" | "onPreviewLines" | "onPinLines"
>) {
  return (
    <button
      type="button"
      aria-label={`Source ${source.label}: ${describeLines(source)} of the page`}
      aria-pressed={pinnedKey === lineIds.join(" ")}
      onFocus={() => onPreviewLines(lineIds)}
      onBlur={() => onPreviewLines(null)}
      onMouseEnter={() => onPreviewLines(lineIds)}
      onMouseLeave={() => onPreviewLines(null)}
      onClick={() => onPinLines(lineIds)}
      className={`rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-brand aria-pressed:bg-brand-wash ${className}`}
    >
      {children}
    </button>
  );
}

/** The answer's text, each [n] that stands for lines of this page a button; any other label stays as written. */
function CitedAnswer({ text, sources, ...props }: { text: string; sources: readonly ChatSource[] } & CitationProps) {
  const byLabel = new Map(sources.map((source) => [source.label, source]));
  return (
    <>
      {splitCitations(text).map((part, index) => {
        if (part.kind === "text") {
          return <Fragment key={index}>{part.text}</Fragment>;
        }
        return (
          <Fragment key={index}>
            [
            {part.labels.map((label, at) => {
              const source = byLabel.get(label);
              const lineIds = sourceLineIds(source, props.lines, props.runId);
              const separator = at ? ", " : "";
              if (!source || !lineIds.length) {
                return (
                  <Fragment key={at}>
                    {separator}
                    {label}
                  </Fragment>
                );
              }
              return (
                <Fragment key={at}>
                  {separator}
                  <CitationButton
                    source={source}
                    lineIds={lineIds}
                    pinnedKey={props.pinnedKey}
                    onPreviewLines={props.onPreviewLines}
                    onPinLines={props.onPinLines}
                    className="px-0.5 text-xs font-medium text-brand underline decoration-dotted underline-offset-2 hover:decoration-solid"
                  >
                    {label}
                  </CitationButton>
                </Fragment>
              );
            })}
            ]
          </Fragment>
        );
      })}
    </>
  );
}

/**
 * The sources the answer cites, in words, under it. Text, not buttons: the
 * answer's own [n] buttons are the way to them by keyboard, and the list
 * repeated them, two tab stops and two identical names per label (a reader
 * with three answers met 18 stops on the way back up). A pointer over an
 * entry, or a click on it, still lights its lines up.
 */
function SourceList({ sources, ...props }: { sources: readonly ChatSource[] } & CitationProps) {
  const headingId = useId();
  const listed = citedSourceList(sources, props.lines, props.runId);
  if (!listed.length) {
    return null;
  }
  return (
    <div className="mt-2 border-t border-rule pt-2 text-xs">
      <p id={headingId} className="text-ink-muted">
        Sources on this page
      </p>
      <ul aria-labelledby={headingId} className="mt-1 flex flex-wrap gap-x-3 gap-y-1">
        {listed.map(({ source, words, lineIds }) => (
          <li key={source.label}>
            {lineIds.length ? (
              <span
                onMouseEnter={() => props.onPreviewLines(lineIds)}
                onMouseLeave={() => props.onPreviewLines(null)}
                onClick={() => props.onPinLines(lineIds)}
                className={`cursor-pointer rounded-sm px-1 hover:bg-well ${props.pinnedKey === lineIds.join(" ") ? "bg-brand-wash" : ""}`}
              >
                <span className="font-medium text-brand">[{source.label}]</span> {words}
              </span>
            ) : (
              <span className="px-1 text-ink-muted">
                [{source.label}] {words}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

const SUGGESTIONS = ["Who is named on this page?", "Summarise this page in two sentences.", "Translate the first five lines."];

const VERDICT_TONE = { ok: "text-positive", warn: "text-warning", error: "text-negative" } as const;

/** What a second model made of the answer: its verdict, and its own wording where that differs. */
function VerificationNote({ message }: { message: WorkspaceChatMessage }) {
  // The answer comes first and the check after it (2026-11-25): meanwhile the
  // message says so, and a check that never came back says that.
  if (message.check === "checking") {
    return (
      <p role="status" className="mt-2 border-t border-rule pt-2 text-xs text-ink-muted">
        Checking the answer…
      </p>
    );
  }
  if (message.check === "not_checked") {
    return (
      <p className="mt-2 border-t border-rule pt-2 text-xs text-ink-muted">
        Not checked: the check did not finish before the connection closed.
      </p>
    );
  }
  const note = describeVerification(message.verification, message.content);
  if (!note) {
    return null;
  }
  return (
    <div className="mt-2 border-t border-rule pt-2 text-xs">
      <p className={`font-medium ${VERDICT_TONE[note.tone]}`} title={note.detail}>
        {note.label}
        <span className="sr-only">. {note.detail}</span>
      </p>
      {note.corrected ? (
        <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-6">
          <span className="text-xs text-ink-muted">The checker&apos;s wording: </span>
          {note.corrected}
        </p>
      ) : null}
      {note.notes.length ? (
        <ul className="mt-1 list-disc space-y-0.5 pl-4 text-ink-muted">
          {note.notes.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

/**
 * Questions about the page, answered from its own text. Each answer says how
 * it was grounded (the badges under it), so an answer without sources is seen
 * as one.
 */
export function ChatPanel({
  messages,
  sending,
  asking,
  loadingLabel,
  error,
  input,
  onInput,
  onSend,
  onClear,
  disabledReason,
  notice,
  announcement,
  hasTranscription,
  lines = [],
  runId = null,
  pinnedLineIds = null,
  onPreviewLines = () => {},
  onPinLines = () => {},
}: ChatPanelProps) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const citation: CitationProps = {
    lines,
    runId,
    pinnedKey: pinnedLineIds ? pinnedLineIds.join(" ") : null,
    onPreviewLines,
    onPinLines,
  };

  // Where the reader is: at the end of the conversation, or scrolled up to
  // read something. A streamed piece or a late verdict moves only the former.
  const atEndRef = useRef(true);
  const countRef = useRef(messages.length);
  useEffect(() => {
    const follow = followsTheChange({ countBefore: countRef.current, messages, wasAtEnd: atEndRef.current });
    countRef.current = messages.length;
    if (follow) {
      endRef.current?.scrollIntoView({ block: "end" });
    }
  }, [messages, sending]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      {notice ? (
        <p role="note" className="mx-6 mt-4 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
          {notice}
        </p>
      ) : null}
      <div
        role="log"
        aria-label="Conversation"
        aria-live={sending ? "off" : "polite"}
        aria-busy={sending}
        onScroll={(event) => {
          atEndRef.current = isAtEnd(event.currentTarget);
        }}
        className="min-h-0 flex-1 overflow-y-auto px-6 py-4"
      >
        {!messages.length ? (
          <div>
            <p className="text-sm">Ask about this page.</p>
            <p className="mt-1 text-sm text-ink-muted">
              {hasTranscription
                ? "Answers are drawn from the page's transcription and say so when the page does not hold the answer."
                : "Read the page first: without its transcription, answers cannot be grounded in the page."}
            </p>
            {!disabledReason ? (
              <ul className="mt-4 space-y-1.5">
                {SUGGESTIONS.map((suggestion) => (
                  <li key={suggestion}>
                    <button
                      type="button"
                      onClick={() => onInput(suggestion)}
                      className="w-full rounded-md border border-line px-3 py-2 text-left text-sm outline-none hover:bg-well focus-visible:ring-2 focus-visible:ring-brand"
                    >
                      {suggestion}
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : (
          <ol className="flex flex-col gap-4">
            {messages.map((message) => {
              const mine = message.role === "user";
              return (
                <li key={message.id} className={mine ? "ml-8" : ""}>
                  <p className="mb-1 text-xs text-ink-muted">{mine ? "You" : "ArchAI"}</p>
                  <div
                    className={`rounded-lg px-3 py-2 text-sm leading-6 ${
                      mine ? "bg-well" : message.failed ? "border border-negative/40" : "border border-line"
                    }`}
                  >
                    {message.content ? (
                      <p className={`whitespace-pre-wrap break-words ${message.failed ? "text-negative" : ""}`}>
                        {!mine && !message.failed && message.sources?.length ? (
                          <CitedAnswer text={message.content} sources={message.sources} {...citation} />
                        ) : (
                          message.content
                        )}
                      </p>
                    ) : sending && !mine ? (
                      <p role="status" className="text-ink-muted">
                        {loadingLabel}…
                      </p>
                    ) : null}
                    {message.extractionBadges?.length ? <ExtractionBadges badges={message.extractionBadges} /> : null}
                    {!mine && !message.failed && message.sources?.length ? <SourceList sources={message.sources} {...citation} /> : null}
                    {!mine && message.retrieval ? <RetrievalBadges retrieval={message.retrieval} /> : null}
                    {!mine ? <VerificationNote message={message} /> : null}
                    {message.imageUrl ? (
                      <Image
                        src={message.imageUrl}
                        alt={message.imageAlt || "Image from the page"}
                        width={1200}
                        height={1200}
                        unoptimized
                        className="mt-2 h-auto w-full rounded-md border border-line"
                      />
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ol>
        )}
        <div ref={endRef} />
      </div>

      <p role="status" className="sr-only">
        {announcement}
      </p>

      <div className="border-t border-rule px-6 py-4">
        {error ? (
          <p role="alert" className="mb-2 text-sm text-negative">
            {error}
          </p>
        ) : null}
        <label htmlFor="ask-input" className="sr-only">
          Question about this page
        </label>
        <textarea
          id="ask-input"
          value={input}
          onChange={(event) => onInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!sending) {
                onSend();
              }
            }
          }}
          rows={3}
          placeholder={disabledReason ?? "Ask a question about this page"}
          // Read-only, not disabled, while a question is sent: a disabled field
          // loses focus, and a keyboard user who pressed Enter was left on the
          // document body until the answer came.
          disabled={Boolean(disabledReason)}
          readOnly={sending}
          className="w-full resize-none rounded-md border border-field bg-sheet px-3 py-2 text-sm outline-none placeholder:text-ink-muted focus-visible:ring-2 focus-visible:ring-brand disabled:bg-well read-only:bg-well"
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <Button variant="quiet" onClick={onClear} disabled={!messages.length || sending}>
            Clear conversation
          </Button>
          <Button variant="primary" onClick={onSend} disabled={Boolean(disabledReason) || sending || !input.trim()}>
            {asking ? "Asking" : "Ask"} <Icon name="send" />
          </Button>
        </div>
      </div>
    </div>
  );
}
