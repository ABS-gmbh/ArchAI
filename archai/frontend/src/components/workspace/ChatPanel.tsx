"use client";

import { useEffect, useRef } from "react";
import Image from "next/image";

import type { WorkspaceChatMessage } from "@/lib/workspace/types";

import { ExtractionBadges } from "./ExtractionBadges";
import { RetrievalBadges } from "./RetrievalBadges";
import { Button, Icon } from "./ui";

interface ChatPanelProps {
  messages: WorkspaceChatMessage[];
  sending: boolean;
  loadingLabel: string;
  error: string | null;
  input: string;
  onInput: (value: string) => void;
  onSend: () => void;
  onClear: () => void;
  /** Why the composer is off, in words; null when a question can be asked. */
  disabledReason: string | null;
  /** Said once to screen readers when a reply ends. */
  announcement: string;
  hasTranscription: boolean;
}

const SUGGESTIONS = ["Who is named on this page?", "Summarise this page in two sentences.", "Translate the first five lines."];

/**
 * Questions about the page, answered from its own text. Each answer says how
 * it was grounded (the badges under it), so an answer without sources is seen
 * as one.
 */
export function ChatPanel({
  messages,
  sending,
  loadingLabel,
  error,
  input,
  onInput,
  onSend,
  onClear,
  disabledReason,
  announcement,
  hasTranscription,
}: ChatPanelProps) {
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, sending]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div
        role="log"
        aria-label="Conversation"
        aria-live={sending ? "off" : "polite"}
        aria-busy={sending}
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
                  <div className={`rounded-lg px-3 py-2 text-sm leading-6 ${mine ? "bg-well" : "border border-line"}`}>
                    {message.content ? (
                      <p className="whitespace-pre-wrap break-words">{message.content}</p>
                    ) : sending && !mine ? (
                      <p role="status" className="text-ink-muted">
                        {loadingLabel}…
                      </p>
                    ) : null}
                    {message.extractionBadges?.length ? <ExtractionBadges badges={message.extractionBadges} /> : null}
                    {!mine && message.retrieval ? <RetrievalBadges retrieval={message.retrieval} /> : null}
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
              onSend();
            }
          }}
          rows={3}
          placeholder={disabledReason ?? "Ask a question about this page"}
          disabled={Boolean(disabledReason) || sending}
          className="w-full resize-none rounded-md border border-field bg-sheet px-3 py-2 text-sm outline-none placeholder:text-ink-muted focus-visible:ring-2 focus-visible:ring-brand disabled:bg-well"
        />
        <div className="mt-2 flex items-center justify-between gap-2">
          <Button variant="quiet" onClick={onClear} disabled={!messages.length || sending}>
            Clear conversation
          </Button>
          <Button variant="primary" onClick={onSend} disabled={Boolean(disabledReason) || sending || !input.trim()}>
            {sending ? "Sending" : "Ask"} <Icon name="send" />
          </Button>
        </div>
      </div>
    </div>
  );
}
