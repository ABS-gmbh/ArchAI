// The [n] labels of a chat answer, tied to the lines of the page they stand for.
// Plain TypeScript, so citations.test.ts can import it.
//
// The backend labels each passage it shows the model [1], [2], ... and
// returns `sources`, one entry per label: the run it comes from and, where it
// knows them, the first and last line of the page (1-based, the pane's own
// numbering). The workspace turns a label with lines on the page shown into a
// button that lights those lines up; any other label stays plain text.

import type { ChatSource } from "@/lib/api/chat";

export type { ChatSource };

/** The same labels the backend recognises (chat_ai._CITATION_RE): [2], [2, 3], [2;3]. */
const CITATION = /\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]/g;

export type AnswerPart = { kind: "text"; text: string } | { kind: "cite"; labels: string[] };

/** The answer cut into text and citation groups, in order. */
export function splitCitations(text: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const at = match.index ?? 0;
    if (at > last) {
      parts.push({ kind: "text", text: text.slice(last, at) });
    }
    parts.push({ kind: "cite", labels: match[1].split(/[,;]/).map((label) => label.trim()) });
    last = at + match[0].length;
  }
  if (last < text.length) {
    parts.push({ kind: "text", text: text.slice(last) });
  }
  return parts;
}

const isLineNumber = (value: unknown): value is number => typeof value === "number" && Number.isInteger(value) && value >= 1;

/** One source as the backend sent it, keeping only what the workspace uses; null when it is not one. */
function parseSource(raw: unknown): ChatSource | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const value = raw as Record<string, unknown>;
  if (typeof value.label !== "string" || !/^\d{1,3}$/.test(value.label) || typeof value.run_id !== "string") {
    return null;
  }
  const source: ChatSource = { label: value.label, run_id: value.run_id };
  if (isLineNumber(value.line_start) && isLineNumber(value.line_end) && value.line_end >= value.line_start) {
    source.line_start = value.line_start;
    source.line_end = value.line_end;
  }
  if (typeof value.cited === "boolean") {
    source.cited = value.cited;
  }
  return source;
}

/** The answer's `sources`, validated; anything malformed is left out. */
export function parseSources(raw: unknown): ChatSource[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map(parseSource).filter((source): source is ChatSource => source !== null);
}

/** Whether *value* is a list of sources as the workspace stores them (workspacePersistence). */
export function isSourceList(value: unknown): boolean {
  return Array.isArray(value) && value.every((entry) => {
    const parsed = parseSource(entry);
    if (!parsed) {
      return false;
    }
    const raw = entry as Record<string, unknown>;
    // A stored entry has its lines whole or not at all.
    return (raw.line_start === undefined) === (raw.line_end === undefined) && (raw.line_start === undefined || parsed.line_start !== undefined);
  });
}

/**
 * The ids of the rows *source* stands for, among the page's *lines*, or none:
 * when it has no line numbers, comes from another run than the page's
 * (*runId*), or names lines the page does not have.
 */
export function sourceLineIds(source: ChatSource | undefined, lines: readonly { id: string }[], runId: string | null): string[] {
  if (!source || !runId || source.run_id !== runId || source.line_start === undefined || source.line_end === undefined) {
    return [];
  }
  if (source.line_end > lines.length) {
    return [];
  }
  return lines.slice(source.line_start - 1, source.line_end).map((line) => line.id);
}

/** "line 4" or "lines 4 to 7"; null without line numbers. */
export function describeLines(source: ChatSource): string | null {
  if (source.line_start === undefined || source.line_end === undefined) {
    return null;
  }
  return source.line_start === source.line_end ? `line ${source.line_start}` : `lines ${source.line_start} to ${source.line_end}`;
}

/** The sources the answer cites, in label order, each said in words for the list under the answer. */
export function citedSourceList(
  sources: readonly ChatSource[],
  lines: readonly { id: string }[],
  runId: string | null,
): { source: ChatSource; words: string; lineIds: string[] }[] {
  return sources
    .filter((source) => source.cited)
    .sort((a, b) => Number(a.label) - Number(b.label))
    .map((source) => {
      const lineIds = sourceLineIds(source, lines, runId);
      const lineWords = describeLines(source);
      const words = lineIds.length && lineWords
        ? lineWords.charAt(0).toUpperCase() + lineWords.slice(1)
        : source.run_id && runId && source.run_id !== runId
          ? "From another reading of the page"
          : "Lines not known";
      return { source, words, lineIds };
    });
}
