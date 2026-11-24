import type { RunSnapshot } from "@/lib/api/ocrAgent";

import { Pill, type PillTone } from "./ui";

const STAGE_WORDS: Record<string, string> = {
  RESERVED: "Sending the page",
  RECEIVED: "Request received",
  OCR_RUNNING: "Reading the lines",
  OCR_DONE: "Lines read",
  PROOFREAD_DONE: "Proofread",
  QUALITY_GATES: "Quality check",
  QUALITY_BLOCKED: "Stopped by the quality check",
  ANALYZE_RUNNING: "Finding passages and names",
  ANALYZE_DONE: "Passages and names found",
  STORED: "Saved",
  AUTHORITY_LINKING_RUNNING: "Linking names to authority records",
  AUTHORITY_LINKING_DONE: "Names checked against authority records",
  AUTHORITY_LINKING_DEFERRED: "Names not linked: the reading is too uncertain",
  AUTHORITY_LINKING_ERROR: "Linking failed",
  LINKING_SKIPPED: "Linking skipped",
  INDEX_RUNNING: "Indexing for search",
  INDEX_DONE: "Indexed for search",
  INDEX_SKIPPED: "Not indexed for search",
  INDEX_ERROR: "Indexing failed",
  DONE: "Finished",
  FAILED: "Failed",
};

export function stageInWords(stage: string | null | undefined): string {
  const key = String(stage || "").toUpperCase();
  return STAGE_WORDS[key] ?? (key ? key.replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase()) : "Working");
}

function statusPill(status: string | undefined): { tone: PillTone; label: string } {
  switch (String(status || "").toUpperCase()) {
    case "COMPLETED":
      return { tone: "positive", label: "Finished" };
    case "FAILED":
      return { tone: "negative", label: "Failed" };
    case "FAILED_QUALITY":
      return { tone: "warning", label: "Stopped by the quality check" };
    case "RUNNING":
    case "RESERVED":
      return { tone: "neutral", label: "Running" };
    default:
      return { tone: "neutral", label: status ? status.toLowerCase() : "Not started" };
  }
}

const plural = (count: string, one: string, many: string) => `${count} ${count === "1" ? one : many}`;

/**
 * The backend's event messages are written for its log ("chunks=13,
 * mentions=0", "Run results persisted in SQLite."). Known ones are put in the
 * reader's words, or dropped where they only repeat the step's name; an
 * unknown message is shown as it came.
 */
const MESSAGE_WORDS: [RegExp, (match: RegExpExecArray) => string][] = [
  [/^Full-page OCR request received\.$/, () => ""],
  [/^Full-page OCR running\.$/, () => ""],
  [/^Full-page OCR done \(segmented\)\.$/, () => "The page's layout was found and each line read on its own."],
  [/^Full-page OCR done \((.+)\)\.$/, (m) => `Read with ${m[1]}.`],
  [/^Running chunk\/entity analysis\.$/, () => ""],
  [/^MENTION_RECALL_LOW: mentions=(\d+) < expected_min=(\d+)/, (m) => `Fewer names than a text this long usually has: ${m[1]} found, at least ${m[2]} expected.`],
  [/^chunks=(\d+), mentions=(\d+)$/, (m) => `${plural(m[1], "passage", "passages")}, ${plural(m[2], "name", "names")}`],
  [/^Run results persisted in SQLite\.$/, () => ""],
  [/^Running authority linking \(Wikidata\)\.$/, () => ""],
  [/^Authority linking complete: linked=(\d+), unresolved=(\d+), ambiguous=(\d+)\.$/, (m) => `${m[1]} linked, ${m[2]} not linked, ${m[3]} ambiguous`],
  [/^Auto-indexing into vector store\.$/, () => ""],
  [/^Indexed (\d+) chunks\.$/, (m) => `${plural(m[1], "passage", "passages")} indexed`],
  [/^Full-page pipeline completed successfully\.$/, () => ""],
];

export function messageInWords(message: string | null | undefined): string {
  const text = String(message || "").trim();
  for (const [pattern, words] of MESSAGE_WORDS) {
    const match = pattern.exec(text);
    if (match) {
      return words(match);
    }
  }
  return text;
}

function clock(ts: string | undefined): string {
  if (!ts) {
    return "";
  }
  const date = new Date(ts);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

/** What the pipeline did with this page, step by step, as its run recorded it. */
export function RunPanel({ snapshot, runId }: { snapshot: RunSnapshot | null; runId: string | null }) {
  if (!snapshot) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-dashed border-control p-5">
          <p className="text-sm">No run to show for this page.</p>
          <p className="mt-1 text-sm text-ink-muted">
            {runId
              ? "This page was read without step-by-step tracking. Read it again to follow each step here."
              : "Each step of reading a page is recorded here: lines read, the quality check, names found, linking and indexing."}
          </p>
        </div>
      </div>
    );
  }
  // Reading the lines logs a start and then a count every few lines; one row with the latest count is enough.
  const all = snapshot.events ?? [];
  const lastReading = all.reduce((last, event, index) => (event.stage === "OCR_RUNNING" ? index : last), -1);
  const events = all
    .filter((event, index) => event.stage !== "OCR_RUNNING" || index === lastReading)
    .map((event) => {
      const count = /^lines (\d+)\/(\d+)$/.exec(event.message);
      return { ...event, message: count ? `${count[1]} of ${count[2]} lines` : messageInWords(event.message) };
    });
  const pill = statusPill(snapshot.run?.status ?? snapshot.status);
  return (
    <div className="p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Pill tone={pill.tone}>{pill.label}</Pill>
        <p className="text-xs tabular-nums text-ink-muted">
          {typeof snapshot.chunks_count === "number" ? `${snapshot.chunks_count} passages` : ""}
          {typeof snapshot.chunks_count === "number" && typeof snapshot.mentions_count === "number" ? " · " : ""}
          {typeof snapshot.mentions_count === "number" ? `${snapshot.mentions_count} names` : ""}
        </p>
      </div>
      <ol className="mt-4">
        {events.map((event, index) => {
          const failed = event.event === "ERROR";
          return (
            <li key={`${index}-${event.stage}-${event.message}`} className="grid grid-cols-[3.75rem_minmax(0,1fr)] gap-x-3 border-t border-rule py-2 first:border-t-0">
              <span className="pt-0.5 text-xs tabular-nums text-ink-muted">{clock(event.ts)}</span>
              <span className="min-w-0">
                <span className={`block text-sm ${failed ? "font-medium text-negative" : ""}`}>{stageInWords(event.stage)}</span>
                {event.message ? <span className="block break-words text-xs text-ink-muted">{event.message}</span> : null}
              </span>
            </li>
          );
        })}
      </ol>
      {runId ? <p className="mt-4 break-all text-xs text-ink-muted">Run {runId}</p> : null}
    </div>
  );
}
