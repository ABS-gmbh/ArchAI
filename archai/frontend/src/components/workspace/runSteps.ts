// A run's steps in a reader's words, for the Steps panel (RunPanel.tsx) and
// the reading progress line. Plain TypeScript, no JSX, so the unit tests can
// import it (runSteps.test.ts).

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
  [/^Full-page OCR text stored\.$/, () => ""],
  [/^Running the full-page post-OCR pipeline\.$/, () => ""],
  [/^Full-page OCR run received \(.+\)\.$/, () => ""],
  // A reading the quality gate rates too uncertain to search or link.
  [/^degraded_capture chunks=(\d+), mentions=(\d+)$/, (m) => `${plural(m[1], "passage", "passages")}, ${plural(m[2], "name", "names")} (names found with the cautious search used for an uncertain reading)`],
  [/^Authority linking SKIPPED: token_search_allowed=False \(quality=(\w+)\)$/, (m) => `Skipped: the reading was rated ${m[1].toLowerCase()}, too uncertain to look names up.`],
  [/^RAG auto-index SKIPPED: token_search_allowed=False \(quality=(\w+)\)$/, (m) => `Skipped: the reading was rated ${m[1].toLowerCase()}, too uncertain to search.`],
  [/^token_search_allowed=False quality=\w+$/, () => ""],
  [/^Deferred linking stored unresolved mentions: total=(\d+)$/, (m) => `${plural(m[1], "name", "names")} kept, not linked`],
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
