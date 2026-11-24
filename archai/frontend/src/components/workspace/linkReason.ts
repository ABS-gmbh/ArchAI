// Why a name was not linked, in a reader's words. The linker writes its
// reasons for its log (app/services/entity_scoring.py, authority_linking.py);
// the Names panel (EntitiesPanel.tsx) shows them through this. Plain
// TypeScript, so linkReason.test.ts can import it.

export interface LinkingCounts {
  linked: number;
  notLinked: number;
  ambiguous: number;
  /** Not looked up at all: dates, roles, the time budget spent. */
  skipped: number;
}

/**
 * The linking step's summary, also the Names panel's, so the two cannot be read
 * against each other: "2 linked, 3 not linked, 1 not looked up". Ambiguous and
 * not looked up are left out at zero.
 */
export function linkingSummary(counts: LinkingCounts): string {
  const parts = [`${counts.linked} linked`, `${counts.notLinked} not linked`];
  if (counts.ambiguous) parts.push(`${counts.ambiguous} ambiguous`);
  if (counts.skipped) parts.push(`${counts.skipped} not looked up`);
  return parts.join(", ");
}

/** A page's names counted as the backend's summary counts them (trace_pipeline.py). */
export function summaryOfMentions(mentions: { status?: string | null }[]): LinkingCounts {
  const counts: LinkingCounts = { linked: 0, notLinked: 0, ambiguous: 0, skipped: 0 };
  for (const { status } of mentions) {
    if (status === "linked") counts.linked += 1;
    else if (status === "ambiguous") counts.ambiguous += 1;
    else if (status === "skipped") counts.skipped += 1;
    else counts.notLinked += 1;
  }
  return counts;
}

export type StatusTone = "positive" | "warning" | "neutral";

/** A name's status as a pill: the summary's words. */
export function mentionStatus(status: string): { tone: StatusTone; label: string } {
  if (status === "linked") return { tone: "positive", label: "Linked" };
  if (status === "ambiguous") return { tone: "warning", label: "Ambiguous" };
  if (status === "skipped") return { tone: "neutral", label: "Not looked up" };
  return { tone: "neutral", label: "Not linked" };
}

/** The linker's reason in a reader's words; an unrecognised reason is shown as the backend wrote it. */
export function reasonInWords(reason: string): string {
  if (/^ent_type=date\b/i.test(reason)) {
    return "A date: dates are not looked up.";
  }
  if (/^ent_type=role\b/i.test(reason)) {
    return "A role or title: these are not looked up.";
  }
  const budget = /^linking time budget of ([\d.]+)s spent/i.exec(reason);
  if (budget) {
    return `Not looked up: the time for linking this page (${budget[1]} s) ran out.`;
  }
  if (/^low_evidence_place:/i.test(reason)) {
    return "Not linked: too little on the page suggests this is a place.";
  }
  if (/^name_likeness=/i.test(reason)) {
    return "Not linked: this does not read like a name; it may be a misreading.";
  }
  const below = /best score \(([\d.]+)\) < threshold \(([\d.]+)\)/.exec(reason);
  if (below) {
    return `The best match scored ${Number(below[1]).toFixed(2)}; a link needs ${Number(below[2]).toFixed(2)}.`;
  }
  const gated = /token_search_allowed=False(?:\s+quality=(\w+))?/i.exec(reason);
  if (gated) {
    return gated[1]
      ? `Not matched: the page's reading was graded ${gated[1].toLowerCase()}, too uncertain for names to be linked.`
      : "Not matched: the page's reading is too uncertain for names to be linked.";
  }
  if (/no candidates found/i.test(reason)) {
    return "No authority record was found for this name.";
  }
  const wrongKind = /^all (\d+) candidate\(s\) type_incompatible/i.exec(reason);
  if (wrongKind) {
    return wrongKind[1] === "1"
      ? "The one record found is not the right kind of record for this name."
      : `None of the ${wrongKind[1]} records found is the right kind of record for this name.`;
  }
  const notClose = /name similarity reaches ([\d.]+) \(.*similarity=([\d.]+)\)/i.exec(reason);
  if (notClose) {
    return `No record of the right kind has a name close enough: the closest is ${Number(notClose[2]).toFixed(2)} alike, a link needs ${Number(notClose[1]).toFixed(2)}.`;
  }
  return reason.replace(/\s*\[ocr_quality=\w+\]\s*$/, "").replace(/^./, (first) => first.toUpperCase());
}
