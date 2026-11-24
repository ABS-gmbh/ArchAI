// Why a name was not linked, in a reader's words. The linker writes its
// reasons for its log (app/services/entity_scoring.py, authority_linking.py);
// the Names panel (EntitiesPanel.tsx) shows them through this. Plain
// TypeScript, so linkReason.test.ts can import it.

/** The linker's reason in a reader's words; an unrecognised reason is shown as the backend wrote it. */
export function reasonInWords(reason: string): string {
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
