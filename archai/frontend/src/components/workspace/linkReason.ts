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

// ── Kinds of mention ────────────────────────────────────────────────────
//
// What the backend marks on a page is not all names: the title rule marks
// "pere", "euesque", "prince", role words are marked, and dates are. They were
// listed and counted as names ("6 names found" on the Antitus page, three of
// them titles; live regression, 2026-11-24). The panel now counts names only
// and lists the rest after them, each with its reason. The kinds are the
// backend's ent_type values (app/services/mention_extraction.py,
// pipeline_hardening.py, ocr_test_fixtures.py); linkReason.test.ts reads them
// from the backend's sources and fails on one without a word here.

interface KindWords {
  /** The kind as a row's label. */
  label: string;
  /** One and several, for the "Also marked" line; names need none. */
  one?: string;
  many?: string;
}

export const KIND_WORDS: Record<string, KindWords> = {
  person: { label: "Person" },
  place: { label: "Place" },
  work: { label: "Work" },
  organization: { label: "Organisation" },
  org: { label: "Organisation" },
  unknown: { label: "Name" },
  title: { label: "Title", one: "role or title", many: "roles or titles" },
  role: { label: "Role", one: "role or title", many: "roles or titles" },
  date: { label: "Date", one: "date", many: "dates" },
  // The canned answers of the test fixtures (ENABLE_TEST_OCR_FIXTURES,
  // app/services/ocr_test_fixtures.py): terms a reader studies, not names.
  religious_title: { label: "Religious title", one: "role or title", many: "roles or titles" },
  social_role: { label: "Social role", one: "role or title", many: "roles or titles" },
  temporal_liturgical_term: { label: "Liturgical time", one: "date", many: "dates" },
  abstract_concept: term("Concept"),
  action_or_warning: term("Action or warning"),
  glossing_language_element: term("Gloss"),
  grammatical_term: term("Grammatical term"),
  lexical_example: term("Example word"),
  moral_action: term("Moral action"),
  moral_practical_concept: term("Moral concept"),
  morphological_explanation: term("Grammar note"),
  pedagogical_function: term("Teaching aid"),
  religious_group: term("Religious group"),
  religious_reference: term("Religious reference"),
  state_or_fortune: term("State or fortune"),
  theological_concept: term("Theological concept"),
  verb_paradigm_example: term("Verb form"),
};

function term(label: string): KindWords {
  return { label, one: "term", many: "terms" };
}

/** A mention's kind as its row shows it; an unknown kind reads "Name". */
export function kindLabel(entType: string | null | undefined): string {
  return KIND_WORDS[String(entType || "").trim().toLowerCase()]?.label ?? "Name";
}

/** Whether a kind is a name: everything but the kinds with words for the "Also marked" line. */
export function isName(entType: string | null | undefined): boolean {
  return !KIND_WORDS[String(entType || "").trim().toLowerCase()]?.one;
}

/** The mentions split into names and the rest, each in the order the backend gave. */
export function splitMentions<T extends { ent_type?: string | null }>(mentions: T[]): { names: T[]; others: T[] } {
  return {
    names: mentions.filter((mention) => isName(mention.ent_type)),
    others: mentions.filter((mention) => !isName(mention.ent_type)),
  };
}

/** "3 roles or titles, 1 date": the rest, counted by what a reader calls them. */
export function otherMarksSummary(others: { ent_type?: string | null }[]): string {
  const counts = new Map<string, { one: string; many: string; count: number }>();
  for (const mention of others) {
    const words = KIND_WORDS[String(mention.ent_type || "").trim().toLowerCase()];
    const one = words?.one ?? "other mark";
    const entry = counts.get(one) ?? { one, many: words?.many ?? "other marks", count: 0 };
    entry.count += 1;
    counts.set(one, entry);
  }
  return [...counts.values()].map(({ one, many, count }) => `${count} ${count === 1 ? one : many}`).join(", ");
}
