"use client";

import { useState } from "react";

import type { LinkCandidate, LinkingReport, MentionResult } from "@/lib/api/ocrAgent";

import { reasonInWords } from "./linkReason";
import { Button, Icon, Pill, type PillTone } from "./ui";

export type LinkingState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "failed"; error: string }
  | { phase: "ready"; report: LinkingReport };

const STATUS: Record<string, { tone: PillTone; label: string }> = {
  linked: { tone: "positive", label: "Linked" },
  ambiguous: { tone: "warning", label: "Ambiguous" },
  unresolved: { tone: "neutral", label: "Not linked" },
  skipped: { tone: "neutral", label: "Skipped" },
};

const TYPE_WORDS: Record<string, string> = {
  person: "Person",
  place: "Place",
  work: "Work",
  date: "Date",
  organization: "Organisation",
  org: "Organisation",
};

function recordUrl(candidate: LinkCandidate): string | null {
  const qid = String(candidate.qid || "").trim();
  if (/^Q\d+$/.test(qid)) {
    return `https://www.wikidata.org/wiki/${qid}`;
  }
  const viaf = String(candidate.viaf_id || "").trim();
  if (/^\d+$/.test(viaf)) {
    return `https://viaf.org/viaf/${viaf}`;
  }
  const geonames = String(candidate.geonames_id || "").trim();
  if (/^\d+$/.test(geonames)) {
    return `https://www.geonames.org/${geonames}`;
  }
  return null;
}

function recordId(candidate: LinkCandidate): string {
  return String(candidate.qid || candidate.authority_id || candidate.viaf_id || candidate.geonames_id || "").trim();
}

function score(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "—";
}

function Mention({ mention }: { mention: MentionResult }) {
  const [open, setOpen] = useState(false);
  const status = STATUS[mention.status] ?? { tone: "neutral" as PillTone, label: mention.status };
  const selected = mention.status === "linked" ? mention.selected : null;
  const candidates = mention.top_candidates ?? [];
  const url = selected ? recordUrl(selected) : null;
  return (
    <li className="border-t border-rule py-3 first:border-t-0">
      <div className="flex items-start justify-between gap-3">
        <p className="min-w-0 break-words font-text text-[17px] leading-6">{mention.surface}</p>
        <Pill tone={status.tone}>{status.label}</Pill>
      </div>
      <p className="mt-0.5 text-xs text-ink-muted">{TYPE_WORDS[String(mention.ent_type || "").toLowerCase()] ?? mention.ent_type ?? "Name"}</p>
      {selected ? (
        <p className="mt-2 text-sm">
          {url ? (
            <a href={url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 font-medium underline decoration-control underline-offset-2 hover:decoration-ink">
              {selected.label || recordId(selected)} <Icon name="external" className="text-ink-muted" />
              <span className="sr-only">(opens in a new tab)</span>
            </a>
          ) : (
            <span className="font-medium">{selected.label || recordId(selected)}</span>
          )}
          <span className="text-ink-muted">
            {" "}
            · {recordId(selected)} · score <span className="tabular-nums">{score(selected.score)}</span>
          </span>
          {selected.description ? <span className="mt-0.5 block text-ink-muted">{selected.description}</span> : null}
        </p>
      ) : null}
      {mention.reason && mention.status !== "linked" ? <p className="mt-2 text-sm text-ink-muted">{reasonInWords(mention.reason)}</p> : null}
      {candidates.length ? (
        <>
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((value) => !value)}
            className="mt-2 rounded text-xs font-medium text-ink underline decoration-control underline-offset-2 outline-none hover:decoration-ink focus-visible:ring-2 focus-visible:ring-brand"
          >
            {open ? "Hide" : "Show"} {candidates.length} {candidates.length === 1 ? "candidate" : "candidates"} considered
          </button>
          {open ? (
            <table className="mt-2 w-full text-xs">
              <thead>
                <tr className="text-left text-ink-muted">
                  <th scope="col" className="py-1 pr-2 font-normal">Record</th>
                  <th scope="col" className="py-1 pr-2 font-normal">Source</th>
                  <th scope="col" className="py-1 text-right font-normal">Score</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((candidate, index) => {
                  const link = recordUrl(candidate);
                  return (
                    <tr key={`${recordId(candidate)}-${index}`} className="border-t border-rule align-top">
                      <td className="py-1.5 pr-2">
                        {link ? (
                          <a href={link} target="_blank" rel="noreferrer" className="underline decoration-control underline-offset-2 hover:decoration-ink">
                            {candidate.label || recordId(candidate)}
                            <span className="sr-only"> (opens in a new tab)</span>
                          </a>
                        ) : (
                          candidate.label || recordId(candidate)
                        )}
                        {candidate.type_compatible === false ? <span className="block text-ink-muted">Wrong kind of record for this name</span> : null}
                      </td>
                      <td className="py-1.5 pr-2 text-ink-muted">{candidate.source || "—"}</td>
                      <td className="py-1.5 text-right tabular-nums">{score(candidate.score)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : null}
        </>
      ) : null}
    </li>
  );
}

/**
 * The names found on the page and what each was linked to: the record, its
 * score and, where nothing was linked, the reason. Candidates stay one click
 * away so a decision can be checked.
 */
export function EntitiesPanel({ state, hasRun, onRetry }: { state: LinkingState; hasRun: boolean; onRetry: () => void }) {
  if (!hasRun) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-dashed border-control p-5">
          <p className="text-sm">No names yet.</p>
          <p className="mt-1 text-sm text-ink-muted">
            People, places and works are found when a page is read, then matched against Wikidata, VIAF and GeoNames.
          </p>
        </div>
      </div>
    );
  }
  if (state.phase === "loading" || state.phase === "idle") {
    return (
      <div className="space-y-4 p-6" role="status">
        <span className="sr-only">Loading the names found on this page</span>
        {[0, 1, 2].map((row) => (
          <div key={row} aria-hidden="true">
            <div className="h-5 w-2/5 rounded-sm bg-well" />
            <div className="mt-2 h-3 w-1/4 rounded-sm bg-well" />
            <div className="mt-2 h-4 w-4/5 rounded-sm bg-well" />
          </div>
        ))}
      </div>
    );
  }
  if (state.phase === "failed") {
    return (
      <div className="p-6">
        <div role="alert" className="rounded-lg border border-line p-4">
          <p className="text-sm font-medium">The names could not be loaded.</p>
          <p className="mt-1 text-sm text-ink-muted">{state.error}</p>
          <Button variant="secondary" className="mt-3" onClick={onRetry}>
            <Icon name="retry" /> Try again
          </Button>
        </div>
      </div>
    );
  }
  const mentions = state.report.mention_results ?? [];
  if (!mentions.length) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-dashed border-control p-5">
          <p className="text-sm">No names were found on this page.</p>
          <p className="mt-1 text-sm text-ink-muted">
            If the page plainly names someone, the reading may be too uncertain for names to be picked out.
          </p>
        </div>
      </div>
    );
  }
  const linked = mentions.filter((mention) => mention.status === "linked").length;
  return (
    <div className="p-6">
      <p className="text-sm text-ink-muted">
        <span className="tabular-nums text-ink">{linked}</span> of <span className="tabular-nums text-ink">{mentions.length}</span>{" "}
        {mentions.length === 1 ? "name" : "names"} linked to an authority record.
      </p>
      <ul className="mt-3">
        {mentions.map((mention, index) => (
          <Mention key={mention.mention_id ?? `${mention.surface}-${index}`} mention={mention} />
        ))}
      </ul>
    </div>
  );
}
