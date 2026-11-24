import type { WorkspaceDocument } from "@/lib/workspace/types";

const TYPED_COMMANDS: { say: string; does: string }[] = [
  { say: "segment this page", does: "finds the layout regions and posts a summary" },
  { say: "crop the initial", does: "cuts out one kind of region, by its label" },
  { say: "explain the initial", does: "describes one kind of region from the image, once regions are found" },
  { say: "translate to English", does: "translates the transcription" },
];

/**
 * What was entered about the manuscript when it was opened, in full, and the
 * things the Ask box understands besides questions.
 */
export function DocumentPanel({ document }: { document: WorkspaceDocument | null }) {
  if (!document) {
    return (
      <div className="p-6">
        <div className="rounded-lg border border-dashed border-control p-5">
          <p className="text-sm">No document is open.</p>
          <p className="mt-1 text-sm text-ink-muted">What you enter about a manuscript when opening it is kept here.</p>
        </div>
      </div>
    );
  }
  const { metadata } = document;
  const rows: [string, string][] = [
    ["Language", metadata.language],
    ["Year", metadata.year],
    ["Place of origin", metadata.placeOrOrigin],
    ["Script", metadata.scriptFamily],
    ["Kind of document", metadata.documentType],
    ["Notes", metadata.notes],
  ];
  return (
    <div className="p-6">
      <h3 className="break-words text-sm font-medium">{document.name}</h3>
      <p className="mt-0.5 text-xs tabular-nums text-ink-muted">
        {document.pages.length} {document.pages.length === 1 ? "page" : "pages"}
      </p>
      <dl className="mt-4">
        {rows.map(([label, value]) => (
          <div key={label} className="grid grid-cols-[8rem_minmax(0,1fr)] gap-x-3 border-t border-rule py-2 text-sm first:border-t-0">
            <dt className="text-ink-muted">{label}</dt>
            <dd className="min-w-0 whitespace-pre-wrap break-words">{value.trim() ? value : <span className="text-ink-muted">—</span>}</dd>
          </div>
        ))}
      </dl>
      <h3 className="mt-6 text-sm font-medium">The Ask box also takes instructions</h3>
      <ul className="mt-2 space-y-2 text-sm">
        {TYPED_COMMANDS.map((command) => (
          <li key={command.say}>
            <span className="font-medium">&ldquo;{command.say}&rdquo;</span> <span className="text-ink-muted">{command.does}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
