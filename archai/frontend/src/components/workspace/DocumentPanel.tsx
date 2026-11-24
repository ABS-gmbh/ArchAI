"use client";

import { useState } from "react";

import type { WorkspaceDocument } from "@/lib/workspace/types";

import { Button } from "./ui";
import type { StoreStatus } from "./workspaceStore";

const TYPED_COMMANDS: { say: string; does: string }[] = [
  { say: "segment this page", does: "finds the layout regions and posts a summary" },
  { say: "crop the initial", does: "cuts out one kind of region, by its label" },
  { say: "explain the initial", does: "describes one kind of region from the image, once regions are found" },
  { say: "translate to English", does: "translates the transcription" },
];

interface DocumentPanelProps {
  document: WorkspaceDocument | null;
  /** Whether the open document is kept in this browser (workspaceStore.ts); null until known. */
  keep: StoreStatus | null;
  /** Close the document and remove it, with its readings, from this browser. */
  onClose: () => void;
}

/**
 * What was entered about the manuscript when it was opened, in full, whether
 * it is kept in this browser, and the things the Ask box understands besides
 * questions.
 */
export function DocumentPanel({ document, keep, onClose }: DocumentPanelProps) {
  const [confirming, setConfirming] = useState(false);
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
      <div className="mt-6 rounded-lg border border-rule p-4 text-sm">
        {keep === "unavailable" ? (
          <p role="status">
            This browser does not let the workspace keep documents, so this one will not survive a reload: its pages would
            have to be opened and read again.
          </p>
        ) : keep === "kept" ? (
          <p className="text-ink-muted">
            The pages, their readings and this conversation are kept in this browser, so a reload or a closed tab does not
            lose them (in a private window, only until the window is closed). Opening another document replaces them.
          </p>
        ) : null}
        {confirming ? (
          <div className="mt-3" role="group" aria-label="Close this document">
            <p>Close this document? Its readings and conversation are discarded, and its pages removed from this browser.</p>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button
                variant="primary"
                onClick={() => {
                  setConfirming(false);
                  onClose();
                }}
              >
                Close and discard
              </Button>
              <Button variant="secondary" autoFocus onClick={() => setConfirming(false)}>
                Keep it open
              </Button>
            </div>
          </div>
        ) : (
          <Button variant="secondary" className="mt-3" onClick={() => setConfirming(true)}>
            Close document
          </Button>
        )}
      </div>
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
