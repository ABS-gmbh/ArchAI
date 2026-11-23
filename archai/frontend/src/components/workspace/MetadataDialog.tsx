"use client";

import type { WorkspaceDocumentMetadata } from "@/lib/workspace/types";

import { Button } from "./ui";

interface MetadataDialogProps {
  draft: WorkspaceDocumentMetadata;
  pageCount: number;
  error: string | null;
  onChange: (next: WorkspaceDocumentMetadata) => void;
  onCancel: () => void;
  onSave: () => void;
}

const FIELDS: { key: keyof WorkspaceDocumentMetadata; label: string; placeholder: string; required?: boolean; wide?: boolean }[] = [
  { key: "language", label: "Language", placeholder: "latin, middle_french, anglo_norman", required: true },
  { key: "year", label: "Year", placeholder: "1248, or c. 1250", required: true },
  { key: "placeOrOrigin", label: "Place of origin", placeholder: "Lausanne, Paris, Toledo" },
  { key: "scriptFamily", label: "Script", placeholder: "caroline, textualis, cursiva" },
  { key: "documentType", label: "Kind of document", placeholder: "gospel, charter, cartulary", wide: true },
];

/**
 * What is known about the manuscript before it is read. Language and year
 * steer the recogniser, so both are required. Focus stays inside the dialog
 * and Escape cancels.
 */
export function MetadataDialog({ draft, pageCount, error, onChange, onCancel, onSave }: MetadataDialogProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="metadata-dialog-title"
        aria-describedby="metadata-dialog-description"
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.stopPropagation();
            onCancel();
            return;
          }
          if (event.key !== "Tab") {
            return;
          }
          const focusable = Array.from(
            event.currentTarget.querySelectorAll<HTMLElement>("input:not([disabled]), textarea:not([disabled]), button:not([disabled])"),
          );
          if (!focusable.length) {
            return;
          }
          const first = focusable[0];
          const last = focusable[focusable.length - 1];
          const active = document.activeElement;
          if (event.shiftKey && (active === first || !event.currentTarget.contains(active))) {
            event.preventDefault();
            last.focus();
          } else if (!event.shiftKey && active === last) {
            event.preventDefault();
            first.focus();
          }
        }}
        className="max-h-full w-full max-w-xl overflow-y-auto rounded-2xl border border-line bg-sheet p-6 shadow-[0_16px_48px_rgba(19,20,22,0.18)]"
      >
        <h2 id="metadata-dialog-title" className="text-base font-semibold">
          About this manuscript
        </h2>
        <p id="metadata-dialog-description" className="mt-1 text-sm text-ink-muted">
          {pageCount} {pageCount === 1 ? "page" : "pages"} selected. Language and year choose how the pages are read; the rest
          is kept with the document.
        </p>
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          {FIELDS.map((field, index) => {
            const invalid = Boolean(error) && field.required && !draft[field.key].trim();
            return (
              <label key={field.key} className={`block text-sm ${field.wide ? "sm:col-span-2" : ""}`}>
                <span className="mb-1.5 block font-medium">
                  {field.label}
                  {field.required ? <span className="font-normal text-ink-muted"> (required)</span> : null}
                </span>
                <input
                  autoFocus={index === 0}
                  required={field.required}
                  aria-invalid={invalid}
                  value={draft[field.key]}
                  onChange={(event) => onChange({ ...draft, [field.key]: event.target.value })}
                  placeholder={field.placeholder}
                  className={`h-10 w-full rounded-lg border bg-sheet px-3 outline-none placeholder:text-ink-muted focus-visible:ring-2 focus-visible:ring-brand ${
                    invalid ? "border-negative" : "border-control"
                  }`}
                />
              </label>
            );
          })}
          <label className="block text-sm sm:col-span-2">
            <span className="mb-1.5 block font-medium">Notes</span>
            <textarea
              value={draft.notes}
              onChange={(event) => onChange({ ...draft, notes: event.target.value })}
              rows={3}
              placeholder="Repository, shelfmark, dating, provenance"
              className="w-full rounded-lg border border-control bg-sheet px-3 py-2 outline-none placeholder:text-ink-muted focus-visible:ring-2 focus-visible:ring-brand"
            />
          </label>
        </div>
        {error ? (
          <p role="alert" className="mt-3 text-sm text-negative">
            {error}
          </p>
        ) : null}
        <div className="mt-6 flex justify-end gap-2">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onSave}>
            Open document
          </Button>
        </div>
      </div>
    </div>
  );
}
