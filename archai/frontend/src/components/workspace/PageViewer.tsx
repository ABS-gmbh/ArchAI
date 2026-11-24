"use client";

import { type DragEvent, type ReactNode, useState } from "react";
import Image from "next/image";

import type { FullPageLine } from "@/lib/api/ocrAgent";

import { Button, Icon } from "./ui";

export interface PageViewerProps {
  page: { id: string; name: string; dataUrl: string } | null;
  zoom: number;
  onZoom: (next: number) => void;
  /** The page with its layout regions drawn on it, when regions have been found. */
  regionsSrc: string | null;
  showRegions: boolean;
  onShowRegions: (show: boolean) => void;
  /** Lines with a box on the page, in the page's own pixel frame. */
  lines: FullPageLine[];
  /** The frame the boxes are in; falls back to the image's natural size. */
  pageSize: { width: number; height: number } | null;
  activeLineId: string | null;
  onActiveLine: (id: string | null) => void;
  onSelectLine: (id: string) => void;
  onOpenPages: () => void;
  onDropFiles: (files: File[]) => void;
  /** Why the last attempt to open files failed, in words. */
  openError: string | null;
  /** Page navigation, shown inside the viewer where the top bar has no room for it. */
  pageNav?: ReactNode;
}

/**
 * The page on a well one step darker than the frame. Lines the recogniser read
 * are drawn over it as boxes: the one in focus, here or in the transcription,
 * carries the brand hue. Zoom changes the width, so the well scrolls normally.
 */
export function PageViewer({
  page,
  zoom,
  onZoom,
  regionsSrc,
  showRegions,
  onShowRegions,
  lines,
  pageSize,
  activeLineId,
  onActiveLine,
  onSelectLine,
  onOpenPages,
  onDropFiles,
  openError,
  pageNav,
}: PageViewerProps) {
  const [dropping, setDropping] = useState(false);
  const dropHandlers = {
    onDragOver: (event: DragEvent<HTMLDivElement>) => {
      if (event.dataTransfer.types.includes("Files")) {
        event.preventDefault();
        setDropping(true);
      }
    },
    onDragLeave: () => setDropping(false),
    onDrop: (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDropping(false);
      onDropFiles(Array.from(event.dataTransfer.files));
    },
  };
  const [natural, setNatural] = useState<{ id: string; width: number; height: number } | null>(null);
  const [showLines, setShowLines] = useState(true);

  if (!page) {
    // First run: the well is the place the pages go, so it is the drop target.
    return (
      <div {...dropHandlers} className="h-full min-h-64 rounded-2xl bg-well p-4">
        <div
          className={`flex h-full flex-col items-center justify-center gap-4 rounded-xl border border-dashed px-6 text-center ${
            dropping ? "border-brand bg-brand-wash" : "border-control"
          }`}
        >
          <div>
            <p className="text-base font-semibold">Open a manuscript&apos;s pages</p>
            <p className="mx-auto mt-1 max-w-sm text-sm text-ink-muted">
              Drop the page images here, or choose them. One image per page, in PNG, JPEG or TIFF; they are sorted by file name.
            </p>
          </div>
          <Button variant="primary" onClick={onOpenPages}>
            <Icon name="upload" /> Choose page images
          </Button>
          {openError ? (
            <p role="alert" className="max-w-sm text-sm text-negative">
              {openError}
            </p>
          ) : null}
          <dl className="mt-2 grid max-w-md gap-x-6 gap-y-3 text-left text-sm sm:grid-cols-3">
            <div>
              <dt className="font-medium">Read</dt>
              <dd className="text-ink-muted">Each line is found and transcribed, with a verdict on how far to trust it.</dd>
            </div>
            <div>
              <dt className="font-medium">Check</dt>
              <dd className="text-ink-muted">A line and its place on the page light up together.</dd>
            </div>
            <div>
              <dt className="font-medium">Ask</dt>
              <dd className="text-ink-muted">Answers come from the page&apos;s own text, with the names it mentions linked.</dd>
            </div>
          </dl>
        </div>
      </div>
    );
  }

  const usingRegions = showRegions && Boolean(regionsSrc);
  const measured = natural && natural.id === page.id ? natural : null;
  const frame = pageSize ?? measured;
  const boxed = lines.filter((line) => Array.isArray(line.bbox_xyxy) && line.bbox_xyxy.length === 4);
  const percent = Math.round(zoom * 100);

  return (
    <div {...dropHandlers} className={`relative h-full min-h-64 overflow-hidden rounded-2xl bg-well ${dropping ? "ring-2 ring-brand" : ""}`}>
      <div className="h-full overflow-auto p-4">
        <div className="relative mx-auto" style={{ width: `${zoom * 100}%` }}>
          <Image
            src={usingRegions && regionsSrc ? regionsSrc : page.dataUrl}
            alt={usingRegions ? `${page.name}, with its layout regions marked` : page.name}
            width={measured?.width ?? 1200}
            height={measured?.height ?? 1600}
            unoptimized
            priority
            className="block h-auto w-full select-none"
            onLoad={(event) => {
              const image = event.currentTarget;
              if (!usingRegions && image.naturalWidth && image.naturalHeight) {
                setNatural({ id: page.id, width: image.naturalWidth, height: image.naturalHeight });
              }
            }}
          />
          {frame && showLines && !usingRegions && boxed.length ? (
            <svg
              aria-hidden="true"
              viewBox={`0 0 ${frame.width} ${frame.height}`}
              preserveAspectRatio="none"
              className="absolute inset-0 h-full w-full"
              onMouseLeave={() => onActiveLine(null)}
            >
              {boxed.map((line) => {
                const [x0, y0, x1, y1] = line.bbox_xyxy as [number, number, number, number];
                const active = line.region_id === activeLineId;
                return (
                  <rect
                    key={line.region_id}
                    x={x0}
                    y={y0}
                    width={Math.max(1, x1 - x0)}
                    height={Math.max(1, y1 - y0)}
                    className="cursor-pointer"
                    fill={active ? "var(--aa-brand)" : "transparent"}
                    fillOpacity={active ? 0.16 : 0}
                    stroke="var(--aa-brand)"
                    strokeOpacity={active ? 1 : 0.35}
                    strokeWidth={active ? 2 : 1}
                    vectorEffect="non-scaling-stroke"
                    pointerEvents="all"
                    onMouseEnter={() => onActiveLine(line.region_id)}
                    onClick={() => onSelectLine(line.region_id)}
                  />
                );
              })}
            </svg>
          ) : null}
        </div>
      </div>

      {pageNav ? (
        <div className="absolute right-4 top-4 rounded-md border border-line bg-sheet sm:hidden">{pageNav}</div>
      ) : null}

      <div className="pointer-events-none absolute inset-x-4 bottom-4 flex flex-wrap items-end justify-between gap-2">
        <div className="pointer-events-auto flex items-center gap-3 rounded-md border border-line bg-sheet px-3 py-1.5 text-sm">
          {regionsSrc ? (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                className="h-4 w-4 accent-[var(--aa-ink)]"
                checked={showRegions}
                onChange={(event) => onShowRegions(event.target.checked)}
              />
              Layout regions
            </label>
          ) : null}
          {boxed.length ? (
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                className="h-4 w-4 accent-[var(--aa-ink)]"
                checked={showLines && !usingRegions}
                disabled={usingRegions}
                onChange={(event) => setShowLines(event.target.checked)}
              />
              Lines read
            </label>
          ) : null}
          {!regionsSrc && !boxed.length ? <span className="text-ink-muted">{page.name}</span> : null}
        </div>
        <div className="pointer-events-auto flex items-center rounded-md border border-line bg-sheet">
          <Button variant="quiet" className="w-8 px-0" aria-label="Zoom out" disabled={zoom <= 0.5} onClick={() => onZoom(zoom - 0.25)}>
            <Icon name="minus" />
          </Button>
          <span className="w-12 text-center text-sm tabular-nums" aria-live="polite">
            <span className="sr-only">Zoom </span>
            {percent}%
          </span>
          <Button variant="quiet" className="w-8 px-0" aria-label="Zoom in" disabled={zoom >= 3} onClick={() => onZoom(zoom + 0.25)}>
            <Icon name="plus" />
          </Button>
          <span aria-hidden="true" className="h-5 w-px bg-line" />
          <Button variant="quiet" onClick={() => onZoom(1)}>
            Fit width
          </Button>
        </div>
      </div>
    </div>
  );
}
