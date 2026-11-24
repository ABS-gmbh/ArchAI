"use client";

import { type DragEvent, type ReactNode, useState } from "react";
import Image from "next/image";

import type { FullPageLine } from "@/lib/api/ocrAgent";

import type { StoredLabelRegionMap } from "./workspaceSegmentation";

import { Button, Icon } from "./ui";

export interface PageViewerProps {
  page: { id: string; name: string; dataUrl: string } | null;
  zoom: number;
  onZoom: (next: number) => void;
  /** The page's layout regions by kind, in the page image's own pixel frame, when they have been found. */
  regions: StoredLabelRegionMap | null;
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

function regionArea(box: [number, number, number, number]): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

/**
 * The page on a well one step darker than the frame. Lines the recogniser read
 * are drawn over it as boxes: the one in focus, here or in the transcription,
 * carries the brand hue. Layout regions are drawn the same way, outlined in
 * ink, with the kind in focus (in the list or under the pointer) in the brand
 * hue. Zoom changes the width, so the well scrolls normally.
 */
export function PageViewer({
  page,
  zoom,
  onZoom,
  regions,
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
  // The kind of region in focus: pinned by a click, or under the pointer. Both belong to one page.
  const [pinnedKind, setPinnedKind] = useState<{ pageId: string; kind: string } | null>(null);
  const [hoverKind, setHoverKind] = useState<string | null>(null);

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
              Drop the page images here, or choose them. One image per page, in PNG or JPEG; they are sorted by file name.
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

  // Largest count first; within the drawing, largest area first, so a small region stays reachable inside a column.
  const kinds = Object.entries(regions ?? {})
    .filter(([, items]) => items.length > 0)
    .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
  const hasRegions = kinds.length > 0;
  const usingRegions = showRegions && hasRegions;
  const regionShapes = kinds
    .flatMap(([kind, items]) => items.map((region) => ({ kind, region })))
    .sort((a, b) => regionArea(b.region.bbox_xyxy) - regionArea(a.region.bbox_xyxy));
  const pinned = pinnedKind && pinnedKind.pageId === page.id ? pinnedKind.kind : null;
  const focusKind = hoverKind ?? pinned;
  const measured = natural && natural.id === page.id ? natural : null;
  const frame = pageSize ?? measured;
  const boxed = lines.filter((line) => Array.isArray(line.bbox_xyxy) && line.bbox_xyxy.length === 4);
  const percent = Math.round(zoom * 100);

  return (
    <div {...dropHandlers} className={`relative h-full min-h-64 overflow-hidden rounded-2xl bg-well ${dropping ? "ring-2 ring-brand" : ""}`}>
      {/* A named tab stop, so the arrow keys scroll a zoomed page: Chrome makes a
          scroller focusable by itself but gives it no name, Safari not at all. */}
      <div
        tabIndex={0}
        role="region"
        aria-label="Page image"
        className="h-full overflow-auto p-4 outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand"
      >
        <div className="relative mx-auto" style={{ width: `${zoom * 100}%` }}>
          <Image
            src={page.dataUrl}
            alt={page.name}
            width={measured?.width ?? 1200}
            height={measured?.height ?? 1600}
            unoptimized
            priority
            className="block h-auto w-full select-none"
            onLoad={(event) => {
              const image = event.currentTarget;
              if (image.naturalWidth && image.naturalHeight) {
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
          {measured && usingRegions ? (
            // Regions are in the image's own pixels (the image is what was segmented), not in the read's frame.
            <svg
              aria-hidden="true"
              viewBox={`0 0 ${measured.width} ${measured.height}`}
              preserveAspectRatio="none"
              className="absolute inset-0 h-full w-full"
              onMouseLeave={() => setHoverKind(null)}
            >
              {regionShapes.map(({ kind, region }) => {
                const inFocus = kind === focusKind;
                const dimmed = Boolean(focusKind) && !inFocus;
                const look = {
                  className: "cursor-pointer",
                  fill: inFocus ? "var(--aa-brand)" : "transparent",
                  fillOpacity: inFocus ? 0.14 : 0,
                  stroke: inFocus ? "var(--aa-brand)" : "var(--aa-ink)",
                  strokeOpacity: inFocus ? 1 : dimmed ? 0.18 : 0.45,
                  strokeWidth: inFocus ? 2 : 1,
                  vectorEffect: "non-scaling-stroke" as const,
                  pointerEvents: "all" as const,
                  onMouseEnter: () => setHoverKind(kind),
                  onClick: () => setPinnedKind(pinned === kind ? null : { pageId: page.id, kind }),
                };
                const polygons = (region.polygons ?? []).filter((points) => points.length >= 6);
                if (polygons.length) {
                  return polygons.map((points, index) => (
                    <polygon key={`${kind}-${region.region_id}-${index}`} points={points.join(" ")} {...look}>
                      <title>{kind}</title>
                    </polygon>
                  ));
                }
                const [x0, y0, x1, y1] = region.bbox_xyxy;
                return (
                  <rect key={`${kind}-${region.region_id}`} x={x0} y={y0} width={Math.max(1, x1 - x0)} height={Math.max(1, y1 - y0)} {...look}>
                    <title>{kind}</title>
                  </rect>
                );
              })}
            </svg>
          ) : null}
        </div>
      </div>

      {usingRegions ? (
        <details open className="absolute left-4 top-4 w-56 max-w-[calc(100%-2rem)] rounded-md border border-line bg-sheet text-sm">
          <summary className="cursor-pointer select-none rounded-md px-3 py-1.5 outline-none focus-visible:ring-2 focus-visible:ring-brand">
            Layout regions <span className="tabular-nums text-ink-muted">{regionShapes.length}</span>
          </summary>
          <ul className="max-h-64 overflow-y-auto border-t border-rule p-1" onMouseLeave={() => setHoverKind(null)}>
            {kinds.map(([kind, items]) => (
              <li key={kind}>
                <button
                  type="button"
                  aria-pressed={pinned === kind}
                  onMouseEnter={() => setHoverKind(kind)}
                  onFocus={() => setHoverKind(kind)}
                  onBlur={() => setHoverKind(null)}
                  onClick={() => setPinnedKind(pinned === kind ? null : { pageId: page.id, kind })}
                  className={`flex w-full items-center justify-between gap-3 rounded px-2 py-1 text-left outline-none focus-visible:ring-2 focus-visible:ring-brand ${
                    focusKind === kind ? "bg-brand-wash" : "hover:bg-well"
                  } ${
                    // Pressed is more than a tint (1.1:1 on the sheet): a brand bar
                    // on the left edge (5.1:1 on the tint) and a heavier name.
                    pinned === kind ? "font-medium shadow-[inset_3px_0_0_var(--aa-brand)]" : ""
                  }`}
                >
                  <span className="min-w-0 truncate">{kind.replace(/\s+/g, " ").trim()}</span>
                  <span className="tabular-nums text-ink-muted">{items.length}</span>
                </button>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {pageNav ? (
        <div className="absolute right-4 top-4 rounded-md border border-line bg-sheet sm:hidden">{pageNav}</div>
      ) : null}

      <div className="pointer-events-none absolute inset-x-4 bottom-4 flex flex-wrap items-end justify-between gap-2">
        <div className="pointer-events-auto flex items-center gap-3 rounded-md border border-line bg-sheet px-3 py-1.5 text-sm">
          {hasRegions ? (
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
          {!hasRegions && !boxed.length ? <span className="text-ink-muted">{page.name}</span> : null}
        </div>
        <div className="pointer-events-auto flex items-center rounded-md border border-line bg-sheet">
          <Button variant="quiet" className="w-8 px-0" aria-label="Zoom out" disabled={zoom <= 0.5} onClick={() => onZoom(zoom - 0.25)}>
            <Icon name="minus" />
          </Button>
          <span className="w-12 text-center text-sm tabular-nums" aria-live="polite">
            <span className="sr-only">Zoom </span>
            {percent}%
          </span>
          <Button variant="quiet" className="w-8 px-0" aria-label="Zoom in" disabled={zoom >= 2.5} onClick={() => onZoom(zoom + 0.25)}>
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
