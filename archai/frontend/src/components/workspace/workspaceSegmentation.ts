// Segmentation results in the workspace: text-label filtering, location
// suggestions, summaries, stored label regions and transparent crop overlays.
// Moved out of DocumentChatWorkspace.tsx.

import { type LabelRegionPayload } from "@/lib/api/chat";

export const TEXT_LABEL_INCLUDE_TOKENS = [
  "script",
  "gloss",
  "header",
  "catchword",
  "page number",
  "quire",
  "line",
  "text",
  "paragraph",
];

export const TEXT_LABEL_EXCLUDE_TOKENS = [
  "border",
  "column",
  "table",
  "diagram",
  "illustration",
  "graphic",
  "music",
  "zone",
];

export interface CocoCategory {
  id: number;
  name: string;
}

export interface CocoAnnotation {
  id: number;
  category_id: number;
  bbox: [number, number, number, number];
  segmentation?: number[] | number[][];
}

export interface CocoPayload {
  categories?: CocoCategory[];
  annotations?: CocoAnnotation[];
}

export interface OCRLocationSuggestion {
  region_id: string;
  category: string;
  bbox_xywh: [number, number, number, number];
}

export type StoredLabelRegion = LabelRegionPayload;

export type StoredLabelRegionMap = Record<string, StoredLabelRegion[]>;

export function isRelevantTextLabel(label: string): boolean {
  const key = label.toLowerCase();
  if (TEXT_LABEL_EXCLUDE_TOKENS.some((token) => key.includes(token))) {
    return false;
  }
  return TEXT_LABEL_INCLUDE_TOKENS.some((token) => key.includes(token));
}

export function extractLocationSuggestions(coco: CocoPayload | null | undefined): OCRLocationSuggestion[] {
  if (!coco) {
    return [];
  }
  const categories = Array.isArray(coco.categories) ? coco.categories : [];
  const annotations = Array.isArray(coco.annotations) ? coco.annotations : [];
  const categoryById = new Map<number, string>();
  for (const category of categories) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      categoryById.set(category.id, category.name);
    }
  }

  const suggestions: OCRLocationSuggestion[] = [];
  for (const annotation of annotations) {
    if (!annotation || !Array.isArray(annotation.bbox) || annotation.bbox.length < 4) {
      continue;
    }
    const category = categoryById.get(Number(annotation.category_id)) || "";
    if (!category || !isRelevantTextLabel(category)) {
      continue;
    }
    const [x, y, w, h] = annotation.bbox;
    if (![x, y, w, h].every((value) => Number.isFinite(value))) {
      continue;
    }
    if (w < 8 || h < 8) {
      continue;
    }
    suggestions.push({
      region_id: String(annotation.id),
      category,
      bbox_xywh: [x, y, w, h],
    });
  }

  suggestions.sort((a, b) => {
    const ay = a.bbox_xywh[1];
    const by = b.bbox_xywh[1];
    if (Math.abs(ay - by) > 8) {
      return ay - by;
    }
    return a.bbox_xywh[0] - b.bbox_xywh[0];
  });

  return suggestions.slice(0, 60);
}

export function buildSegmentationSummary(coco: CocoPayload | null | undefined): string {
  if (!coco) {
    return "Segmentation completed, but no COCO payload was returned.";
  }

  const categories = Array.isArray(coco.categories) ? coco.categories : [];
  const annotations = Array.isArray(coco.annotations) ? coco.annotations : [];
  const categoryById = new Map<number, string>();
  for (const category of categories) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      categoryById.set(category.id, category.name);
    }
  }

  const counts = new Map<string, number>();
  for (const annotation of annotations) {
    const label = categoryById.get(Number(annotation?.category_id)) || `category_${String(annotation?.category_id ?? "unknown")}`;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const lines = Array.from(counts.entries())
    .sort((a, b) => {
      if (b[1] !== a[1]) {
        return b[1] - a[1];
      }
      return a[0].localeCompare(b[0]);
    })
    .map(([label, count]) => `- ${label}: ${count}`);

  const textLikeCount = extractLocationSuggestions(coco).length;
  return [
    "Segmentation complete.",
    "",
    `Total regions: ${annotations.length}`,
    `Text-like regions: ${textLikeCount}`,
    "",
    "Discovered labels:",
    ...(lines.length ? lines : ["- none"]),
  ].join("\n");
}

export function normalizeCommandText(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/4/g, "a")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ");
}

export function normalizeLabelText(text: string): string {
  return normalizeCommandText(text)
    .replace(/\bembelished\b/g, "embellished")
    .replace(/\blabels\b/g, "label")
    .replace(/\bregions\b/g, "region");
}

export function cocoCategoryMap(coco: CocoPayload | null | undefined): Map<number, string> {
  const out = new Map<number, string>();
  for (const category of coco?.categories || []) {
    if (typeof category?.id === "number" && typeof category?.name === "string") {
      out.set(category.id, category.name);
    }
  }
  return out;
}

export function bestMatchingSegmentationLabel(text: string, labels: string[]): string | null {
  const prompt = normalizeLabelText(text);
  let bestLabel: string | null = null;
  let bestScore = 0;

  for (const label of labels) {
    const normalized = normalizeLabelText(label);
    let score = 0;
    if (prompt.includes(normalized)) {
      score += 100;
    }
    const tokens = normalized.split(" ").filter((token) => token.length >= 3);
    for (const token of tokens) {
      if (prompt.includes(token)) {
        score += 10;
      }
    }
    if (prompt.includes("embellished") && normalized.includes("embellished")) {
      score += 50;
    }
    if (prompt.includes("initial") && normalized.includes("initial")) {
      score += 30;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLabel = label;
    }
  }

  return bestScore > 0 ? bestLabel : null;
}

export function annotationPolygons(annotation: CocoAnnotation): number[][] {
  const raw = annotation.segmentation;
  if (!raw) {
    return [];
  }
  if (Array.isArray(raw) && raw.length && typeof raw[0] === "number") {
    return [raw as number[]];
  }
  if (Array.isArray(raw) && Array.isArray(raw[0])) {
    return (raw as number[][]).filter((poly) => Array.isArray(poly) && poly.length >= 6);
  }
  return [];
}

export function buildStoredLabelRegions(coco: CocoPayload | null | undefined): StoredLabelRegionMap {
  const labelsById = cocoCategoryMap(coco);
  const out: StoredLabelRegionMap = {};
  for (const annotation of coco?.annotations || []) {
    if (!annotation || !Array.isArray(annotation.bbox) || annotation.bbox.length < 4) {
      continue;
    }
    const label = labelsById.get(Number(annotation.category_id));
    if (!label) {
      continue;
    }
    const [x, y, w, h] = annotation.bbox;
    if (![x, y, w, h].every((value) => Number.isFinite(value))) {
      continue;
    }
    const region: StoredLabelRegion = {
      region_id: String(annotation.id),
      bbox_xyxy: [x, y, x + w, y + h],
      polygons: annotationPolygons(annotation),
    };
    if (!out[label]) {
      out[label] = [];
    }
    out[label].push(region);
  }
  return out;
}

export function availableStoredLabels(labelsByName: StoredLabelRegionMap | null | undefined): string[] {
  return Object.keys(labelsByName || {}).sort((a, b) => a.localeCompare(b));
}

export function resolveCropLabelFromPrompt(text: string, labelsByName: StoredLabelRegionMap | null | undefined): string | null {
  return bestMatchingSegmentationLabel(text, availableStoredLabels(labelsByName));
}

export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new window.Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Failed to load page image for crop."));
    image.src = src;
  });
}

export async function buildTransparentCropOverlay(
  pageDataUrl: string,
  labelsByName: StoredLabelRegionMap,
  label: string,
): Promise<{ imageUrl: string; matchCount: number }> {
  const matches = labelsByName[label] || [];
  if (!matches.length) {
    throw new Error(`No regions found for label: ${label}`);
  }

  const source = await loadImageElement(pageDataUrl);
  const canvas = document.createElement("canvas");
  canvas.width = source.naturalWidth || source.width;
  canvas.height = source.naturalHeight || source.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Failed to create crop canvas.");
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (const annotation of matches) {
    const polygons = annotation.polygons || [];
    if (polygons.length) {
      for (const polygon of polygons) {
        ctx.save();
        ctx.beginPath();
        for (let index = 0; index < polygon.length; index += 2) {
          const x = polygon[index] ?? 0;
          const y = polygon[index + 1] ?? 0;
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.closePath();
        ctx.clip();
        ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      }
      continue;
    }

    const [x1, y1, x2, y2] = annotation.bbox_xyxy;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x1, y1, x2 - x1, y2 - y1);
    ctx.clip();
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    ctx.restore();
  }

  return {
    imageUrl: canvas.toDataURL("image/png"),
    matchCount: matches.length,
  };
}
