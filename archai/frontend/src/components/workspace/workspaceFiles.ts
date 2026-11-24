// Files and images of the document workspace: ids, data URLs, page files.
// Moved out of DocumentChatWorkspace.tsx.

import type { WorkspacePage } from "@/lib/workspace/types";

export function makeId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now()}-${Math.round(Math.random() * 1_000_000)}`;
}

export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(new Error(`Failed to read file: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

export async function pageToFile(page: WorkspacePage): Promise<File> {
  const value = String(page.dataUrl || "");
  if (value.startsWith("data:")) {
    const match = value.match(/^data:([^;,]+)?(;base64)?,([\s\S]*)$/);
    if (!match) {
      throw new Error(`Failed to parse page bytes for ${page.name}.`);
    }

    const mime = match[1] || page.mimeType || "image/png";
    const payload = match[3] || "";
    const isBase64 = Boolean(match[2]);
    let blob: Blob;

    if (isBase64) {
      const binary = atob(payload);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
      }
      blob = new Blob([bytes], { type: mime });
    } else {
      blob = new Blob([decodeURIComponent(payload)], { type: mime });
    }

    return new File([blob], page.name, { type: page.mimeType || blob.type || "image/png" });
  }

  const response = await fetch(value);
  if (!response.ok) {
    throw new Error(`Failed to load page bytes for ${page.name}.`);
  }
  const blob = await response.blob();
  return new File([blob], page.name, { type: page.mimeType || blob.type || "image/png" });
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function sortedImages(files: FileList | File[]): File[] {
  return Array.from(files)
    .filter((file) => file.type.startsWith("image/"))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function toBase64(dataUrl: string): string {
  const index = dataUrl.indexOf(",");
  if (index === -1) {
    return dataUrl;
  }
  return dataUrl.slice(index + 1);
}

/**
 * What to tell someone whose page image the server refused as too large. The
 * image is sent as text inside the request, about a third larger than the
 * file, so the size given is what was sent, not the file's.
 */
export function pageTooLargeMessage(sentCharacters: number): string {
  const megabytes = Math.max(1, Math.round(sentCharacters / (1024 * 1024)));
  return `This page image is too large to send: about ${megabytes} MB as it travels. Save it as a JPEG or at a lower resolution and open it again, or ask whoever runs the server to raise MAX_JSON_MEGABYTES.`;
}

export interface RequestLimits {
  /** MAX_JSON_MEGABYTES in bytes: the largest request body the server reads. */
  jsonBytes: number;
  /** OCR_MAX_DECODE_PIXELS: the most pixels an image may have. */
  decodePixels: number;
}

// The rest of a read request around the image: settings, metadata. Generous,
// so a page the check lets through is not refused for a few bytes.
const REQUEST_OVERHEAD_BYTES = 256 * 1024;

/** What to tell someone whose page has more pixels than the server reads. */
export function tooManyPixelsMessage(width: number, height: number, limit: number): string {
  const megapixels = (pixels: number) => String(Math.round(pixels / 100_000) / 10);
  return `This page image has ${megapixels(width * height)} megapixels; the server reads at most ${megapixels(limit)}. Save it at a lower resolution, or ask whoever runs the server to raise OCR_MAX_DECODE_PIXELS.`;
}

/**
 * Why a page must not be sent, or null: checked before a run is reserved or
 * anything is uploaded. A 60 MB PNG went out as an 85 MB request before the
 * server refused it (live regression, 2026-11-24). The size is the image as
 * it travels (base64); the pixels are checked once the image has loaded and
 * its size is known. Without the server's limits (an older backend) nothing
 * is checked here and the server's own answer is shown.
 */
export function pageRefusal(
  page: { base64Length: number; width?: number; height?: number },
  limits: RequestLimits | null,
): string | null {
  if (!limits) {
    return null;
  }
  if (page.base64Length + REQUEST_OVERHEAD_BYTES > limits.jsonBytes) {
    return pageTooLargeMessage(page.base64Length);
  }
  if (page.width && page.height && page.width * page.height > limits.decodePixels) {
    return tooManyPixelsMessage(page.width, page.height, limits.decodePixels);
  }
  return null;
}

/** A page the workspace refused before sending it; the message is pageRefusal's sentence. */
export class PageRefused extends Error {}
