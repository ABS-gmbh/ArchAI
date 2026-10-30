// Files and images of the document workspace: ids, data URLs, page files.
// Moved out of DocumentChatWorkspace.tsx.

import { type WorkspacePage } from "@/lib/workspace/types";

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

export function sortedImages(files: FileList): File[] {
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
