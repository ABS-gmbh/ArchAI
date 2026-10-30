// Extraction results as the workspace shows them. Moved out of
// DocumentChatWorkspace.tsx.

import { type OCRExtractResponse } from "@/lib/api/ocrAgent";

export function removeTextUncertainties(text: string): string {
  return text
    .split("\n")
    .map((line) =>
      line
        .replace(/\[(?:…|\.{3})\]/g, "")
        .replace(/…/g, "")
        .replace(/\?/g, "")
        .replace(/\s{2,}/g, " ")
        .trim(),
    )
    .filter((line, index, lines) => line.length > 0 || index < lines.length - 1)
    .join("\n")
    .trim();
}

export function getExtractionStatus(result: OCRExtractResponse): "FULL" | "PARTIAL" | "EMPTY" {
  return result.status;
}
