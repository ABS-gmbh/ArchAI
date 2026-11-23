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

export type ExtractionTone = "ok" | "warn" | "error" | "neutral";

export interface ExtractionBadge {
  tone: ExtractionTone;
  label: string;
  detail: string;
}

const QUALITY_BADGES: Record<string, ExtractionBadge> = {
  HIGH: {
    tone: "ok",
    label: "Quality: high",
    detail: "The transcription passed every quality check and is used for search, entities and chat.",
  },
  OK: {
    tone: "ok",
    label: "Quality: acceptable",
    detail: "The transcription passed the quality gate and is used for search, entities and chat. Expect some misreadings.",
  },
  RISKY: {
    tone: "warn",
    label: "Quality: risky",
    detail: "The transcription shows signs of misreading. Check it against the page before relying on it.",
  },
  UNRELIABLE: {
    tone: "error",
    label: "Quality: unreliable",
    detail: "The transcription failed the quality gate. It is shown, but it is not searched or linked to authorities.",
  },
};

/** How many places the recogniser marked as unreadable or doubtful: "[…]", "[...]", "…" and "?". */
export function countUncertainties(rawText: string): number {
  return (rawText.match(/\[(?:…|\.{3})\]|…|\?/g) ?? []).length;
}

/**
 * Badges for a finished extraction: the quality verdict, whether search is
 * blocked, the recogniser's confidence, the uncertain places that were removed
 * from the displayed text, and how many warnings came back.
 */
export function describeExtraction(result: OCRExtractResponse, rawText: string): ExtractionBadge[] {
  const badges: ExtractionBadge[] = [];
  const label = String(result.quality_label || "").trim().toUpperCase();
  if (QUALITY_BADGES[label]) {
    badges.push(QUALITY_BADGES[label]);
  }
  const downstream = String(result.downstream_mode || "").trim();
  if (downstream && downstream !== "token_based" && label !== "UNRELIABLE") {
    badges.push({
      tone: "warn",
      label: "Not searchable",
      detail: "The quality gate kept this transcription out of search and authority linking.",
    });
  }
  if (typeof result.confidence === "number" && Number.isFinite(result.confidence)) {
    const percent = Math.round(Math.min(1, Math.max(0, result.confidence)) * 100);
    badges.push({
      tone: "neutral",
      label: `Confidence ${percent}%`,
      detail: "The recogniser's own confidence in its reading, averaged over the page. It is not an accuracy measurement.",
    });
  }
  const uncertain = countUncertainties(rawText);
  if (uncertain > 0) {
    badges.push({
      tone: "warn",
      label: `${uncertain} uncertain ${uncertain === 1 ? "place" : "places"}`,
      detail: "Places the recogniser marked as unreadable or doubtful. The marks are removed from the text shown here.",
    });
  }
  const warnings = (result.warnings ?? []).filter((warning) => String(warning || "").trim());
  if (warnings.length) {
    badges.push({
      tone: "warn",
      label: `${warnings.length} ${warnings.length === 1 ? "warning" : "warnings"}`,
      detail: warnings.join("; "),
    });
  }
  return badges;
}
