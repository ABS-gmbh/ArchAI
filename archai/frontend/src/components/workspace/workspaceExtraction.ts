// Extraction results as the workspace shows them. Moved out of
// DocumentChatWorkspace.tsx.

import type { OCRExtractResponse } from "@/lib/api/ocrAgent";

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

/**
 * Whether the text can carry the pipeline's own uncertainty marks ("[…]", "…", "?").
 *
 * The segmented engine never inserts them: there, a question mark or an
 * ellipsis is something the recogniser read on the page (a punctus
 * interrogativus, for one), and removing it would alter the transcription.
 */
export function carriesUncertaintyMarks(result: Pick<OCRExtractResponse, "ocr_engine">): boolean {
  return result.ocr_engine !== "segmented";
}

/** The transcription as shown and stored: uncertainty marks removed only where the engine inserts them. */
export function cleanTranscription(result: Pick<OCRExtractResponse, "ocr_engine">, text: string): string {
  if (carriesUncertaintyMarks(result)) {
    return removeTextUncertainties(text);
  }
  return text
    .split("\n")
    .map((line) => line.trim())
    .join("\n")
    .trim();
}

export function getExtractionStatus(result: OCRExtractResponse): "FULL" | "PARTIAL" | "EMPTY" {
  return result.status;
}

const RECOGNISER_NAMES: Record<string, string> = {
  kraken_catmus: "CATMuS Medieval",
  kraken_cremma_medieval: "CREMMA Medieval",
  kraken_cremma_lat: "CREMMA Latin",
  kraken_mccatmus: "McCATMuS",
  "glm-ocr": "GLM-OCR",
  glmocr: "GLM-OCR",
};

/** A recogniser's name as a reader knows it; an unknown one is shown as the backend names it. */
export function recogniserName(id: string | null | undefined): string {
  const key = String(id || "").trim();
  return RECOGNISER_NAMES[key.toLowerCase().replace(/:latest$/, "")] ?? key;
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
  const uncertain = carriesUncertaintyMarks(result) ? countUncertainties(rawText) : 0;
  if (uncertain > 0) {
    badges.push({
      tone: "warn",
      label: `${uncertain} uncertain ${uncertain === 1 ? "place" : "places"}`,
      detail: "Places the recogniser marked as unreadable or doubtful. The marks are removed from the text shown here.",
    });
  }
  const warnings = (result.warnings ?? []).filter((warning) => String(warning || "").trim());
  let fallbackLines = 0;
  const fallbackNames: string[] = [];
  const others: string[] = [];
  for (const warning of warnings) {
    const fallback = /^LINES_READ_BY_FALLBACK:([^:]+):(\d+)$/.exec(String(warning));
    if (fallback) {
      fallbackLines += Number(fallback[2]);
      fallbackNames.push(recogniserName(fallback[1]));
    } else {
      others.push(String(warning));
    }
  }
  if (fallbackLines > 0) {
    badges.push({
      tone: "warn",
      label: `${fallbackLines} ${fallbackLines === 1 ? "line" : "lines"} from a fallback recogniser`,
      detail: `The page's main recogniser read nothing on ${fallbackLines === 1 ? "this line" : "these lines"}, so ${fallbackNames.join(" and ")} read ${fallbackLines === 1 ? "it" : "them"}. The page's quality verdict cannot see this; check ${fallbackLines === 1 ? "that line" : "those lines"} against the page.`,
    });
  }
  if (others.length) {
    badges.push({
      tone: "warn",
      label: `${others.length} ${others.length === 1 ? "warning" : "warnings"}`,
      detail: others.join("; "),
    });
  }
  return badges;
}
