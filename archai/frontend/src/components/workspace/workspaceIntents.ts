// What a workspace chat message asks for, and the prompts built from it:
// intent detection, OCR hints from the message and the document's metadata,
// and the translation and contextual prompts. Moved out of
// DocumentChatWorkspace.tsx.

import { type WorkspaceDocumentMetadata } from "@/lib/workspace/types";
import { normalizeCommandText, normalizeLabelText } from "./workspaceSegmentation";

export type WorkspaceIntent = "segment" | "extract" | "translate" | "entities" | "crop" | "label_analysis" | null;

export type OcrPromptHints = {
  scriptHintSeed?: string;
  languageHint?: string;
  ocrBackend?: "auto" | "kraken_mccatmus" | "kraken_catmus" | "kraken_cremma_medieval" | "kraken_cremma_lat";
};

export function isSegmentationIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value === "segment" ||
    value === "segment image" ||
    value === "segment the image" ||
    value === "segment this image" ||
    value === "segment page" ||
    value === "segment this page" ||
    value === "run segmentation" ||
    value === "show segmentation" ||
    value.startsWith("segment ") ||
    value.includes("segmentation") ||
    value.includes("labels") ||
    value.includes("regions") ||
    value.includes("bounding boxes")
  );
}

export function isExtractionIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value === "extract text" ||
    value === "extract the text" ||
    value === "extract text in chat" ||
    value === "ocr this page" ||
    value === "run ocr" ||
    value.includes("extract text") ||
    /\bextract\b.*\btext\b/.test(value) ||
    /\bextract\b.*\b(manuscript|page|latin|french|english|old french|middle french|anglo norman|italian|spanish|iberian|portuguese|catalan)\b/.test(value) ||
    value.includes("transcribe") ||
    value.includes("ocr") ||
    value.includes("read the text") ||
    value.includes("what does it say")
  );
}

export function isTranslationIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  return (
    value.startsWith("translate") ||
    value.includes("translate") ||
    value === "in english please" ||
    value === "english please" ||
    (value.includes("english") && (value.includes("text") || value.includes("page") || value.includes("into") || value.includes("to ")))
  );
}

export function isCropIntent(text: string): boolean {
  const value = normalizeLabelText(text);
  return value.includes("crop") || value.includes("cut out") || value.includes("isolate");
}

export function isLabelAnalysisPrompt(text: string): boolean {
  const value = normalizeLabelText(text);
  return /(what is this|what is that|explain|style|art style|origin|origins|motif|ornament|ornamental|decorative|decoration|shape|symbol|design|embellished|initial)/.test(value);
}

export function isEntityIntent(text: string): boolean {
  const value = normalizeCommandText(text);
  const hasEntityTopic = /(entity|entities|person|persons|people|place|places|location|locations|name|names|mention|mentioned|mentions|who|where)/.test(value);
  const hasQuestionShape = /(are there|there any|any\b|which|what|who|where|mentioned|mentions|mention|named|names)/.test(value);
  return hasEntityTopic && hasQuestionShape;
}

export function detectWorkspaceIntent(text: string): WorkspaceIntent {
  if (isCropIntent(text)) {
    return "crop";
  }
  if (isSegmentationIntent(text)) {
    return "segment";
  }
  if (isEntityIntent(text)) {
    return "entities";
  }
  if (isExtractionIntent(text)) {
    return "extract";
  }
  if (isTranslationIntent(text)) {
    return "translate";
  }
  return null;
}

export function extractOcrPromptHints(text: string): OcrPromptHints {
  const value = normalizeCommandText(text);
  if (/\banglo norman\b/.test(value)) {
    return { languageHint: "anglo_norman", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold french\b/.test(value)) {
    return { languageHint: "old_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle french\b/.test(value)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmedieval french\b/.test(value)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\blatin\b/.test(value)) {
    return { languageHint: "latin", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmedieval latin\b/.test(value)) {
    return { languageHint: "latin", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bfrench\b/.test(value)) {
    return { languageHint: "french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\boccitan\b|\bold occitan\b|\bprovencal\b|\bprovençal\b/.test(value)) {
    return { languageHint: "occitan", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bitalian\b/.test(value)) {
    return { languageHint: "italian", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bspanish\b|\biberian\b|\bportuguese\b|\bcatalan\b/.test(value)) {
    return { languageHint: "spanish", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle english\b/.test(value)) {
    return { languageHint: "middle_english", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold english\b/.test(value)) {
    return { languageHint: "old_english", scriptHintSeed: "insular_old_english", ocrBackend: "auto" };
  }
  if (/\bmiddle high german\b/.test(value)) {
    return { languageHint: "middle_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold high german\b/.test(value)) {
    return { languageHint: "old_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bgerman\b|\bdutch\b|\bflemish\b/.test(value)) {
    return { languageHint: "german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  return {};
}

export function metadataToOcrHints(metadata: WorkspaceDocumentMetadata | null | undefined): OcrPromptHints {
  if (!metadata) {
    return {};
  }

  const languageValue = normalizeCommandText(metadata.language || "");
  const scriptValue = normalizeCommandText(metadata.scriptFamily || "");
  const fallbackScriptHint = scriptValue.includes("insular") ? "insular_old_english" : "latin";

  if (/\banglo norman\b/.test(languageValue)) {
    return { languageHint: "anglo_norman", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold french\b/.test(languageValue)) {
    return { languageHint: "old_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle french\b/.test(languageValue)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmedieval french\b/.test(languageValue)) {
    return { languageHint: "middle_french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\blatin\b/.test(languageValue)) {
    return { languageHint: "latin", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bmedieval latin\b/.test(languageValue)) {
    return { languageHint: "latin", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bfrench\b/.test(languageValue)) {
    return { languageHint: "french", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\boccitan\b|\bold occitan\b|\bprovencal\b|\bprovençal\b/.test(languageValue)) {
    return { languageHint: "occitan", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bitalian\b/.test(languageValue)) {
    return { languageHint: "italian", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bspanish\b|\biberian\b|\bportuguese\b|\bcatalan\b/.test(languageValue)) {
    return { languageHint: "spanish", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bmiddle english\b/.test(languageValue)) {
    return { languageHint: "middle_english", scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" };
  }
  if (/\bold english\b/.test(languageValue)) {
    return { languageHint: "old_english", scriptHintSeed: "insular_old_english", ocrBackend: "auto" };
  }
  if (/\bmiddle high german\b/.test(languageValue)) {
    return { languageHint: "middle_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bold high german\b/.test(languageValue)) {
    return { languageHint: "old_high_german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  if (/\bgerman\b|\bdutch\b|\bflemish\b/.test(languageValue)) {
    return { languageHint: "german", scriptHintSeed: "latin", ocrBackend: "auto" };
  }
  return scriptValue ? { scriptHintSeed: fallbackScriptHint, ocrBackend: "auto" } : {};
}

export function resolveOcrPromptHints(text: string, metadata: WorkspaceDocumentMetadata | null | undefined): OcrPromptHints {
  const promptHints = extractOcrPromptHints(text);
  if (promptHints.languageHint || promptHints.scriptHintSeed) {
    return promptHints;
  }
  return metadataToOcrHints(metadata);
}

export function translationLanguageHint(metadata: WorkspaceDocumentMetadata | null | undefined): string {
  const language = String(metadata?.language || "").trim();
  if (language) {
    return language;
  }
  const hints = metadataToOcrHints(metadata);
  if (hints.languageHint) {
    return hints.languageHint.replace(/_/g, " ");
  }
  return "unknown";
}

export function buildEnglishTranslationPrompt(
  userRequest: string,
  sourceText: string,
  metadata: WorkspaceDocumentMetadata | null | undefined,
): string {
  const sourceLanguage = translationLanguageHint(metadata);
  const sourceLanguageLower = sourceLanguage.toLowerCase();
  const year = String(metadata?.year || "").trim();
  const scriptFamily = String(metadata?.scriptFamily || "").trim();
  const placeOrOrigin = String(metadata?.placeOrOrigin || "").trim();
  const documentType = String(metadata?.documentType || "").trim();
  const notes = String(metadata?.notes || "").trim();
  return [
    `Translate the extracted manuscript passage from ${sourceLanguage} into fluent, faithful English.`,
    "This is a translation request, not an OCR request.",
    "Use the OCR-extracted text below as the only text to translate.",
    `Source language: ${sourceLanguage}.`,
    "Target language: English.",
    year ? `The manuscript year is: ${year}. Use this only as a weak historical-language hint.` : "",
    scriptFamily ? `The manuscript script family is: ${scriptFamily}. Use this only as a weak reading hint.` : "",
    placeOrOrigin ? `The manuscript place or origin is: ${placeOrOrigin}. Use this only as a weak dialect hint.` : "",
    documentType ? `The manuscript document type is: ${documentType}. Use this only as a weak genre/context hint.` : "",
    notes ? `Additional manuscript notes: ${notes}. Use this only as a weak contextual hint and never to invent text.` : "",
    "Treat the extracted transcript as the authoritative source text to interpret and render into English.",
    sourceLanguageLower.includes("old french")
      ? "Treat spelling variation, abbreviation, and likely OCR distortions as expected features of Old French and resolve them contextually when the intended reading is reasonably clear."
      : "Use sentence-level and passage-level context to resolve obvious orthographic or OCR-like distortions where the intended sense is reasonably clear.",
    "Translate at the level of clauses, sentences, and the whole passage, not as a word-by-word gloss.",
    "Produce the best coherent English rendering that the passage supports.",
    "Prefer fluent English syntax over literal token-by-token paraphrase whenever the context makes the intended sense reasonably clear.",
    "Keep the translation roughly proportional to the source passage; do not expand a damaged or repetitive passage into a longer narrative than the source supports.",
    "Silently normalize obvious OCR-like distortions internally when the likely source reading is clear from context.",
    "Do not turn an unclear token into a confident person, place, or plot detail unless the passage clearly supports that reading.",
    "If a clause remains too corrupt to interpret confidently, mark only that local span with [unclear] instead of inventing connective narrative or repeated moral commentary.",
    "Do not summarize, paraphrase away content, or answer questions about the text.",
    "Preserve uncertainty only where it is genuinely unavoidable after contextual interpretation.",
    "If a word or phrase remains unresolved, mark only that local span with [unclear] or [token?].",
    "Do not invent meaning for genuinely unclear words or spans.",
    "Do not echo opaque source tokens as if they were valid English unless they are clearly names or untranslated forms.",
    "Do not describe the page, layout, decoration, or image.",
    "Do not repeat the source text unchanged unless a token is genuinely unreadable.",
    "Do not add a translator's note, note, explanation, or editorial commentary.",
    "Return only the English translation.",
    "Do not return JSON.",
    "Do not explain your reasoning.",
    "",
    "User request:",
    userRequest,
    "",
    "OCR-extracted source text:",
    sourceText,
  ].filter(Boolean).join("\n");
}

export function buildContextualUserPrompt(
  userRequest: string,
  options: {
    sourceText?: string;
    authorityReport?: string;
    mode: "translation" | "entities";
  },
): string {
  const blocks = [
    "Answer the user's request directly.",
    "OCR has already been performed.",
    "Use the extracted text below as the source evidence.",
    "Do not perform OCR again.",
    "Do not repeat the source text unchanged unless the user explicitly asks for a transcription.",
    "Do not return JSON unless the user explicitly asks for JSON.",
    "Do not mention internal pipeline steps, runs, tables, or debugging output.",
    "",
    "User request:",
    userRequest,
  ];

  if (options.mode === "translation") {
    blocks.push(
      "",
      "Task: translate the OCR text into English.",
      "This is always a best-effort translation request.",
      "Never refuse, gate, or say the OCR is too uncertain.",
      "Return only the English translation.",
      "If a span is uncertain, preserve that uncertainty in English instead of copying the source wording verbatim.",
    );
  }

  if (options.mode === "entities") {
    blocks.push(
      "",
      "Task: answer the user's entity question using the OCR text and authority-linking information below.",
      "If no reliable persons or places are present, say that plainly.",
    );
  }

  if (options.sourceText) {
    blocks.push("", "OCR text:", options.sourceText);
  }

  if (options.authorityReport) {
    blocks.push("", "Authority-linking report:", options.authorityReport);
  }

  return blocks.join("\n");
}
