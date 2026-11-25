// Label analysis's warnings (app/agents/label_analysis_agent.py and the GLM-OCR
// it reads regions with, app/services/glm_ollama_ocr.py) in a reader's words.
// The workspace showed the codes themselves as an error (2026-11-25). Plain
// TypeScript, so labelWarnings.test.ts can import it.

const WORDS: [RegExp, (match: RegExpExecArray) => string][] = [
  [/^LABEL_MODEL_FALLBACK:(.+)$/, (m) => `The configured label-analysis model is not offered, so ${m[1]} answered instead.`],
  [/^LABEL_TIMEOUT_FALLBACK:(.+)$/, (m) => `The first model took too long, so ${m[1]} answered instead.`],
  [/^LABEL_TRANSIENT_FALLBACK:(.+)$/, (m) => `The first model failed for a moment, so ${m[1]} answered instead.`],
  [/^LABEL_OCR_FAILED:(.+)$/, (m) => `Reading the letters in the region failed: ${m[1]}`],
  [/^OCR_PAYLOAD_RESIZED$/, () => "The region was scaled down to be read."],
  [/^OCR_PAYLOAD_COMPRESSED$/, () => "The region was compressed to be read."],
  [/^OCR_PAYLOAD_AUTOCONTRAST$/, () => "The region's contrast was raised to read it."],
  [/^OCR_VARIANT_FALLBACK:(.+)$/, (m) => `The region was read again in another form (${m[1]}): the first gave no text.`],
  [/^OCR_RETRY_ATTEMPTS_USED:(\d+)$/, (m) => `Reading the region took ${m[1]} attempts.`],
  [/^OCR_PROMPT_ECHO_REMOVED:(\d+)$/, (m) => `${m[1]} ${m[1] === "1" ? "line that repeated" : "lines that repeated"} the instructions ${m[1] === "1" ? "was" : "were"} removed.`],
];

/** One warning in words; a sentence, or a code not known here, as it came. */
export function labelWarningInWords(warning: string): string {
  const text = String(warning || "").trim();
  for (const [pattern, words] of WORDS) {
    const match = pattern.exec(text);
    if (match) {
      return words(match);
    }
  }
  return text;
}

/** The warnings in words, each once, as one passage. */
export function labelWarningsInWords(warnings: string[]): string {
  return [...new Set(warnings.map(labelWarningInWords).filter(Boolean))].join(" ");
}
