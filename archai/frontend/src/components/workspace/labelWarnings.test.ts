// Label analysis's warnings in a reader's words (2026-11-25): the workspace
// showed the codes themselves ("LABEL_MODEL_FALLBACK:gemma-4-31b-it") as an error.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { labelWarningInWords, labelWarningsInWords } from "./labelWarnings";

describe("labelWarningInWords", () => {
  it("words the model fallbacks, naming the model that answered", () => {
    assert.equal(labelWarningInWords("LABEL_MODEL_FALLBACK:gemma-4-31b-it"), "The configured label-analysis model is not offered, so gemma-4-31b-it answered instead.");
    assert.equal(labelWarningInWords("LABEL_TIMEOUT_FALLBACK:qwen3.8-27b"), "The first model took too long, so qwen3.8-27b answered instead.");
    assert.equal(labelWarningInWords("LABEL_TRANSIENT_FALLBACK:qwen3.8-27b"), "The first model failed for a moment, so qwen3.8-27b answered instead.");
  });

  it("words how the region was prepared and read", () => {
    assert.equal(labelWarningInWords("OCR_PAYLOAD_RESIZED"), "The region was scaled down to be read.");
    assert.equal(labelWarningInWords("OCR_RETRY_ATTEMPTS_USED:3"), "Reading the region took 3 attempts.");
    assert.equal(labelWarningInWords("OCR_PROMPT_ECHO_REMOVED:2"), "2 lines that repeated the instructions were removed.");
    assert.equal(labelWarningInWords("OCR_VARIANT_FALLBACK:contrast"), "The region was read again in another form (contrast): the first gave no text.");
    assert.equal(
      labelWarningInWords("LABEL_OCR_FAILED:Label OCR failed (TimeoutError). The details are in the server log."),
      "Reading the letters in the region failed: Label OCR failed (TimeoutError). The details are in the server log.",
    );
  });

  it("keeps a sentence as it is, and an unknown code as it came", () => {
    const sentence = "Multiple initial regions matched; returning one letter per region in reading order.";
    assert.equal(labelWarningInWords(sentence), sentence);
    assert.equal(labelWarningInWords("SOMETHING_NEW:x"), "SOMETHING_NEW:x");
  });

  it("joins several, once each", () => {
    assert.equal(
      labelWarningsInWords(["OCR_PAYLOAD_RESIZED", "OCR_PAYLOAD_RESIZED", "LABEL_MODEL_FALLBACK:gemma-4-31b-it"]),
      "The region was scaled down to be read. The configured label-analysis model is not offered, so gemma-4-31b-it answered instead.",
    );
    assert.equal(labelWarningsInWords([]), "");
  });
});
