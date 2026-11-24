// What a typed request does in the workspace: handled as a command (translate,
// names, crop, segment, read the page) or sent to chat as a question.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { detectWorkspaceIntent, resolveOcrPromptHints, translationLanguageHint } from "./workspaceIntents";

const cases: [string, ReturnType<typeof detectWorkspaceIntent>][] = [
  // From the live check on 2026-11-24: the workspace's own suggestions.
  ["Who is named on this page?", "entities"],
  ["Translate the first five lines.", "translate"],
  ["Summarise this page in two sentences.", null],
  // Translation.
  ["translate to english", "translate"],
  ["Translate this page into English.", "translate"],
  ["in english please", "translate"],
  // Names.
  ["Which places are mentioned?", "entities"],
  ["Are there any people mentioned here?", "entities"],
  // Reading the page.
  ["extract text", "extract"],
  ["Transcribe this page", "extract"],
  ["run OCR", "extract"],
  ["What does it say?", "extract"],
  // Layout.
  ["segment this page", "segment"],
  ["show segmentation", "segment"],
  // Cropping a region.
  ["crop the initial", "crop"],
  ["cut out the miniature", "crop"],
  // Questions that are not commands go to chat.
  ["What is the date of this charter?", null],
  ["Explain the abbreviation in line 3.", null],
];

describe("detectWorkspaceIntent", () => {
  for (const [text, intent] of cases) {
    it(`${JSON.stringify(text)} -> ${intent ?? "chat"}`, () => {
      assert.equal(detectWorkspaceIntent(text), intent);
    });
  }
});

// A command word inside another word, or a layout word used about the text, is
// a question: it goes to chat (or to names), not to a command.
const questions: [string, ReturnType<typeof detectWorkspaceIntent>][] = [
  ["What crops are listed in this account?", null],
  ["Is the hand mediocre or careful?", null],
  ["Which regions of France does it mention?", "entities"],
  ["Are the labels on the casks named in the text?", "entities"],
];
// The same words as commands still work.
const commands: [string, ReturnType<typeof detectWorkspaceIntent>][] = [
  ["crop out the initial", "crop"],
  ["OCR this page", "extract"],
  ["show the regions", "segment"],
  ["find the layout regions", "segment"],
  ["draw the bounding boxes", "segment"],
  ["show the labels", "segment"],
  ["What labels are on this page?", "segment"],
  ["Which regions are on the image?", "segment"],
];

describe("detectWorkspaceIntent, words inside questions", () => {
  for (const [text, intent] of [...questions, ...commands]) {
    it(`${JSON.stringify(text)} -> ${intent ?? "chat"}`, () => {
      assert.equal(detectWorkspaceIntent(text), intent);
    });
  }
});

describe("OCR hints", () => {
  it("take the language from the document's metadata when the request names none", () => {
    const hints = resolveOcrPromptHints("read the page", {
      language: "latin",
      year: "1250",
      placeOrOrigin: "",
      scriptFamily: "",
      documentType: "",
      notes: "",
    });
    assert.equal(hints.languageHint, "latin");
  });

  it("tell the translator the document's language, or that it is unknown", () => {
    const metadata = { language: "old french", year: "", placeOrOrigin: "", scriptFamily: "", documentType: "", notes: "" };
    assert.equal(translationLanguageHint(metadata), "old french");
    assert.equal(translationLanguageHint(null), "unknown");
  });
});
