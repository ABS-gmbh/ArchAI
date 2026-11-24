// A run's steps in the Steps panel, from the stage names and log messages the
// backend records (app/services/trace_pipeline.py, app/routers/ocr.py).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { messageInWords, stageInWords } from "./runSteps";

describe("stageInWords", () => {
  it("names each stage of a page read in a reader's words", () => {
    assert.equal(stageInWords("RESERVED"), "Sending the page");
    assert.equal(stageInWords("OCR_RUNNING"), "Reading the lines");
    assert.equal(stageInWords("AUTHORITY_LINKING_DEFERRED"), "Names not linked: the reading is too uncertain");
    assert.equal(stageInWords("INDEX_SKIPPED"), "Not indexed for search");
    assert.equal(stageInWords("DONE"), "Finished");
  });

  it("does not depend on the case the stage arrives in", () => {
    assert.equal(stageInWords("ocr_done"), "Lines read");
  });

  it("turns an unknown stage into a sentence, and no stage into 'Working'", () => {
    assert.equal(stageInWords("PROOFREAD_RUNNING"), "Proofread running");
    assert.equal(stageInWords(""), "Working");
    assert.equal(stageInWords(null), "Working");
  });
});

describe("messageInWords", () => {
  // Messages recorded by the runs of 2026-11-24 (see 038c15b).
  it("puts counts in words, singular and plural", () => {
    assert.equal(messageInWords("chunks=13, mentions=0"), "13 passages, 0 names");
    assert.equal(messageInWords("chunks=1, mentions=1"), "1 passage, 1 name");
    assert.equal(messageInWords("Indexed 13 chunks."), "13 passages indexed");
    assert.equal(messageInWords("Authority linking complete: linked=2, unresolved=5, ambiguous=1."), "2 linked, 5 not linked, 1 ambiguous");
  });

  it("explains a low name count", () => {
    assert.equal(
      messageInWords("MENTION_RECALL_LOW: mentions=0 < expected_min=2 (text_len=1219, quality=HIGH)"),
      "Fewer names than a text this long usually has: 0 found, at least 2 expected.",
    );
  });

  it("says how the page was read", () => {
    assert.equal(messageInWords("Full-page OCR done (segmented)."), "The page's layout was found and each line read on its own.");
    assert.equal(messageInWords("Full-page OCR done (glmocr)."), "Read with glmocr.");
  });

  it("drops messages that only repeat the step's name", () => {
    for (const message of [
      "Full-page OCR request received.",
      "Full-page OCR running.",
      "Running chunk/entity analysis.",
      "Run results persisted in SQLite.",
      "Running authority linking (Wikidata).",
      "Auto-indexing into vector store.",
      "Full-page pipeline completed successfully.",
    ]) {
      assert.equal(messageInWords(message), "", message);
    }
  });

  it("shows an unknown message as it came, and nothing for none", () => {
    assert.equal(messageInWords("No text was read; nothing to analyse."), "No text was read; nothing to analyse.");
    assert.equal(messageInWords(null), "");
  });
});
