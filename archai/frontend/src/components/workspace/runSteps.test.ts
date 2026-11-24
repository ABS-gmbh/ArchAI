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
    // With the names not looked up (a date), which made the parts add up (2026-11-24).
    assert.equal(messageInWords("Authority linking complete: linked=2, unresolved=3, ambiguous=0, skipped=1."), "2 linked, 3 not linked, 1 not looked up");
    // Stored runs keep the three-part message; a zero is left out.
    assert.equal(messageInWords("Authority linking complete: linked=2, unresolved=3, ambiguous=0."), "2 linked, 3 not linked");
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

// The messages a page gets when the quality gate rates its reading too
// uncertain to search or link (app/services/trace_pipeline.py). They are log
// lines ("token_search_allowed=False (quality=RISKY)") and reached the Steps
// panel as they came.
describe("messageInWords, an uncertain reading", () => {
  it("says why linking and indexing were skipped", () => {
    assert.equal(
      messageInWords("Authority linking SKIPPED: token_search_allowed=False (quality=RISKY)"),
      "Skipped: the reading was rated risky, too uncertain to look names up.",
    );
    assert.equal(
      messageInWords("RAG auto-index SKIPPED: token_search_allowed=False (quality=UNRELIABLE)"),
      "Skipped: the reading was rated unreliable, too uncertain to search.",
    );
  });

  it("drops the bare reason, which the step's name already gives", () => {
    assert.equal(messageInWords("token_search_allowed=False quality=RISKY"), "");
  });

  it("counts the names kept without links", () => {
    assert.equal(messageInWords("Deferred linking stored unresolved mentions: total=3"), "3 names kept, not linked");
    assert.equal(messageInWords("Deferred linking stored unresolved mentions: total=1"), "1 name kept, not linked");
  });

  it("puts the cautious name search's counts in words", () => {
    assert.equal(
      messageInWords("degraded_capture chunks=4, mentions=1"),
      "4 passages, 1 name (names found with the cautious search used for an uncertain reading)",
    );
  });

  it("drops the full-page pipeline's other bookkeeping lines", () => {
    for (const message of ["Full-page OCR text stored.", "Running the full-page post-OCR pipeline.", "Full-page OCR run received (segmented)."]) {
      assert.equal(messageInWords(message), "", message);
    }
  });
});

// Steps of runs that an upgrade drill read back from an older install, and
// that the pipeline still writes today (trace_pipeline.py, routers/ocr.py):
// they reached the Steps panel as log lines.
describe("stageInWords and messageInWords, a cautious or blocked reading", () => {
  it("names the cautious name search", () => {
    assert.equal(stageInWords("ANALYZE_DEGRADED"), "Looking for names with a cautious search");
    assert.equal(
      messageInWords("NER gate blocked full analysis; running degraded mention capture only (quality=UNRELIABLE)"),
      "The reading was rated unreliable, so names are looked for with a cautious search only.",
    );
  });

  it("says why the quality check stopped a reading, without the gate's internals", () => {
    const message =
      "Quality gates FAILED for segmented OCR. quality_label_v2=UNRELIABLE, blocked_stages=['token_search', 'token_ner', " +
      "'entity_claims', 'no_entity_claims', 'paraphrase', 'translation'], gibberish=0.0000, lead_frag=0.0000";
    assert.equal(messageInWords(message), "The reading was rated unreliable, too uncertain to search, look names up or translate.");
  });

  it("puts the region-by-region reading in words", () => {
    assert.equal(messageInWords("OCR trace run received."), "");
    assert.equal(messageInWords("Using provided location suggestions: 1"), "1 layout region given with the page");
    assert.equal(messageInWords("Using provided location suggestions: 4"), "4 layout regions given with the page");
    assert.equal(messageInWords("Segmented OCR on 1 regions."), "1 region, read on its own");
    assert.equal(messageInWords("Segmented OCR on 12 regions."), "12 regions, each read on its own");
  });

  it("says a failure after reading happened there, without the exception's class", () => {
    assert.equal(
      messageInWords("Post-OCR pipeline failed: RuntimeError: database is locked"),
      "The steps after reading failed: database is locked",
    );
  });
});
