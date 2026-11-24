// Why a name was not linked, in the Names panel's words, from the reasons the
// linker writes (app/services/entity_scoring.py, authority_linking.py).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { reasonInWords } from "./linkReason";

describe("reasonInWords", () => {
  it("keeps the reasons it already put in words", () => {
    assert.equal(
      reasonInWords("best score (0.682) < threshold (0.8) [ocr_quality=HIGH]"),
      "The best match scored 0.68; a link needs 0.80.",
    );
    assert.equal(
      reasonInWords("token_search_allowed=False quality=RISKY"),
      "Not matched: the page's reading was graded risky, too uncertain for names to be linked.",
    );
    assert.equal(reasonInWords("no candidates found"), "No authority record was found for this name.");
  });

  // Shown raw in a live run on 2026-11-24 (the Antitus page: "pere", "prince").
  it("says when every record found is the wrong kind for the name", () => {
    assert.equal(
      reasonInWords("all 5 candidate(s) type_incompatible (best score=0.476, qid=Q18219701)"),
      "None of the 5 records found is the right kind of record for this name.",
    );
    assert.equal(
      reasonInWords("all 1 candidate(s) type_incompatible (best score=0.279, qid=Q807)"),
      "The one record found is not the right kind of record for this name.",
    );
  });

  it("says when no record of the right kind has a close enough name", () => {
    assert.equal(
      reasonInWords(
        "no type-compatible candidate's name similarity reaches 0.6 (best score=0.491, qid=Q328804, similarity=0.593) [ocr_quality=HIGH]",
      ),
      "No record of the right kind has a name close enough: the closest is 0.59 alike, a link needs 0.60.",
    );
  });

  it("shows an unknown reason as written, without the quality tag, capitalised", () => {
    assert.equal(reasonInWords("something new happened [ocr_quality=OK]"), "Something new happened");
  });
});
