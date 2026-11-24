// Why a name was not linked, in the Names panel's words, from the reasons the
// linker writes (app/services/entity_scoring.py, authority_linking.py).
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { linkingSummary, mentionStatus, reasonInWords, summaryOfMentions } from "./linkReason";

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

// On the Antitus page the Steps said "6 names", then "2 linked, 3 not linked,
// 0 ambiguous", and the Names panel "2 of 6": the sixth, a date, is never looked
// up and no count said so (live regression, 2026-11-24). The step and the panel
// now say the same: "2 linked, 3 not linked, 1 not looked up".
describe("linkingSummary", () => {
  it("names every part that is not zero, linked and not linked always", () => {
    assert.equal(linkingSummary({ linked: 2, notLinked: 3, ambiguous: 0, skipped: 1 }), "2 linked, 3 not linked, 1 not looked up");
    assert.equal(linkingSummary({ linked: 2, notLinked: 5, ambiguous: 1, skipped: 0 }), "2 linked, 5 not linked, 1 ambiguous");
    assert.equal(linkingSummary({ linked: 0, notLinked: 0, ambiguous: 0, skipped: 2 }), "0 linked, 0 not linked, 2 not looked up");
  });

  it("counts a page's names the way the step does", () => {
    const statuses = ["linked", "unresolved", "unresolved", "unresolved", "linked", "skipped"];
    assert.deepEqual(summaryOfMentions(statuses.map((status) => ({ status }))), { linked: 2, notLinked: 3, ambiguous: 0, skipped: 1 });
    // The cautious reading's statuses are not linked too.
    assert.equal(summaryOfMentions([{ status: "unresolved_low_quality" }, { status: "ambiguous" }]).notLinked, 1);
  });
});

describe("why a name was not looked up", () => {
  it("says it in words", () => {
    assert.equal(reasonInWords("ent_type=date (dates never query Wikidata)"), "A date: dates are not looked up.");
    assert.equal(reasonInWords("ent_type=role (not linkable)"), "A role or title: these are not looked up.");
    assert.equal(reasonInWords("linking time budget of 60s spent"), "Not looked up: the time for linking this page (60 s) ran out.");
    assert.equal(
      reasonInWords("low_evidence_place: cap=True gazetteer=False conf=0.55 blacklisted=False lexical=True context=False shape=True"),
      "Not linked: too little on the page suggests this is a place.",
    );
    assert.equal(reasonInWords("name_likeness=0.120 < 0.35 (OCR garbage)"), "Not linked: this does not read like a name; it may be a misreading.");
  });

  it("labels each status as the summary counts it", () => {
    assert.equal(mentionStatus("skipped").label, "Not looked up");
    assert.equal(mentionStatus("unresolved").label, "Not linked");
    assert.equal(mentionStatus("unresolved_low_quality").label, "Not linked");
    assert.equal(mentionStatus("linked").label, "Linked");
    assert.equal(mentionStatus("ambiguous").label, "Ambiguous");
  });
});
