import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { answerProper, describeVerification } from "./chatAnswer";

// The text of a real answer from 2026-11-25, shortened: the answer, then the blocks the backend appends.
const ANSWER = 'The third commandment requires keeping the Sabbath holy. The text states: "Garde q̃ tu saintefies le ior dou sa medi."';
const WITH_BLOCKS =
  ANSWER +
  "\n\n[Verification]\nAssessment: unavailable\nVerified answer: " +
  ANSWER +
  "\nNotes: Verifier response could not be parsed cleanly.\nCitations checked: none\nVerifier model: qwen3-235b-a22b" +
  "\n\n[Retrieval]\nSemantic search was unavailable; evidence was found by keyword match only.";

describe("answerProper", () => {
  it("shows the answer once, without the verification and retrieval blocks", () => {
    assert.equal(answerProper(WITH_BLOCKS), ANSWER);
    assert.equal(answerProper(ANSWER + "\n\n[Retrieval]\nNo passage of this page matched the question."), ANSWER);
  });

  it("leaves an answer without blocks alone, brackets in it included", () => {
    assert.equal(answerProper(ANSWER), ANSWER);
    const bracketed = "The scribe wrote [sic] twice. See [1] and [Verification of the date] below.";
    assert.equal(answerProper(bracketed), bracketed);
  });

  it("keeps the trailing space of a piece still streaming, and cuts a block as it starts to arrive", () => {
    assert.equal(answerProper("The third ", true), "The third ");
    assert.equal(answerProper(ANSWER + "\n\n[Verification]\nAssess", true), ANSWER);
    assert.equal(answerProper("The third \n", false), "The third");
  });
});

describe("describeVerification", () => {
  it("says nothing when no verifier ran or it could not run", () => {
    assert.equal(describeVerification(null, ANSWER), null);
    assert.equal(describeVerification({ assessment: "unavailable", corrected_answer: ANSWER, notes: ["could not be parsed"] }, ANSWER), null);
  });

  it("gives the verdict, and the checker's wording only when it differs from the answer", () => {
    const same = describeVerification({ assessment: "supported", corrected_answer: `  ${ANSWER}\n`, notes: [] }, ANSWER);
    assert.equal(same?.label, "Checked: supported");
    assert.equal(same?.tone, "ok");
    assert.equal(same?.corrected, null);

    const partly = describeVerification(
      { assessment: "partially_supported", corrected_answer: "The page names the third commandment only.", notes: ["The year is not in the passage.", "", " "] },
      ANSWER,
    );
    assert.equal(partly?.tone, "warn");
    assert.equal(partly?.corrected, "The page names the third commandment only.");
    assert.deepEqual(partly?.notes, ["The year is not in the passage."]);

    assert.equal(describeVerification({ assessment: "UNSUPPORTED", corrected_answer: "", notes: [] }, ANSWER)?.tone, "error");
  });
});
