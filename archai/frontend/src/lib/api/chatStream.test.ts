import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { applyStreamEvent, type ChatStreamEvent, type ChatStreamState, checkOutcome, INITIAL_STREAM } from "./chatStream";

const run = (events: ChatStreamEvent[]) => {
  let state: ChatStreamState = INITIAL_STREAM;
  const answeredAt: number[] = [];
  events.forEach((event, index) => {
    const next = applyStreamEvent(state, event);
    state = next.state;
    if (next.answered) answeredAt.push(index);
  });
  return { state, answeredAt };
};

const VERDICT = { assessment: "supported", corrected_answer: "x", notes: [] };

describe("applyStreamEvent", () => {
  it("makes the answer whole at the answer event, before the verdict", () => {
    const { state, answeredAt } = run([
      { type: "delta", delta: "The page " },
      { type: "delta", delta: "names Lausanne." },
      { type: "answer", text: "The page names Lausanne.", model: "chat", checking: true },
      { type: "verification", verification: VERDICT },
      { type: "delta", delta: "\n\n[Verification]\n…" },
      { type: "done", text: "The page names Lausanne.\n\n[Verification]\n…", verification: VERDICT },
    ]);
    assert.deepEqual(answeredAt, [2]);
    assert.equal(state.answer, "The page names Lausanne.");
    assert.equal(state.verification?.assessment, "supported");
    assert.equal(checkOutcome(state), "checked");
  });

  it("is checking between the answer and the verdict", () => {
    const { state } = run([
      { type: "delta", delta: "x" },
      { type: "answer", text: "x", checking: true },
    ]);
    assert.equal(checkOutcome(state), "checking");
  });

  it("marks a stream that ends before the verdict as not checked", () => {
    const { state } = run([
      { type: "answer", text: "x", checking: true },
      { type: "done", text: "x" },
    ]);
    assert.equal(checkOutcome(state), "not_checked");
  });

  it("works with an older backend that sends only deltas and done", () => {
    const { state, answeredAt } = run([
      { type: "delta", delta: "Old " },
      { type: "delta", delta: "answer." },
      { type: "done", text: "Old answer.", verification: VERDICT },
    ]);
    assert.deepEqual(answeredAt, [2]);
    assert.equal(state.answer, "Old answer.");
    assert.equal(checkOutcome(state), "checked");
  });

  it("says nothing of a check when none was announced", () => {
    const { state } = run([{ type: "answer", text: "x", checking: false }, { type: "done", text: "x" }]);
    assert.equal(checkOutcome(state), "none");
  });
});
