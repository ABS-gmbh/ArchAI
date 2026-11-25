import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { citedSourceList, describeLines, isSourceList, parseSources, sourceLineIds, splitCitations } from "./citations";

// An answer as the backend returns one since the evidence is labelled [1], [2], ... (2026-11-25).
const ANSWER = "Caldus de Lausanne witnesses the grant [2]. The date is given as 1214 [1][3], and the price [2, 4].";
const RUN = "run-7";
const LINES = Array.from({ length: 10 }, (_, index) => ({ id: `r${index + 1}` }));

const SOURCES = [
  { label: "1", run_id: RUN, chunk_ids: ["c1"], start_offset: 0, end_offset: 80, line_start: 1, line_end: 2, cited: true },
  { label: "2", run_id: RUN, chunk_ids: ["c2"], start_offset: 90, end_offset: 300, line_start: 4, line_end: 7, cited: true },
  { label: "3", run_id: RUN, chunk_ids: ["c3"], start_offset: 310, end_offset: 330, cited: true },
  { label: "4", run_id: "another-run", chunk_ids: ["c9"], start_offset: 0, end_offset: 20, line_start: 9, line_end: 9, cited: true },
  { label: "5", run_id: RUN, chunk_ids: ["c5"], start_offset: 400, end_offset: 420, line_start: 10, line_end: 10, cited: false },
];

describe("splitCitations", () => {
  it("cuts the answer into text and label groups, in order", () => {
    assert.deepEqual(splitCitations(ANSWER), [
      { kind: "text", text: "Caldus de Lausanne witnesses the grant " },
      { kind: "cite", labels: ["2"] },
      { kind: "text", text: ". The date is given as 1214 " },
      { kind: "cite", labels: ["1"] },
      { kind: "cite", labels: ["3"] },
      { kind: "text", text: ", and the price " },
      { kind: "cite", labels: ["2", "4"] },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves text without labels, and brackets that are not labels, as text", () => {
    assert.deepEqual(splitCitations("No label here [sic] [a1]."), [{ kind: "text", text: "No label here [sic] [a1]." }]);
    assert.deepEqual(splitCitations(""), []);
  });
});

describe("parseSources", () => {
  it("keeps the label, the run, the lines and whether it is cited", () => {
    const parsed = parseSources(SOURCES);
    assert.deepEqual(parsed[1], { label: "2", run_id: RUN, line_start: 4, line_end: 7, cited: true });
    assert.deepEqual(parsed[2], { label: "3", run_id: RUN, cited: true });
  });

  it("leaves out what is malformed, and lines that are not whole", () => {
    assert.deepEqual(parseSources("nope"), []);
    assert.deepEqual(
      parseSources([null, { label: 2, run_id: RUN }, { label: "x", run_id: RUN }, { label: "6", run_id: RUN, line_start: 5, line_end: 3 }]),
      [{ label: "6", run_id: RUN }],
    );
  });
});

describe("isSourceList", () => {
  it("accepts what the workspace stores and refuses anything else", () => {
    assert.equal(isSourceList(parseSources(SOURCES)), true);
    assert.equal(isSourceList([]), true);
    assert.equal(isSourceList("[]"), false);
    assert.equal(isSourceList([{ label: "1" }]), false);
    assert.equal(isSourceList([{ label: "1", run_id: RUN, line_start: 2 }]), false);
    assert.equal(isSourceList([{ label: "1", run_id: RUN, line_start: 0, line_end: 2 }]), false);
  });
});

describe("sourceLineIds", () => {
  const [one, two, three, other] = parseSources(SOURCES);

  it("gives the rows a label stands for on the page shown", () => {
    assert.deepEqual(sourceLineIds(two, LINES, RUN), ["r4", "r5", "r6", "r7"]);
    assert.deepEqual(sourceLineIds(one, LINES, RUN), ["r1", "r2"]);
  });

  it("gives none without lines, for another run, or for lines the page does not have", () => {
    assert.deepEqual(sourceLineIds(three, LINES, RUN), []);
    assert.deepEqual(sourceLineIds(other, LINES, RUN), []);
    assert.deepEqual(sourceLineIds(two, LINES, null), []);
    assert.deepEqual(sourceLineIds(two, LINES.slice(0, 5), RUN), []);
    assert.deepEqual(sourceLineIds(undefined, LINES, RUN), []);
  });
});

describe("describeLines and citedSourceList", () => {
  it("says the lines in words", () => {
    const [one, two, three, other] = parseSources(SOURCES);
    assert.equal(describeLines(two), "lines 4 to 7");
    assert.equal(describeLines(other), "line 9");
    assert.equal(describeLines(three), null);
    assert.equal(describeLines(one), "lines 1 to 2");
  });

  it("lists the cited sources only, in label order, each in words", () => {
    const listed = citedSourceList(parseSources(SOURCES), LINES, RUN).map(({ source, words, lineIds }) => [source.label, words, lineIds.length]);
    assert.deepEqual(listed, [
      ["1", "Lines 1 to 2", 2],
      ["2", "Lines 4 to 7", 4],
      ["3", "Lines not known", 0],
      ["4", "From another reading of the page", 0],
    ]);
  });
});
