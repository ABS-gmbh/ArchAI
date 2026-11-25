import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { followsTheChange, isAtEnd } from "./conversationScroll";

const asked = [{ role: "user" }, { role: "assistant" }];

describe("followsTheChange", () => {
  it("does not move a reader who scrolled up when an answer streams or its verdict arrives", () => {
    // The same messages, longer text or a verdict added: nothing new of the reader's.
    assert.equal(followsTheChange({ countBefore: 2, messages: asked, wasAtEnd: false }), false);
  });

  it("follows a reader who is at the end", () => {
    assert.equal(followsTheChange({ countBefore: 2, messages: asked, wasAtEnd: true }), true);
  });

  it("follows a reader who has just asked, wherever they were", () => {
    const after = [...asked, { role: "user" }, { role: "assistant" }];
    assert.equal(followsTheChange({ countBefore: 2, messages: after, wasAtEnd: false }), true);
  });

  it("does not follow a new message that is not the reader's", () => {
    // A status message from a page read, added while the reader is reading above.
    assert.equal(followsTheChange({ countBefore: 2, messages: [...asked, { role: "assistant" }], wasAtEnd: false }), false);
  });
});

describe("isAtEnd", () => {
  it("counts the last few pixels as the end", () => {
    assert.equal(isAtEnd({ scrollHeight: 1060, clientHeight: 641, scrollTop: 419 }), true);
    assert.equal(isAtEnd({ scrollHeight: 1060, clientHeight: 641, scrollTop: 380 }), true);
    assert.equal(isAtEnd({ scrollHeight: 1060, clientHeight: 641, scrollTop: 0 }), false);
  });
});
