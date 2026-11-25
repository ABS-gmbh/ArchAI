// A refused chat key is said in the Ask panel before a question is asked
// (2026-11-25), and a provider that is only unreachable is said differently.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { chatNotice, defaultChatModel, providerState } from "./chatAvailability";

describe("providerState", () => {
  it("reads the model list's mark, and knows nothing from an older backend", () => {
    assert.equal(providerState({ models: [], provider: "refused" }), "refused");
    assert.equal(providerState({ provider: "unreachable" }), "unreachable");
    assert.equal(providerState({ provider: "ok" }), "ok");
    assert.equal(providerState({ models: ["a"] }), "unknown");
    assert.equal(providerState({ provider: "something" }), "unknown");
    assert.equal(providerState(null), "unknown");
  });
});

describe("chatNotice", () => {
  it("says a refused key, and stops the composer, since no question can be answered", () => {
    const notice = chatNotice("refused");
    assert.ok(notice);
    assert.match(notice.sentence, /refused this server's key/);
    assert.equal(notice.canAsk, false);
    assert.match(String(notice.disabledReason), /refused/);
  });

  it("says an unreachable provider differently, and leaves the composer usable", () => {
    const notice = chatNotice("unreachable");
    assert.ok(notice);
    assert.doesNotMatch(notice.sentence, /key/);
    assert.match(notice.sentence, /did not answer/);
    assert.equal(notice.canAsk, true);
    assert.equal(notice.disabledReason, null);
  });

  it("says nothing when the provider answered, or nothing is known", () => {
    assert.equal(chatNotice("ok"), null);
    assert.equal(chatNotice("unknown"), null);
  });

  it("names no setting, host or provider text", () => {
    for (const state of ["refused", "unreachable"] as const) {
      assert.doesNotMatch(chatNotice(state)?.sentence ?? "", /API_KEY|\.env|https?:|Unknown API key/);
    }
  });
});

describe("defaultChatModel", () => {
  it("follows the backend's default model, then its chat task model", () => {
    assert.equal(defaultChatModel({ default_model: "qwen3.5-397b-a17b", task_models: { chat_rag_model: "other" } }, "offline"), "qwen3.5-397b-a17b");
    assert.equal(defaultChatModel({ default_model: "", task_models: { chat_rag_model: "qwen3.8-27b" } }, "offline"), "qwen3.8-27b");
  });

  it("uses the frontend's own default only when the backend names none or did not answer", () => {
    assert.equal(defaultChatModel({ models: [] }, "offline"), "offline");
    assert.equal(defaultChatModel(null, "offline"), "offline");
    assert.equal(defaultChatModel({ default_model: 7 }, "offline"), "offline");
  });
});
