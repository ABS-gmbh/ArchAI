import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { describeChecks, requestLimitsFrom } from "./health";

describe("requestLimitsFrom", () => {
  it("reads the limits /health states", () => {
    assert.deepEqual(requestLimitsFrom({ status: "ok", limits: { json_bytes: 50331648, decode_pixels: 89478485 } }), {
      jsonBytes: 50331648,
      decodePixels: 89478485,
    });
  });

  it("gives null for an older backend or anything that is not a positive number", () => {
    for (const body of [{ status: "ok" }, null, { limits: { json_bytes: "x", decode_pixels: 1 } }, { limits: { json_bytes: 0, decode_pixels: 1 } }]) {
      assert.equal(requestLimitsFrom(body), null);
    }
  });
});

// The failing checks of a server with no chat key and no Ollama, as /health worded them on 2026-11-25.
const NO_KEY_NO_OLLAMA = {
  database: { ok: true, detail: "The pipeline database answers." },
  ollama: { ok: false, detail: "Ollama is not reachable; GLM-OCR is unavailable." },
  chat_api_key: { ok: false, detail: "No chat API key is configured; chat and translation are unavailable." },
  layout_models: { ok: true, detail: "The layout models are installed." },
  search_index: {
    ok: false,
    detail:
      "Pages are read but not indexed for search: the embedding model needs the chat key. Set the key, or set RAG_EMBEDDING_MODEL= to index with the local embedder (see docs/deployment.md).",
  },
};
// A setting's name, a file or a service: what only whoever runs the server can act on.
const OPERATOR_WORDS = /[A-Z][A-Z0-9]*_[A-Z0-9_]+|docs\/|\.md\b|Ollama|API key|embedd/;

describe("describeChecks", () => {
  it("tells a reader what they cannot do, without a setting, a file or a service", () => {
    const { problems } = describeChecks(NO_KEY_NO_OLLAMA);
    assert.deepEqual(problems, [
      "One of the two ways of reading a page (GLM-OCR) is not available here.",
      "Questions, translations and search are not available here right now.",
    ]);
    for (const problem of problems) {
      assert.doesNotMatch(problem, OPERATOR_WORDS);
    }
  });

  it("keeps the backend's own sentences for whoever runs the server", () => {
    const { details } = describeChecks(NO_KEY_NO_OLLAMA);
    assert.deepEqual(details, [
      "OCR engine (Ollama): Ollama is not reachable; GLM-OCR is unavailable.",
      "Chat provider key: No chat API key is configured; chat and translation are unavailable.",
      NO_KEY_NO_OLLAMA.search_index.detail,
    ]);
  });

  it("says the missing search by itself when the chat key is there", () => {
    const { problems } = describeChecks({ chat_api_key: { ok: true }, search_index: { ok: false, detail: "Pages are read but not indexed for search: …" } });
    assert.deepEqual(problems, ["Pages are read, but answers cannot yet draw on their text."]);
  });

  it("has words for every check the backend reports, and for one it does not know", () => {
    for (const name of ["database", "ollama", "chat_api_key", "layout_models", "line_recognizer", "search_index", "task_models"]) {
      const { problems } = describeChecks({ [name]: { ok: false, detail: "x" } });
      assert.equal(problems.length, 1);
      assert.doesNotMatch(problems[0], OPERATOR_WORDS);
      assert.notEqual(problems[0], "Part of the server is not available.");
    }
    assert.deepEqual(describeChecks({ a_new_check: { ok: false, detail: "SOME_SETTING is wrong" } }), {
      problems: ["Part of the server is not available."],
      details: ["a_new_check: SOME_SETTING is wrong"],
    });
  });

  it("is empty when nothing fails", () => {
    assert.deepEqual(describeChecks({ database: { ok: true }, ollama: null }), { problems: [], details: [] });
  });
});
