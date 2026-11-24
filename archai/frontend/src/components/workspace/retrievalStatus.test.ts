// The badges under a chat answer that say whether it rests on the page's text.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { ChatRetrievalStatus } from "@/lib/api/chat";

import { describeRetrieval } from "./retrievalStatus";

function retrieval(overrides: Partial<ChatRetrievalStatus> = {}): ChatRetrievalStatus {
  return {
    status: "ok",
    chunk_mode: "hybrid",
    entity_mode: "dense",
    chunk_hits: 3,
    entity_hits: 0,
    run_scoped: true,
    notes: [],
    verifier: "skipped",
    ...overrides,
  };
}

const badges = (value: ChatRetrievalStatus | null) => describeRetrieval(value).map((badge) => [badge.tone, badge.label]);

describe("describeRetrieval", () => {
  it("shows nothing when retrieval was not used or not reported", () => {
    assert.deepEqual(badges(null), []);
    assert.deepEqual(badges(retrieval({ status: "not_used" })), []);
  });

  it("counts the page's passages the answer rests on", () => {
    assert.deepEqual(badges(retrieval({ chunk_hits: 3 })), [["ok", "Page sources · 3 passages"]]);
    assert.deepEqual(badges(retrieval({ chunk_hits: 1 })), [["ok", "Page sources · 1 passage"]]);
  });

  it("warns when search found nothing in the page's sources, and says what the answer rests on instead", () => {
    const [badge] = describeRetrieval(retrieval({ chunk_hits: 0 }));
    assert.equal(badge.label, "No matching passages");
    assert.equal(badge.tone, "warn");
    assert.match(badge.detail, /not grounded in this page's text/);
    const [withExcerpt] = describeRetrieval(retrieval({ chunk_hits: 0, transcript_in_context: true }));
    assert.match(withExcerpt.detail, /unsearched transcript excerpt/);
  });

  it("warns when the page was never indexed for search", () => {
    assert.deepEqual(badges(retrieval({ status: "no_searchable_run", chunk_hits: 0 })), [["warn", "No searchable transcription"]]);
  });

  it("says which half of hybrid search was missing", () => {
    assert.deepEqual(badges(retrieval({ status: "degraded", chunk_mode: "lexical", chunk_hits: 2 })), [["warn", "Keyword search only · 2 passages"]]);
    assert.deepEqual(badges(retrieval({ status: "degraded", chunk_mode: "dense", chunk_hits: 2 })), [["warn", "Semantic search only · 2 passages"]]);
    assert.deepEqual(badges(retrieval({ status: "degraded", chunk_mode: "error", chunk_hits: 0 })), [["warn", "Search partly unavailable · 0 passages"]]);
  });

  it("marks an answer whose sources could not be searched as an error", () => {
    assert.deepEqual(badges(retrieval({ status: "unavailable", chunk_hits: 0 })), [["error", "Search unavailable"]]);
  });

  it("adds the verifier's state: checked, or not verified when it was unavailable", () => {
    assert.deepEqual(badges(retrieval({ verifier: "ran" })), [
      ["ok", "Page sources · 3 passages"],
      ["ok", "Checked by verifier"],
    ]);
    assert.deepEqual(badges(retrieval({ verifier: "unavailable" })), [
      ["ok", "Page sources · 3 passages"],
      ["warn", "Not verified"],
    ]);
    assert.deepEqual(badges(retrieval({ verifier: "disabled" })), [["ok", "Page sources · 3 passages"]]);
  });

  it("gives a general chat, not about one page, no grounding badges", () => {
    assert.deepEqual(badges(retrieval({ run_scoped: false, verifier: "ran" })), []);
    assert.deepEqual(badges(retrieval({ run_scoped: false, verifier: "unavailable" })), [["warn", "Not verified"]]);
  });
});
