// How a page's reading is shown: its text, the recogniser's name, and the badges
// that say how far to trust it.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { OCRExtractResponse } from "@/lib/api/ocrAgent";

import { cleanTranscription, countUncertainties, describeExtraction, recogniserName, withIndexBadge } from "./workspaceExtraction";

function reading(overrides: Partial<OCRExtractResponse> = {}): OCRExtractResponse {
  return {
    status: "FULL",
    model_used: "kraken_catmus",
    ocr_engine: "segmented",
    fallbacks_used: [],
    detected_language: "latin",
    language_confidence: null,
    script_hint: "latin",
    confidence: null,
    warnings: [],
    lines: [],
    text: "",
    ...overrides,
  };
}

describe("cleanTranscription", () => {
  it("keeps the '?' and '…' the segmented engine read on the page (a punctus interrogativus is text)", () => {
    const text = "  Quis est hic?  \nEt cetera…";
    assert.equal(cleanTranscription({ ocr_engine: "segmented" }, text), "Quis est hic?\nEt cetera…");
  });

  it("removes the uncertainty marks GLM-OCR inserts, and the spaces they leave", () => {
    const text = "In principio [...] erat ?verbum\net verbum […] erat…";
    assert.equal(cleanTranscription({ ocr_engine: "glmocr" }, text), "In principio erat verbum\net verbum erat");
  });

  it("counts each uncertainty mark once", () => {
    assert.equal(countUncertainties("a [...] b […] c … d ?"), 4);
  });
});

describe("recogniserName", () => {
  it("names the recognisers as readers know them", () => {
    assert.equal(recogniserName("kraken_catmus"), "CATMuS Medieval");
    assert.equal(recogniserName("kraken_mccatmus"), "McCATMuS");
    assert.equal(recogniserName("KRAKEN_CREMMA_MEDIEVAL"), "CREMMA Medieval");
    assert.equal(recogniserName("glm-ocr:latest"), "GLM-OCR");
  });

  it("shows an unknown recogniser as the backend names it", () => {
    assert.equal(recogniserName("calamari_gt4histocr"), "calamari_gt4histocr");
    assert.equal(recogniserName(null), "");
  });
});

describe("describeExtraction", () => {
  const labels = (badges: { label: string }[]) => badges.map((badge) => badge.label);

  it("leads with the quality verdict and the recogniser's confidence", () => {
    const badges = describeExtraction(reading({ quality_label: "ok", confidence: 0.914 }), "");
    assert.deepEqual(labels(badges), ["Quality: acceptable", "Confidence 91%"]);
    assert.equal(badges[0].tone, "ok");
  });

  it("says the page is not searchable when the quality gate kept it out of search", () => {
    const badges = describeExtraction(reading({ quality_label: "RISKY", downstream_mode: "mention_only" }), "");
    assert.deepEqual(labels(badges), ["Quality: risky", "Not searchable"]);
  });

  it("does not repeat 'not searchable' for an unreliable page, whose badge already says so", () => {
    const badges = describeExtraction(reading({ quality_label: "UNRELIABLE", downstream_mode: "blocked" }), "");
    assert.deepEqual(labels(badges), ["Quality: unreliable"]);
  });

  it("counts lines read by a fallback recogniser, naming it, apart from other warnings", () => {
    const badges = describeExtraction(
      reading({ warnings: ["LINES_READ_BY_FALLBACK:kraken_mccatmus:2", "LINES_READ_BY_FALLBACK:kraken_cremma_medieval:1", "Tile 3 was retried."] }),
      "",
    );
    assert.deepEqual(labels(badges), ["3 lines from a fallback recogniser", "1 warning"]);
    assert.match(badges[0].detail, /McCATMuS and CREMMA Medieval/);
    assert.equal(badges[1].detail, "Tile 3 was retried.");
  });

  it("says 'line' for one fallback line", () => {
    const badges = describeExtraction(reading({ warnings: ["LINES_READ_BY_FALLBACK:kraken_mccatmus:1"] }), "");
    assert.deepEqual(labels(badges), ["1 line from a fallback recogniser"]);
  });

  it("counts uncertain places only where the engine inserts the marks", () => {
    assert.deepEqual(labels(describeExtraction(reading({ ocr_engine: "glmocr" }), "a [...] b ?")), ["2 uncertain places"]);
    assert.deepEqual(labels(describeExtraction(reading({ ocr_engine: "segmented" }), "a [...] b ?")), []);
  });
});

// Live regression (2026-11-24): without a chat key every page said "Quality:
// high … is used for search" while its Steps said it was not indexed. The badge
// now gives the verdict only, and a "Not indexed" pill carries the step's reason.
describe("the quality verdict and indexing", () => {
  it("states the verdict without claiming the page was indexed", () => {
    for (const quality of ["HIGH", "OK"]) {
      const [badge] = describeExtraction(reading({ quality_label: quality }), "");
      assert.doesNotMatch(badge.detail, /is used for search|indexed/);
      assert.match(badge.detail, /good enough to search and to look names up/);
    }
  });

  const quality = describeExtraction(reading({ quality_label: "HIGH" }), "");
  const event = (stage: string, message: string) => ({ stage, event: "INFO", message });
  const unavailable =
    "Not indexed: the embedding service is unavailable, so this page cannot be searched until the run is indexed again.";

  it("adds a Not indexed pill with the step's reason", () => {
    const badges = withIndexBadge(quality, [event("ANALYZE_DONE", "chunks=2, mentions=6"), event("INDEX_SKIPPED", unavailable)]);
    assert.deepEqual(
      badges.map((badge) => badge.label),
      ["Quality: high", "Not indexed"],
    );
    assert.equal(badges[1].tone, "warn");
    assert.equal(badges[1].detail, unavailable);
  });

  it("adds nothing when the page was indexed, or indexed again after a skip", () => {
    assert.equal(withIndexBadge(quality, [event("INDEX_DONE", "Indexed 2 chunks.")]).length, 1);
    assert.equal(withIndexBadge(quality, [event("INDEX_SKIPPED", unavailable), event("INDEX_DONE", "Indexed 2 chunks.")]).length, 1);
    assert.equal(withIndexBadge(quality, undefined).length, 1);
  });

  it("does not repeat what Not searchable already says", () => {
    const risky = describeExtraction(reading({ quality_label: "RISKY", downstream_mode: "mention_only" }), "");
    const skipped = event("INDEX_SKIPPED", "RAG auto-index SKIPPED: token_search_allowed=False (quality=RISKY)");
    assert.deepEqual(withIndexBadge(risky, [skipped]).map((badge) => badge.label), ["Quality: risky", "Not searchable"]);
  });
});
