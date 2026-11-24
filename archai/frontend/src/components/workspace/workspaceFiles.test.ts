import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { pageRefusal, pageTooLargeMessage, tooManyPixelsMessage } from "./workspaceFiles";

describe("pageTooLargeMessage", () => {
  it("gives the size as sent and both ways out", () => {
    const message = pageTooLargeMessage(53 * 1024 * 1024);
    assert.match(message, /about 53 MB as it travels/);
    assert.match(message, /JPEG/);
    assert.match(message, /MAX_JSON_MEGABYTES/);
  });

  it("never says 0 MB", () => {
    assert.match(pageTooLargeMessage(10), /about 1 MB/);
  });
});

// Live regression (2026-11-24): a 60 MB PNG went out as an 85 MB request before
// the server refused it as over 48 MB. The workspace now refuses such a page
// before it reserves a run or sends anything, from the limits /api/health gives.
describe("pageRefusal", () => {
  const limits = { jsonBytes: 48 * 1024 * 1024, decodePixels: 89_478_485 };
  const MB = 1024 * 1024;

  it("refuses a page whose request would be over the JSON limit, with the size message", () => {
    const refusal = pageRefusal({ base64Length: 85 * MB, width: 4000, height: 5000 }, limits);
    assert.equal(refusal, pageTooLargeMessage(85 * MB));
  });

  it("leaves room for the rest of the request", () => {
    assert.notEqual(pageRefusal({ base64Length: 48 * MB - 1000 }, limits), null);
    assert.equal(pageRefusal({ base64Length: 40 * MB }, limits), null);
  });

  it("refuses a page over the pixel limit once its size is known", () => {
    const refusal = pageRefusal({ base64Length: 2 * MB, width: 12_000, height: 9_000 }, limits);
    assert.equal(refusal, tooManyPixelsMessage(12_000, 9_000, limits.decodePixels));
    assert.equal(pageRefusal({ base64Length: 2 * MB, width: 9_000, height: 9_000 }, limits), null);
    assert.equal(pageRefusal({ base64Length: 2 * MB }, limits), null); // not loaded yet: the server checks
  });

  it("checks nothing without the server's limits (an older backend)", () => {
    assert.equal(pageRefusal({ base64Length: 500 * MB, width: 1e5, height: 1e5 }, null), null);
  });
});

describe("tooManyPixelsMessage", () => {
  it("gives both sizes in megapixels and both ways out", () => {
    assert.equal(
      tooManyPixelsMessage(12_000, 9_000, 89_478_485),
      "This page image has 108 megapixels; the server reads at most 89.5. Save it at a lower resolution, or ask whoever runs the server to raise OCR_MAX_DECODE_PIXELS.",
    );
  });
});
