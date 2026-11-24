import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { requestLimitsFrom } from "./health";

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
