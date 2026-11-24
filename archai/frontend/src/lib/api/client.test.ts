import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { readErrorDetail } from "./client";

const answer = (body: string, status = 500, statusText = "Internal Server Error") => new Response(body, { status, statusText });

describe("readErrorDetail", () => {
  it("gives the backend's sentence", async () => {
    assert.equal(await readErrorDetail(answer('{"detail": "The page is too large."}', 413)), "The page is too large.");
  });

  it("names the field of a validation error", async () => {
    const body = JSON.stringify({ detail: [{ loc: ["body", "temperature"], msg: "must be at most 2" }] });
    assert.equal(await readErrorDetail(answer(body, 422)), "temperature: must be at most 2");
  });

  it("never shows an empty JSON body as the reason", async () => {
    for (const body of ["{}", '{"detail": null}', '{"detail": ""}', "[]"]) {
      const text = await readErrorDetail(answer(body, 502, "Bad Gateway"));
      assert.equal(text, "The server answered 502 Bad Gateway without giving a reason.");
    }
  });

  it("keeps a plain-text reason, and says something when there is none", async () => {
    assert.equal(await readErrorDetail(answer("upstream timed out", 504, "Gateway Timeout")), "upstream timed out");
    assert.equal(await readErrorDetail(answer("", 503, "")), "The server answered 503 without giving a reason.");
  });
});
