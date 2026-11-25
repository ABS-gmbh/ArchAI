import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { clockTime, waitFrom, waitShown } from "./providerWait";

const NOW = new Date(2026, 10, 25, 14, 0, 0).getTime();

describe("waitFrom", () => {
  it("holds Ask for a short wait and only tells the time of a long one", () => {
    assert.deepEqual(waitFrom(7, NOW), { until: NOW + 7000, holds: true });
    assert.deepEqual(waitFrom(60, NOW), { until: NOW + 60000, holds: true });
    assert.deepEqual(waitFrom(1920, NOW), { until: NOW + 1920000, holds: false });
  });

  it("is no wait without a usable Retry-After", () => {
    for (const value of [null, undefined, 0, -3, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.equal(waitFrom(value, NOW), null);
    }
  });
});

describe("waitShown", () => {
  it("counts a short wait down, in whole seconds, and lets Ask go when it is over", () => {
    const wait = waitFrom(7, NOW);
    assert.deepEqual(waitShown(wait, NOW), { holdsAsk: true, note: "Ask again in 7 s" });
    assert.deepEqual(waitShown(wait, NOW + 6100), { holdsAsk: true, note: "Ask again in 1 s" });
    assert.deepEqual(waitShown(wait, NOW + 7000), { holdsAsk: false, note: null });
  });

  it("says the clock time of a long wait and leaves Ask on", () => {
    assert.deepEqual(waitShown(waitFrom(1920, NOW), NOW), { holdsAsk: false, note: "You can ask again at 14:32" });
  });

  it("never holds Ask once the time has passed, so a wait that outlived its page cannot lock it", () => {
    assert.deepEqual(waitShown({ until: NOW - 1, holds: true }, NOW), { holdsAsk: false, note: null });
    assert.deepEqual(waitShown(null, NOW), { holdsAsk: false, note: null });
  });
});

describe("clockTime", () => {
  it("is hours and minutes on the reader's clock", () => {
    assert.equal(clockTime(new Date(2026, 10, 25, 9, 5).getTime()), "09:05");
  });
});
