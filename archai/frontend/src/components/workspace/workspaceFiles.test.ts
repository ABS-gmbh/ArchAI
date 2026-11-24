import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { pageTooLargeMessage } from "./workspaceFiles";

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
