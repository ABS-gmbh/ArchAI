// What the /api proxy (route.ts) forwards to the backend about the client.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { clientAddress, upstreamHeaders } from "./proxyRules";

// A security review found that the proxy passed the client's own
// X-Forwarded-For to the backend, whose uvicorn trusts that header from
// 127.0.0.1: the analytics login limit, keyed on the address, could be stepped
// around with a made-up address per guess (40 guesses, never a 429).
describe("clientAddress", () => {
  it("trusts no forwarded-for entry when nothing in front of the frontend adds one", () => {
    // Next keeps a forwarded-for the client sent, and fills one in only when there is none,
    // so with no proxy in front the entries cannot be told from the client's own.
    assert.equal(clientAddress("10.9.8.7", 0), null);
    assert.equal(clientAddress(null, 0), null);
  });

  it("takes the entry the trusted proxies added, not the ones the client sent", () => {
    // nginx appends the address it saw: "<client's own entries>, <real address>".
    assert.equal(clientAddress("1.1.1.1, 2.2.2.2, 203.0.113.5", 1), "203.0.113.5");
    assert.equal(clientAddress("1.1.1.1, 203.0.113.5, 10.0.0.2", 2), "203.0.113.5");
  });

  it("gives no address when there are fewer entries than trusted proxies", () => {
    assert.equal(clientAddress("203.0.113.5", 2), null);
    assert.equal(clientAddress("", 1), null);
  });
});

describe("upstreamHeaders", () => {
  const request = new Headers({
    "content-type": "application/json",
    host: "frontend.example",
    "x-forwarded-for": "1.1.1.1, 203.0.113.5",
    forwarded: "for=1.1.1.1",
    "x-real-ip": "1.1.1.1",
    "x-forwarded-host": "evil.example",
    "x-forwarded-proto": "https",
    "x-forwarded-port": "443",
    connection: "keep-alive",
    authorization: "Bearer analytics-session",
  });

  it("drops every client-supplied forwarding header", () => {
    const headers = upstreamHeaders(request, "127.0.0.1:8010", { trustedProxyHops: 0, token: "" });
    for (const name of ["x-forwarded-for", "forwarded", "x-real-ip", "x-forwarded-host", "x-forwarded-proto", "x-forwarded-port"]) {
      assert.equal(headers[name], undefined, name);
    }
  });

  it("sends the address the trusted proxy saw as the only forwarded-for entry", () => {
    const headers = upstreamHeaders(request, "127.0.0.1:8010", { trustedProxyHops: 1, token: "" });
    assert.equal(headers["x-forwarded-for"], "203.0.113.5");
  });

  it("keeps the request's own headers, sets the backend's host and adds the token", () => {
    const headers = upstreamHeaders(request, "127.0.0.1:8010", { trustedProxyHops: 0, token: "server-side-token" });
    assert.equal(headers.host, "127.0.0.1:8010");
    assert.equal(headers["content-type"], "application/json");
    assert.equal(headers.authorization, "Bearer analytics-session");
    assert.equal(headers["x-api-token"], "server-side-token");
    assert.equal(headers.connection, undefined);
  });
});
