// What the /api proxy (route.ts) forwards to the backend about the client.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { clientAddress, crossSiteRefusal, upstreamHeaders } from "./proxyRules";

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

// The review sent POST /api/ocr/runs (no body) and a multipart POST to
// /api/predict/batch with "Origin: https://evil.example": 201 and 202. Through
// the proxy, which adds the API token itself, any page a user visits could do
// the same in their browser: cross-site request forgery for every request a
// form can send without a preflight.
describe("crossSiteRefusal", () => {
  const own = { trustForwardedHost: false, allowedOrigins: [] as string[] };
  const headers = (values: Record<string, string>) => new Headers({ host: "archai.example:3000", ...values });

  it("refuses a state-changing request from another origin", () => {
    for (const method of ["POST", "PUT", "PATCH", "DELETE"]) {
      assert.match(String(crossSiteRefusal(method, headers({ origin: "https://evil.example" }), own)), /another site/, method);
    }
    assert.notEqual(crossSiteRefusal("POST", headers({ origin: "null" }), own), null);
    assert.notEqual(crossSiteRefusal("POST", headers({ "sec-fetch-site": "cross-site" }), own), null);
  });

  it("lets the frontend's own pages through, by name or by address", () => {
    assert.equal(crossSiteRefusal("POST", headers({ origin: "http://archai.example:3000" }), own), null);
    const byAddress = new Headers({ host: "192.0.2.10:3000", origin: "http://192.0.2.10:3000", "sec-fetch-site": "same-origin" });
    assert.equal(crossSiteRefusal("POST", byAddress, own), null);
    const defaultPort = new Headers({ host: "archai.example", origin: "https://archai.example" });
    assert.equal(crossSiteRefusal("POST", defaultPort, own), null);
  });

  it("lets through reads, and requests that carry no Origin (scripts, server-side clients)", () => {
    assert.equal(crossSiteRefusal("GET", headers({ origin: "https://evil.example" }), own), null);
    assert.equal(crossSiteRefusal("HEAD", headers({ origin: "https://evil.example" }), own), null);
    assert.equal(crossSiteRefusal("POST", headers({}), own), null);
  });

  it("uses a reverse proxy's forwarded host only when told to trust it", () => {
    // nginx with its default "Host: $proxy_host" sends the upstream's address as Host.
    const behindProxy = new Headers({ host: "127.0.0.1:3000", "x-forwarded-host": "archai.example", origin: "https://archai.example" });
    assert.notEqual(crossSiteRefusal("POST", behindProxy, own), null);
    assert.equal(crossSiteRefusal("POST", behindProxy, { ...own, trustForwardedHost: true }), null);
    assert.equal(crossSiteRefusal("POST", behindProxy, { ...own, allowedOrigins: ["https://archai.example"] }), null);
    const forged = new Headers({ host: "127.0.0.1:3000", "x-forwarded-host": "evil.example", origin: "https://archai.example" });
    assert.notEqual(crossSiteRefusal("POST", forged, { ...own, trustForwardedHost: true }), null);
  });
});
