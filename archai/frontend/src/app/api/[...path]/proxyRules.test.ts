// What the /api proxy (route.ts) forwards to the backend about the client.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  backendTarget,
  clientAddress,
  crossSiteRefusal,
  downstreamHeaders,
  hostRefusal,
  tokenRefusal,
  upstreamHeaders,
} from "./proxyRules";

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
      assert.match(String(crossSiteRefusal(method, headers({ origin: "https://evil.example" }), own)?.detail), /another site/, method);
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

// new URL() resolves dot segments, encoded ones included: a pathname of
// "/api/%2e%2e/docs" became the backend's /docs. Next normalises such a path
// before it routes the request (under `next start`, /api/%2e%2e/docs is the
// frontend's own /docs page and never reaches the proxy), but the proxy does
// not rely on that.
describe("backendTarget", () => {
  it("builds the backend URL for an /api path", () => {
    assert.equal(backendTarget("http://127.0.0.1:8010", "/api/classes", "?a=1")?.href, "http://127.0.0.1:8010/api/classes?a=1");
    assert.equal(backendTarget("http://127.0.0.1:8010/", "/api/x", "")?.href, "http://127.0.0.1:8010/api/x");
    assert.equal(backendTarget("http://backend.internal/archai", "/api/health", "")?.href, "http://backend.internal/archai/api/health");
  });

  it("refuses a path that resolves outside the backend's /api", () => {
    for (const pathname of ["/api/%2e%2e/docs", "/api/%2E%2E/%2e%2e/openapi.json", "/api/../docs", "/api/x/%2e%2e/%2e%2e/redoc"]) {
      assert.equal(backendTarget("http://127.0.0.1:8010", pathname, ""), null, pathname);
    }
    assert.equal(backendTarget("http://backend.internal/archai", "/api/%2e%2e/%2e%2e/docs", ""), null);
  });
});

// The review found the backend's CORS answer (Access-Control-Allow-Origin:
// http://localhost:3000, -Allow-Credentials: true) passed on to the browser
// from the frontend's public origin, and the proxy's 401 telling any visitor
// whether the frontend server has BACKEND_API_TOKEN set.
describe("what crosses the proxy", () => {
  it("does not send the browser's Origin to the backend", () => {
    const headers = upstreamHeaders(new Headers({ origin: "https://archai.example" }), "127.0.0.1:8010", { trustedProxyHops: 0, token: "" });
    assert.equal(headers.origin, undefined);
  });

  it("does not pass the backend's CORS headers, software name or token header back", () => {
    const headers = downstreamHeaders({
      "content-type": "application/json",
      "access-control-allow-origin": "http://localhost:3000",
      "access-control-allow-credentials": "true",
      "access-control-expose-headers": "Content-Disposition",
      server: "uvicorn",
      "x-api-token": "never",
      connection: "close",
      "content-disposition": "attachment; filename=results.zip",
      "set-cookie": ["a=1", "b=2"],
    });
    assert.deepEqual([...new Set(headers.keys())].sort(), ["content-disposition", "content-type", "set-cookie"]);
    assert.deepEqual(headers.getSetCookie(), ["a=1", "b=2"]);
  });

  it("keeps a stream unbuffered", () => {
    const headers = downstreamHeaders({ "content-type": "text/event-stream" });
    assert.equal(headers.get("cache-control"), "no-cache, no-transform");
    assert.equal(headers.get("x-accel-buffering"), "no");
  });

  it("tells the browser the same about a refused token whether or not one is configured", () => {
    const missing = tokenRefusal(false);
    const wrong = tokenRefusal(true);
    assert.equal(missing.detail, wrong.detail);
    assert.equal(missing.status, wrong.status);
    assert.doesNotMatch(missing.detail, /BACKEND_API_TOKEN|API_TOKEN|sends none/);
    // The operator's hint goes to the server log, where it names the setting.
    assert.match(missing.log, /BACKEND_API_TOKEN is not set/);
    assert.match(wrong.log, /answered 401 to BACKEND_API_TOKEN/);
  });
});

// Behind a reverse proxy that rewrites Host (nginx's default), the app's own
// pages are refused too, until TRUST_FORWARDED_HOST or FRONTEND_ORIGINS is
// set; the browser's message alone did not say so.
describe("crossSiteRefusal, for the operator", () => {
  const own = { trustForwardedHost: false, allowedOrigins: [] as string[] };
  const behindProxy = new Headers({ host: "127.0.0.1:3000", origin: "https://archai.example" });

  it("points the browser at the documentation, without naming any setting's value", () => {
    const refusal = crossSiteRefusal("POST", behindProxy, own);
    assert.ok(refusal);
    assert.match(
      refusal.detail,
      /If this is the app's own page, the frontend is behind a proxy that changes the Host header: see docs\/deployment\.md, 'Behind a reverse proxy'\.$/,
    );
    assert.doesNotMatch(refusal.detail, /127\.0\.0\.1|archai\.example/);
  });

  it("tells the server log what was compared and which settings fix it", () => {
    const refusal = crossSiteRefusal("POST", behindProxy, own);
    assert.ok(refusal);
    assert.match(refusal.log, /Origin "https:\/\/archai\.example"/);
    assert.match(refusal.log, /Host "127\.0\.0\.1:3000"/);
    assert.match(refusal.log, /TRUST_FORWARDED_HOST=true/);
    assert.match(refusal.log, /FRONTEND_ORIGINS/);
  });

  it("quotes what the client sent, so it cannot write lines of its own into the log", () => {
    const forged = new Headers({ host: "127.0.0.1:3000", origin: "https://a.example", "sec-fetch-site": "cross-site" });
    const refusal = crossSiteRefusal("POST", forged, own);
    assert.ok(refusal);
    assert.doesNotMatch(refusal.log, /\n/);
  });
});

// Behind a proxy that rewrites Host, the browser's own requests were refused
// although the browser said they were same-origin: Sec-Fetch-Site is a
// forbidden header name, set by the browser and never by a page, so
// "same-origin" is proof enough. The other values get no such pass.
describe("crossSiteRefusal and Sec-Fetch-Site", () => {
  const own = { trustForwardedHost: false, allowedOrigins: [] as string[] };
  const host = "127.0.0.1:3000"; // what a Host-rewriting proxy sends
  const request = (site: string | null, origin: string | null) => {
    const headers = new Headers({ host });
    if (site !== null) headers.set("sec-fetch-site", site);
    if (origin !== null) headers.set("origin", origin);
    return headers;
  };
  const accepted = (site: string | null, origin: string | null) => crossSiteRefusal("POST", request(site, origin), own) === null;
  const foreign = "https://archai.example"; // not the Host this server sees
  const matching = "http://127.0.0.1:3000";

  it("same-origin: accepted whatever the Origin and Host", () => {
    assert.equal(accepted("same-origin", foreign), true);
    assert.equal(accepted("same-origin", matching), true);
    assert.equal(accepted("same-origin", null), true);
  });

  it("same-site: Origin is still compared with Host", () => {
    assert.equal(accepted("same-site", foreign), false);
    assert.equal(accepted("same-site", matching), true);
    assert.equal(accepted("same-site", null), true);
  });

  it("cross-site: refused, with or without Origin", () => {
    assert.equal(accepted("cross-site", foreign), false);
    assert.equal(accepted("cross-site", matching), false);
    assert.equal(accepted("cross-site", null), false);
  });

  it("none (typed in the address bar, a bookmark): Origin is still compared with Host", () => {
    assert.equal(accepted("none", foreign), false);
    assert.equal(accepted("none", matching), true);
    assert.equal(accepted("none", null), true);
  });

  it("no Sec-Fetch-Site (an older browser): Origin is compared with Host, and a mismatch carries the operator's pointer", () => {
    assert.equal(accepted(null, matching), true);
    assert.equal(accepted(null, null), true);
    const refusal = crossSiteRefusal("POST", request(null, foreign), own);
    assert.ok(refusal);
    assert.match(refusal.detail, /see docs\/deployment\.md, 'Behind a reverse proxy'\.$/);
  });

  it("the value is read without regard to case or spaces", () => {
    assert.equal(accepted(" Same-Origin ", foreign), true);
    assert.equal(accepted("CROSS-SITE", null), false);
  });
});

// Second security review (2026-11-24): DNS rebinding. A page on evil.example
// whose name is made to resolve to the frontend's address is same-origin with
// itself: Host evil.example:3000, Origin http://evil.example:3000 and
// Sec-Fetch-Site same-origin were accepted, also with FRONTEND_ORIGINS set, and
// the proxy attached the token, to reads too. A name cannot be rebound when the
// browser was given an address, so the proxy now answers only requests for
// loopback, an IP address, or a host named in FRONTEND_ORIGINS.
describe("hostRefusal", () => {
  const none = { trustForwardedHost: false, allowedOrigins: [] as string[] };
  const listed = { trustForwardedHost: false, allowedOrigins: ["https://archai.example"] };
  const forHost = (host: string, extra: Record<string, string> = {}) => new Headers({ host, ...extra });
  const accepted = (host: string, options = none, extra: Record<string, string> = {}) =>
    hostRefusal(forHost(host, extra), options) === null;

  it("accepts loopback and IP addresses, the default install's", () => {
    for (const host of ["localhost:3000", "LOCALHOST:3000", "127.0.0.1:3000", "[::1]:3000", "192.168.1.20:3000", "10.0.0.5", "app.localhost:3000"]) {
      assert.equal(accepted(host), true, host);
    }
  });

  it("refuses a name it was not given, for reads too, with the operator's pointer", () => {
    for (const host of ["evil.example:3000", "archai.example", "127.0.0.1.evil.example", "localhost.evil.example:3000"]) {
      assert.equal(accepted(host), false, host);
    }
    const refusal = hostRefusal(forHost("evil.example:3000", { origin: "http://evil.example:3000", "sec-fetch-site": "same-origin" }), none);
    assert.ok(refusal);
    assert.match(refusal.detail, /see docs\/deployment\.md, 'Behind a reverse proxy'\.$/);
    assert.doesNotMatch(refusal.detail, /evil\.example/);
    assert.match(refusal.log, /Host "evil\.example:3000"/);
    assert.match(refusal.log, /FRONTEND_ORIGINS/);
    assert.doesNotMatch(refusal.log, /\n/);
  });

  it("accepts a name listed in FRONTEND_ORIGINS, and only that one", () => {
    assert.equal(accepted("archai.example", listed), true);
    assert.equal(accepted("archai.example:443", listed), true);
    assert.equal(accepted("evil.example:3000", listed), false);
    assert.equal(accepted("archai.example:3000", listed), false);
    assert.equal(accepted("127.0.0.1:3000", listed), true);
  });

  it("refuses a Host that is not a host", () => {
    for (const host of ["", "evil.example/x", "a b", "user@127.0.0.1"]) {
      assert.equal(accepted(host), false, JSON.stringify(host));
    }
  });

  it("judges the forwarded host when told to trust it, and only then", () => {
    const proxied = { "x-forwarded-host": "archai.example" };
    assert.equal(accepted("127.0.0.1:3000", none, proxied), true); // not trusted: Host is what counts
    assert.equal(accepted("127.0.0.1:3000", { ...none, trustForwardedHost: true }, proxied), false);
    assert.equal(accepted("127.0.0.1:3000", { ...listed, trustForwardedHost: true }, proxied), true);
    assert.equal(accepted("127.0.0.1:3000", { ...listed, trustForwardedHost: true }, { "x-forwarded-host": "evil.example" }), false);
  });
});

describe("crossSiteRefusal with FRONTEND_ORIGINS", () => {
  const listed = { trustForwardedHost: false, allowedOrigins: ["https://archai.example"] };

  it("compares the whole origin, scheme included, for a listed host", () => {
    const request = (origin: string) => new Headers({ host: "archai.example", origin });
    assert.equal(crossSiteRefusal("POST", request("https://archai.example"), listed), null);
    assert.notEqual(crossSiteRefusal("POST", request("http://archai.example"), listed), null);
    assert.notEqual(crossSiteRefusal("POST", request("http://archai.example:443"), listed), null);
  });

  it("still lets the server's own address through", () => {
    const local = new Headers({ host: "127.0.0.1:3000", origin: "http://127.0.0.1:3000" });
    assert.equal(crossSiteRefusal("POST", local, listed), null);
  });
});

// Second security review (2026-11-24): the backend's redirects name its own
// address (location: http://127.0.0.1:8010/api/classes), and the proxy passed
// them to the browser. A location at the backend becomes a path on the
// frontend; one elsewhere is passed on as it is.
describe("downstreamHeaders and Location", () => {
  const backend = new URL("http://backend.internal:8010");
  const location = (value: string) => downstreamHeaders({ location: value }, backend).get("location");

  it("turns a location at the backend into a path on the frontend", () => {
    assert.equal(location("http://backend.internal:8010/api/classes"), "/api/classes");
    assert.equal(location("http://BACKEND.internal:8010/api/x?a=1#f"), "/api/x?a=1#f");
  });

  it("keeps a relative location and one on another host", () => {
    assert.equal(location("/api/classes"), "/api/classes");
    assert.equal(location("https://www.wikidata.org/wiki/Q807"), "https://www.wikidata.org/wiki/Q807");
  });

  it("drops a location at the backend outside its /api", () => {
    assert.equal(location("http://backend.internal:8010/docs"), null);
  });

  it("takes off the path BACKEND_URL puts the backend under", () => {
    const prefixed = new URL("http://backend.internal:8010/archai/");
    assert.equal(downstreamHeaders({ location: "http://backend.internal:8010/archai/api/classes" }, prefixed).get("location"), "/api/classes");
    assert.equal(downstreamHeaders({ location: "http://backend.internal:8010/api/classes" }, prefixed).get("location"), null);
  });
});
