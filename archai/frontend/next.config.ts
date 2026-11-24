import type { NextConfig } from "next";

// /api/* reaches the backend through the route handler in
// src/app/api/[...path]/route.ts, which reads BACKEND_URL and
// BACKEND_API_TOKEN when the server runs, not when it is built.

// A NEXT_PUBLIC_ variable is inlined into the JavaScript sent to every
// browser, so a token under that name would be published. Refuse to build or
// start with one set.
for (const name of ["NEXT_PUBLIC_BACKEND_API_TOKEN", "NEXT_PUBLIC_API_TOKEN"]) {
  if (process.env[name]) {
    throw new Error(
      `${name} is set. NEXT_PUBLIC_ variables are bundled into the browser's JavaScript, which would publish the token. ` +
        "Unset it and set BACKEND_API_TOKEN instead: the frontend's server adds it to the requests it forwards.",
    );
  }
}

// Where the browser may call the backend directly (src/lib/api/client.ts,
// apiUrl): NEXT_PUBLIC_BACKEND_URL when set, and 127.0.0.1:8010 when the page
// is opened on localhost. Everything else goes through the same-origin /api.
function directBackendOrigins(): string[] {
  const origins = new Set(["http://127.0.0.1:8010"]);
  const configured = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (configured) {
    try {
      origins.add(new URL(configured).origin);
    } catch {
      // Not a URL: client.ts would not reach it either.
    }
  }
  return [...origins];
}

// Production only: `next dev` needs eval and a websocket for hot reload.
// Scripts and styles keep 'unsafe-inline': Next's App Router writes inline
// bootstrap scripts without a nonce on statically rendered pages, and React
// style props are inline styles. What the policy does hold: no script, style,
// font or frame from another origin, no plugins, no <base> or form target
// elsewhere, no framing, and requests only to this server and the backend.
// Page images are data: URLs, and pageToFile fetches them (connect-src data:).
const contentSecurityPolicy = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self'",
  `connect-src 'self' data: blob: ${directBackendOrigins().join(" ")}`,
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

// Browser features the workspace does not use, turned off for the page and anything it loads.
const permissionsPolicy = [
  "accelerometer=()",
  "camera=()",
  "display-capture=()",
  "geolocation=()",
  "gyroscope=()",
  "hid=()",
  "magnetometer=()",
  "microphone=()",
  "midi=()",
  "payment=()",
  "serial=()",
  "usb=()",
].join(", ");

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "same-origin" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Permissions-Policy", value: permissionsPolicy },
  ...(process.env.NODE_ENV === "production" ? [{ key: "Content-Security-Policy", value: contentSecurityPolicy }] : []),
];

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
