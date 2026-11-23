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

const nextConfig: NextConfig = {};

export default nextConfig;
