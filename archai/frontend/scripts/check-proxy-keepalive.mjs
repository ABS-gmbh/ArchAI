// The /api proxy and an early answer from the backend (run after `npm run build`):
//
//   node scripts/check-proxy-keepalive.mjs
//
// Starts a stand-in backend that, like uvicorn behind the backend's body
// limit, answers 413 to a large POST before reading its body, and `next
// start` against it; then, on ONE keep-alive client connection:
//   1. a 4 MB POST, answered 413 early, then a GET: the GET must be answered
//      (it hung 6 s and the connection was reset: final live run, 2026-11-24);
//   2. a POST larger than the proxy drains, answered 413 early: the proxy must
//      close the connection rather than leave it hanging, so the client's next
//      request goes on a new connection and is answered;
//   3. an upload the client aborts half way: the backend request must be
//      cancelled, as before.
// Exits 1 if any check fails.
import { spawn } from "node:child_process";
import http from "node:http";
import net from "node:net";

const STUB_PORT = 8091;
const NEXT_PORT = 3061;
const MB = 1024 * 1024;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ── The stand-in backend ────────────────────────────────────────────────
// On raw sockets, as uvicorn behaves behind the backend's body limit: it reads
// a request's headers, answers a large POST with 413 at once, reads none of its
// body, and drops the connection a moment later. (Node's http server would
// read or reset on its own.)
const aborted = [];
const answer = (socket, status, detail) => {
  const body = JSON.stringify({ detail });
  socket.write(`HTTP/1.1 ${status} X\r\ncontent-type: application/json\r\ncontent-length: ${Buffer.byteLength(body)}\r\nconnection: close\r\n\r\n${body}`);
};
const stub = net.createServer((socket) => {
  socket.on("error", () => {});
  let head = Buffer.alloc(0);
  const onData = (chunk) => {
    head = Buffer.concat([head, chunk]);
    const end = head.indexOf("\r\n\r\n");
    if (end === -1) return;
    socket.off("data", onData);
    const [requestLine, ...lines] = head.subarray(0, end).toString().split("\r\n");
    const [method, path] = requestLine.split(" ");
    const length = Number(lines.find((l) => /^content-length:/i.test(l))?.split(":")[1] ?? 0);
    if (path.startsWith("/api/slow-upload")) {
      // Reads the body; notes when the request is cut off before its end.
      // Asks for it first when the proxy waits to be asked, as uvicorn does.
      if (lines.some((l) => /^expect:\s*100-continue/i.test(l))) socket.write("HTTP/1.1 100 Continue\r\n\r\n");
      let received = head.length - end - 4;
      socket.on("data", (more) => {
        received += more.length;
        if (received >= length) answer(socket, 200, "ok");
      });
      socket.on("close", () => {
        if (received < length) aborted.push(Date.now());
      });
      return;
    }
    if (method === "POST" && length > 2 * MB) {
      socket.pause();
      answer(socket, 413, "The request is larger than the server accepts (2 MB).");
      setTimeout(() => socket.destroy(), 200);
      return;
    }
    answer(socket, 404, "Run not found: nope");
    socket.end();
  };
  socket.on("data", onData);
});
await new Promise((resolve) => stub.listen(STUB_PORT, "127.0.0.1", resolve));

// ── next start against it ───────────────────────────────────────────────
const next = spawn("npx", ["next", "start", "-p", String(NEXT_PORT), "-H", "127.0.0.1"], {
  env: { ...process.env, BACKEND_URL: `http://127.0.0.1:${STUB_PORT}`, BACKEND_API_TOKEN: "" },
  stdio: process.env.CHECK_VERBOSE ? "inherit" : "ignore",
  detached: true, // its own process group: npx starts the server as a child, and both must stop
});
const stopNext = () => {
  try {
    process.kill(-next.pid, "SIGTERM");
  } catch {}
};
let up = false;
for (let i = 0; i < 60 && !up; i++) {
  await sleep(500);
  try {
    up = (await fetch(`http://127.0.0.1:${NEXT_PORT}/api/health`)).status > 0;
  } catch {}
}

// ── One keep-alive connection ───────────────────────────────────────────
const agent = new http.Agent({ keepAlive: true, maxSockets: 1 });
const sockets = new Set();

function request(method, path, body, { timeoutMs = 15000 } = {}) {
  return new Promise((resolve) => {
    const started = Date.now();
    const req = http.request(
      { host: "127.0.0.1", port: NEXT_PORT, method, path, agent, headers: body ? { "content-type": "application/json", "content-length": body.length } : {} },
      (res) => {
        res.resume();
        res.on("end", () => resolve({ status: res.statusCode, ms: Date.now() - started, closing: res.headers.connection === "close" }));
      },
    );
    req.on("socket", (socket) => {
      sockets.add(socket);
      socket.on("error", () => {}); // a write into a closed connection; the request's own error says it
    });
    req.setTimeout(timeoutMs, () => req.destroy(new Error(`no answer in ${timeoutMs} ms`)));
    req.on("error", (error) => resolve({ status: null, ms: Date.now() - started, error: error.message }));
    if (body) req.end(body);
    else req.end();
  });
}

const results = [];
const check = (name, ok, detail) => {
  results.push(ok);
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${JSON.stringify(detail)}`);
};

if (!up) {
  check("next start answers", false, "not up after 30 s");
} else {
  const body = (size) => Buffer.from(`{"image_b64":"${"A".repeat(size)}"}`);

  const big = await request("POST", "/api/ocr/extract_full_page", body(4 * MB));
  const then = await request("GET", "/api/ocr/trace/nope");
  check("4 MB POST answered 413 early", big.status === 413, big);
  check("then a GET on the same connection is answered", then.status === 404 && then.ms < 3000, then);

  const huge = await request("POST", "/api/ocr/extract_full_page", body(64 * MB));
  const after = await request("GET", "/api/ocr/trace/nope");
  check("64 MB POST answered 413 early", huge.status === 413, huge);
  check("then a GET is answered (on a new connection if the proxy closed the old one)", after.status === 404 && after.ms < 3000, after);
  check("connections used", sockets.size >= 1, { sockets: sockets.size });

  // An upload the client gives up on: the backend's request must be cut off too.
  const before = aborted.length;
  await new Promise((resolve) => {
    const req = http.request({ host: "127.0.0.1", port: NEXT_PORT, method: "POST", path: "/api/slow-upload", headers: { "content-type": "application/octet-stream", "content-length": 8 * MB } });
    req.on("error", () => resolve());
    req.on("socket", (socket) => socket.on("error", () => {}));
    let sent = 0;
    const pump = setInterval(() => {
      req.write(Buffer.alloc(256 * 1024));
      sent += 256 * 1024;
      if (sent >= 2 * MB) {
        clearInterval(pump);
        req.destroy();
        resolve();
      }
    }, 50);
  });
  await sleep(2000);
  check("an aborted upload cancels the backend request", aborted.length > before, { cancelled: aborted.length - before });
}

agent.destroy();
stopNext();
stub.close();
process.exit(results.every(Boolean) ? 0 : 1);
