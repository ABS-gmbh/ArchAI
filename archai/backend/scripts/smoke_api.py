"""Smoke-test a running backend over real HTTP.

Usage::

    # in one shell: a backend on scratch data, no .env, nothing live
    ARCHAI_DB_PATH=/tmp/smoke/archai.sqlite CHROMA_PERSIST_DIR=/tmp/smoke/chroma \\
    ARCHAI_WIKIDATA_CACHE=/tmp/smoke/wikidata.sqlite OCR_EVIDENCE_PATH=/tmp/smoke/evidence.jsonl \\
    TASK_BASE_DIR=/tmp/smoke/tasks MAX_UPLOAD_MEGABYTES=2 GLMOCR_OLLAMA_HOST=http://127.0.0.1:9 \\
        uvicorn app.main:app --port 8765
    # in another
    python scripts/smoke_api.py --base http://127.0.0.1:8765 --db /tmp/smoke/archai.sqlite

The TestClient suite runs the app in-process; this exercises the real server:
uvicorn, the middleware stack, streaming responses, disconnects and
concurrency. It expects the backend above: no chat key, no Ollama, no layout
models, MAX_UPLOAD_MEGABYTES=2. Every check states the status it expects, and
the script exits 1 if any differs. --db (the backend's own scratch database)
lets it create a run that never finishes, for the stream checks; without it
those checks are skipped. Never point it at a live backend: it writes runs,
tasks and evidence spans.

Against a backend with API_TOKEN set, pass the same value as --token: every
request then carries it, and the authentication checks run as well (no token,
a wrong token, and the token as ?token= on the two event streams). Through the
frontend's /api proxy, which adds the token itself, leave --token out.
"""

from __future__ import annotations

import argparse
import base64
import io
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from PIL import Image


@dataclass
class Result:
    name: str
    expected: tuple[int, ...]
    status: int | None
    seconds: float
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in self.expected


def _png(size: tuple[int, int] = (60, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()
PNG_B64 = base64.b64encode(PNG).decode("ascii")
NOT_AN_IMAGE_B64 = base64.b64encode(b"%PDF-1.4 not an image").decode("ascii")


class Smoke:
    def __init__(self, base: str, db: str | None, token: str | None = None) -> None:
        self.base = base.rstrip("/")
        self.db = db
        self.token = token
        self.headers = {"X-API-Token": token} if token else {}
        self.client = httpx.Client(base_url=self.base, timeout=60, headers=self.headers)
        # Sends no token, for the checks that it is required.
        self.bare = httpx.Client(base_url=self.base, timeout=60)
        self.results: list[Result] = []
        self.batch_task_id: str | None = None

    def check(self, name: str, expected: int | tuple[int, ...], send: Callable[[], httpx.Response], note: str = "") -> httpx.Response | None:
        wanted = (expected,) if isinstance(expected, int) else expected
        started = time.perf_counter()
        try:
            response = send()
        except httpx.HTTPError as exc:
            self.results.append(Result(name, wanted, None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"))
            return None
        detail = ""
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                body = response.json()
                detail = str(body.get("detail", "")) if isinstance(body, dict) else ""
            except ValueError:
                pass
        self.results.append(Result(name, wanted, response.status_code, time.perf_counter() - started, note or detail[:90]))
        return response

    # ------------------------------------------------------------------ routes

    def basics(self) -> None:
        c = self.client
        self.check("GET /api/health", 200, lambda: c.get("/api/health"))
        self.check("GET /api/classes", 200, lambda: c.get("/api/classes"))
        self.check("GET unknown route", 404, lambda: c.get("/api/nope"))

    def predict(self) -> None:
        c = self.client
        self.check("predict/single non-image", 422, lambda: c.post("/api/predict/single", files={"image": ("p.png", b"not an image", "image/png")}))
        self.check("predict/single empty", 422, lambda: c.post("/api/predict/single", files={"image": ("p.png", b"", "image/png")}))
        self.check("predict/single confidence 1.5", 422, lambda: c.post("/api/predict/single", files={"image": ("p.png", PNG, "image/png")}, data={"confidence": "1.5"}))
        self.check("predict/single bad classes", 422, lambda: c.post("/api/predict/single", files={"image": ("p.png", PNG, "image/png")}, data={"classes": "{"}))
        self.check("predict/single over the limit", 413, lambda: c.post("/api/predict/single", files={"image": ("p.png", b"\0" * (3 << 20), "image/png")}))
        self.check(
            "predict/single malformed multipart",
            400,
            lambda: c.post("/api/predict/single", content=b"garbage", headers={"content-type": "multipart/form-data; boundary=xyz"}),
        )
        # No layout models in the smoke environment: a clear 500, not a traceback.
        self.check("predict/single valid page, no models", (200, 500), lambda: c.post("/api/predict/single", files={"image": ("p.png", PNG, "image/png")}))
        batch = self.check("predict/batch not a zip", 202, lambda: c.post("/api/predict/batch", files={"zip_file": ("b.zip", b"not a zip", "application/zip")}))
        if batch is not None and batch.status_code == 202:
            task_id = batch.json()["task_id"]
            self.batch_task_id = task_id
            self.check("predict/batch progress ends", 200, lambda: c.get(f"/api/predict/batch/{task_id}/progress", timeout=30))
            self.check("predict/batch results of failed task", 409, lambda: c.get(f"/api/predict/batch/{task_id}/results"))
        self.check("predict/batch empty", 422, lambda: c.post("/api/predict/batch", files={"zip_file": ("b.zip", b"", "application/zip")}))
        for path in ("coco_json", "annotated_image", "annotated/0", "results_zip"):
            self.check(f"download {path} unknown task", 404, lambda path=path: c.get(f"/api/download/abcdef123456/{path}"))
        self.check("download traversal", 404, lambda: c.get("/api/download/..%2F..%2Fetc/coco_json"))
        self.check("predict/batch progress unknown", 404, lambda: c.get("/api/predict/batch/abcdef123456/progress"))

    def ocr(self) -> None:
        c = self.client
        full = {"image_b64": PNG_B64, "apply_proofread": False}
        self.check("ocr/extract_full_page non-image", 422, lambda: c.post("/api/ocr/extract_full_page", json={**full, "image_b64": NOT_AN_IMAGE_B64}))
        self.check("ocr/extract_full_page bad base64", 422, lambda: c.post("/api/ocr/extract_full_page", json={**full, "image_b64": "!!!"}))
        self.check("ocr/extract_full_page glmocr, no Ollama", 502, lambda: c.post("/api/ocr/extract_full_page", json={**full, "ocr_backend": "glmocr"}))
        self.check("ocr/saia, no Ollama", 502, lambda: c.post("/api/ocr/saia", json=full))
        self.check("ocr/extract, no key", 422, lambda: c.post("/api/ocr/extract", json={"image_b64": PNG_B64}))
        self.check("agents/ocr, no key", 422, lambda: c.post("/api/agents/ocr", json={"image_b64": PNG_B64}))
        self.check("ocr/page_with_trace non-image", 422, lambda: c.post("/api/ocr/page_with_trace", json={**full, "image_b64": NOT_AN_IMAGE_B64}))
        self.check("ocr/page_with_trace glmocr, no key/Ollama", (200, 500), lambda: c.post("/api/ocr/page_with_trace", json={**full, "ocr_backend": "glmocr"}))
        big = "A" * (4 << 20)
        self.check("ocr body over the limit", 413, lambda: c.post("/api/ocr/extract_full_page", json={"image_b64": big}))
        self.check("ocr/trace unknown", 404, lambda: c.get("/api/ocr/trace/nope"))
        self.check("ocr/trace tables unknown", 404, lambda: c.get("/api/ocr/trace/nope/tables"))
        self.check("ocr/trace stream unknown", 404, lambda: c.get("/api/ocr/trace/stream/nope"))

    def chat_and_data(self) -> None:
        c = self.client
        messages = {"messages": [{"role": "user", "content": "hi"}]}
        self.check("chat/models, no key", 422, lambda: c.get("/api/chat/models"))
        self.check("chat/completions, no key", 422, lambda: c.post("/api/chat/completions", json={**messages, "stream": False}))
        self.check("chat/completions temperature 50", 422, lambda: c.post("/api/chat/completions", json={**messages, "stream": False, "temperature": 50}))
        streamed = self.check("chat/completions streamed, no key", 200, lambda: c.post("/api/chat/completions", json={**messages, "stream": True}))
        if streamed is not None and '"type": "error"' not in streamed.text:
            self.results[-1].note = "stream carried no error event"
            self.results[-1].expected = ()
        self.check("chat/label-analysis, no key", 422, lambda: c.post("/api/chat/label-analysis", json={"question": "q", "label_name": "MainZone", "image_b64": PNG_B64}))
        self.check("evidence/spans empty body", 422, lambda: c.post("/api/evidence/spans", json={}))
        self.check("evidence/spans list", 200, lambda: c.get("/api/evidence/spans/page-1"))
        self.check("index status unknown", 404, lambda: c.get("/api/index/run/nope/status"))
        self.check("index run unknown", 404, lambda: c.post("/api/index/run/nope"))
        self.check("index search", 200, lambda: c.post("/api/index/search", params={"query": "arthur"}))
        self.check("index search top_k -1", 422, lambda: c.post("/api/index/search", params={"query": "arthur", "top_k": -1}))
        self.check("index delete unknown", 200, lambda: c.delete("/api/index/run/nope"))
        self.check("authority link unknown", 404, lambda: c.post("/api/authority/link/nope"))
        self.check("authority report unknown", 404, lambda: c.get("/api/authority/report/nope"))
        self.check("analytics login, unconfigured", 503, lambda: c.post("/api/analytics/login", json={"username": "a", "password": "b"}))
        self.check("analytics data, no token", (401, 403), lambda: c.get("/api/analytics/data"))
        self.check("rag debug, disabled", 404, lambda: c.get("/api/rag/debug/retrieve", params={"query": "x"}))

    # ------------------------------------------------------------- streaming

    def _open_run(self) -> str | None:
        """A run that never finishes, written into the backend's scratch database."""
        if not self.db:
            return None
        run_id = f"smoke-{uuid.uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, created_at, updated_at, status, current_stage, asset_ref) VALUES (?, ?, ?, 'RUNNING', 'RECEIVED', 'smoke')",
                (run_id, now, now),
            )
        return run_id

    def streams(self) -> None:
        run_id = self._open_run()
        if run_id is None:
            self.results.append(Result("stream checks", (), None, 0, "skipped: pass --db"))
            return
        started = time.perf_counter()
        try:
            with self.client.stream("GET", f"/api/ocr/trace/stream/{run_id}") as response:
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        break  # disconnect after the first event
            self.results.append(Result("trace stream, client disconnects", (200,), response.status_code, time.perf_counter() - started))
        except httpx.HTTPError as exc:
            self.results.append(Result("trace stream, client disconnects", (200,), None, 0, str(exc)))

        # Twenty open streams polling the database, with /health asked meanwhile.
        stop = threading.Event()

        def hold() -> None:
            with httpx.Client(base_url=self.base, timeout=30, headers=self.headers) as client, client.stream("GET", f"/api/ocr/trace/stream/{run_id}") as response:
                for _ in response.iter_lines():
                    if stop.is_set():
                        return

        workers = [threading.Thread(target=hold, daemon=True) for _ in range(20)]
        for worker in workers:
            worker.start()
        time.sleep(1.5)
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            self.client.get("/api/health")
            latencies.append(time.perf_counter() - t0)
        stop.set()
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE pipeline_runs SET status='FAILED', updated_at=? WHERE run_id=?", (datetime.now(timezone.utc).isoformat(), run_id))
        for worker in workers:
            worker.join(timeout=10)
        worst = max(latencies)
        self.results.append(
            Result("health with 20 open streams", (200,), 200 if worst < 1.0 else 599, worst, f"worst {worst * 1000:.0f} ms of 10")
        )

    def auth(self) -> None:
        """API_TOKEN: required everywhere but /api/health, and accepted as ?token= on the two event streams."""
        if not self.token:
            return
        bare = self.bare
        self.check("health, no token", 200, lambda: bare.get("/api/health"))
        self.check("classes, no token", 401, lambda: bare.get("/api/classes"))
        self.check("classes, wrong token", 401, lambda: bare.get("/api/classes", headers={"X-API-Token": "wrong-token-of-the-right-length-0000"}))
        self.check("classes, token as Bearer", 200, lambda: bare.get("/api/classes", headers={"Authorization": f"Bearer {self.token}"}))
        self.check("classes, ?token= (streams only)", 401, lambda: bare.get("/api/classes", params={"token": self.token}))
        self.check("upload, no token", 401, lambda: bare.post("/api/predict/single", files={"image": ("p.png", PNG, "image/png")}))
        if self.batch_task_id:
            task_id = self.batch_task_id
            self.check("batch progress, ?token=", 200, lambda: bare.get(f"/api/predict/batch/{task_id}/progress", params={"token": self.token}, timeout=30))
            self.check("batch progress, no token", 401, lambda: bare.get(f"/api/predict/batch/{task_id}/progress"))
        run_id = self._open_run()
        if run_id is None:
            return
        self.check("trace stream, no token", 401, lambda: bare.get(f"/api/ocr/trace/stream/{run_id}"))
        started = time.perf_counter()
        try:
            with bare.stream("GET", f"/api/ocr/trace/stream/{run_id}", params={"token": self.token}) as response:
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        break
            self.results.append(Result("trace stream, ?token=", (200,), response.status_code, time.perf_counter() - started))
        except httpx.HTTPError as exc:
            self.results.append(Result("trace stream, ?token=", (200,), None, 0, str(exc)))
        with sqlite3.connect(self.db or "") as conn:
            conn.execute("UPDATE pipeline_runs SET status='FAILED', updated_at=? WHERE run_id=?", (datetime.now(timezone.utc).isoformat(), run_id))

    def report(self) -> int:
        width = max(len(r.name) for r in self.results)
        failed = 0
        for r in self.results:
            mark = "ok  " if r.ok or r.expected == () and r.status is None else "FAIL"
            if mark == "FAIL":
                failed += 1
            print(f"{mark} {r.name:{width}s} {str(r.status):>4s} want {'/'.join(map(str, r.expected)) or '-':7s} {r.seconds * 1000:7.0f} ms  {r.note}")
        print(f"\n{len(self.results) - failed} ok, {failed} failed")
        return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--db", help="the backend's scratch pipeline database, for the stream checks")
    parser.add_argument("--token", help="the backend's API_TOKEN, when it has one; leave out through the frontend proxy")
    args = parser.parse_args(argv)
    smoke = Smoke(args.base, args.db, args.token)
    smoke.basics()
    smoke.predict()
    smoke.ocr()
    smoke.chat_and_data()
    smoke.streams()
    smoke.auth()
    return smoke.report()


if __name__ == "__main__":
    sys.exit(main())
