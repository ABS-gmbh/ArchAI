// The open document kept across a reload (task T, 2026-11-24): the record is
// built, migrated, checked and restored by pure functions, tested here; the
// IndexedDB around them is workspaceStore.ts.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import type { FullPageLine, RunSnapshot } from "@/lib/api/ocrAgent";

import {
  blobToDataUrl,
  checkRecord,
  dataUrlToBlob,
  type PageReading,
  type PageStateMaps,
  readingsFromState,
  recordFromStorage,
  restoreWorkspace,
  serialiseWorkspace,
  stateFromReadings,
  WORKSPACE_SCHEMA_VERSION,
  type WorkspaceSnapshot,
} from "./workspacePersistence";

// A 1x1 PNG.
const PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const JPEG = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2Q==";

const line = (id: string, order: number, box: [number, number, number, number] | null): FullPageLine => ({
  region_id: id,
  lane: "body",
  reading_order: order,
  text: `text of ${id}`,
  char_start: null,
  char_end: null,
  bbox_xyxy: box,
  confidence: 0.9,
  backend_name: "kraken_catmus",
});

const steps: RunSnapshot = { run_id: "run-1", events: [{ stage: "INDEX_SKIPPED", event: "INFO", message: "Not indexed: no key." }] };

const reading: PageReading = {
  text: "Caldus de Lausanne\nsecond line",
  runId: "run-1",
  authorityReport: "=== report ===",
  readBy: "segmented",
  lines: [line("r1", 0, [10, 20, 300, 60]), line("r2", 1, [10, 70, 300, 110])],
  textLines: [
    { id: "r1", text: "Caldus de Lausanne", confidence: 0.91, fallbackBackend: null, onPage: true },
    { id: "r2", text: "second line", confidence: null, fallbackBackend: "kraken_mccatmus", onPage: true },
  ],
  marginLines: ["fol. 1r"],
  frame: { width: 1200, height: 1700 },
  badges: [{ tone: "ok", label: "Quality: high", detail: "The transcription passed every quality check." }],
  steps,
  names: { run_id: "run-1", mention_results: [{ surface: "Lausanne", status: "linked" }] },
};

const snapshot = (): WorkspaceSnapshot => ({
  document: {
    id: "doc-1",
    name: "Antitus (2 pages)",
    createdAt: 1764000000000,
    metadata: { language: "middle_french", year: "1500", placeOrOrigin: "", scriptFamily: "bastarda", documentType: "manuscript", notes: "" },
    pages: [
      { id: "page-1", name: "a.png", dataUrl: PNG, mimeType: "image/png" },
      { id: "page-2", name: "b.jpg", dataUrl: JPEG, mimeType: "image/jpeg" },
    ],
  },
  pageIndex: 1,
  zoom: 1.5,
  messages: [
    { id: "m1", role: "user", content: "Who is named?", createdAt: 1764000001000 },
    { id: "m2", role: "assistant", content: "Lausanne.", createdAt: 1764000002000, extractionBadges: reading.badges },
  ],
  readings: { "page-1": reading },
  regions: { "page-2": { "Main script black": [{ region_id: "z1", bbox_xyxy: [0, 0, 10, 10] }] } },
});

/** What IndexedDB hands back: a structured clone, here a JSON copy. */
const stored = (value: unknown) => JSON.parse(JSON.stringify(value)) as unknown;

describe("page images", () => {
  it("go to a Blob and back unchanged", async () => {
    for (const dataUrl of [PNG, JPEG]) {
      const blob = dataUrlToBlob(dataUrl);
      assert.ok(blob);
      assert.equal(await blobToDataUrl(blob), dataUrl);
    }
  });

  it("are refused when not a base64 data URL", () => {
    assert.equal(dataUrlToBlob("blob:http://localhost/123"), null);
    assert.equal(dataUrlToBlob("data:image/png,raw"), null);
  });
});

describe("a round trip", () => {
  it("gives back the same workspace", async () => {
    const built = serialiseWorkspace(snapshot(), 42);
    assert.ok(built);
    assert.deepEqual(Object.keys(built.blobs), ["page-1", "page-2"]);
    assert.equal(built.record.version, WORKSPACE_SCHEMA_VERSION);
    assert.equal(JSON.stringify(built.record).includes("base64"), false, "page images are kept apart, as Blobs");

    const checked = recordFromStorage(stored(built.record));
    assert.ok(checked.ok, checked.ok ? "" : checked.reason);
    const restored = await restoreWorkspace(checked.record, built.blobs);
    assert.deepEqual(restored, snapshot());
  });

  it("keeps nothing shared with the state it was built from", () => {
    const source = snapshot();
    const built = serialiseWorkspace(source, 1);
    assert.ok(built);
    source.readings["page-1"].textLines[0].text = "changed afterwards";
    assert.equal(built.record.readings["page-1"].textLines[0].text, "Caldus de Lausanne");
  });

  it("drops the state of pages that are not in the document", () => {
    const source = snapshot();
    source.readings["page-gone"] = reading;
    const built = serialiseWorkspace(source, 1);
    assert.deepEqual(Object.keys(built?.record.readings ?? {}), ["page-1"]);
  });
});

describe("a record that cannot be shown as it is", () => {
  const record = () => {
    const built = serialiseWorkspace(snapshot(), 1);
    assert.ok(built);
    return { record: stored(built.record) as Record<string, unknown>, blobs: built.blobs };
  };

  it("from a later version is discarded, not guessed at", () => {
    const { record: future } = record();
    future.version = WORKSPACE_SCHEMA_VERSION + 1;
    const checked = recordFromStorage(future);
    assert.equal(checked.ok, false);
    assert.match(checked.ok ? "" : checked.reason, /version 2 is not one this workspace reads/);
  });

  it("without a version, or not an object, is discarded", () => {
    const { record: unversioned } = record();
    delete unversioned.version;
    for (const raw of [unversioned, null, "text", [], 3]) {
      assert.equal(recordFromStorage(raw).ok, false);
    }
  });

  it("truncated anywhere is discarded whole", () => {
    const cuts: ((r: Record<string, unknown>) => void)[] = [
      (r) => delete r.document,
      (r) => delete r.messages,
      (r) => delete r.readings,
      (r) => ((r.document as { pages: unknown[] }).pages = []),
      (r) => delete ((r.readings as Record<string, Record<string, unknown>>)["page-1"]).lines,
      (r) => ((r.readings as Record<string, { lines: unknown[] }>)["page-1"].lines = [{ region_id: "r1" }]),
      (r) => ((r.messages as unknown[]).push({ id: "m3", role: "assistant" })),
      (r) => delete ((r.document as { metadata: Record<string, unknown> }).metadata).year,
    ];
    for (const cut of cuts) {
      const { record: truncated } = record();
      cut(truncated);
      assert.equal(recordFromStorage(truncated).ok, false, String(cut));
    }
  });

  it("whose rows have lost their boxes is discarded, so no row lights up the wrong place", () => {
    const { record: mismatched } = record();
    const page = (mismatched.readings as Record<string, { lines: FullPageLine[] }>)["page-1"];
    page.lines = page.lines.filter((l) => l.region_id !== "r2");
    const checked = recordFromStorage(mismatched);
    assert.equal(checked.ok, false);
    assert.match(checked.ok ? "" : checked.reason, /boxes are missing/);
  });

  it("with a reading of a page it does not have is discarded", () => {
    const { record: stray } = record();
    (stray.readings as Record<string, unknown>)["page-9"] = reading;
    assert.equal(recordFromStorage(stray).ok, false);
  });

  it("with a page whose image is missing or empty is not restored", async () => {
    const { record: raw, blobs } = record();
    const checked = checkRecord(raw);
    assert.ok(checked.ok);
    assert.equal(await restoreWorkspace(checked.record, { "page-1": blobs["page-1"] }), null);
    assert.equal(await restoreWorkspace(checked.record, { ...blobs, "page-2": new Blob([]) }), null);
  });

  it("brings the page and zoom into range rather than failing", () => {
    const { record: raw } = record();
    raw.pageIndex = 7;
    raw.zoom = 40;
    const checked = checkRecord(raw);
    assert.ok(checked.ok);
    assert.equal(checked.record.pageIndex, 1);
    assert.equal(checked.record.zoom, 1);
  });
});

describe("readingsFromState and back", () => {
  const maps = (): PageStateMaps => ({
    text: { "page-1": reading.text, "page-2": "half read" },
    runId: { "page-1": "run-1", "page-2": "run-2" },
    authorityReport: { "page-1": reading.authorityReport },
    readState: { "page-1": { phase: "done", readBy: "segmented" }, "page-2": { phase: "reading", readBy: null } },
    lines: { "page-1": reading.lines },
    textLines: { "page-1": reading.textLines },
    marginLines: { "page-1": reading.marginLines },
    frame: { "page-1": { width: 1200, height: 1700 } },
    badges: { "page-1": reading.badges },
    steps: { "page-1": steps },
    names: { "run-1": { phase: "ready", report: reading.names ?? undefined } },
  });

  it("keeps finished readings only, and gives them back as state", () => {
    const readings = readingsFromState(["page-1", "page-2"], maps());
    assert.deepEqual(Object.keys(readings), ["page-1"]);
    assert.deepEqual(readings["page-1"], reading);
    const back = stateFromReadings(readings);
    assert.deepEqual(back.readState, { "page-1": { phase: "done", readBy: "segmented" } });
    assert.deepEqual(back.names, { "run-1": { phase: "ready", report: reading.names } });
    assert.deepEqual(back.lines["page-1"], reading.lines);
  });

  it("leaves out steps of an earlier run and names not loaded yet", () => {
    const state = maps();
    state.steps["page-1"] = { ...steps, run_id: "run-0" };
    state.names = { "run-1": { phase: "loading" } };
    const readings = readingsFromState(["page-1"], state);
    assert.equal(readings["page-1"].steps, null);
    assert.equal(readings["page-1"].names, null);
  });
});

// The verdict arrives after the answer (2026-11-25). A message saved while its
// check is still running must not come back "checking" for ever after a reload.
describe("a message whose check had not come back", () => {
  it("is kept as not checked", () => {
    const source = snapshot();
    source.messages.push({ id: "m3", role: "assistant", content: "Lausanne.", createdAt: 1764000003000, check: "checking" });
    const built = serialiseWorkspace(source, 1);
    assert.ok(built);
    assert.equal(built.record.messages.at(-1)?.check, "not_checked");
    assert.ok(recordFromStorage(stored(built.record)).ok);
  });

  it("refuses a check state it does not know", () => {
    const built = serialiseWorkspace(snapshot(), 1);
    assert.ok(built);
    const raw = stored(built.record) as { messages: Record<string, unknown>[] };
    raw.messages[1].check = "maybe";
    assert.equal(recordFromStorage(raw).ok, false);
  });
});
