// The open document as it is kept in this browser (workspaceStore.ts writes
// it to IndexedDB), so a reload, a closed tab reopened or a browser crash does
// not cost the reader the document and its readings. Pure functions only, so
// the unit tests cover them (workspacePersistence.test.ts): building the
// stored record from the workspace's state, migrating a record written by
// another version, validating one read back, and turning it into state again.
//
// What is kept: the document (its pages as Blobs, stored apart from the
// record), each page's finished reading (transcription lines with their boxes
// and frame, the margin lines, quality badges, recogniser, run id, steps,
// names report once loaded, the authority report), the layout regions found,
// the conversation, the zoom and the current page. What is not: a read or a
// question still running (nothing is saved while one is), a failed read's
// message, and the annotated layout image, which is a link to the server's
// task files that expire (the regions themselves are kept and drawn).

import type { LabelRegionPayload } from "@/lib/api/chat";
import type { FullPageLine, LinkingReport, RunSnapshot } from "@/lib/api/ocrAgent";
import type { WorkspaceChatMessage, WorkspaceDocument, WorkspaceDocumentMetadata, WorkspacePage } from "@/lib/workspace/types";

import { isSourceList } from "./citations";
import type { ExtractionBadge } from "./workspaceExtraction";

/** Raised by one whenever the stored record's shape changes; migrateRecord says what an older one becomes. */
export const WORKSPACE_SCHEMA_VERSION = 1;

export interface StoredTranscriptionLine {
  id: string;
  text: string;
  confidence: number | null;
  fallbackBackend: string | null;
  onPage: boolean;
}

/** One page's finished reading. */
export interface PageReading {
  /** The transcription as chat, search and the names work from it. */
  text: string;
  runId: string;
  authorityReport: string;
  /** The recogniser that read the page, as the read state shows it. */
  readBy: string | null;
  /** The lines with their boxes, in the frame below. */
  lines: FullPageLine[];
  textLines: StoredTranscriptionLine[];
  marginLines: string[];
  /** The page size the boxes are in; null when the engine gave none. */
  frame: { width: number; height: number } | null;
  /** Quality verdict, confidence, fallback lines, warnings. */
  badges: ExtractionBadge[];
  /** The run's steps as last followed. */
  steps: RunSnapshot | null;
  /** The names report, once it was loaded; null until then. */
  names: LinkingReport | null;
}

/** The workspace's state for one document, as the component holds it (pages as data URLs). */
export interface WorkspaceSnapshot {
  document: WorkspaceDocument;
  pageIndex: number;
  zoom: number;
  messages: WorkspaceChatMessage[];
  readings: Record<string, PageReading>;
  regions: Record<string, Record<string, LabelRegionPayload[]>>;
}

/** The record kept in IndexedDB; the page images are kept beside it, keyed by page id. */
export interface StoredWorkspace {
  version: typeof WORKSPACE_SCHEMA_VERSION;
  savedAt: number;
  document: {
    id: string;
    name: string;
    createdAt: number;
    metadata: WorkspaceDocumentMetadata;
    pages: { id: string; name: string; mimeType: string }[];
  };
  pageIndex: number;
  zoom: number;
  messages: WorkspaceChatMessage[];
  readings: Record<string, PageReading>;
  regions: Record<string, Record<string, LabelRegionPayload[]>>;
}

// ── Page images ─────────────────────────────────────────────────────────

/** A data URL's bytes as a Blob of its type; null when it is not a base64 data URL. */
export function dataUrlToBlob(dataUrl: string): Blob | null {
  const match = /^data:([^;,]*)(?:;[^,]*)?;base64,([^]*)$/.exec(dataUrl);
  if (!match) {
    return null;
  }
  try {
    const binary = atob(match[2]);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Blob([bytes], { type: match[1] || "application/octet-stream" });
  } catch {
    return null;
  }
}

/** A Blob as a base64 data URL of its type (the form the workspace shows and sends pages in). */
export async function blobToDataUrl(blob: Blob, type = blob.type): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let start = 0; start < bytes.length; start += chunk) {
    binary += String.fromCharCode(...bytes.subarray(start, start + chunk));
  }
  return `data:${type || "application/octet-stream"};base64,${btoa(binary)}`;
}

// ── Building the record ─────────────────────────────────────────────────

/**
 * The record to store for *snapshot*, and its page images as Blobs. The record
 * is copied through JSON, so it holds plain data only and nothing shared with
 * the component's state. Null when a page image is not a data URL.
 */
export function serialiseWorkspace(
  snapshot: WorkspaceSnapshot,
  savedAt: number,
): { record: StoredWorkspace; blobs: Record<string, Blob> } | null {
  const blobs: Record<string, Blob> = {};
  for (const page of snapshot.document.pages) {
    const blob = dataUrlToBlob(page.dataUrl);
    if (!blob) {
      return null;
    }
    blobs[page.id] = blob;
  }
  const pageIds = new Set(snapshot.document.pages.map((page) => page.id));
  const onPages = <T>(map: Record<string, T>) => Object.fromEntries(Object.entries(map).filter(([id]) => pageIds.has(id)));
  const record: StoredWorkspace = JSON.parse(
    JSON.stringify({
      version: WORKSPACE_SCHEMA_VERSION,
      savedAt,
      document: {
        id: snapshot.document.id,
        name: snapshot.document.name,
        createdAt: snapshot.document.createdAt,
        metadata: snapshot.document.metadata,
        pages: snapshot.document.pages.map((page) => ({ id: page.id, name: page.name, mimeType: page.mimeType })),
      },
      pageIndex: snapshot.pageIndex,
      zoom: snapshot.zoom,
      // A check still running when the page goes will not come back to this page.
      messages: snapshot.messages.map((message) => (message.check === "checking" ? { ...message, check: "not_checked" } : message)),
      readings: onPages(snapshot.readings),
      regions: onPages(snapshot.regions),
    }),
  );
  return { record, blobs };
}

// ── Reading a record back ───────────────────────────────────────────────

/**
 * *raw* as a record of this version, or null: a record from a later version
 * (written by a newer workspace in this browser) is not guessed at, and one
 * without a version is not ours. Version 1 is the first; when the record
 * changes, the step from each older version goes here.
 */
export function migrateRecord(raw: unknown): unknown | null {
  if (!isObject(raw) || typeof raw.version !== "number") {
    return null;
  }
  if (raw.version === WORKSPACE_SCHEMA_VERSION) {
    return raw;
  }
  return null;
}

export type RecordCheck = { ok: true; record: StoredWorkspace } | { ok: false; reason: string };

/**
 * Whether *raw* (already migrated) is a record the workspace can show as it
 * is. Anything that does not fit fails the whole record: a document shown with
 * part of its state would be read as complete. The page index and zoom are
 * brought into range rather than failing.
 */
export function checkRecord(raw: unknown): RecordCheck {
  const fail = (reason: string): RecordCheck => ({ ok: false, reason });
  if (!isObject(raw) || raw.version !== WORKSPACE_SCHEMA_VERSION) {
    return fail("not a record of this version");
  }
  const document = raw.document;
  if (!isObject(document) || !isString(document.id) || !isString(document.name) || !isNumber(document.createdAt)) {
    return fail("the document is incomplete");
  }
  if (!isMetadata(document.metadata)) {
    return fail("the document's metadata is incomplete");
  }
  const pages = document.pages;
  if (!Array.isArray(pages) || !pages.length || !pages.every((page) => isObject(page) && isString(page.id) && isString(page.name) && isString(page.mimeType))) {
    return fail("the pages are incomplete");
  }
  const pageIds = new Set(pages.map((page) => (page as { id: string }).id));
  if (pageIds.size !== pages.length) {
    return fail("two pages share an id");
  }
  if (!Array.isArray(raw.messages) || !raw.messages.every(isMessage)) {
    return fail("the conversation is incomplete");
  }
  if (!isObject(raw.readings)) {
    return fail("the readings are missing");
  }
  for (const [pageId, reading] of Object.entries(raw.readings)) {
    if (!pageIds.has(pageId)) {
      return fail("a reading belongs to no page");
    }
    const problem = readingProblem(reading);
    if (problem) {
      return fail(`a reading ${problem}`);
    }
  }
  if (!isObject(raw.regions) || !Object.entries(raw.regions).every(([pageId, map]) => pageIds.has(pageId) && isRegionMap(map))) {
    return fail("the layout regions are incomplete");
  }
  const pageIndex = isNumber(raw.pageIndex) ? Math.min(Math.max(0, Math.trunc(raw.pageIndex)), pages.length - 1) : 0;
  const zoom = isNumber(raw.zoom) && raw.zoom >= 0.25 && raw.zoom <= 4 ? raw.zoom : 1;
  return {
    ok: true,
    record: { ...(raw as unknown as StoredWorkspace), pageIndex, zoom, savedAt: isNumber(raw.savedAt) ? raw.savedAt : 0 },
  };
}

/**
 * The workspace's state from a checked record and its page images. Null when
 * a page's image is missing or empty: the document cannot be shown whole.
 */
export async function restoreWorkspace(
  record: StoredWorkspace,
  blobs: Record<string, Blob | undefined>,
): Promise<WorkspaceSnapshot | null> {
  const pages: WorkspacePage[] = [];
  for (const page of record.document.pages) {
    const blob = blobs[page.id];
    if (!blob || !blob.size) {
      return null;
    }
    pages.push({ id: page.id, name: page.name, mimeType: page.mimeType, dataUrl: await blobToDataUrl(blob, page.mimeType) });
  }
  return {
    document: {
      id: record.document.id,
      name: record.document.name,
      createdAt: record.document.createdAt,
      metadata: record.document.metadata,
      pages,
    },
    pageIndex: record.pageIndex,
    zoom: record.zoom,
    messages: record.messages,
    readings: record.readings,
    regions: record.regions,
  };
}

/** migrateRecord, then checkRecord: what the store does with whatever it reads back. */
export function recordFromStorage(raw: unknown): RecordCheck {
  const migrated = migrateRecord(raw);
  if (migrated === null) {
    return { ok: false, reason: isObject(raw) && isNumber(raw.version) ? `version ${raw.version} is not one this workspace reads` : "not a workspace record" };
  }
  return checkRecord(migrated);
}

// ── Shapes ──────────────────────────────────────────────────────────────

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNullableNumber(value: unknown): boolean {
  return value === null || isNumber(value);
}

function isNullableString(value: unknown): boolean {
  return value === null || isString(value);
}

function isBox(value: unknown): boolean {
  return Array.isArray(value) && value.length === 4 && value.every(isNumber);
}

function isMetadata(value: unknown): value is WorkspaceDocumentMetadata {
  return (
    isObject(value) &&
    ["language", "year", "placeOrOrigin", "scriptFamily", "documentType", "notes"].every((key) => isString(value[key]))
  );
}

function isMessage(value: unknown): boolean {
  return (
    isObject(value) &&
    isString(value.id) &&
    (value.role === "user" || value.role === "assistant" || value.role === "system") &&
    isString(value.content) &&
    isNumber(value.createdAt) &&
    (value.imageUrl === undefined || isString(value.imageUrl)) &&
    (value.check === undefined || value.check === "checking" || value.check === "not_checked") &&
    (value.verification === undefined ||
      value.verification === null ||
      (isObject(value.verification) &&
        isString(value.verification.assessment) &&
        isString(value.verification.corrected_answer) &&
        Array.isArray(value.verification.notes) &&
        value.verification.notes.every(isString) &&
        (value.verification.self_check === undefined || typeof value.verification.self_check === "boolean"))) &&
    (value.sources === undefined || isSourceList(value.sources)) &&
    (value.extractionBadges === undefined || (Array.isArray(value.extractionBadges) && value.extractionBadges.every(isBadge)))
  );
}

function isBadge(value: unknown): boolean {
  return (
    isObject(value) &&
    (value.tone === "ok" || value.tone === "warn" || value.tone === "error" || value.tone === "neutral") &&
    isString(value.label) &&
    isString(value.detail)
  );
}

function isLine(value: unknown): boolean {
  return (
    isObject(value) &&
    isString(value.region_id) &&
    (value.lane === "body" || value.lane === "secondary") &&
    isNumber(value.reading_order) &&
    isString(value.text) &&
    (value.bbox_xyxy === null || isBox(value.bbox_xyxy)) &&
    isNullableNumber(value.confidence)
  );
}

function isTextLine(value: unknown): boolean {
  return (
    isObject(value) &&
    isString(value.id) &&
    isString(value.text) &&
    isNullableNumber(value.confidence) &&
    isNullableString(value.fallbackBackend) &&
    typeof value.onPage === "boolean"
  );
}

function isRegionMap(value: unknown): boolean {
  return (
    isObject(value) &&
    Object.values(value).every(
      (regions) => Array.isArray(regions) && regions.every((region) => isObject(region) && isString(region.region_id) && isBox(region.bbox_xyxy)),
    )
  );
}

/** Why *value* is not a page reading the workspace can show, or null. */
function readingProblem(value: unknown): string | null {
  if (!isObject(value)) {
    return "is not an object";
  }
  if (!isString(value.text) || !isString(value.runId) || !isString(value.authorityReport) || !isNullableString(value.readBy)) {
    return "has no text or run";
  }
  if (!Array.isArray(value.lines) || !value.lines.every(isLine)) {
    return "has a line without its box";
  }
  if (!Array.isArray(value.textLines) || !value.textLines.every(isTextLine)) {
    return "has a transcription line that is incomplete";
  }
  // Each row the viewer can light up must have its box among the lines.
  const boxed = new Set((value.lines as FullPageLine[]).filter((line) => line.bbox_xyxy).map((line) => line.region_id));
  if ((value.textLines as StoredTranscriptionLine[]).some((line) => line.onPage && !boxed.has(line.id))) {
    return "has rows whose boxes are missing";
  }
  if (!Array.isArray(value.marginLines) || !value.marginLines.every(isString)) {
    return "has margin lines that are not text";
  }
  if (value.frame !== null && !(isObject(value.frame) && isNumber(value.frame.width) && isNumber(value.frame.height))) {
    return "has a frame without its size";
  }
  if (!Array.isArray(value.badges) || !value.badges.every(isBadge)) {
    return "has an incomplete quality badge";
  }
  // Steps and names are taken for the reading's own run when the record is
  // built (readingsFromState), so only their shape is checked here.
  if (value.steps !== null && !(isObject(value.steps) && (value.steps.events === undefined || Array.isArray(value.steps.events)))) {
    return "has steps that are not a list";
  }
  if (value.names !== null && !(isObject(value.names) && (value.names.mention_results === undefined || Array.isArray(value.names.mention_results)))) {
    return "has names that are not a list";
  }
  return null;
}

// ── Between the component's state and the readings ──────────────────────

/** The component's per-page state, as DocumentChatWorkspace holds it (keyed by page id, names by run id). */
export interface PageStateMaps {
  text: Record<string, string>;
  runId: Record<string, string>;
  authorityReport: Record<string, string>;
  readState: Record<string, { phase: string; readBy: string | null }>;
  lines: Record<string, FullPageLine[]>;
  textLines: Record<string, StoredTranscriptionLine[]>;
  marginLines: Record<string, string[]>;
  frame: Record<string, { width: number; height: number }>;
  badges: Record<string, ExtractionBadge[]>;
  steps: Record<string, RunSnapshot>;
  names: Record<string, { phase: string; report?: LinkingReport }>;
}

/**
 * The finished readings of *pageIds*: a page counts once its read is done.
 * Steps are kept only when they are of the page's current run, and names only
 * once their report has loaded, so nothing of an earlier read rides along.
 */
export function readingsFromState(pageIds: string[], maps: PageStateMaps): Record<string, PageReading> {
  const readings: Record<string, PageReading> = {};
  for (const pageId of pageIds) {
    if (maps.readState[pageId]?.phase !== "done") {
      continue;
    }
    const runId = maps.runId[pageId] ?? "";
    const steps = maps.steps[pageId];
    const names = runId ? maps.names[runId] : undefined;
    readings[pageId] = {
      text: maps.text[pageId] ?? "",
      runId,
      authorityReport: maps.authorityReport[pageId] ?? "",
      readBy: maps.readState[pageId]?.readBy ?? null,
      lines: maps.lines[pageId] ?? [],
      textLines: maps.textLines[pageId] ?? [],
      marginLines: maps.marginLines[pageId] ?? [],
      frame: maps.frame[pageId] ?? null,
      badges: maps.badges[pageId] ?? [],
      steps: steps && runId && steps.run_id === runId ? steps : null,
      names: names?.phase === "ready" && names.report ? names.report : null,
    };
  }
  return readings;
}

/** The component's per-page state from stored readings: the inverse of readingsFromState. */
export function stateFromReadings(readings: Record<string, PageReading>): PageStateMaps {
  const maps: PageStateMaps = {
    text: {},
    runId: {},
    authorityReport: {},
    readState: {},
    lines: {},
    textLines: {},
    marginLines: {},
    frame: {},
    badges: {},
    steps: {},
    names: {},
  };
  for (const [pageId, reading] of Object.entries(readings)) {
    maps.text[pageId] = reading.text;
    maps.runId[pageId] = reading.runId;
    maps.authorityReport[pageId] = reading.authorityReport;
    maps.readState[pageId] = { phase: "done", readBy: reading.readBy };
    maps.lines[pageId] = reading.lines;
    maps.textLines[pageId] = reading.textLines;
    maps.marginLines[pageId] = reading.marginLines;
    if (reading.frame) maps.frame[pageId] = reading.frame;
    maps.badges[pageId] = reading.badges;
    if (reading.steps) maps.steps[pageId] = reading.steps;
    if (reading.names && reading.runId) maps.names[reading.runId] = { phase: "ready", report: reading.names };
  }
  return maps;
}
