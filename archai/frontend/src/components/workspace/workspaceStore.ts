// The open document in this browser's IndexedDB, so it survives a reload, a
// closed tab reopened or a crash (task T, 2026-11-24). One document is kept:
// the one last open. The record (workspacePersistence.ts) and its page images
// (Blobs, keyed by page id) are written in one transaction when a document is
// opened; afterwards only the record is rewritten. Everything here fails soft:
// without IndexedDB, or when it refuses or is full, the workspace works as
// before and says once that the document will not survive a reload.

import {
  recordFromStorage,
  restoreWorkspace,
  serialiseWorkspace,
  type StoredWorkspace,
  type WorkspaceSnapshot,
} from "./workspacePersistence";

const DB_NAME = "archai-workspace";
const DB_VERSION = 1;
const RECORDS = "records";
const PAGES = "pages";
const RECORD_KEY = "open-document";

/** "kept": the document is in this browser's storage; "unavailable": storage is missing, refused or full. */
export type StoreStatus = "kept" | "unavailable";

let opening: Promise<IDBDatabase | null> | null = null;

function openDatabase(): Promise<IDBDatabase | null> {
  if (opening) {
    return opening;
  }
  const attempt = new Promise<IDBDatabase | null>((resolve) => {
    try {
      if (typeof indexedDB === "undefined") {
        resolve(null);
        return;
      }
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(RECORDS)) db.createObjectStore(RECORDS);
        if (!db.objectStoreNames.contains(PAGES)) db.createObjectStore(PAGES);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
      request.onblocked = () => resolve(null);
    } catch {
      // SecurityError where storage is refused (some private windows, blocked site data).
      resolve(null);
    }
  }).then((db) => {
    if (!db) {
      opening = null; // a later attempt may succeed (the user allowed storage)
    }
    return db;
  });
  opening = attempt;
  return attempt;
}

function finished(transaction: IDBTransaction): Promise<boolean> {
  return new Promise((resolve) => {
    transaction.oncomplete = () => resolve(true);
    transaction.onerror = () => resolve(false); // QuotaExceededError among them
    transaction.onabort = () => resolve(false);
  });
}

function read<T>(request: IDBRequest<T>): Promise<T | undefined> {
  return new Promise((resolve) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => resolve(undefined);
  });
}

/**
 * Keep *snapshot* as the open document. With *withPages* (a document was
 * opened, or another one selected), its page images replace the stored ones;
 * otherwise only the record is rewritten.
 */
export async function saveWorkspace(snapshot: WorkspaceSnapshot, withPages: boolean): Promise<StoreStatus> {
  const built = serialiseWorkspace(snapshot, Date.now());
  const db = built ? await openDatabase() : null;
  if (!built || !db) {
    return "unavailable";
  }
  try {
    const transaction = db.transaction([RECORDS, PAGES], "readwrite");
    const done = finished(transaction);
    if (withPages) {
      const pages = transaction.objectStore(PAGES);
      pages.clear();
      for (const [pageId, blob] of Object.entries(built.blobs)) {
        pages.put(blob, pageId);
      }
    }
    transaction.objectStore(RECORDS).put(built.record, RECORD_KEY);
    return (await done) ? "kept" : "unavailable";
  } catch {
    return "unavailable";
  }
}

/**
 * The document kept in this browser, or null: none kept, storage unavailable,
 * or a record that fails its check (discarded, with one console line).
 */
export async function loadWorkspace(): Promise<{ snapshot: WorkspaceSnapshot | null; status: StoreStatus }> {
  const db = await openDatabase();
  if (!db) {
    return { snapshot: null, status: "unavailable" };
  }
  let raw: unknown;
  const blobs: Record<string, Blob | undefined> = {};
  try {
    const transaction = db.transaction([RECORDS, PAGES], "readonly");
    raw = await read(transaction.objectStore(RECORDS).get(RECORD_KEY));
    if (raw === undefined) {
      return { snapshot: null, status: "kept" };
    }
    const checked = recordFromStorage(raw);
    if (!checked.ok) {
      return discard(`its record ${checked.reason}`);
    }
    // All at once, in the same turn as the record's answer: a transaction
    // with no request pending commits, and a later get would fail.
    const pages = transaction.objectStore(PAGES);
    const ids = (checked.record as StoredWorkspace).document.pages.map((page) => page.id);
    const found = await Promise.all(ids.map((id) => read(pages.get(id))));
    ids.forEach((id, index) => {
      const blob = found[index];
      blobs[id] = blob instanceof Blob ? blob : undefined;
    });
    const snapshot = await restoreWorkspace(checked.record, blobs);
    return snapshot ? { snapshot, status: "kept" } : discard("a page image is missing");
  } catch {
    return { snapshot: null, status: "unavailable" };
  }
}

async function discard(reason: string): Promise<{ snapshot: null; status: StoreStatus }> {
  console.info(`[workspace] The document kept in this browser was discarded: ${reason}.`);
  await clearWorkspace();
  return { snapshot: null, status: "kept" };
}

/** Remove the kept document and its page images ("Close document"). */
export async function clearWorkspace(): Promise<void> {
  const db = await openDatabase();
  if (!db) {
    return;
  }
  try {
    const transaction = db.transaction([RECORDS, PAGES], "readwrite");
    const done = finished(transaction);
    transaction.objectStore(RECORDS).clear();
    transaction.objectStore(PAGES).clear();
    await done;
  } catch {
    // Nothing more to do: a later save replaces whatever is there.
  }
}
