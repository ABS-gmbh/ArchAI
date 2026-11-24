// Why a name was not linked, in the Names panel's words, from the reasons the
// linker writes (app/services/entity_scoring.py, authority_linking.py).
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import {
  isName,
  KIND_WORDS,
  kindLabel,
  linkingSummary,
  mentionStatus,
  otherMarksSummary,
  reasonInWords,
  splitMentions,
  summaryOfMentions,
} from "./linkReason";

describe("reasonInWords", () => {
  it("keeps the reasons it already put in words", () => {
    assert.equal(
      reasonInWords("best score (0.682) < threshold (0.8) [ocr_quality=HIGH]"),
      "The best match scored 0.68; a link needs 0.80.",
    );
    assert.equal(
      reasonInWords("token_search_allowed=False quality=RISKY"),
      "Not matched: the page's reading was graded risky, too uncertain for names to be linked.",
    );
    assert.equal(reasonInWords("no candidates found"), "No authority record was found for this name.");
  });

  // Shown raw in a live run on 2026-11-24 (the Antitus page: "pere", "prince").
  it("says when every record found is the wrong kind for the name", () => {
    assert.equal(
      reasonInWords("all 5 candidate(s) type_incompatible (best score=0.476, qid=Q18219701)"),
      "None of the 5 records found is the right kind of record for this name.",
    );
    assert.equal(
      reasonInWords("all 1 candidate(s) type_incompatible (best score=0.279, qid=Q807)"),
      "The one record found is not the right kind of record for this name.",
    );
  });

  it("says when no record of the right kind has a close enough name", () => {
    assert.equal(
      reasonInWords(
        "no type-compatible candidate's name similarity reaches 0.6 (best score=0.491, qid=Q328804, similarity=0.593) [ocr_quality=HIGH]",
      ),
      "No record of the right kind has a name close enough: the closest is 0.59 alike, a link needs 0.60.",
    );
  });

  it("shows an unknown reason as written, without the quality tag, capitalised", () => {
    assert.equal(reasonInWords("something new happened [ocr_quality=OK]"), "Something new happened");
  });
});

// On the Antitus page the Steps said "6 names", then "2 linked, 3 not linked,
// 0 ambiguous", and the Names panel "2 of 6": the sixth, a date, is never looked
// up and no count said so (live regression, 2026-11-24). The step and the panel
// now say the same: "2 linked, 3 not linked, 1 not looked up".
describe("linkingSummary", () => {
  it("names every part that is not zero, linked and not linked always", () => {
    assert.equal(linkingSummary({ linked: 2, notLinked: 3, ambiguous: 0, skipped: 1 }), "2 linked, 3 not linked, 1 not looked up");
    assert.equal(linkingSummary({ linked: 2, notLinked: 5, ambiguous: 1, skipped: 0 }), "2 linked, 5 not linked, 1 ambiguous");
    assert.equal(linkingSummary({ linked: 0, notLinked: 0, ambiguous: 0, skipped: 2 }), "0 linked, 0 not linked, 2 not looked up");
  });

  it("counts a page's names the way the step does", () => {
    const statuses = ["linked", "unresolved", "unresolved", "unresolved", "linked", "skipped"];
    assert.deepEqual(summaryOfMentions(statuses.map((status) => ({ status }))), { linked: 2, notLinked: 3, ambiguous: 0, skipped: 1 });
    // The cautious reading's statuses are not linked too.
    assert.equal(summaryOfMentions([{ status: "unresolved_low_quality" }, { status: "ambiguous" }]).notLinked, 1);
  });
});

describe("why a name was not looked up", () => {
  it("says it in words", () => {
    assert.equal(reasonInWords("ent_type=date (dates never query Wikidata)"), "A date: dates are not looked up.");
    assert.equal(reasonInWords("ent_type=role (not linkable)"), "A role or title: these are not looked up.");
    assert.equal(reasonInWords("linking time budget of 60s spent"), "Not looked up: the time for linking this page (60 s) ran out.");
    assert.equal(
      reasonInWords("low_evidence_place: cap=True gazetteer=False conf=0.55 blacklisted=False lexical=True context=False shape=True"),
      "Not linked: too little on the page suggests this is a place.",
    );
    assert.equal(reasonInWords("name_likeness=0.120 < 0.35 (OCR garbage)"), "Not linked: this does not read like a name; it may be a misreading.");
  });

  it("labels each status as the summary counts it", () => {
    assert.equal(mentionStatus("skipped").label, "Not looked up");
    assert.equal(mentionStatus("unresolved").label, "Not linked");
    assert.equal(mentionStatus("unresolved_low_quality").label, "Not linked");
    assert.equal(mentionStatus("linked").label, "Linked");
    assert.equal(mentionStatus("ambiguous").label, "Ambiguous");
  });
});

// The backend's kinds, read from its sources: every literal it gives ent_type.
// (pipeline_hardening's _HIGH_RECALL_TRIGGERS also maps words to "context", but
// only its person and place triggers make mentions, so it is not scanned.)
function backendKinds(): Set<string> {
  const app = join(fileURLToPath(new URL(".", import.meta.url)), "../../../../backend/app");
  const files: string[] = [];
  const walk = (dir: string) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (name.endsWith(".py")) files.push(path);
    }
  };
  walk(app);
  const kinds = new Set<string>();
  const patterns = [
    /["']ent_type["']\]?\s*[:=]\s*["']([A-Za-z_]+)["']/g, // "ent_type": "x", mention["ent_type"] = "x"
    /\bent_type\s*=\s*["']([A-Za-z_]+)["']/g, // ent_type="X" (keyword), ent_type = "x"
    /\bent_type\s*=\s*["']([A-Za-z_]+)["']\s+if\b[^\n]*\belse\s+["']([A-Za-z_]+)["']/g, // ent_type = "a" if ... else "b"
    /get\(["']ent_type["'](?:,\s*["'][A-Za-z_]+["'])?\)\s*or\s*["']([A-Za-z_]+)["']/g, // m.get("ent_type") or "unknown"
    /get\(["']ent_type["'],\s*["']([A-Za-z_]+)["']\)/g, // m.get("ent_type", "unknown")
    /_LINKABLE_TYPES\s*=\s*\{([^}]*)\}/g,
  ];
  for (const file of files) {
    const text = readFileSync(file, "utf8");
    for (const pattern of patterns) {
      for (const match of text.matchAll(pattern)) {
        for (const group of match.slice(1)) {
          if (!group) continue;
          if (group.includes('"') || group.includes("'") || group.includes(":")) {
            // A set or dict body: its quoted kinds (values of a dict, members of a set).
            for (const item of group.matchAll(/:\s*["']([A-Za-z_]+)["']|^\s*["']([A-Za-z_]+)["']|,\s*["']([A-Za-z_]+)["']/g)) {
              const kind = item[1] ?? item[2] ?? item[3];
              if (kind && !group.includes(`"${kind}":`) && !group.includes(`'${kind}':`)) kinds.add(kind.toLowerCase());
            }
          } else {
            kinds.add(group.toLowerCase());
          }
        }
      }
    }
  }
  return kinds;
}

describe("the kinds of mention", () => {
  it("has words for every kind the backend sends", () => {
    const kinds = backendKinds();
    for (const expected of ["person", "place", "work", "date", "title", "role", "unknown", "grammatical_term", "verb_paradigm_example"]) {
      assert.ok(kinds.has(expected), `the scan no longer finds ${expected}: did the backend change how it sets ent_type?`);
    }
    const missing = [...kinds].filter((kind) => !KIND_WORDS[kind]);
    assert.deepEqual(missing, [], `kinds found: ${[...kinds].sort().join(", ")}`);
  });

  it("labels each row, and an unknown kind as a name", () => {
    assert.equal(kindLabel("title"), "Title");
    assert.equal(kindLabel("PLACE"), "Place");
    assert.equal(kindLabel("something_new"), "Name");
    assert.equal(kindLabel(null), "Name");
  });

  // The Antitus page as the live regression read it (2026-11-24).
  const antitus = [
    { surface: "La satyre megere.", ent_type: "work", status: "linked" },
    { surface: "pere", ent_type: "title", status: "unresolved" },
    { surface: "euesque", ent_type: "title", status: "unresolved" },
    { surface: "prince", ent_type: "title", status: "unresolved" },
    { surface: "lausanne", ent_type: "place", status: "linked" },
    { surface: "lan de grace Mille anq", ent_type: "date", status: "skipped" },
  ];

  it("counts names only, and the rest by what a reader calls them", () => {
    const { names, others } = splitMentions(antitus);
    assert.deepEqual(names.map((m) => m.surface), ["La satyre megere.", "lausanne"]);
    assert.equal(linkingSummary(summaryOfMentions(names)), "2 linked, 0 not linked");
    assert.equal(otherMarksSummary(others), "3 roles or titles, 1 date");
    assert.equal(otherMarksSummary([{ ent_type: "role" }]), "1 role or title");
    assert.equal(otherMarksSummary([{ ent_type: "date" }, { ent_type: "date" }]), "2 dates");
  });

  it("treats persons, places, works and unknown kinds as names", () => {
    for (const kind of ["person", "place", "work", "unknown", "", null, "organization"]) {
      assert.equal(isName(kind), true, String(kind));
    }
    for (const kind of ["title", "role", "date", "DATE"]) {
      assert.equal(isName(kind), false, kind);
    }
  });
});
