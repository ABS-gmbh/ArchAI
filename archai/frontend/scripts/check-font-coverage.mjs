#!/usr/bin/env node
// Which characters a recogniser can output that no transcription font can draw.
//
// The transcription face (`font-text`) is Gentium Plus, then Junicode for what
// Gentium lacks (src/app/layout.tsx). A character neither has falls to a
// system font, or to an empty box when none has it. This reads the character
// maps of the bundled WOFF2 files and checks them against the character sets
// of the Kraken recognisers (scripts/fixtures/recogniser-charsets.json).
//
// No dependencies: WOFF2 compresses its tables with Brotli, which Node ships,
// and the cmap table is never transformed inside WOFF2.
//
//   node scripts/check-font-coverage.mjs
//
// Exits 1 when a character of a medieval recogniser is missing from both
// fonts, other than the known gaps listed below.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { brotliDecompressSync } from "node:zlib";

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");
const FONTS = [
  ["Gentium Plus", "src/fonts/gentium-plus/GentiumPlus-Regular.woff2"],
  ["Junicode", "src/fonts/junicode/JunicodeVF-Roman.woff2"],
];
// The recognisers the backend reads medieval pages with; catmus_print is not routed to.
const MEDIEVAL = ["catmus_medieval", "cremma_medieval", "mccatmus"];
// Characters checked and found in no font this project can ship under the OFL.
const KNOWN_GAPS = new Map([[0xf2da, "MUFI private-use code point; in neither Gentium Plus nor Junicode 2.226"]]);

// WOFF2 known-table tags, by index (WOFF2 spec, 5.1).
const KNOWN_TAGS = (
  "cmap head hhea hmtx maxp name OS/2 post cvt  fpgm glyf loca prep CFF  VORG EBDT EBLC gasp hdmx kern LTSH PCLT VDMX " +
  "vhea vmtx BASE GDEF GPOS GSUB EBSC JSTF MATH CBDT CBLC COLR CPAL SVG  sbix acnt avar bdat bloc bsln cvar fdsc feat " +
  "fmtx fvar gvar hsty just lcar mort morx opbd prop trak Zapf Silf Glat Gloc Feat Sill"
)
  .match(/.{4}\s?/g)
  .map((tag) => tag.slice(0, 4));

function readBase128(buffer, offset) {
  let value = 0;
  for (let i = 0; i < 5; i += 1) {
    const byte = buffer[offset + i];
    value = value * 128 + (byte & 0x7f);
    if (!(byte & 0x80)) {
      return [value, offset + i + 1];
    }
  }
  throw new Error("bad UIntBase128");
}

/** The cmap table of a WOFF2 file, as stored (cmap is never transformed). */
function woff2Cmap(path) {
  const file = readFileSync(path);
  if (file.toString("latin1", 0, 4) !== "wOF2") {
    throw new Error(`${path} is not WOFF2`);
  }
  const numTables = file.readUInt16BE(12);
  const compressedSize = file.readUInt32BE(20);
  let offset = 48;
  const tables = [];
  for (let i = 0; i < numTables; i += 1) {
    const flags = file[offset];
    offset += 1;
    let tag = KNOWN_TAGS[flags & 0x3f];
    if ((flags & 0x3f) === 63) {
      tag = file.toString("latin1", offset, offset + 4);
      offset += 4;
    }
    const version = flags >> 6;
    let length;
    [length, offset] = readBase128(file, offset);
    // glyf and loca are transformed at version 0, every other table at a non-zero version.
    const transformed = tag === "glyf" || tag === "loca" ? version === 0 : version !== 0;
    if (transformed) {
      [length, offset] = readBase128(file, offset);
    }
    tables.push({ tag, length });
  }
  if (file.toString("latin1", 4, 8) === "ttcf") {
    throw new Error(`${path} is a font collection, which this script does not read`);
  }
  const data = brotliDecompressSync(file.subarray(offset, offset + compressedSize));
  let position = 0;
  for (const table of tables) {
    if (table.tag === "cmap") {
      return data.subarray(position, position + table.length);
    }
    position += table.length;
  }
  throw new Error(`${path} has no cmap table`);
}

/** The code points a cmap maps to a glyph, from its best Unicode subtable (format 12, else format 4). */
function codePoints(cmap) {
  const count = cmap.readUInt16BE(2);
  const subtables = [];
  for (let i = 0; i < count; i += 1) {
    const platform = cmap.readUInt16BE(4 + i * 8);
    const encoding = cmap.readUInt16BE(6 + i * 8);
    const offset = cmap.readUInt32BE(8 + i * 8);
    subtables.push({ platform, encoding, offset, format: cmap.readUInt16BE(offset) });
  }
  const unicode = (s) => s.platform === 0 || (s.platform === 3 && (s.encoding === 1 || s.encoding === 10));
  const best = subtables.find((s) => unicode(s) && s.format === 12) ?? subtables.find((s) => unicode(s) && s.format === 4);
  if (!best) {
    throw new Error("no Unicode cmap subtable of format 12 or 4");
  }
  const mapped = new Set();
  const o = best.offset;
  if (best.format === 12) {
    const groups = cmap.readUInt32BE(o + 12);
    for (let g = 0; g < groups; g += 1) {
      const start = cmap.readUInt32BE(o + 16 + g * 12);
      const end = cmap.readUInt32BE(o + 20 + g * 12);
      const glyph = cmap.readUInt32BE(o + 24 + g * 12);
      for (let cp = start; cp <= end; cp += 1) {
        if (glyph + (cp - start) !== 0) {
          mapped.add(cp);
        }
      }
    }
    return mapped;
  }
  const segments = cmap.readUInt16BE(o + 6) / 2;
  const ends = o + 14;
  const starts = ends + segments * 2 + 2;
  const deltas = starts + segments * 2;
  const rangeOffsets = deltas + segments * 2;
  for (let s = 0; s < segments; s += 1) {
    const end = cmap.readUInt16BE(ends + s * 2);
    const start = cmap.readUInt16BE(starts + s * 2);
    const delta = cmap.readInt16BE(deltas + s * 2);
    const rangeOffset = cmap.readUInt16BE(rangeOffsets + s * 2);
    for (let cp = start; cp <= end && cp !== 0xffff; cp += 1) {
      let glyph;
      if (rangeOffset === 0) {
        glyph = (cp + delta) & 0xffff;
      } else {
        const at = rangeOffsets + s * 2 + rangeOffset + (cp - start) * 2;
        glyph = cmap.readUInt16BE(at);
        if (glyph !== 0) {
          glyph = (glyph + delta) & 0xffff;
        }
      }
      if (glyph !== 0) {
        mapped.add(cp);
      }
    }
  }
  return mapped;
}

const hex = (cp) => `U+${cp.toString(16).toUpperCase().padStart(4, "0")}`;

const fonts = FONTS.map(([name, path]) => ({ name, cps: codePoints(woff2Cmap(join(FRONTEND, path))) }));
const charsets = JSON.parse(readFileSync(join(FRONTEND, "scripts/fixtures/recogniser-charsets.json"), "utf8"));
for (const font of fonts) {
  console.log(`${font.name}: ${font.cps.size} code points`);
}

let failed = false;
for (const [model, chars] of Object.entries(charsets)) {
  if (model.startsWith("_")) {
    continue;
  }
  const wanted = [...new Set([...chars].map((c) => c.codePointAt(0)))].filter((cp) => !/\s/.test(String.fromCodePoint(cp)));
  const notInFirst = wanted.filter((cp) => !fonts[0].cps.has(cp));
  const inNone = notInFirst.filter((cp) => !fonts.some((font) => font.cps.has(cp)));
  const known = inNone.filter((cp) => KNOWN_GAPS.has(cp));
  const missing = inNone.filter((cp) => !KNOWN_GAPS.has(cp));
  const medieval = MEDIEVAL.includes(model);
  console.log(
    `${model}${medieval ? "" : " (not routed to)"}: ${wanted.length} characters; not in ${fonts[0].name}: ` +
      `${notInFirst.length ? notInFirst.map(hex).join(" ") : "none"}; missing from both: ${missing.length ? missing.map(hex).join(" ") : "none"}` +
      (known.length ? `; known gaps: ${known.map((cp) => `${hex(cp)} (${KNOWN_GAPS.get(cp)})`).join(", ")}` : ""),
  );
  if (medieval && missing.length) {
    failed = true;
  }
}
process.exit(failed ? 1 : 0);
