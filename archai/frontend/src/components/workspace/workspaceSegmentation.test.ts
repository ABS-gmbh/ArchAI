// Layout regions as the workspace stores them for cropping and label questions.
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  type CocoPayload,
  annotationPolygons,
  availableStoredLabels,
  buildStoredLabelRegions,
  labelNamedInPrompt,
  resolveCropLabelFromPrompt,
} from "./workspaceSegmentation";

const square = [10, 10, 60, 10, 60, 60, 10, 60];

const coco: CocoPayload = {
  categories: [
    { id: 1, name: "Main script black" },
    { id: 2, name: "Embellished" },
  ],
  annotations: [
    { id: 11, category_id: 1, bbox: [10, 20, 100, 40], segmentation: [square] },
    { id: 12, category_id: 1, bbox: [10, 70, 100, 40], segmentation: square },
    { id: 13, category_id: 2, bbox: [200, 20, 30, 30] },
    // An annotation of a category the payload does not declare.
    { id: 14, category_id: 9, bbox: [0, 0, 10, 10] },
    // A box that is not four numbers.
    { id: 15, category_id: 1, bbox: [1, 2, Number.NaN, 4] },
  ],
};

describe("buildStoredLabelRegions", () => {
  const regions = buildStoredLabelRegions(coco);

  it("groups regions by label name, with boxes as corners", () => {
    assert.deepEqual(availableStoredLabels(regions), ["Embellished", "Main script black"]);
    assert.deepEqual(
      regions["Main script black"].map((region) => [region.region_id, region.bbox_xyxy]),
      [
        ["11", [10, 20, 110, 60]],
        ["12", [10, 70, 110, 110]],
      ],
    );
  });

  it("keeps each region's polygons, from a nested list or a single flat one", () => {
    assert.deepEqual(regions["Main script black"][0].polygons, [square]);
    assert.deepEqual(regions["Main script black"][1].polygons, [square]);
    assert.deepEqual(regions.Embellished[0].polygons, []);
  });

  it("drops annotations with an unknown category or a box that is not four numbers", () => {
    const ids = Object.values(regions).flat().map((region) => region.region_id);
    assert.deepEqual(ids.sort(), ["11", "12", "13"]);
  });

  it("returns nothing for a missing payload", () => {
    assert.deepEqual(buildStoredLabelRegions(null), {});
    assert.deepEqual(buildStoredLabelRegions({}), {});
  });
});

describe("annotationPolygons", () => {
  it("drops nested polygons with fewer than three points", () => {
    assert.deepEqual(annotationPolygons({ id: 1, category_id: 1, bbox: [0, 0, 1, 1], segmentation: [square, [1, 2, 3, 4]] }), [square]);
  });

  it("reads no polygon from run-length-encoded masks", () => {
    const rle = { counts: [1, 2], size: [10, 10] } as unknown as number[];
    assert.deepEqual(annotationPolygons({ id: 1, category_id: 1, bbox: [0, 0, 1, 1], segmentation: rle }), []);
  });

  // A malformed polygon would reach the crop overlay and the label-analysis
  // request as an outline the region does not have.
  const polygons = (segmentation: unknown) =>
    annotationPolygons({ id: 1, category_id: 1, bbox: [0, 0, 1, 1], segmentation: segmentation as number[] });

  it("drops a flat polygon with fewer than three points", () => {
    assert.deepEqual(polygons([10, 20, 30]), []);
    assert.deepEqual(polygons([10, 20, 30, 40]), []);
  });

  it("drops a polygon with an odd number of coordinates", () => {
    assert.deepEqual(polygons([10, 10, 60, 10, 60, 60, 10]), []);
    assert.deepEqual(polygons([[10, 10, 60, 10, 60, 60, 10]]), []);
  });

  it("drops a polygon with a coordinate that is not a number", () => {
    assert.deepEqual(polygons([10, 10, 60, Number.NaN, 60, 60]), []);
    assert.deepEqual(polygons([square, [10, 10, "60", 10, 60, 60]]), [square]);
  });
});

describe("resolveCropLabelFromPrompt", () => {
  const regions = buildStoredLabelRegions(coco);

  it("finds the label a request names", () => {
    assert.equal(resolveCropLabelFromPrompt("crop the embellished initial", regions), "Embellished");
    assert.equal(resolveCropLabelFromPrompt("cut out the main script", regions), "Main script black");
  });

  it("finds nothing when the request names no label on the page", () => {
    assert.equal(resolveCropLabelFromPrompt("crop the miniature", regions), null);
  });
});

describe("labelNamedInPrompt", () => {
  const labels = ["Page Number", "Main script black", "Column", "Embellished", "Plain initial- coloured", "Plain initial - Black", "Running header"];

  it("does not take a question about the text for a question about a region", () => {
    // Each shares a word with a label ("page", "script", "black", "running").
    for (const question of [
      "What is this page about?",
      "Explain the script of this manuscript",
      "What is this black stain?",
      "Explain why the text is running into the margin",
    ]) {
      assert.equal(labelNamedInPrompt(question, labels), null, question);
    }
  });

  it("finds a label named whole, in any order of its words, or in the plural", () => {
    assert.equal(labelNamedInPrompt("What is this page number?", labels), "Page Number");
    assert.equal(labelNamedInPrompt("explain the number on the page", labels), "Page Number");
    assert.equal(labelNamedInPrompt("What is this column?", labels), "Column");
    assert.equal(labelNamedInPrompt("explain the columns", labels), "Column");
    assert.equal(labelNamedInPrompt("what is the running header", labels), "Running header");
  });

  it("takes 'initial' for a kind of initial, and prefers the one whose other words are said", () => {
    assert.equal(labelNamedInPrompt("What is this black initial?", labels), "Plain initial - Black");
    assert.equal(labelNamedInPrompt("explain the coloured initial", labels), "Plain initial- coloured");
    assert.ok(String(labelNamedInPrompt("What is this initial?", labels)).includes("initial"));
    assert.equal(labelNamedInPrompt("explain the embellished letter", labels), "Embellished");
  });
});
