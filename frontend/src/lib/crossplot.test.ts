// Unit tests for the cross-plot / histogram / correlation data-shaping (doc 06 §10.3,
// doc 07 §3.2) and the LINKED-BRUSHING selection mapping (doc 06 §10.3). Pure functions —
// no DOM / no THREE — runnable headlessly via `npm test`.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  columnBounds,
  toPixel,
  fromPixel,
  projectScatter,
  normalizeRect,
  rectIsDegenerate,
  pointsInRect,
  histogramOf,
  correlationMatrix,
  correlationColor,
  type Viewport,
} from "./crossplot";
import {
  selectionToCellIndices,
  selectionToMask,
  selectionToVolume,
  pickNearestVoxel,
} from "./brushing";
import { makeMockFusedSample, type FusedSampleOut } from "./fusion";

// A tiny hand-built sample so expected values are exact.
function tinySample(): FusedSampleOut {
  // grid_shape (nz,ny,nx) = (2,1,2) -> 4 cells; cell_index is the flat (z,y,x) index.
  return {
    properties: ["a", "b"],
    n: 4,
    features: [
      [0, 10],
      [1, 11],
      [2, 12],
      [3, 13],
    ],
    cell_index: [0, 1, 2, 3],
    coords: [
      [-100, 0, 0],
      [-100, 0, 100],
      [-200, 0, 0],
      [-200, 0, 100],
    ],
    grid_shape: [2, 1, 2],
    mode: "all",
  };
}

const VP: Viewport = { width: 100, height: 100, pad: 10 };

test("columnBounds finds finite min/max and guards constant columns", () => {
  const s = tinySample();
  assert.deepEqual(columnBounds(s, "a"), { min: 0, max: 3 });
  assert.deepEqual(columnBounds(s, "b"), { min: 10, max: 13 });
  // unknown property -> safe unit range
  assert.deepEqual(columnBounds(s, "zzz"), { min: 0, max: 1 });
});

test("toPixel / fromPixel round-trip (both axes, with flip)", () => {
  const b = { min: 0, max: 10 };
  for (const v of [0, 2.5, 5, 7.5, 10]) {
    const pxX = toPixel(v, b, 100, 10, false);
    assert.ok(Math.abs(fromPixel(pxX, b, 100, 10, false) - v) < 1e-9);
    const pxY = toPixel(v, b, 100, 10, true);
    assert.ok(Math.abs(fromPixel(pxY, b, 100, 10, true) - v) < 1e-9);
  }
  // flip puts the max at the TOP (smallest pixel y == pad).
  assert.equal(toPixel(10, b, 100, 10, true), 10);
  assert.equal(toPixel(0, b, 100, 10, true), 90);
});

test("projectScatter projects rows and carries the brushing key + colour channel", () => {
  const s = tinySample();
  const xb = columnBounds(s, "a");
  const yb = columnBounds(s, "b");
  const pts = projectScatter(s, "a", "b", VP, xb, yb, "depth");
  assert.equal(pts.length, 4);
  // local index i must be preserved so the brush maps back to the right cell.
  assert.deepEqual(pts.map((p) => p.i), [0, 1, 2, 3]);
  // colour-by-depth uses the z coord (coords[i][0]).
  assert.equal(pts[0].c, -100);
  assert.equal(pts[2].c, -200);
  // min a (0) -> left edge (pad); max a (3) -> right edge.
  assert.equal(pts[0].px, 10);
  assert.equal(pts[3].px, 90);
});

test("projectScatter drops non-finite rows", () => {
  const s = tinySample();
  s.features[1] = [NaN, 11];
  const pts = projectScatter(s, "a", "b", VP, columnBounds(s, "a"), columnBounds(s, "b"));
  assert.equal(pts.length, 3);
  assert.ok(!pts.some((p) => p.i === 1));
});

test("normalizeRect + rectIsDegenerate", () => {
  const r = normalizeRect({ x: 30, y: 80 }, { x: 10, y: 20 });
  assert.deepEqual(r, { x0: 10, x1: 30, y0: 20, y1: 80 });
  assert.ok(rectIsDegenerate({ x0: 0, y0: 0, x1: 1, y1: 50 }));
  assert.ok(!rectIsDegenerate({ x0: 0, y0: 0, x1: 50, y1: 50 }));
});

test("pointsInRect returns the brushed LOCAL row indices (brushing key)", () => {
  const s = tinySample();
  const xb = columnBounds(s, "a");
  const yb = columnBounds(s, "b");
  const pts = projectScatter(s, "a", "b", VP, xb, yb);
  // Brush the left half (a in [0, ~1.5]) -> rows 0 and 1.
  const sel = pointsInRect(pts, { x0: 0, y0: 0, x1: 55, y1: 100 });
  assert.deepEqual(sel.sort((m, n) => m - n), [0, 1]);
});

test("histogramOf bins are NaN-aware and sum to the finite count", () => {
  const s = tinySample();
  const h = histogramOf(s, "a", 4);
  assert.equal(h.counts.length, 4);
  assert.equal(h.edges.length, 5);
  assert.equal(
    h.counts.reduce((m, n) => m + n, 0),
    4,
  );
  s.features[0] = [NaN, 10];
  const h2 = histogramOf(s, "a", 4);
  assert.equal(
    h2.counts.reduce((m, n) => m + n, 0),
    3,
  );
});

test("correlationMatrix: perfectly correlated columns give r≈1; matrix is symmetric", () => {
  const s = tinySample(); // b = a + 10 -> perfectly correlated
  const { properties, matrix } = correlationMatrix(s);
  assert.deepEqual(properties, ["a", "b"]);
  assert.ok(Math.abs((matrix[0][1] as number) - 1) < 1e-9);
  assert.equal(matrix[0][1], matrix[1][0]);
  assert.ok(Math.abs((matrix[0][0] as number) - 1) < 1e-9);
});

test("correlationMatrix returns null for too-few-finite / zero-variance pairs", () => {
  const s: FusedSampleOut = {
    properties: ["a", "b"],
    n: 2,
    features: [
      [1, 5],
      [1, 9],
    ], // a is constant -> zero variance
    cell_index: [0, 1],
    coords: [
      [0, 0, 0],
      [0, 0, 0],
    ],
    grid_shape: [1, 1, 2],
    mode: "all",
  };
  const { matrix } = correlationMatrix(s);
  assert.equal(matrix[0][1], null);
});

test("correlationColor: +1 red-ish, -1 blue-ish, null neutral", () => {
  assert.match(correlationColor(1), /rgb\(255,90,90\)/);
  assert.match(correlationColor(-1), /rgb\(90,90,255\)/);
  assert.match(correlationColor(0), /rgb\(255,255,255\)/);
  assert.equal(correlationColor(null), "rgb(70,76,92)");
});

// ── brushing.ts (mirrors backend geosim.fusion.selection_to_mask) ──────────────────────

test("selectionToCellIndices maps rows -> the backend flat cell indices", () => {
  const s = tinySample();
  assert.deepEqual(selectionToCellIndices(s, [0, 2]), [0, 2]);
  // out-of-range rows are ignored.
  assert.deepEqual(selectionToCellIndices(s, [99, 1]), [1]);
});

test("selectionToMask sets exactly the selected flat cells (selection_to_mask twin)", () => {
  const s = tinySample();
  const mask = selectionToMask(s, [1, 3]);
  assert.equal(mask.length, 4);
  assert.deepEqual(Array.from(mask), [0, 1, 0, 1]);
});

test("selectionToVolume marks selected cells 1.0 and the rest NaN (overlay sentinel)", () => {
  const s = tinySample();
  const grid = { origin: [-300, 0, 0] as [number, number, number], spacing: [100, 100, 100] as [number, number, number] };
  const vol = selectionToVolume(s, [0, 3], grid);
  assert.deepEqual(vol.shape, [2, 1, 2]);
  assert.deepEqual(vol.origin, [-300, 0, 0]);
  assert.equal(vol.data[0], 1.0);
  assert.ok(Number.isNaN(vol.data[1]));
  assert.ok(Number.isNaN(vol.data[2]));
  assert.equal(vol.data[3], 1.0);
});

test("pickNearestVoxel returns the nearest cell's multi-property values (3D pick -> inspector)", () => {
  const s = tinySample();
  // Pick near cell 2 (coords (z,y,x) = (-200,0,0) -> XYZ (0,0,-200)).
  const r = pickNearestVoxel(s, [5, 5, -195]);
  assert.ok(r);
  assert.equal(r!.row, 2);
  assert.equal(r!.cellIndex, 2);
  assert.deepEqual(
    r!.values.map((v) => [v.property, v.value]),
    [
      ["a", 2],
      ["b", 12],
    ],
  );
});

test("pickNearestVoxel respects the maxDist gate (pick outside the sampled region)", () => {
  const s = tinySample();
  const r = pickNearestVoxel(s, [1e6, 1e6, 1e6], 10);
  assert.ok(r);
  assert.equal(r!.row, -1);
  assert.equal(r!.values.length, 0);
});

// ── offline mock (self-contained dev path) ─────────────────────────────────────────────

test("makeMockFusedSample yields a consistent grid + co-located sample", () => {
  const { grid, sample } = makeMockFusedSample({ shape: [4, 5, 6], properties: ["resistivity", "density"] });
  assert.deepEqual(grid.shape, [4, 5, 6]);
  assert.equal(grid.n_cells, 4 * 5 * 6);
  assert.equal(sample.n, 4 * 5 * 6);
  assert.equal(sample.properties.length, 2);
  assert.equal(sample.features[0].length, 2);
  assert.equal(sample.cell_index.length, sample.n);
  assert.equal(sample.coords.length, sample.n);
  // resistivity vs density are anti-correlated by construction (anomaly lowers R, raises ρ).
  const { matrix } = correlationMatrix(sample);
  assert.ok((matrix[0][1] as number) < 0);
});

test("makeMockFusedSample is deterministic for a fixed seed", () => {
  const a = makeMockFusedSample({ shape: [3, 3, 3], seed: 7 });
  const b = makeMockFusedSample({ shape: [3, 3, 3], seed: 7 });
  assert.deepEqual(a.sample.features, b.sample.features);
});
