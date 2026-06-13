// Unit tests for the terrain surface-grid → mesh vertex/Z math (doc 06 §6, doc 01 §6).
//
// Pure-function tests — no THREE / no DOM — runnable headlessly via `npm test`. These
// cover: surfaceModel parsing (flat/synthetic/dem + fallback), flat/synthetic grid build,
// the explicit-elevation (DEM) wrapper + its length guard, deterministic synthetic relief,
// the grid→mesh vertex positions (X,Y,Z) + vertical-exaggeration Z scaling + index winding,
// NaN-hole quad dropping, and the finite elevation range.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  parseSurfaceModel,
  buildSurfaceGrid,
  makeGridFromElevation,
  syntheticElevation,
  hashStringToUnit,
  gridSampleXY,
  gridToMesh,
  elevationRange,
  type SurfaceGrid,
} from "./terrain";

test("parseSurfaceModel parses flat / synthetic / dem and falls back to flat:0", () => {
  assert.deepEqual(parseSurfaceModel("flat:0"), { kind: "flat", z: 0 });
  assert.deepEqual(parseSurfaceModel("flat:1620"), { kind: "flat", z: 1620 });
  assert.deepEqual(parseSurfaceModel("synthetic:great-basin-v1"), {
    kind: "synthetic",
    id: "great-basin-v1",
  });
  assert.deepEqual(parseSurfaceModel("dem:copernicus-30m"), {
    kind: "dem",
    provider: "copernicus-30m",
  });
  // Null / empty / unknown → flat:0 ground so the viewer always has a surface.
  assert.deepEqual(parseSurfaceModel(null), { kind: "flat", z: 0 });
  assert.deepEqual(parseSurfaceModel(""), { kind: "flat", z: 0 });
  assert.deepEqual(parseSurfaceModel("garbage"), { kind: "flat", z: 0 });
  // "flat" with no value → 0.
  assert.deepEqual(parseSurfaceModel("flat"), { kind: "flat", z: 0 });
});

test("buildSurfaceGrid lays a flat grid at the requested elevation over the extent", () => {
  const g = buildSurfaceGrid(
    { kind: "flat", z: 1620 },
    { xmin: -1000, xmax: 1000, ymin: -500, ymax: 500 },
    3, // 3×3 samples → 2×2 quads
  );
  assert.equal(g.nx, 3);
  assert.equal(g.ny, 3);
  assert.equal(g.x0, -1000);
  assert.equal(g.y0, -500);
  assert.equal(g.dx, 1000); // (1000 - -1000)/(3-1)
  assert.equal(g.dy, 500); // (500 - -500)/(3-1)
  // Every elevation is the flat z.
  for (const v of g.elevation) assert.equal(v, 1620);
  // Sample (2,2) is the far corner.
  assert.deepEqual(gridSampleXY(g, 2, 2), [1000, 500]);
});

test("buildSurfaceGrid clamps resolution to >=2 (at least one quad)", () => {
  const g = buildSurfaceGrid(
    { kind: "flat", z: 0 },
    { xmin: 0, xmax: 10, ymin: 0, ymax: 10 },
    1,
  );
  assert.equal(g.nx, 2);
  assert.equal(g.ny, 2);
});

test("buildSurfaceGrid uses deterministic synthetic relief about z=0", () => {
  const ext = { xmin: -2000, xmax: 2000, ymin: -2000, ymax: 2000 };
  const a = buildSurfaceGrid({ kind: "synthetic", id: "great-basin-v1" }, ext, 16);
  const b = buildSurfaceGrid({ kind: "synthetic", id: "great-basin-v1" }, ext, 16);
  // Deterministic: same id+extent+res → identical surface.
  assert.deepEqual(Array.from(a.elevation), Array.from(b.elevation));
  // Different scenarios differ somewhere.
  const c = buildSurfaceGrid({ kind: "synthetic", id: "other" }, ext, 16);
  assert.notDeepEqual(Array.from(a.elevation), Array.from(c.elevation));
  // Relief stays within the default ±amplitude band (±120 m about base 0).
  const r = elevationRange(a)!;
  assert.ok(r.min >= -121 && r.max <= 121, `relief range ${r.min}..${r.max}`);
});

test("buildSurfaceGrid('dem') falls back to a flat z=0 grid (no synthetic DEM)", () => {
  const g = buildSurfaceGrid(
    { kind: "dem", provider: "copernicus-30m" },
    { xmin: 0, xmax: 100, ymin: 0, ymax: 100 },
    4,
  );
  for (const v of g.elevation) assert.equal(v, 0);
});

test("syntheticElevation is pure/stable and honours base+amplitude", () => {
  const z1 = syntheticElevation("scn", 100, 200);
  const z2 = syntheticElevation("scn", 100, 200);
  assert.equal(z1, z2);
  // base shifts the whole surface; amplitude=0 → exactly base everywhere.
  assert.equal(syntheticElevation("scn", 100, 200, { base: 500, amplitude: 0 }), 500);
});

test("hashStringToUnit is stable and in [0,1)", () => {
  const h = hashStringToUnit("great-basin-v1");
  assert.equal(h, hashStringToUnit("great-basin-v1"));
  assert.ok(h >= 0 && h < 1);
  assert.notEqual(hashStringToUnit("a"), hashStringToUnit("b"));
});

test("makeGridFromElevation wraps an explicit Engineering-elevation array", () => {
  // 2×2 DEM-style grid, row-major (Y outer, X inner).
  const elev = Float32Array.from([10, 20, 30, 40]);
  const g = makeGridFromElevation(elev, {
    nx: 2,
    ny: 2,
    x0: 0,
    y0: 0,
    dx: 30,
    dy: 30,
  });
  assert.equal(g.elevation[0], 10);
  assert.equal(g.elevation[3], 40);
  assert.deepEqual(gridSampleXY(g, 1, 1), [30, 30]);
});

test("makeGridFromElevation rejects a mismatched-length array", () => {
  assert.throws(
    () =>
      makeGridFromElevation(Float32Array.from([1, 2, 3]), {
        nx: 2,
        ny: 2,
        x0: 0,
        y0: 0,
        dx: 1,
        dy: 1,
      }),
    /elevation length/,
  );
});

test("gridToMesh places vertices at (X, Y, elevation) in the Engineering Frame", () => {
  // 2×2 grid with distinct elevations to check Z displacement + XY placement.
  const g: SurfaceGrid = {
    nx: 2,
    ny: 2,
    x0: 100,
    y0: 200,
    dx: 50,
    dy: 25,
    elevation: Float32Array.from([1000, 1010, 1020, 1030]),
  };
  const m = gridToMesh(g);
  assert.equal(m.positions.length, 2 * 2 * 3);
  // Vertex 0 = sample (0,0): X=x0, Y=y0, Z=elev[0].
  assert.deepEqual(Array.from(m.positions.slice(0, 3)), [100, 200, 1000]);
  // Vertex 1 = sample (1,0): X=x0+dx, Y=y0, Z=elev[1].
  assert.deepEqual(Array.from(m.positions.slice(3, 6)), [150, 200, 1010]);
  // Vertex 2 = sample (0,1): X=x0, Y=y0+dy, Z=elev[2].
  assert.deepEqual(Array.from(m.positions.slice(6, 9)), [100, 225, 1020]);
  // Vertex 3 = sample (1,1): X=x0+dx, Y=y0+dy, Z=elev[3].
  assert.deepEqual(Array.from(m.positions.slice(9, 12)), [150, 225, 1030]);
  // One quad → two triangles → 6 indices.
  assert.equal(m.indices.length, 6);
  // UVs span [0,1] across the grid.
  assert.deepEqual(Array.from(m.uvs.slice(0, 2)), [0, 0]);
  assert.deepEqual(Array.from(m.uvs.slice(6, 8)), [1, 1]); // far corner vertex 3
});

test("gridToMesh applies vertical exaggeration to Z only (render-only, doc 06 §2.3)", () => {
  const g: SurfaceGrid = {
    nx: 2,
    ny: 2,
    x0: 0,
    y0: 0,
    dx: 10,
    dy: 10,
    elevation: Float32Array.from([100, 100, 100, 100]),
  };
  const m = gridToMesh(g, 3);
  // X/Y unchanged; Z multiplied by vex.
  assert.equal(m.positions[0], 0); // X
  assert.equal(m.positions[1], 0); // Y
  assert.equal(m.positions[2], 300); // Z = 100 * 3
  // The opposite-corner X/Y still in true metres.
  assert.equal(m.positions[9], 10);
  assert.equal(m.positions[10], 10);
  assert.equal(m.positions[11], 300);
});

test("gridToMesh winds quads CCW from above (upward-facing surface)", () => {
  const g: SurfaceGrid = {
    nx: 2,
    ny: 2,
    x0: 0,
    y0: 0,
    dx: 1,
    dy: 1,
    elevation: Float32Array.from([0, 0, 0, 0]),
  };
  const m = gridToMesh(g);
  // Indices: a,b,d, b,c,d with a=0,b=1,c=3,d=2.
  assert.deepEqual(Array.from(m.indices), [0, 1, 2, 1, 3, 2]);
});

test("gridToMesh drops quads whose four corners are all NaN (DEM holes)", () => {
  // 3×2 grid: left quad (cols 0-1) has finite corners, right quad (cols 1-2) is entirely
  // NaN. A quad is dropped only when ALL four of its corners are NaN, so col 1 must be NaN
  // too (it is shared by both quads, but the left quad still has finite corners in col 0).
  const g: SurfaceGrid = {
    nx: 3,
    ny: 2,
    x0: 0,
    y0: 0,
    dx: 1,
    dy: 1,
    // row 0: [0, NaN, NaN], row 1: [0, NaN, NaN] → right quad (cols 1-2) all-NaN
    elevation: Float32Array.from([0, NaN, NaN, 0, NaN, NaN]),
  };
  const m = gridToMesh(g);
  // 2 quads possible; right one (all-NaN corners) dropped → only 1 quad → 6 indices.
  assert.equal(m.indices.length, 6);
});

test("elevationRange ignores NaN and returns null when all-NaN", () => {
  const g: SurfaceGrid = {
    nx: 2,
    ny: 2,
    x0: 0,
    y0: 0,
    dx: 1,
    dy: 1,
    elevation: Float32Array.from([NaN, 5, 1, NaN]),
  };
  assert.deepEqual(elevationRange(g), { min: 1, max: 5 });
  const empty: SurfaceGrid = { ...g, elevation: Float32Array.from([NaN, NaN, NaN, NaN]) };
  assert.equal(elevationRange(empty), null);
});
