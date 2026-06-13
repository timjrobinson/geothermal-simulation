// Unit tests for the pure brick-extraction logic the worker uses (doc 06 §3.4, §1.3 server-
// decode fallback). Pure — no DOM / no fetch — runnable headlessly via `npm test`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { extractBrick, brickFiniteRange, isEmptyBrick } from "./brickDecode";
import type { Shape3 } from "./volume";

// Build a (z,y,x) C-contiguous level where each voxel encodes its flat index.
function ramp(shape: Shape3): Float32Array {
  const [nz, ny, nx] = shape;
  const a = new Float32Array(nz * ny * nx);
  for (let i = 0; i < a.length; i++) a[i] = i;
  return a;
}

test("extractBrick carves an aligned interior brick in (z,y,x) order", () => {
  const shape: Shape3 = [4, 4, 4];
  const lvl = ramp(shape);
  // edge 2, brick (0,0,0): voxels (z,y,x) in [0,2)^3
  const b = extractBrick(lvl, shape, 2, 0, 0, 0);
  assert.equal(b.length, 8);
  // flat index in level for (z,y,x) = (z*4 + y)*4 + x
  assert.equal(b[0], 0); // (0,0,0)
  assert.equal(b[1], 1); // (0,0,1)
  assert.equal(b[2], 4); // (0,1,0) -> level index 4
  assert.equal(b[4], 16); // (1,0,0) -> level index 16
});

test("extractBrick picks the correct offset brick", () => {
  const shape: Shape3 = [4, 4, 4];
  const lvl = ramp(shape);
  // edge 2, brick (1,1,1): voxels z,y,x in [2,4)^3; first voxel (2,2,2) = (2*4+2)*4+2 = 42
  const b = extractBrick(lvl, shape, 2, 1, 1, 1);
  assert.equal(b[0], 42);
  assert.equal(b[1], 43); // (2,2,3)
  assert.equal(b[7], (3 * 4 + 3) * 4 + 3); // (3,3,3) = 63
});

test("extractBrick pads overhang voxels with NaN (edge bricks)", () => {
  const shape: Shape3 = [3, 3, 3]; // not a multiple of edge 2
  const lvl = ramp(shape);
  // brick (1,1,1) covers z,y,x in [2,4) but level only has index 2 -> 1 valid voxel, rest NaN
  const b = extractBrick(lvl, shape, 2, 1, 1, 1);
  assert.equal(b.length, 8);
  // only (2,2,2) = (2*3+2)*3+2 = 26 is valid; it sits at brick-local (0,0,0)
  assert.equal(b[0], 26);
  // the rest of the brick overhangs -> NaN
  for (let i = 1; i < 8; i++) assert.ok(Number.isNaN(b[i]), `voxel ${i} should be NaN`);
});

test("extractBrick fully-outside brick returns all-NaN", () => {
  const shape: Shape3 = [2, 2, 2];
  const lvl = ramp(shape);
  const b = extractBrick(lvl, shape, 2, 5, 5, 5); // way past the extent
  assert.ok(b.every((v) => Number.isNaN(v)));
});

test("brickFiniteRange + isEmptyBrick are NaN-aware", () => {
  const mixed = new Float32Array([NaN, 3, NaN, -2, 10, NaN]);
  assert.deepEqual(brickFiniteRange(mixed), { min: -2, max: 10 });
  assert.equal(isEmptyBrick(mixed), false);

  const allNaN = new Float32Array([NaN, NaN, NaN]);
  assert.equal(brickFiniteRange(allNaN), null);
  assert.equal(isEmptyBrick(allNaN), true);

  // Inf is treated as non-finite (rejected) so it cannot widen the range
  const withInf = new Float32Array([Infinity, 5, -Infinity]);
  assert.deepEqual(brickFiniteRange(withInf), { min: 5, max: 5 });
});
