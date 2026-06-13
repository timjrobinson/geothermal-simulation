// Unit tests for the volume-decode + Engineering-frame math (doc 06 §1.3, §2, §3.1).
//
// Pure-function tests — no THREE / no DOM — runnable headlessly. Run via:
//   npm test    (esbuild-bundles this file, then `node --test`)
// These cover the doc 06 §1.3 single-resident decode (ArrayBuffer -> Float32Array shape,
// NaN no-data preservation, truncation guard) and the §2 Engineering AABB corner math.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  decodeVolume,
  voxelCount,
  engineeringAABB,
  aabbCenter,
  aabbSize,
  sampleAt,
  finiteMinMax,
  isLittleEndian,
  type VolumeMeta,
} from "./volume";

// Build a raw little-endian f32 ArrayBuffer for a (nz,ny,nx) volume from a generator.
function rawVolume(
  shape: [number, number, number],
  gen: (i: number, j: number, k: number) => number,
): ArrayBuffer {
  const [nz, ny, nx] = shape;
  const arr = new Float32Array(nz * ny * nx);
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) arr[(k * ny + j) * nx + i] = gen(i, j, k);
  // Float32Array on a LE host already lays out LE bytes.
  return arr.buffer;
}

test("voxelCount multiplies the (z,y,x) extents", () => {
  assert.equal(voxelCount([2, 3, 4]), 24);
});

test("decodeVolume yields a Float32Array of the right shape", () => {
  const shape: [number, number, number] = [2, 3, 4];
  const meta: VolumeMeta = { shape, origin: [0, 0, 0], spacing: [1, 1, 1] };
  const buf = rawVolume(shape, (i, j, k) => i + 10 * j + 100 * k);
  const v = decodeVolume(buf, meta);
  assert.equal(v.data.length, 24);
  // C-contiguous (z,y,x): index (k,j,i) = (1,2,3) -> 3 + 20 + 100 = 123.
  assert.equal(sampleAt(v, 3, 2, 1), 123);
  // x is the fastest axis.
  assert.equal(sampleAt(v, 1, 0, 0), 1);
  assert.equal(sampleAt(v, 0, 1, 0), 10);
  assert.equal(sampleAt(v, 0, 0, 1), 100);
});

test("decodeVolume preserves NaN no-data verbatim", () => {
  const shape: [number, number, number] = [1, 1, 3];
  const meta: VolumeMeta = { shape, origin: [0, 0, 0], spacing: [1, 1, 1] };
  const buf = rawVolume(shape, (i) => (i === 1 ? NaN : i));
  const v = decodeVolume(buf, meta);
  assert.ok(Number.isNaN(v.data[1]));
  assert.equal(v.data[0], 0);
  assert.equal(v.data[2], 2);
});

test("decodeVolume rejects a truncated/garbled buffer", () => {
  const meta: VolumeMeta = {
    shape: [2, 2, 2],
    origin: [0, 0, 0],
    spacing: [1, 1, 1],
  };
  const tooShort = new ArrayBuffer(4 * 7); // expects 8 voxels
  assert.throws(() => decodeVolume(tooShort, meta), /byte length/);
});

test("engineeringAABB places half-voxel-padded corners (Z-up XYZ)", () => {
  // 4 samples in x (nx=4), 3 in y, 2 in z; origin (oz,oy,ox), spacing (dz,dy,dx).
  const meta: VolumeMeta = {
    shape: [2, 3, 4],
    origin: [-1000, 200, 100], // [oz, oy, ox]
    spacing: [25, 10, 5], // [dz, dy, dx]
  };
  const box = engineeringAABB(meta);
  // X: ox=100, dx=5, nx=4 -> centres 100..115, padded 97.5 .. 117.5
  assert.equal(box.min[0], 97.5);
  assert.equal(box.max[0], 117.5);
  // Y: oy=200, dy=10, ny=3 -> centres 200..220, padded 195 .. 225
  assert.equal(box.min[1], 195);
  assert.equal(box.max[1], 225);
  // Z: oz=-1000, dz=25, nz=2 -> centres -1000,-975, padded -1012.5 .. -962.5
  assert.equal(box.min[2], -1012.5);
  assert.equal(box.max[2], -962.5);
});

test("aabbCenter / aabbSize are consistent with the corners", () => {
  const box = { min: [0, -10, -100] as [number, number, number], max: [20, 10, 100] as [number, number, number] };
  assert.deepEqual(aabbCenter(box), [10, 0, 0]);
  assert.deepEqual(aabbSize(box), [20, 20, 200]);
});

test("finiteMinMax ignores NaN and returns null when all-NaN", () => {
  assert.deepEqual(finiteMinMax(new Float32Array([NaN, 1, NaN, 3, 2])), {
    min: 1,
    max: 3,
  });
  assert.equal(finiteMinMax(new Float32Array([NaN, NaN])), null);
});

test("host is little-endian (browser-target assumption holds here)", () => {
  assert.equal(isLittleEndian(), true);
});
