// Pure brick-extraction logic shared by the worker (doc 06 §3.4, §7.5 "decode bricks in Web
// Workers"). PURE — no DOM / no fetch / no THREE — so the per-brick voxel slicing is unit-
// testable headlessly. The worker imports these to carve a brick out of a server-decoded
// level buffer; the main thread imports the same functions in tests.
//
// Why slice rather than fetch per-chunk: the doc 06 §1.3 SPIKE defers browser-Blosc, so the
// streaming client uses the SERVER-DECODED raw path — GET /property-models/{id}/volume?
// level=L returns the WHOLE level as contiguous LE f32 (z,y,x). We carve the requested 64³
// brick out of that decoded level. The on-disk layout (doc 02 §10.2) and the brick ADDRESS
// (doc 04 §6) are unchanged — only WHO decodes changes (doc 06 §1.3). When browser-Blosc is
// later wired, the same brick atlas/page-table consumes per-chunk decoded buffers instead.

import type { Shape3 } from "./volume";

// Extract a `brickEdge`³ brick at brick index (bz,by,bx) from a C-contiguous (z,y,x) level
// buffer of shape `shape`. Returns a fresh Float32Array of length brickEdge³, padded with NaN
// (the no-data sentinel, doc 02 §10.2) where the brick overhangs the level extent (edge
// bricks at non-multiple-of-edge sizes). The brick is itself C-contiguous (z,y,x) so it
// uploads directly as a Data3DTexture sub-image of (edge,edge,edge) — matching data3d.ts.
export function extractBrick(
  level: Float32Array,
  shape: Shape3,
  brickEdge: number,
  bz: number,
  by: number,
  bx: number,
): Float32Array {
  const [nz, ny, nx] = shape;
  const e = brickEdge;
  const out = new Float32Array(e * e * e);
  out.fill(NaN); // overhang stays no-data

  const z0 = bz * e;
  const y0 = by * e;
  const x0 = bx * e;
  // Voxel extent actually inside the level (clamped).
  const zc = Math.min(e, nz - z0);
  const yc = Math.min(e, ny - y0);
  const xc = Math.min(e, nx - x0);
  if (zc <= 0 || yc <= 0 || xc <= 0) return out; // brick entirely outside (shouldn't happen)

  for (let z = 0; z < zc; z++) {
    const srcZ = (z0 + z) * ny;
    const dstZ = z * e;
    for (let y = 0; y < yc; y++) {
      const srcRow = (srcZ + (y0 + y)) * nx + x0;
      const dstRow = (dstZ + y) * e;
      // Copy the contiguous x-run for this (z,y) line.
      out.set(level.subarray(srcRow, srcRow + xc), dstRow);
    }
  }
  return out;
}

// NaN-aware finite min/max over a brick (cheap per-brick stats, feeds the optional per-brick
// value range / empty-brick skip). Returns null if the brick is entirely NaN (a fully
// no-data brick the renderer may skip uploading).
export function brickFiniteRange(
  brick: Float32Array,
): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < brick.length; i++) {
    const v = brick[i];
    if (v === v && v !== Infinity && v !== -Infinity) {
      // v===v rejects NaN without Number.isFinite call overhead in the hot loop
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return min === Infinity ? null : { min, max };
}

// True if a brick is entirely no-data (all NaN) — such bricks need no atlas slot (doc 06
// §3.4 sparse skipping; the page table simply leaves them -1 and the shader falls back).
export function isEmptyBrick(brick: Float32Array): boolean {
  return brickFiniteRange(brick) === null;
}
