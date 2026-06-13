// Unit tests for the screen-space-error LOD selection (doc 06 §3.4, §7.3). Pure — no THREE /
// no DOM — runnable headlessly via `npm test`.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  distanceToAABB2,
  aabbInFrustum,
  aabbIntersect,
  voxelSizePx,
  pickLevel,
  selectBricks,
  type ViewDesc,
} from "./lod";
import { coarsestLevel, levelBrickCount, type PyramidSpec } from "./bricks";
import type { AABB } from "./volume";

const spec: PyramidSpec = {
  shape0: [256, 256, 256],
  origin: [0, 0, 0],
  spacing0: [1, 1, 1],
  levels: 3,
};

test("distanceToAABB2 is 0 inside and squared-euclidean outside", () => {
  const box: AABB = { min: [0, 0, 0], max: [10, 10, 10] };
  assert.equal(distanceToAABB2([5, 5, 5], box), 0);
  assert.equal(distanceToAABB2([13, 5, 5], box), 9); // 3 m past the +x face
  assert.equal(distanceToAABB2([-4, -3, 5], box), 25); // 3-4-5 in xy
});

test("aabbInFrustum culls boxes fully outside a plane, keeps straddlers", () => {
  const box: AABB = { min: [0, 0, 0], max: [1, 1, 1] };
  // plane x >= 0 (normal +x, d=0): box at x in [0,1] is inside
  assert.equal(aabbInFrustum(box, [[1, 0, 0, 0]]), true);
  // plane x >= 5 (a*x + d >= 0 with a=1,d=-5): box max x=1 < 5 -> culled
  assert.equal(aabbInFrustum(box, [[1, 0, 0, -5]]), false);
  // no planes -> never culled
  assert.equal(aabbInFrustum(box), true);
});

test("aabbIntersect detects overlap incl. touching", () => {
  const a: AABB = { min: [0, 0, 0], max: [1, 1, 1] };
  assert.equal(aabbIntersect(a, { min: [0.5, 0.5, 0.5], max: [2, 2, 2] }), true);
  assert.equal(aabbIntersect(a, { min: [1, 0, 0], max: [2, 1, 1] }), true); // touching face
  assert.equal(aabbIntersect(a, { min: [2, 2, 2], max: [3, 3, 3] }), false);
});

test("voxelSizePx: perspective shrinks with distance; ortho is constant", () => {
  const persp: ViewDesc = { eye: [0, 0, 0], fovYRad: Math.PI / 2, viewportH: 1000 };
  const near = voxelSizePx(1, 10, persp);
  const far = voxelSizePx(1, 100, persp);
  assert.ok(near > far); // farther voxel projects smaller
  assert.ok(Math.abs(near / far - 10) < 1e-6); // inverse-linear in distance
  // camera inside the voxel -> infinite (always refine)
  assert.equal(voxelSizePx(1, 0, persp), Infinity);
  // ortho: distance-independent
  const ortho: ViewDesc = { eye: [0, 0, 0], orthoMetresPerPixel: 0.5 };
  assert.equal(voxelSizePx(1, 10, ortho), 2);
  assert.equal(voxelSizePx(1, 1000, ortho), 2);
});

test("pickLevel returns coarser levels when far, finer when near", () => {
  const view: ViewDesc = { eye: [0, 0, 0], fovYRad: Math.PI / 2, viewportH: 1000 };
  // Very far -> coarsest level meets a 1.5px budget
  assert.equal(pickLevel(spec, 1e6, view, 1.5), coarsestLevel(spec));
  // Very near -> finest level (0)
  assert.equal(pickLevel(spec, 0.001, view, 1.5), 0);
  // pickLevel never goes finer than 0 or coarser than coarsest
  const l = pickLevel(spec, 50, view, 1.5);
  assert.ok(l >= 0 && l <= coarsestLevel(spec));
});

test("selectBricks always includes the full coarsest level (never blank)", () => {
  // Camera far away, generous frustum (none) -> only coarsest should be selected.
  const view: ViewDesc = { eye: [1e6, 1e6, 1e6], fovYRad: Math.PI / 4, viewportH: 1080 };
  const sel = selectBricks(spec, view, { targetVoxelPx: 1.5 });
  const coarse = sel.filter((s) => s.coarsest);
  assert.equal(coarse.length, levelBrickCount(spec, coarsestLevel(spec))); // all top bricks
  assert.ok(coarse.every((s) => s.addr.level === coarsestLevel(spec)));
});

test("selectBricks refines near the camera and respects maxBricks", () => {
  // Camera sitting inside the volume -> SSE demands finer levels.
  const view: ViewDesc = { eye: [128, 128, 128], fovYRad: Math.PI / 2, viewportH: 2000 };
  const sel = selectBricks(spec, view, { targetVoxelPx: 1.0, maxBricks: 10 });
  const finer = sel.filter((s) => !s.coarsest);
  assert.ok(finer.length > 0, "expected refinement near the camera");
  assert.ok(finer.length <= 10, "maxBricks caps the finer set");
  // finer bricks must be a finer level than coarsest
  assert.ok(finer.every((s) => s.addr.level < coarsestLevel(spec)));
  // priority: the list is sorted nearest-first among the finer set
  for (let i = 1; i < finer.length; i++) {
    assert.ok(finer[i - 1].distance <= finer[i].distance + 1e-6);
  }
});

test("selectBricks clip-culls bricks fully outside the clip box", () => {
  const view: ViewDesc = { eye: [128, 128, 128], fovYRad: Math.PI / 2, viewportH: 2000 };
  // Clip to a small region in the +corner; bricks elsewhere must be dropped.
  const clip: AABB = { min: [200, 200, 200], max: [255, 255, 255] };
  const sel = selectBricks(spec, view, { targetVoxelPx: 1.0, maxBricks: 64, clip });
  // coarsest bricks may still appear (always-resident) but every FINER brick selected must
  // have a footprint that intersects the clip box (doc 06 §3.4 clip-box culling).
  const finerOutside = sel.filter(
    (s) => !s.coarsest && !aabbIntersect(brickBoxOf(s.addr), clip),
  );
  assert.equal(finerOutside.length, 0);
});

// local helper using the real brickAABB
import { brickAABB } from "./bricks";
function brickBoxOf(addr: { level: number; t: number; bz: number; by: number; bx: number }) {
  return brickAABB(spec, addr);
}
