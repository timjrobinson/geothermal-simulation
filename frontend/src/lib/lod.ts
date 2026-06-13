// LOD selection for the M2+ brick streamer (doc 06 §3.4, §7.2, §7.3). PURE — no THREE / no
// DOM — so the screen-space-error, distance, frustum and clip-box culling math is unit-
// testable headlessly. The renderer feeds it a camera descriptor + the pyramid spec; it
// returns the set of bricks to request, coarsest-first, prioritized by on-screen importance.
//
// Strategy (doc 06 §7.3): for each candidate brick we
//   1. cull by the view frustum (skip bricks fully outside) and the clip box (skip bricks
//      fully outside the user clip region) — doc 06 §3.4 "only bricks intersecting the view
//      frustum AND the clip box ... are requested",
//   2. compute its SCREEN-SPACE ERROR — project the brick's voxel size to pixels at the
//      brick's distance — and pick the COARSEST level whose voxel projects to <= the target
//      pixel budget (doc 06 §7.3 "screen-space-error driven (project brick size to pixels)"),
//   3. prioritize by smallest distance / largest screen size so near/important bricks load
//      first (doc 06 §7.3 foveated/region refinement).
//
// We do NOT do the GPU upload here — that is the brick pool (lib/brickPool.ts) + the worker.
// This module only DECIDES what to want.

import type { AABB } from "./volume";
import {
  type PyramidSpec,
  type BrickAddress,
  levelBrickGrid,
  levelSpacing,
  brickAABB,
  coarsestLevel,
} from "./bricks";

// Minimal camera/view descriptor the renderer extracts from THREE each frame. All in
// Engineering metres (world == Engineering Frame, doc 06 §2.1). `planes` are the 6 frustum
// planes as [a,b,c,d] with a*x+b*y+c*z+d >= 0 INSIDE (THREE.Frustum convention). `fovYRad`
// + `viewportH` convert metres-at-distance to pixels for the SSE test; ortho cameras pass
// `orthoMetresPerPixel` instead (no perspective foreshortening, doc 06 §2.2).
export interface ViewDesc {
  eye: [number, number, number];
  planes?: ReadonlyArray<readonly [number, number, number, number]>;
  fovYRad?: number; // perspective vertical FOV
  viewportH?: number; // viewport height in px
  orthoMetresPerPixel?: number; // ortho: world metres per device pixel (overrides perspective)
}

export interface LodConfig {
  // Target screen-space error: the largest a brick's VOXEL may project before we demand a
  // finer level (pixels). ~1-2 px ≈ "one voxel per pixel". Larger => coarser => cheaper.
  targetVoxelPx?: number;
  // Hard cap on how many bricks we will request beyond the always-resident coarsest level,
  // bounding fetch/VRAM pressure (doc 06 §7.2 fixed budget). Coarsest-level bricks are always
  // included and do NOT count against this cap.
  maxBricks?: number;
  // Clip box in Engineering metres; bricks fully outside are culled (doc 06 §3.4). Omit to
  // disable clip culling.
  clip?: AABB;
}

// A selected brick + the screen-space metrics that chose it (priority ordering + tests).
export interface BrickSelection {
  addr: BrickAddress;
  // Estimated on-screen size of the brick's VOXEL in pixels at its nearest point. The LOD
  // invariant is voxelPx <= targetVoxelPx (a coarser level would exceed it).
  voxelPx: number;
  // Distance (m) from the eye to the brick's nearest point (priority: near loads first).
  distance: number;
  // Whether the brick is always-resident coarsest level (loaded regardless of budget).
  coarsest: boolean;
}

// Squared distance from a point to an AABB (0 if inside). Pure helper.
export function distanceToAABB2(p: readonly [number, number, number], box: AABB): number {
  let d2 = 0;
  for (let k = 0; k < 3; k++) {
    const v = p[k];
    const lo = box.min[k];
    const hi = box.max[k];
    if (v < lo) d2 += (lo - v) * (lo - v);
    else if (v > hi) d2 += (v - hi) * (v - hi);
  }
  return d2;
}

// True if an AABB is at least partially inside the frustum (conservative — a box is culled
// ONLY if it lies entirely on the outside of some plane). Standard AABB-vs-frustum test:
// for each plane, find the box corner farthest along the plane normal (the "positive
// vertex"); if even that is outside, the whole box is outside. No planes => never culled.
export function aabbInFrustum(
  box: AABB,
  planes?: ReadonlyArray<readonly [number, number, number, number]>,
): boolean {
  if (!planes || planes.length === 0) return true;
  for (const [a, b, c, d] of planes) {
    const px = a >= 0 ? box.max[0] : box.min[0];
    const py = b >= 0 ? box.max[1] : box.min[1];
    const pz = c >= 0 ? box.max[2] : box.min[2];
    if (a * px + b * py + c * pz + d < 0) return false; // positive vertex outside => culled
  }
  return true;
}

// True if two AABBs overlap (touching counts as overlap). Used for clip-box culling.
export function aabbIntersect(a: AABB, b: AABB): boolean {
  return (
    a.min[0] <= b.max[0] &&
    a.max[0] >= b.min[0] &&
    a.min[1] <= b.max[1] &&
    a.max[1] >= b.min[1] &&
    a.min[2] <= b.max[2] &&
    a.max[2] >= b.min[2]
  );
}

// Project a world-space length (the brick's largest voxel edge) to pixels at a given
// distance from the eye (doc 06 §7.3 SSE). Perspective: pixels = len / (2*dist*tan(fov/2)) *
// viewportH. Ortho: pixels = len / metresPerPixel (distance-independent). Returns +Inf when
// dist -> 0 (camera inside the voxel => always refine).
export function voxelSizePx(
  worldLen: number,
  distance: number,
  view: ViewDesc,
): number {
  if (view.orthoMetresPerPixel && view.orthoMetresPerPixel > 0) {
    return worldLen / view.orthoMetresPerPixel;
  }
  const fov = view.fovYRad ?? Math.PI / 4;
  const h = view.viewportH ?? 1080;
  if (distance <= 1e-6) return Infinity;
  const worldPerPxAtDist = (2 * distance * Math.tan(fov / 2)) / h;
  return worldLen / Math.max(worldPerPxAtDist, 1e-12);
}

// Largest voxel edge (metres) at a level — the SSE projects THIS length.
function maxVoxelEdge(spec: PyramidSpec, level: number): number {
  const [dz, dy, dx] = levelSpacing(spec, level);
  return Math.max(dz, dy, dx);
}

// For a brick footprint at a given distance, pick the COARSEST level whose voxel projects to
// <= targetPx (doc 06 §7.3). Walks from coarsest (cheap) toward finest (0) and stops at the
// first level meeting the budget; never returns finer than 0. This is the per-region LOD
// decision; the caller maps the footprint to the actual brick at the chosen level.
export function pickLevel(
  spec: PyramidSpec,
  distance: number,
  view: ViewDesc,
  targetPx: number,
): number {
  const coarse = coarsestLevel(spec);
  let chosen = coarse;
  for (let level = coarse; level >= 0; level--) {
    const px = voxelSizePx(maxVoxelEdge(spec, level), distance, view);
    chosen = level;
    if (px <= targetPx) break; // coarsest level that already meets the budget
  }
  return chosen;
}

// SELECT the bricks to have resident this frame (doc 06 §3.4 + §7.3). Always includes the
// full coarsest level (never blank), then walks each coarsest-level node's subtree, refining
// where the screen-space error demands it, subject to frustum + clip culling and the
// maxBricks budget. Returns selections sorted so the MOST important (nearest / largest on
// screen) non-coarsest bricks come first — the fetch queue drains in that order.
export function selectBricks(
  spec: PyramidSpec,
  view: ViewDesc,
  cfg: LodConfig = {},
): BrickSelection[] {
  const targetPx = cfg.targetVoxelPx ?? 1.5;
  const maxBricks = cfg.maxBricks ?? 64;

  const coarse = coarsestLevel(spec);
  const selections: BrickSelection[] = [];
  const seen = new Set<string>();

  const consider = (addr: BrickAddress, forceCoarsest: boolean): BrickSelection | null => {
    const box = brickAABB(spec, addr);
    if (!aabbInFrustum(box, view.planes)) return null;
    if (cfg.clip && !aabbIntersect(box, cfg.clip)) return null;
    const distance = Math.sqrt(distanceToAABB2(view.eye, box));
    const px = voxelSizePx(maxVoxelEdge(spec, addr.level), distance, view);
    return { addr, voxelPx: px, distance, coarsest: forceCoarsest };
  };

  // 1) Always-resident coarsest level: every brick of the top level (doc 06 §3.4). These are
  //    NOT culled away (the volume must never be blank), but still skip frustum/clip misses
  //    to avoid wasted uploads — except we keep at least the addresses that pass.
  const [gz, gy, gx] = levelBrickGrid(spec, coarse);
  const coarseFrontier: BrickAddress[] = [];
  for (let bz = 0; bz < gz; bz++)
    for (let by = 0; by < gy; by++)
      for (let bx = 0; bx < gx; bx++) {
        const addr: BrickAddress = { level: coarse, t: 0, bz, by, bx };
        coarseFrontier.push(addr);
        const sel = consider(addr, true);
        if (sel) {
          selections.push(sel);
          seen.add(keyOf(addr));
        }
      }

  // 2) Refinement frontier: from each visible coarsest node, descend the octree where the
  //    screen-space error of the CURRENT level still exceeds the target (i.e. a finer level
  //    is warranted). Collect candidate finer bricks, then budget-cap by priority.
  const candidates: BrickSelection[] = [];
  const visit = (addr: BrickAddress): void => {
    if (addr.level <= 0) return; // already finest
    const box = brickAABB(spec, addr);
    if (!aabbInFrustum(box, view.planes)) return;
    if (cfg.clip && !aabbIntersect(box, cfg.clip)) return;
    const distance = Math.sqrt(distanceToAABB2(view.eye, box));
    const pxHere = voxelSizePx(maxVoxelEdge(spec, addr.level), distance, view);
    if (pxHere <= targetPx) return; // this level is already fine enough; do not refine
    // Descend: the children (finer level) are warranted. Add them, then recurse.
    for (const child of childrenOf(spec, addr)) {
      const sel = consider(child, false);
      if (!sel) continue;
      const k = keyOf(child);
      if (!seen.has(k)) {
        seen.add(k);
        candidates.push(sel);
      }
      visit(child);
    }
  };
  for (const addr of coarseFrontier) visit(addr);

  // 3) Priority order: nearest first (smallest distance), then largest on-screen. Cap to
  //    maxBricks finer bricks (doc 06 §7.2 fixed budget) — coarsest bricks are always kept.
  candidates.sort((a, b) => a.distance - b.distance || b.voxelPx - a.voxelPx);
  for (const sel of candidates.slice(0, maxBricks)) selections.push(sel);
  return selections;
}

// Local copies of bricks.ts helpers to avoid importing the validating brickAddress (these
// are already-valid in-grid addresses from selection). Kept tiny + pure.
function keyOf(a: BrickAddress): string {
  return `${a.level}/${a.t}/${a.bz}/${a.by}/${a.bx}`;
}
function childrenOf(spec: PyramidSpec, a: BrickAddress): BrickAddress[] {
  if (a.level <= 0) return [];
  const child = a.level - 1;
  const [gz, gy, gx] = levelBrickGrid(spec, child);
  const out: BrickAddress[] = [];
  for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
      for (let dx = 0; dx < 2; dx++) {
        const bz = a.bz * 2 + dz;
        const by = a.by * 2 + dy;
        const bx = a.bx * 2 + dx;
        if (bz < gz && by < gy && bx < gx) out.push({ level: child, t: a.t, bz, by, bx });
      }
  return out;
}
