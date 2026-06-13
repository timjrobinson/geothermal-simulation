// Unit tests for the brick-address math (doc 06 §3.4, doc 04 §5/§6, doc 02 §10.2). Pure
// functions — no THREE / no DOM — runnable headlessly via `npm test`.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  BRICK_SIZE,
  brickEdge,
  levelShape,
  levelSpacing,
  levelBrickGrid,
  levelBrickCount,
  coarsestLevel,
  brickAddress,
  brickChunkPath,
  brickZarrUrl,
  levelVolumeUrl,
  brickAABB,
  volumeAABB,
  brickParent,
  brickChildren,
  brickKey,
  shouldStream,
  type PyramidSpec,
} from "./bricks";

// A 256³ level-0 volume, 1 m voxels, origin 0, 3-level pyramid (256 -> 128 -> 64).
const spec: PyramidSpec = {
  shape0: [256, 256, 256],
  origin: [0, 0, 0],
  spacing0: [1, 1, 1],
  levels: 3,
};

test("brickEdge defaults to BRICK_SIZE (64)", () => {
  assert.equal(brickEdge(spec), BRICK_SIZE);
  assert.equal(brickEdge({ ...spec, brick: 32 }), 32);
});

test("levelShape halves each axis per level (floor, min 1)", () => {
  assert.deepEqual(levelShape(spec, 0), [256, 256, 256]);
  assert.deepEqual(levelShape(spec, 1), [128, 128, 128]);
  assert.deepEqual(levelShape(spec, 2), [64, 64, 64]);
  // odd extent floors down
  assert.deepEqual(levelShape({ ...spec, shape0: [9, 9, 9] }, 1), [4, 4, 4]);
  // never below 1
  assert.deepEqual(levelShape({ ...spec, shape0: [1, 1, 1], levels: 4 }, 3), [1, 1, 1]);
});

test("levelSpacing scales voxel size by 2^level (footprint preserved)", () => {
  assert.deepEqual(levelSpacing(spec, 0), [1, 1, 1]);
  assert.deepEqual(levelSpacing(spec, 1), [2, 2, 2]);
  assert.deepEqual(levelSpacing(spec, 2), [4, 4, 4]);
});

test("levelBrickGrid is ceil(shape/edge) and count is the product", () => {
  // 256/64 = 4 along each axis at level 0
  assert.deepEqual(levelBrickGrid(spec, 0), [4, 4, 4]);
  assert.equal(levelBrickCount(spec, 0), 64);
  // 64/64 = 1 -> coarsest fits one brick
  assert.deepEqual(levelBrickGrid(spec, 2), [1, 1, 1]);
  assert.equal(levelBrickCount(spec, 2), 1);
  // partial edge brick still counts (ceil)
  assert.deepEqual(levelBrickGrid({ ...spec, shape0: [65, 65, 65] }, 0), [2, 2, 2]);
});

test("coarsestLevel is levels-1", () => {
  assert.equal(coarsestLevel(spec), 2);
  assert.equal(coarsestLevel({ ...spec, levels: 1 }), 0);
});

test("brickAddress validates level + indices", () => {
  const a = brickAddress(spec, 0, 1, 2, 3);
  assert.deepEqual(a, { level: 0, t: 0, bz: 1, by: 2, bx: 3 });
  assert.throws(() => brickAddress(spec, 3, 0, 0, 0), /level 3 out of range/);
  assert.throws(() => brickAddress(spec, 0, 4, 0, 0), /out of grid/); // grid is 4 -> max idx 3
  assert.throws(() => brickAddress(spec, 0, -1, 0, 0), /out of grid/);
});

test("brickChunkPath == doc-02 §10.2 chunk key <property>/<level>/c/<bz>/<by>/<bx>", () => {
  // level-0 grid is 4 so (0,1,2) is in range
  const a = brickAddress(spec, 0, 0, 1, 2);
  assert.equal(brickChunkPath("resistivity", a), "resistivity/0/c/0/1/2");
  // 4D form prepends the time index
  assert.equal(brickChunkPath("resistivity", { ...a, t: 5 }, true), "resistivity/0/c/5/0/1/2");
});

test("brickZarrUrl + levelVolumeUrl build the doc-04 endpoint paths", () => {
  const a = brickAddress(spec, 0, 0, 0, 0);
  assert.equal(
    brickZarrUrl("pm-1", "density", a),
    "/property-models/pm-1/zarr/density/0/c/0/0/0",
  );
  assert.equal(
    levelVolumeUrl("pm 1", "den/sity", 2),
    "/property-models/pm%201/volume?level=2&property=den%2Fsity",
  );
});

test("volumeAABB covers the level-0 footprint with half-voxel padding", () => {
  const box = volumeAABB(spec);
  assert.deepEqual(box.min, [-0.5, -0.5, -0.5]);
  // 256 voxels, last centre at 255, +0.5 half-voxel = 255.5
  assert.deepEqual(box.max, [255.5, 255.5, 255.5]);
});

test("brickAABB tiles the volume; bricks abut with no gap/overlap", () => {
  // level 0, edge 64, brick (0,0,0) spans x in [-0.5, 63.5]; brick (.,.,1) starts at 63.5
  const b0 = brickAABB(spec, { level: 0, t: 0, bz: 0, by: 0, bx: 0 });
  assert.deepEqual(b0.min, [-0.5, -0.5, -0.5]);
  assert.equal(b0.max[0], 63.5);
  const b1 = brickAABB(spec, { level: 0, t: 0, bz: 0, by: 0, bx: 1 });
  assert.equal(b1.min[0], 63.5); // abuts b0.max exactly
  assert.equal(b1.max[0], 127.5);
  // coarsest level brick covers (approximately) the whole footprint — its far face sits
  // within one coarse voxel (4 m) of the level-0 footprint due to floor-downsampling.
  const top = brickAABB(spec, { level: 2, t: 0, bz: 0, by: 0, bx: 0 });
  assert.deepEqual(top.min, [-2, -2, -2]); // edge 64 voxels * 4 m spacing, half = 2
  const fullMax = volumeAABB(spec).max[0];
  assert.ok(top.max[0] <= fullMax && fullMax - top.max[0] <= 4);
});

test("brickParent halves indices and climbs a level; null at coarsest", () => {
  const a = brickAddress(spec, 0, 3, 2, 1);
  assert.deepEqual(brickParent(spec, a), { level: 1, t: 0, bz: 1, by: 1, bx: 0 });
  assert.equal(brickParent(spec, { level: 2, t: 0, bz: 0, by: 0, bx: 0 }), null);
});

test("brickChildren are the up-to-8 finer nodes, clamped to the finer grid", () => {
  // level 2 (grid 1) -> level 1 (grid 2): node (0,0,0) has all 8 children
  const kids = brickChildren(spec, { level: 2, t: 0, bz: 0, by: 0, bx: 0 });
  assert.equal(kids.length, 8);
  assert.ok(kids.every((k) => k.level === 1));
  // finest level has no children
  assert.deepEqual(brickChildren(spec, { level: 0, t: 0, bz: 0, by: 0, bx: 0 }), []);
  // a clamped case: 3-voxel-grid finer level -> some children fall off
  const odd: PyramidSpec = { ...spec, shape0: [192, 192, 192] }; // L0 grid 3, L1 grid 2, L2 grid 1
  const k2 = brickChildren(odd, { level: 1, t: 0, bz: 1, by: 1, bx: 1 });
  // child indices 2,3 along each axis; grid at L0 is ceil(192/64)=3 -> idx 3 culled, idx 2 kept
  assert.ok(k2.length < 8);
  assert.ok(k2.every((k) => k.bz < 3 && k.by < 3 && k.bx < 3));
});

test("brickKey is stable + collision-free across levels and time", () => {
  assert.equal(brickKey({ level: 1, t: 0, bz: 2, by: 3, bx: 4 }), "1/0/2/3/4");
  assert.notEqual(
    brickKey({ level: 0, t: 0, bz: 1, by: 1, bx: 1 }),
    brickKey({ level: 1, t: 0, bz: 1, by: 1, bx: 1 }),
  );
});

test("shouldStream: large pyramided volumes stream; small / no-pyramid stay resident", () => {
  // 256³ at default ceiling 256³ -> not over voxel ceiling, not over axis -> resident
  assert.equal(shouldStream(spec), false);
  // 512³ exceeds both -> stream
  assert.equal(shouldStream({ ...spec, shape0: [512, 512, 512], levels: 4 }), true);
  // big but only 1 level -> cannot stream, must stay resident
  assert.equal(shouldStream({ ...spec, shape0: [512, 512, 512], levels: 1 }), false);
  // axis over the WebGL2 floor even if voxel count modest
  assert.equal(shouldStream({ ...spec, shape0: [300, 8, 8], levels: 2 }), true);
});
