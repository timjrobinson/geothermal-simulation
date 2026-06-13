// Unit tests for the fixed-VRAM brick pool (LRU) + page-table indexing + atlas slot math
// (doc 06 §3.4, §7.2, §7.5). Pure — no THREE / no DOM — runnable headlessly via `npm test`.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  chooseAtlasGrid,
  atlasCapacity,
  atlasDims,
  slotToCoord,
  slotVoxelOrigin,
  slotUVWTransform,
  BrickPool,
  pageIndex,
  makePageTable,
  setPage,
  getPage,
  fillPageTable,
  type AtlasLayout,
} from "./brickPool";

test("chooseAtlasGrid gives a near-cube grid covering the capacity", () => {
  assert.deepEqual(chooseAtlasGrid(8), [2, 2, 2]);
  const g = chooseAtlasGrid(256);
  assert.ok(g[0] * g[1] * g[2] >= 256);
  // near-cube: axes differ by at most 1
  assert.ok(Math.max(...g) - Math.min(...g) <= 1);
  // non-cube count still covered
  const g2 = chooseAtlasGrid(10);
  assert.ok(g2[0] * g2[1] * g2[2] >= 10);
});

const layout: AtlasLayout = { brickEdge: 64, grid: [2, 2, 2] }; // 8 slots

test("atlasCapacity + atlasDims derive from the grid + edge", () => {
  assert.equal(atlasCapacity(layout), 8);
  assert.deepEqual(atlasDims(layout), [128, 128, 128]); // 2*64
});

test("slotToCoord / slotVoxelOrigin index the 3D slot grid (x fastest)", () => {
  assert.deepEqual(slotToCoord(layout, 0), [0, 0, 0]);
  assert.deepEqual(slotToCoord(layout, 1), [1, 0, 0]); // x fastest
  assert.deepEqual(slotToCoord(layout, 2), [0, 1, 0]); // then y
  assert.deepEqual(slotToCoord(layout, 4), [0, 0, 1]); // then z
  // slot 5 = sx=5%2=1, sy=floor(5/2)%2=0, sz=floor(5/4)=1 -> (1,0,1)
  assert.deepEqual(slotToCoord(layout, 5), [1, 0, 1]);
  assert.deepEqual(slotVoxelOrigin(layout, 5), [64, 0, 64]);
});

test("slotUVWTransform maps brick-local [0,1]^3 into the slot's atlas sub-cube", () => {
  const { origin, scale } = slotUVWTransform(layout, 0);
  assert.deepEqual(origin, [0, 0, 0]);
  assert.deepEqual(scale, [0.5, 0.5, 0.5]); // 64/128
  const t5 = slotUVWTransform(layout, 5); // coord (1,0,1)
  assert.deepEqual(t5.origin, [0.5, 0, 0.5]);
  // a brick-local centre (0.5,0.5,0.5) lands at slot centre
  const center = [
    t5.origin[0] + 0.5 * t5.scale[0],
    t5.origin[1] + 0.5 * t5.scale[1],
    t5.origin[2] + 0.5 * t5.scale[2],
  ];
  assert.deepEqual(center, [0.75, 0.25, 0.75]);
});

test("BrickPool admits up to capacity, then evicts the LRU unpinned brick", () => {
  const pool = new BrickPool(layout); // 8 slots
  const slots = new Set<number>();
  for (let i = 0; i < 8; i++) {
    const e = pool.admit(`b${i}`);
    slots.add(e.slot);
  }
  assert.equal(pool.size(), 8);
  assert.equal(slots.size, 8); // all distinct slots
  // touch b1..b7 so b0 is the LRU; admitting a 9th evicts b0 and reuses its slot
  for (let i = 1; i < 8; i++) pool.touch(`b${i}`);
  const b0Slot = pool.get("b0")!.slot;
  const e8 = pool.admit("b8");
  assert.equal(pool.has("b0"), false, "LRU b0 evicted");
  assert.equal(e8.slot, b0Slot, "freed slot reused");
  assert.equal(pool.size(), 8);
});

test("BrickPool never evicts pinned (coarsest) bricks", () => {
  const pool = new BrickPool(layout);
  for (let i = 0; i < 8; i++) pool.admit(`pin${i}`, true); // all pinned
  assert.throws(() => pool.admit("extra"), /all .* bricks pinned/);
  // a mix: pin 4, fill the rest, evictions only hit the unpinned
  const p2 = new BrickPool(layout);
  for (let i = 0; i < 4; i++) p2.admit(`pin${i}`, true);
  for (let i = 0; i < 4; i++) p2.admit(`tmp${i}`);
  p2.admit("more"); // evicts an unpinned tmp, never a pin
  assert.ok([0, 1, 2, 3].every((i) => p2.has(`pin${i}`)));
});

test("admit on an existing key touches it without consuming a new slot", () => {
  const pool = new BrickPool(layout);
  const a = pool.admit("k");
  const b = pool.admit("k");
  assert.equal(a.slot, b.slot);
  assert.equal(pool.size(), 1);
  // admitting with pinned=true upgrades the entry
  pool.admit("k", true);
  assert.equal(pool.get("k")!.pinned, true);
});

test("evictExcept releases unwanted unpinned bricks, keeps pinned + wanted", () => {
  const pool = new BrickPool(layout);
  pool.admit("pin", true);
  pool.admit("keep");
  pool.admit("drop1");
  pool.admit("drop2");
  const evicted = pool.evictExcept(new Set(["keep"]));
  assert.deepEqual(evicted.sort(), ["drop1", "drop2"]);
  assert.ok(pool.has("pin") && pool.has("keep"));
  assert.equal(pool.size(), 2);
});

test("clear empties the pool and frees all slots", () => {
  const pool = new BrickPool(layout);
  for (let i = 0; i < 5; i++) pool.admit(`b${i}`);
  pool.clear();
  assert.equal(pool.size(), 0);
  // can refill to capacity again
  for (let i = 0; i < 8; i++) pool.admit(`c${i}`);
  assert.equal(pool.size(), 8);
});

test("vramBytes scales with resident brick count", () => {
  const pool = new BrickPool(layout);
  assert.equal(pool.vramBytes(), 0);
  pool.admit("a");
  assert.equal(pool.vramBytes(), 64 * 64 * 64 * 4); // one 64^3 f32 brick
});

// ── Page table ──────────────────────────────────────────────────────────────────────────

test("pageIndex linearizes (bz,by,bx) within a level grid (x fastest)", () => {
  const grid: [number, number, number] = [2, 3, 4]; // [gz,gy,gx]
  assert.equal(pageIndex(grid, 0, 0, 0), 0);
  assert.equal(pageIndex(grid, 0, 0, 1), 1); // x fastest
  assert.equal(pageIndex(grid, 0, 1, 0), 4); // +gx
  assert.equal(pageIndex(grid, 1, 0, 0), 12); // +gy*gx
});

test("makePageTable blocks levels contiguously and inits to -1", () => {
  const grids: Array<[number, number, number]> = [
    [4, 4, 4], // level 0: 64 bricks
    [2, 2, 2], // level 1: 8 bricks
    [1, 1, 1], // level 2: 1 brick
  ];
  const pt = makePageTable(grids);
  assert.equal(pt.data.length, 64 + 8 + 1);
  assert.deepEqual(pt.levelOffset, [0, 64, 72]);
  assert.ok(pt.data.every((v) => v === -1));
});

test("setPage / getPage round-trip per level + offset", () => {
  const grids: Array<[number, number, number]> = [
    [4, 4, 4],
    [2, 2, 2],
    [1, 1, 1],
  ];
  const pt = makePageTable(grids);
  setPage(pt, 0, 1, 2, 3, 42);
  setPage(pt, 1, 0, 0, 1, 7);
  setPage(pt, 2, 0, 0, 0, 99);
  assert.equal(getPage(pt, 0, 1, 2, 3), 42);
  assert.equal(getPage(pt, 1, 0, 0, 1), 7);
  assert.equal(getPage(pt, 2, 0, 0, 0), 99);
  assert.equal(getPage(pt, 0, 0, 0, 0), -1); // untouched stays -1
  // level-2 brick is the LAST entry (offset 72)
  assert.equal(pt.data[72], 99);
});

test("fillPageTable rebuilds the table from pool entries (key-encoded)", () => {
  const grids: Array<[number, number, number]> = [
    [4, 4, 4],
    [2, 2, 2],
    [1, 1, 1],
  ];
  const pt = makePageTable(grids);
  fillPageTable(pt, [
    { key: "0/0/1/2/3", slot: 5, pinned: false, lastUsed: 1 },
    { key: "2/0/0/0/0", slot: 0, pinned: true, lastUsed: 2 },
  ]);
  assert.equal(getPage(pt, 0, 1, 2, 3), 5);
  assert.equal(getPage(pt, 2, 0, 0, 0), 0);
  // a second fill clears stale entries
  fillPageTable(pt, [{ key: "2/0/0/0/0", slot: 0, pinned: true, lastUsed: 3 }]);
  assert.equal(getPage(pt, 0, 1, 2, 3), -1);
  assert.equal(getPage(pt, 2, 0, 0, 0), 0);
});
