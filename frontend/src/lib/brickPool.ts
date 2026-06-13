// Fixed-VRAM brick-pool atlas + page-table indexing + LRU eviction (doc 06 §3.4, §7.2,
// §7.5). The POOL LOGIC here is PURE (no THREE / no DOM) so the slot allocation, LRU
// eviction, page-table indexing and atlas-coordinate math are unit-testable headlessly; the
// actual GPU textures (the Data3DTexture atlas + the page-table texture) are created and
// uploaded by scene/StreamingVolumeLayer from this bookkeeping.
//
// Model (doc 06 §3.4):
//   - ONE atlas Data3DTexture holds N bricks of `brickEdge`³ packed in a 3D grid of slots
//     (atlasGrid = [sx, sy, sz] slots per axis). Fixed VRAM regardless of full volume size.
//   - A page-table maps a brick KEY (level/t/bz/by/bx) -> an atlas slot index. On the GPU the
//     page table is a small texture indexed per-level; here we just track key -> slot and the
//     slot's atlas coordinates so the shader (and tests) can resolve a sample.
//   - LRU: when full, the least-recently-used resident brick is evicted to make room. The
//     COARSEST level is pinned (never evicted) so the volume is never blank (doc 06 §3.4).
//
// "Recently used" is touched every frame for bricks the LOD selection still wants; pinned
// bricks are excluded from eviction candidacy.

// A resident brick's bookkeeping entry.
export interface PoolEntry {
  key: string; // brickKey(addr)
  slot: number; // atlas slot index [0, capacity)
  pinned: boolean; // coarsest-level bricks are pinned (never evicted)
  lastUsed: number; // monotonic tick of last touch (LRU ordering)
}

// Atlas geometry: a cube-ish grid of slots, each holding one brick of `brickEdge`³ voxels.
export interface AtlasLayout {
  brickEdge: number; // voxels per brick edge (BRICK_SIZE)
  grid: [number, number, number]; // slots per axis [sx, sy, sz]
}

// Choose an atlas slot grid for a desired capacity: the smallest near-cube grid whose product
// is >= capacity (keeps the atlas texture roughly cubic, friendly to MAX_3D_TEXTURE_SIZE).
export function chooseAtlasGrid(capacity: number): [number, number, number] {
  const n = Math.max(1, Math.ceil(Math.cbrt(capacity)));
  // Grow one axis at a time until the product covers capacity (handles non-cube counts).
  let sx = n;
  let sy = n;
  let sz = n;
  while (sx * sy * sz < capacity) {
    if (sx <= sy && sx <= sz) sx++;
    else if (sy <= sz) sy++;
    else sz++;
  }
  return [sx, sy, sz];
}

// Total slot capacity of a layout.
export function atlasCapacity(layout: AtlasLayout): number {
  return layout.grid[0] * layout.grid[1] * layout.grid[2];
}

// Atlas texture dimensions in voxels (width=sx*edge, height=sy*edge, depth=sz*edge).
export function atlasDims(layout: AtlasLayout): [number, number, number] {
  const e = layout.brickEdge;
  return [layout.grid[0] * e, layout.grid[1] * e, layout.grid[2] * e];
}

// Slot index -> integer slot coordinate [sx, sy, sz] in the atlas grid (x fastest).
export function slotToCoord(layout: AtlasLayout, slot: number): [number, number, number] {
  const [gx, gy] = layout.grid;
  const sx = slot % gx;
  const sy = Math.floor(slot / gx) % gy;
  const sz = Math.floor(slot / (gx * gy));
  return [sx, sy, sz];
}

// Slot index -> the atlas VOXEL origin [x0, y0, z0] of that slot (top-left-near corner).
export function slotVoxelOrigin(layout: AtlasLayout, slot: number): [number, number, number] {
  const [sx, sy, sz] = slotToCoord(layout, slot);
  const e = layout.brickEdge;
  return [sx * e, sy * e, sz * e];
}

// Slot index -> the atlas NORMALIZED [0,1]^3 origin of that slot (what a sampler3D needs).
// The brick occupies [origin, origin + edge/dim] in atlas UVW. Returned as { origin, scale }
// where a brick-local uvw in [0,1]^3 maps to atlas uvw = origin + uvw*scale.
export function slotUVWTransform(
  layout: AtlasLayout,
  slot: number,
): { origin: [number, number, number]; scale: [number, number, number] } {
  const [x0, y0, z0] = slotVoxelOrigin(layout, slot);
  const [dx, dy, dz] = atlasDims(layout);
  const e = layout.brickEdge;
  return {
    origin: [x0 / dx, y0 / dy, z0 / dz],
    scale: [e / dx, e / dy, e / dz],
  };
}

// The LRU brick pool. Tracks key -> entry, a free-slot list, and a monotonic clock. Pinned
// (coarsest) bricks are admitted normally but never offered for eviction.
export class BrickPool {
  readonly layout: AtlasLayout;
  readonly capacity: number;
  private entries = new Map<string, PoolEntry>();
  private free: number[];
  private clock = 0;

  constructor(layout: AtlasLayout) {
    this.layout = layout;
    this.capacity = atlasCapacity(layout);
    this.free = Array.from({ length: this.capacity }, (_, i) => i);
  }

  size(): number {
    return this.entries.size;
  }

  // Bookkeeping VRAM used by resident bricks in bytes (f32 atlas), for the store readout.
  vramBytes(): number {
    const e = this.layout.brickEdge;
    return this.entries.size * e * e * e * 4;
  }

  has(key: string): boolean {
    return this.entries.has(key);
  }

  get(key: string): PoolEntry | undefined {
    return this.entries.get(key);
  }

  // Advance the LRU clock and mark a resident brick as used this frame (so the LOD-wanted set
  // survives eviction). No-op if not resident. Returns the entry (or undefined).
  touch(key: string): PoolEntry | undefined {
    const e = this.entries.get(key);
    if (e) e.lastUsed = ++this.clock;
    return e;
  }

  // Tick the clock once per frame so a batch of touches in one frame share a monotonic order.
  tick(): number {
    return ++this.clock;
  }

  // Admit a brick, returning the slot it now occupies. If already resident, just touches it.
  // If the pool is full, evicts the least-recently-used UNPINNED brick; throws only if every
  // slot is pinned (a misconfiguration — the coarsest level alone must fit the pool).
  admit(key: string, pinned = false): PoolEntry {
    const existing = this.entries.get(key);
    if (existing) {
      existing.lastUsed = ++this.clock;
      if (pinned) existing.pinned = true;
      return existing;
    }
    let slot = this.free.pop();
    if (slot === undefined) {
      slot = this.evictLRU();
    }
    const entry: PoolEntry = { key, slot, pinned, lastUsed: ++this.clock };
    this.entries.set(key, entry);
    return entry;
  }

  // Evict the least-recently-used unpinned brick and return its freed slot. Throws if none
  // are evictable (all pinned). Caller-internal.
  private evictLRU(): number {
    let victim: PoolEntry | null = null;
    for (const e of this.entries.values()) {
      if (e.pinned) continue;
      if (!victim || e.lastUsed < victim.lastUsed) victim = e;
    }
    if (!victim) {
      throw new Error(
        `BrickPool full and all ${this.entries.size} bricks pinned — atlas too small for ` +
          `the pinned (coarsest) set`,
      );
    }
    this.entries.delete(victim.key);
    return victim.slot;
  }

  // Explicitly evict everything not in `keep` (and not pinned). Used to release a frame's
  // unwanted bricks proactively when the working set shrinks. Returns evicted keys.
  evictExcept(keep: Set<string>): string[] {
    const evicted: string[] = [];
    for (const e of [...this.entries.values()]) {
      if (e.pinned || keep.has(e.key)) continue;
      this.entries.delete(e.key);
      this.free.push(e.slot);
      evicted.push(e.key);
    }
    return evicted;
  }

  // Snapshot of resident entries (for the page-table build + debugging readout).
  list(): PoolEntry[] {
    return [...this.entries.values()];
  }

  // Reset to empty (e.g. on WEBGL_lose_context rebuild, doc 06 §7.5).
  clear(): void {
    this.entries.clear();
    this.free = Array.from({ length: this.capacity }, (_, i) => i);
    this.clock = 0;
  }
}

// ── Page table ───────────────────────────────────────────────────────────────────────────
// The page table maps a brick KEY -> the atlas slot holding it. On the GPU it is encoded as a
// per-level lookup texture; here we expose the resolved mapping the shader walks: given a
// world sample's (level, bz, by, bx) the shader reads the page table to find the slot, then
// the slot's atlas UVW transform to sample the atlas. We keep the indexing pure + testable.

// Linear page-table index for a brick within its level grid (x fastest, then y, then z) — the
// texel offset into that level's page-table block. `grid` is the level brick grid [gz,gy,gx].
export function pageIndex(
  grid: readonly [number, number, number],
  bz: number,
  by: number,
  bx: number,
): number {
  const [, gy, gx] = grid;
  return (bz * gy + by) * gx + bx;
}

// A flat page table across ALL levels: a single Int32Array where each level occupies a
// contiguous block (sized by its brick grid), holding the atlas slot for each brick or -1 if
// not resident. `levelGrids[level]` = [gz,gy,gx]. This is the CPU mirror the GPU texture
// encodes; the shader reads slot = pageTable[levelOffset[level] + pageIndex(...)].
export interface PageTable {
  data: Int32Array; // slot index per brick, -1 == not resident
  levelOffset: number[]; // start index of each level's block
  levelGrids: Array<readonly [number, number, number]>; // [gz,gy,gx] per level
}

// Build an empty page table (all -1) sized for the given per-level brick grids (index ==
// level). Total size is the sum of each level's brick count.
export function makePageTable(
  levelGrids: Array<readonly [number, number, number]>,
): PageTable {
  const levelOffset: number[] = [];
  let total = 0;
  for (const g of levelGrids) {
    levelOffset.push(total);
    total += g[0] * g[1] * g[2];
  }
  const data = new Int32Array(total).fill(-1);
  return { data, levelOffset, levelGrids };
}

// Set a brick's resident slot in the page table (-1 to clear).
export function setPage(
  pt: PageTable,
  level: number,
  bz: number,
  by: number,
  bx: number,
  slot: number,
): void {
  const idx = pt.levelOffset[level] + pageIndex(pt.levelGrids[level], bz, by, bx);
  pt.data[idx] = slot;
}

// Look up a brick's resident slot (-1 if not resident). The shader uses this to decide
// whether to sample the atlas at this level or fall back to a coarser level (no holes).
export function getPage(
  pt: PageTable,
  level: number,
  bz: number,
  by: number,
  bx: number,
): number {
  const idx = pt.levelOffset[level] + pageIndex(pt.levelGrids[level], bz, by, bx);
  return pt.data[idx];
}

// Rebuild the page-table data from a pool's resident entries. `parseKey` turns a brick key
// back into its (level,bz,by,bx); we keep that decode here so the pool stays address-agnostic.
export function fillPageTable(pt: PageTable, entries: PoolEntry[]): void {
  pt.data.fill(-1);
  for (const e of entries) {
    const [level, , bz, by, bx] = e.key.split("/").map(Number);
    if (level < 0 || level >= pt.levelGrids.length) continue;
    setPage(pt, level, bz, by, bx, e.slot);
  }
}
