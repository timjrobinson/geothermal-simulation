// Brick-address math for the M2+ octree-LOD streaming renderer (doc 06 §3.4, doc 04 §5/§6,
// doc 02 §10.2). PURE — no THREE / no DOM / no fetch — so the addressing, pyramid-extent,
// and octree parent/child math is unit-testable headlessly.
//
// A brick is the doc-04 §6 unit: a 64³ Zarr chunk addressed
//   (artifact_id, property, level, t, bz, by, bx)
// which maps 1:1 onto the doc-02 §10.2 chunk path
//   <property>/<level>/c/<bz>/<by>/<bx>   (inside <datasetId>.zarr)
// so brick address == Zarr chunk key (doc 04 §6 "brick address == Zarr chunk path").
//
// Pyramid convention (doc 04 §5): level 0 = FULL resolution; each higher level halves each
// spatial axis (mean-downsampled). The COARSEST level is the largest level index and fits in
// ~1-2 bricks — it is always kept resident so the volume is never blank (doc 06 §3.4).
//
// Coordinate convention matches the rest of the viewer (lib/volume.ts, lib/data3d.ts):
// volume buffers + shapes are (z, y, x) = [nz, ny, nx]; Engineering AABBs are XYZ metres,
// Z-up. Brick indices follow the same ordering: (bz, by, bx).

import type { AABB, Shape3, Vec3ZYX } from "./volume";

// Canonical brick edge length in voxels (doc 04 §4.2 / §6 — 64³ cubic chunks).
export const BRICK_SIZE = 64;

// A brick's octree address. `t` is the time index (0 for static volumes, doc 04 §6).
export interface BrickAddress {
  level: number; // pyramid level (0 = finest / full-res, larger = coarser)
  t: number; // time index
  bz: number;
  by: number;
  bx: number;
}

// A volume's pyramid descriptor: the level-0 (full-res) shape + spacing + origin (doc 04
// §9.2 meta), the number of levels, and the brick edge. Everything else (per-level shapes,
// per-level brick grids) is DERIVED from these so there is one source of truth.
export interface PyramidSpec {
  shape0: Shape3; // level-0 [nz, ny, nx]
  origin: Vec3ZYX; // level-0 [oz, oy, ox] metres (shared across levels)
  spacing0: Vec3ZYX; // level-0 [dz, dy, dx] metres
  levels: number; // pyramid_levels (doc 04 §5)
  brick?: number; // brick edge in voxels (default BRICK_SIZE)
}

// Brick edge for a spec (defaults to BRICK_SIZE).
export function brickEdge(spec: PyramidSpec): number {
  return spec.brick && spec.brick > 0 ? spec.brick : BRICK_SIZE;
}

// The voxel shape of a pyramid level. Each level halves each axis (doc 04 §5: mean
// downsampling, ×⅛ voxels per level), floor-rounded but never below 1 — matching a
// power-of-two pyramid where odd extents round down per skimage/numpy block-reduce.
export function levelShape(spec: PyramidSpec, level: number): Shape3 {
  const f = 1 << level; // 2^level
  const [nz, ny, nx] = spec.shape0;
  return [
    Math.max(1, Math.floor(nz / f)),
    Math.max(1, Math.floor(ny / f)),
    Math.max(1, Math.floor(nx / f)),
  ];
}

// The voxel SPACING (metres) of a level. Coarser levels have proportionally larger voxels
// (level L voxel covers 2^L level-0 voxels along each axis) — so the Engineering footprint
// of the whole level is (approximately) preserved across the pyramid.
export function levelSpacing(spec: PyramidSpec, level: number): Vec3ZYX {
  const f = 1 << level;
  const [dz, dy, dx] = spec.spacing0;
  return [dz * f, dy * f, dx * f];
}

// Number of bricks along each axis at a level (ceil — a partial edge brick still exists).
// Returned (z, y, x) to match the (bz, by, bx) address ordering.
export function levelBrickGrid(spec: PyramidSpec, level: number): Shape3 {
  const b = brickEdge(spec);
  const [nz, ny, nx] = levelShape(spec, level);
  return [Math.ceil(nz / b), Math.ceil(ny / b), Math.ceil(nx / b)];
}

// Total brick count at a level.
export function levelBrickCount(spec: PyramidSpec, level: number): number {
  const [gz, gy, gx] = levelBrickGrid(spec, level);
  return gz * gy * gx;
}

// The COARSEST level index (largest level index, smallest extent) — always kept resident so
// the volume is never blank (doc 06 §3.4 coarse-first). With `levels` levels, indices are
// 0..levels-1 and the coarsest is levels-1.
export function coarsestLevel(spec: PyramidSpec): number {
  return Math.max(0, spec.levels - 1);
}

// Build a brick address. Validates indices against the level brick grid; throws on OOB so a
// mis-derived request fails loudly rather than fetching a 404 silently.
export function brickAddress(
  spec: PyramidSpec,
  level: number,
  bz: number,
  by: number,
  bx: number,
  t = 0,
): BrickAddress {
  if (level < 0 || level >= spec.levels) {
    throw new Error(`brick level ${level} out of range [0, ${spec.levels})`);
  }
  const [gz, gy, gx] = levelBrickGrid(spec, level);
  if (bz < 0 || bz >= gz || by < 0 || by >= gy || bx < 0 || bx >= gx) {
    throw new Error(
      `brick (${bz},${by},${bx}) out of grid (${gz},${gy},${gx}) at level ${level}`,
    );
  }
  return { level, t, bz, by, bx };
}

// The Zarr chunk path for a brick == its doc-04 §6 address (doc 02 §10.2):
//   <property>/<level>/c/<bz>/<by>/<bx>
// This is the path component appended to GET /property-models/{id}/zarr/<...>. Time is the
// LEADING chunk index when t-bearing; for static M1/M2 volumes t==0 and is omitted (doc 04
// §6 — chunk grid is (z,y,x) for static, (t,z,y,x) when 4D). `withTime` forces the t index.
export function brickChunkPath(
  property: string,
  a: BrickAddress,
  withTime = false,
): string {
  const idx = withTime
    ? [a.t, a.bz, a.by, a.bx]
    : [a.bz, a.by, a.bx];
  return `${property}/${a.level}/c/${idx.join("/")}`;
}

// Full relative URL for fetching a brick via the Zarr-over-HTTP passthrough (doc 04 §9.2,
// backend GET /property-models/{id}/zarr/{path}). The chunk bytes are Blosc+zstd-encoded;
// when browser-Blosc is not wired (doc 06 §1.3 SPIKE) callers use brickVolumeUrl instead.
export function brickZarrUrl(
  id: string,
  property: string,
  a: BrickAddress,
  withTime = false,
): string {
  return `/property-models/${encodeURIComponent(id)}/zarr/${brickChunkPath(
    property,
    a,
    withTime,
  )}`;
}

// Server-decoded fallback URL (doc 06 §1.3): fetch the WHOLE decoded level as raw LE f32 via
// GET /property-models/{id}/volume?level=L&property=P, then slice bricks out of it
// client-side (see sliceBrick). This sidesteps browser-Blosc — the on-disk layout is
// unaffected, only WHO decodes changes (doc 06 §1.3). Used by the worker.
export function levelVolumeUrl(id: string, property: string, level: number): string {
  return (
    `/property-models/${encodeURIComponent(id)}/volume` +
    `?level=${level}&property=${encodeURIComponent(property)}`
  );
}

// The Engineering-frame AABB (XYZ metres) covered by a brick (doc 06 §2/§3.4). The brick
// spans voxels [b*edge, (b+1)*edge) at its level, clamped to the level extent; the box runs
// from the first voxel's near face to the last voxel's far face (half-voxel padding, exactly
// like engineeringAABB in volume.ts). Origin is shared across levels; spacing scales by 2^L.
export function brickAABB(spec: PyramidSpec, a: BrickAddress): AABB {
  const b = brickEdge(spec);
  const [nz, ny, nx] = levelShape(spec, a.level);
  const [dz, dy, dx] = levelSpacing(spec, a.level);
  const [oz, oy, ox] = spec.origin;

  const k0 = a.bz * b;
  const j0 = a.by * b;
  const i0 = a.bx * b;
  const k1 = Math.min(k0 + b, nz); // exclusive voxel end, clamped to extent
  const j1 = Math.min(j0 + b, ny);
  const i1 = Math.min(i0 + b, nx);

  // Voxel centre of (i0,j0,k0) is origin + idx*spacing; near face is -half a voxel.
  const minX = ox + i0 * dx - dx / 2;
  const minY = oy + j0 * dy - dy / 2;
  const minZ = oz + k0 * dz - dz / 2;
  const maxX = ox + (i1 - 1) * dx + dx / 2;
  const maxY = oy + (j1 - 1) * dy + dy / 2;
  const maxZ = oz + (k1 - 1) * dz + dz / 2;
  return {
    min: [Math.min(minX, maxX), Math.min(minY, maxY), Math.min(minZ, maxZ)],
    max: [Math.max(minX, maxX), Math.max(minY, maxY), Math.max(minZ, maxZ)],
  };
}

// The whole-volume Engineering AABB (level-independent — every level covers the same footprint
// up to half-voxel padding). Computed from level 0 for the tightest box.
export function volumeAABB(spec: PyramidSpec): AABB {
  const [nz, ny, nx] = spec.shape0;
  const [dz, dy, dx] = spec.spacing0;
  const [oz, oy, ox] = spec.origin;
  return {
    min: [ox - dx / 2, oy - dy / 2, oz - dz / 2],
    max: [
      ox + (nx - 1) * dx + dx / 2,
      oy + (ny - 1) * dy + dy / 2,
      oz + (nz - 1) * dz + dz / 2,
    ],
  };
}

// Octree parent of a brick: the coarser-level node that contains it (doc 04 §6 octree
// compatibility — node (level, bz,by,bx) has children (level-1, 2b{xyz}+{0,1})). The parent
// is at level+1 with halved indices. Returns null if already at the coarsest level.
export function brickParent(spec: PyramidSpec, a: BrickAddress): BrickAddress | null {
  if (a.level >= coarsestLevel(spec)) return null;
  return {
    level: a.level + 1,
    t: a.t,
    bz: a.bz >> 1,
    by: a.by >> 1,
    bx: a.bx >> 1,
  };
}

// Octree children of a brick (the finer-level nodes it contains): up to 8, clamped to the
// finer level's brick grid (edge bricks at non-power-of-two extents have fewer children).
// Returns [] if already at the finest level (0).
export function brickChildren(spec: PyramidSpec, a: BrickAddress): BrickAddress[] {
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

// A stable string key for a brick address — the cache/page-table/LRU key. Includes t so 4D
// frames do not collide. Mirrors the chunk path ordering for easy debugging.
export function brickKey(a: BrickAddress): string {
  return `${a.level}/${a.t}/${a.bz}/${a.by}/${a.bx}`;
}

// Decide whether a volume should use the streaming (brick) path or the M1 single-resident
// fast path (doc 06 §1.3 / §3.4). Small volumes that fit one upload stay single-resident; a
// volume is "large" when its level-0 voxel count exceeds `maxResidentVoxels` OR any axis
// exceeds the WebGL2-guaranteed MAX_3D_TEXTURE_SIZE floor (256, doc 06 §3.4) and a pyramid
// exists to stream from. Default ceiling ~256³ ≈ 16.7M voxels (the single-resident budget).
export function shouldStream(
  spec: PyramidSpec,
  maxResidentVoxels = 256 * 256 * 256,
  maxAxis = 256,
): boolean {
  if (spec.levels <= 1) return false; // no pyramid to stream — must stay resident
  const [nz, ny, nx] = spec.shape0;
  const voxels = nz * ny * nx;
  return voxels > maxResidentVoxels || nz > maxAxis || ny > maxAxis || nx > maxAxis;
}
