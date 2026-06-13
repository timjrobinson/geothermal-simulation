// Terrain surface model + pure surface-grid → mesh geometry math (doc 06 §6, doc 01 §6).
//
// The viewer's world space IS the Engineering Frame (Z-up ENU metres, doc 06 §2.1). The
// terrain is the GROUND SURFACE in that frame: a grid of (X,Y) samples each carrying an
// Engineering ELEVATION (Z). Subsurface volumes hang BENEATH it naturally because they
// share the frame and Z is elevation (doc 06 §6.1). Doc 01 §6 stores the DEM (Copernicus
// GLO-30) already converted to Engineering elevation as such a surface grid, so it lands
// in the scene with NO reprojection at render time.
//
// Two sources of a surface grid (doc 01 §6 surfaceModel):
//   • "flat:<z>"        — a flat surface at constant Engineering elevation z (local mode).
//   • "synthetic:<id>"  — a deterministic synthetic relief surface (the data generator's
//                         local scenarios) — relief computed from id as a stable seed.
//   • "dem:<provider>"  — a georeferenced DEM surface grid (Copernicus GLO-30) supplied as
//                         explicit elevation data (the DEM fetch itself is a backend
//                         follow-up; we render whatever grid data is provided + flat
//                         fallback, per doc 06 §6.1).
//
// Everything here is PURE (no THREE / no DOM) so the grid→mesh vertex/Z math is
// unit-testable headlessly. scene/TerrainLayer.tsx turns the SurfaceGrid this produces
// into a THREE.BufferGeometry (and optional draped basemap UVs, doc 06 §6.2).

// A regular surface grid in the Engineering Frame. Elevation is row-major (row j over Y,
// col i over X): elevation[j * nx + i] is the Engineering Z of sample (i, j). NaN marks a
// no-data cell (e.g. outside DEM coverage) — callers/shaders skip it.
export interface SurfaceGrid {
  nx: number; // sample count along X (East)
  ny: number; // sample count along Y (North)
  x0: number; // Engineering X of column 0 (metres)
  y0: number; // Engineering Y of row 0 (metres)
  dx: number; // X spacing (metres)
  dy: number; // Y spacing (metres)
  elevation: Float32Array; // length nx*ny, row-major (Y outer, X inner), Engineering Z (m)
}

// Parsed surfaceModel spec (doc 01 §2 SpatialFrame.surfaceModel string).
export type SurfaceModel =
  | { kind: "flat"; z: number }
  | { kind: "synthetic"; id: string }
  | { kind: "dem"; provider: string };

// Parse a doc-01 surfaceModel string ("flat:0" | "synthetic:<id>" | "dem:copernicus-30m").
// Tolerant: an unrecognized/empty/null string falls back to a flat:0 surface so the viewer
// always has a ground plane.
export function parseSurfaceModel(spec: string | null | undefined): SurfaceModel {
  if (!spec) return { kind: "flat", z: 0 };
  const idx = spec.indexOf(":");
  const kind = (idx >= 0 ? spec.slice(0, idx) : spec).trim().toLowerCase();
  const rest = idx >= 0 ? spec.slice(idx + 1) : "";
  if (kind === "flat") {
    const z = Number.parseFloat(rest);
    return { kind: "flat", z: Number.isFinite(z) ? z : 0 };
  }
  if (kind === "synthetic") return { kind: "synthetic", id: rest || "default" };
  if (kind === "dem") return { kind: "dem", provider: rest || "copernicus-30m" };
  return { kind: "flat", z: 0 };
}

// A horizontal extent of the surface in Engineering metres (XY), typically the project ROI
// (doc 01 §2) or the union scene AABB footprint (doc 06 §2.2).
export interface XYExtent {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
}

// Deterministic, smooth synthetic relief over (x, y) (Engineering metres), in metres of
// elevation about `base`. A small sum of sinusoids seeded from a string id so the same
// scenario always yields the same surface (doc 01 §6 "a synthetic surface emitted by the
// data generator"). Pure + cheap; not physically meaningful, just plausible-looking relief.
export function syntheticElevation(
  id: string,
  x: number,
  y: number,
  opts: { base?: number; amplitude?: number; wavelength?: number } = {},
): number {
  const base = opts.base ?? 0;
  const amp = opts.amplitude ?? 120; // ±120 m of relief by default
  const wl = opts.wavelength ?? 4000; // ~4 km dominant wavelength
  const seed = hashStringToUnit(id);
  // Phase/orientation jitter from the seed so different scenarios differ but stay stable.
  const px = seed * Math.PI * 2;
  const py = (1 - seed) * Math.PI * 2;
  const k = (2 * Math.PI) / wl;
  const h =
    Math.sin(x * k + px) * Math.cos(y * k * 0.85 + py) +
    0.5 * Math.sin(x * k * 2.1 + py) * Math.sin(y * k * 1.7 + px);
  // h is in roughly [-1.5, 1.5]; normalize to ±amp.
  return base + (amp * h) / 1.5;
}

// Stable hash of a string → a unit float in [0, 1) (FNV-1a). Used to seed synthetic relief.
export function hashStringToUnit(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  // To unsigned, then to [0,1).
  return (h >>> 0) / 0x100000000;
}

// Build a SurfaceGrid for a parsed surfaceModel over an XY extent at a given resolution.
// `res` is the sample count per axis (clamped to >=2 so there is at least one quad). For
// "dem" we cannot synthesize real elevation, so we emit a flat grid at z=0 as the fallback
// (the real DEM grid is supplied directly via makeGridFromElevation when available).
export function buildSurfaceGrid(
  model: SurfaceModel,
  extent: XYExtent,
  res = 64,
): SurfaceGrid {
  const nx = Math.max(2, Math.floor(res));
  const ny = Math.max(2, Math.floor(res));
  const x0 = extent.xmin;
  const y0 = extent.ymin;
  const dx = (extent.xmax - extent.xmin) / (nx - 1);
  const dy = (extent.ymax - extent.ymin) / (ny - 1);
  const elevation = new Float32Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const y = y0 + j * dy;
    for (let i = 0; i < nx; i++) {
      const x = x0 + i * dx;
      let z: number;
      if (model.kind === "flat") z = model.z;
      else if (model.kind === "synthetic") z = syntheticElevation(model.id, x, y);
      else z = 0; // dem fallback (flat) until real elevation data is provided
      elevation[j * nx + i] = z;
    }
  }
  return { nx, ny, x0, y0, dx, dy, elevation };
}

// Wrap an explicit Engineering-elevation array (doc 01 §6 / doc 06 §6.1 — the DEM grid the
// backend has already reprojected + converted to Engineering elevation) as a SurfaceGrid.
// Throws if the array length does not match nx*ny so a malformed grid fails loudly.
export function makeGridFromElevation(
  elevation: Float32Array,
  spec: { nx: number; ny: number; x0: number; y0: number; dx: number; dy: number },
): SurfaceGrid {
  const expected = spec.nx * spec.ny;
  if (elevation.length !== expected) {
    throw new Error(
      `surface elevation length ${elevation.length} != expected ${expected} ` +
        `for grid ${spec.nx}×${spec.ny}`,
    );
  }
  return { ...spec, elevation };
}

// Engineering XY of grid sample (i, j).
export function gridSampleXY(g: SurfaceGrid, i: number, j: number): [number, number] {
  return [g.x0 + i * g.dx, g.y0 + j * g.dy];
}

// Output of gridToMesh: interleaved-ready flat arrays for a THREE.BufferGeometry.
export interface SurfaceMesh {
  positions: Float32Array; // length nx*ny*3 — (X, Y, Z) Engineering metres per vertex
  uvs: Float32Array; // length nx*ny*2 — [0,1] grid UVs (basemap drape, doc 06 §6.2)
  indices: Uint32Array; // length (nx-1)*(ny-1)*6 — two triangles per quad
}

// Turn a SurfaceGrid into mesh vertex positions + UVs + triangle indices in the Engineering
// Frame (doc 06 §6.1 "PlaneGeometry-style mesh displaced in Z"). Each grid sample becomes a
// vertex at (X, Y, elevation·verticalExaggeration). Vertical exaggeration is a RENDER-ONLY
// transform (doc 06 §2.3): we bake it into Z here for the standalone-mesh path; when the
// terrain is parented under an exaggeration-scaled scene root instead, pass vex=1.
//
// Quads whose corner elevations are all NaN (no DEM coverage) are dropped so holes read as
// holes rather than spikes; a quad with some finite corners is kept (the NaN vertex stays
// NaN and the shader/driver discards it). Pure + deterministic for unit testing.
export function gridToMesh(g: SurfaceGrid, vex = 1): SurfaceMesh {
  const { nx, ny } = g;
  const positions = new Float32Array(nx * ny * 3);
  const uvs = new Float32Array(nx * ny * 2);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const vi = j * nx + i;
      const x = g.x0 + i * g.dx;
      const y = g.y0 + j * g.dy;
      const z = g.elevation[vi] * vex;
      positions[vi * 3 + 0] = x;
      positions[vi * 3 + 1] = y;
      positions[vi * 3 + 2] = z;
      uvs[vi * 2 + 0] = nx > 1 ? i / (nx - 1) : 0;
      uvs[vi * 2 + 1] = ny > 1 ? j / (ny - 1) : 0;
    }
  }
  // Two triangles per quad; skip quads with no finite corner.
  const quads: number[] = [];
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i;
      const b = j * nx + (i + 1);
      const c = (j + 1) * nx + (i + 1);
      const d = (j + 1) * nx + i;
      const anyFinite =
        Number.isFinite(g.elevation[a]) ||
        Number.isFinite(g.elevation[b]) ||
        Number.isFinite(g.elevation[c]) ||
        Number.isFinite(g.elevation[d]);
      if (!anyFinite) continue;
      // CCW winding when viewed from +Z (above), so the surface faces up.
      quads.push(a, b, d, b, c, d);
    }
  }
  return { positions, uvs, indices: Uint32Array.from(quads) };
}

// Finite Z range of a surface grid (for camera framing / clip). Returns null if all-NaN.
export function elevationRange(g: SurfaceGrid): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < g.elevation.length; i++) {
    const v = g.elevation[i];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return min === Infinity ? null : { min, max };
}
