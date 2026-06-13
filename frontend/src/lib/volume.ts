// Volume decode + Engineering-frame geometry helpers (doc 06 §1.3, §2, §3.1).
//
// The M1 single-resident path (doc 06 §1.3): the server DECODES a Zarr level into a
// contiguous little-endian float32 (z, y, x) buffer (NaN no-data) via
// GET /property-models/{id}/volume — sidestepping browser Zarr/Blosc, which the doc 06
// §1.3 SPIKE defers to M2 (browser-Blosc not relied on for M1). This module turns that
// ArrayBuffer into a Float32Array of the right shape and computes the volume's
// Engineering-frame axis-aligned bounding box (Z-up ENU metres, doc 06 §2).
//
// Everything here is pure (no THREE / no DOM) so it is unit-testable headlessly.

// (z, y, x) sample counts of a level, matching the backend X-Volume-Shape header
// and PropertyModelMeta.shape (doc 04 §9.2 — level-0 (z, y, x)).
export type Shape3 = readonly [number, number, number]; // [nz, ny, nx]

// Engineering-frame origin / spacing in metres, matching X-Volume-Origin/Spacing and
// PropertyModelMeta.origin/spacing. Ordered (z, y, x) to match the (z, y, x) buffer.
export type Vec3ZYX = readonly [number, number, number];

export interface VolumeMeta {
  shape: Shape3; // [nz, ny, nx]
  origin: Vec3ZYX; // [oz, oy, ox] metres
  spacing: Vec3ZYX; // [dz, dy, dx] metres
}

export interface DecodedVolume extends VolumeMeta {
  data: Float32Array; // length nz*ny*nx, C-contiguous z-major (x fastest)
}

// Voxel count of a (z, y, x) shape.
export function voxelCount(shape: Shape3): number {
  return shape[0] * shape[1] * shape[2];
}

// Decode the raw /volume body (little-endian f32, C-contiguous (z,y,x)) into a typed
// array (doc 06 §1.3). NaNs in the buffer are the no-data sentinel (doc 02 §10.2) and
// are preserved verbatim — the shader skips them. Throws if the byte length does not
// match the declared shape so a truncated/garbled response fails loudly.
//
// NOTE little-endianness: the wire format is explicitly little-endian (X-Volume-Byte-Order:
// little). All target browsers run on little-endian hardware, so a direct Float32Array
// view is correct; we assert this rather than silently mis-decoding on a big-endian host.
export function decodeVolume(buffer: ArrayBuffer, meta: VolumeMeta): DecodedVolume {
  const n = voxelCount(meta.shape);
  const expectedBytes = n * 4;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(
      `volume byte length ${buffer.byteLength} != expected ${expectedBytes} ` +
        `for shape [${meta.shape.join(", ")}]`,
    );
  }
  if (!isLittleEndian()) {
    // Defensive: byte-swap into a fresh buffer on the (vanishingly rare) BE host.
    const view = new DataView(buffer);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = view.getFloat32(i * 4, true);
    return { ...meta, data };
  }
  return { ...meta, data: new Float32Array(buffer) };
}

// Returns true if the host is little-endian (true on every browser-target platform).
export function isLittleEndian(): boolean {
  const probe = new Uint16Array([0x0102]);
  return new Uint8Array(probe.buffer)[0] === 0x02;
}

// The volume's Engineering-frame AABB (doc 06 §2/§3.1). The buffer is sample-centred:
// sample (k, j, i) sits at Engineering XYZ = origin + (i, j, k)·spacing. The proxy box
// the ray-marcher draws spans from the first sample centre minus half a voxel to the
// last sample centre plus half a voxel, so the marched box exactly covers the data.
//
// Returns { min: [x,y,z], max: [x,y,z] } in Engineering metres (XYZ order for THREE).
export interface AABB {
  min: [number, number, number]; // [x, y, z]
  max: [number, number, number];
}

export function engineeringAABB(meta: VolumeMeta): AABB {
  const [nz, ny, nx] = meta.shape;
  const [oz, oy, ox] = meta.origin;
  const [dz, dy, dx] = meta.spacing;
  // Half-voxel padding so the box covers full sample footprints, not just centres.
  const minX = ox - dx / 2;
  const minY = oy - dy / 2;
  const minZ = oz - dz / 2;
  const maxX = ox + (nx - 1) * dx + dx / 2;
  const maxY = oy + (ny - 1) * dy + dy / 2;
  const maxZ = oz + (nz - 1) * dz + dz / 2;
  return {
    min: [Math.min(minX, maxX), Math.min(minY, maxY), Math.min(minZ, maxZ)],
    max: [Math.max(minX, maxX), Math.max(minY, maxY), Math.max(minZ, maxZ)],
  };
}

// Centre of the Engineering AABB (XYZ metres) — the default camera target (doc 06 §2.2).
export function aabbCenter(box: AABB): [number, number, number] {
  return [
    (box.min[0] + box.max[0]) / 2,
    (box.min[1] + box.max[1]) / 2,
    (box.min[2] + box.max[2]) / 2,
  ];
}

// Size (extent) of the AABB along each axis (XYZ metres).
export function aabbSize(box: AABB): [number, number, number] {
  return [
    box.max[0] - box.min[0],
    box.max[1] - box.min[1],
    box.max[2] - box.min[2],
  ];
}

// Sample value at integer (k=z, j=y, i=x) from a C-contiguous (z,y,x) buffer.
// Returns NaN for out-of-range indices (treated as no-data by callers).
export function sampleAt(vol: DecodedVolume, i: number, j: number, k: number): number {
  const [nz, ny, nx] = vol.shape;
  if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return NaN;
  return vol.data[(k * ny + j) * nx + i];
}

// NaN-aware finite min/max over the buffer (a fallback when meta.stats is absent).
// Returns null if the buffer is entirely NaN/empty.
export function finiteMinMax(data: Float32Array): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (min === Infinity) return null;
  return { min, max };
}
