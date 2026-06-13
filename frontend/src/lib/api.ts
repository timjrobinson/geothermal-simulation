// Backend API client for the M1 viewer (doc 04 §9.2 / doc 06 §1.3, §12).
//
// Two reads back the M1 single-resident path:
//   GET /property-models/{id}                — PropertyModelMeta (seeds the transfer fn)
//   GET /property-models/{id}/volume?level=  — raw LE f32 (z,y,x) ArrayBuffer + X-Volume-*
// and the orthogonal slice is served by:
//   POST /property-models/{id}/slice         — raw f32 plane + X-Slice-Header
//
// Response shapes mirror geosim/api/schemas.py exactly (PropertyModelMeta, VolumeMeta,
// SliceRequest, SliceHeader). The /volume endpoint is the doc 06 §1.3 server-decode that
// sidesteps browser-Blosc; browser Zarr/Blosc decode is deferred to M2 per the §1.3 SPIKE.

import { decodeVolume, type DecodedVolume } from "./volume";

// Mirrors PropertyModelStats (schemas.py).
export interface PropertyModelStats {
  min: number | null;
  max: number | null;
  p1: number | null;
  p99: number | null;
}

// Mirrors PropertyModelMeta (schemas.py): shape/origin/spacing are level-0 (z,y,x).
export interface PropertyModelMeta {
  id: string;
  property: string;
  canonicalUnit: string;
  scaling: string; // "linear" | "log"
  colormap: string | null;
  displayRange: [number, number] | null;
  shape: [number, number, number]; // [nz, ny, nx]
  origin: [number, number, number]; // [oz, oy, ox]
  spacing: [number, number, number]; // [dz, dy, dx]
  levels: number;
  stats: PropertyModelStats;
  frame: Record<string, unknown> | null;
  hasSigma: boolean;
}

const JSON_HEADER = (h: Headers, k: string): unknown => {
  const v = h.get(k);
  return v == null ? null : JSON.parse(v);
};

export async function fetchMeta(id: string): Promise<PropertyModelMeta> {
  const r = await fetch(`/property-models/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`meta fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as PropertyModelMeta;
}

// Fetch + decode the M1 single-resident volume (doc 06 §1.3). Prefers the X-Volume-*
// headers (authoritative for the buffer actually returned at this level); falls back to
// the meta shape/origin/spacing if a header is missing.
export async function fetchVolume(
  id: string,
  level: number,
  meta: PropertyModelMeta,
): Promise<DecodedVolume> {
  const r = await fetch(
    `/property-models/${encodeURIComponent(id)}/volume?level=${level}`,
  );
  if (!r.ok) throw new Error(`volume fetch failed: ${r.status} ${r.statusText}`);
  const buf = await r.arrayBuffer();
  const shape = (JSON_HEADER(r.headers, "X-Volume-Shape") as
    | [number, number, number]
    | null) ?? meta.shape;
  const origin = (JSON_HEADER(r.headers, "X-Volume-Origin") as
    | [number, number, number]
    | null) ?? meta.origin;
  const spacing = (JSON_HEADER(r.headers, "X-Volume-Spacing") as
    | [number, number, number]
    | null) ?? meta.spacing;
  return decodeVolume(buf, { shape, origin, spacing });
}

// Mirrors SliceHeader (schemas.py).
export interface SliceHeader {
  width: number;
  height: number;
  dx: number;
  dy: number;
  plane_basis: { origin: number[]; u: number[]; v: number[] };
  encoding: string;
  dtype: string;
}

export interface SliceResult {
  header: SliceHeader;
  data: Float32Array; // (height, width) row-major f32, NaN no-data
}

// POST a slice request (doc 04 §9.3). M1 uses the client-side same-texture slice for
// orthogonal planes (doc 06 §4.1); this server path is provided for HQ/native-res and as
// the no-GPU fallback (doc 06 §7.4) and is exercised by tests / optional UI.
export async function fetchSlice(
  id: string,
  plane: "x" | "y" | "z",
  position: number,
  level = 0,
): Promise<SliceResult> {
  const r = await fetch(`/property-models/${encodeURIComponent(id)}/slice`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plane, position, level, encoding: "f32" }),
  });
  if (!r.ok) throw new Error(`slice fetch failed: ${r.status} ${r.statusText}`);
  const header = JSON.parse(r.headers.get("X-Slice-Header") ?? "{}") as SliceHeader;
  const buf = await r.arrayBuffer();
  return { header, data: new Float32Array(buf) };
}
