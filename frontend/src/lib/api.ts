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

// ── Dataset discovery (doc 06 §9.1 datasets→layers) ──────────────────────────────────
// The "open project" flow lists ingested property models as addable layers. It reads
// GET /projects (project list) and GET /projects/{pid}/artifacts (catalog discovery —
// the ingested artifacts of a project). Shapes mirror the backend ProjectSummary; the
// artifacts catalog is read defensively (field-name tolerant) since the catalog row shape
// is the discovery contract, not a typed M1 schema. When no backend is reachable these
// reject and the UI falls back to the in-memory mock layer (still works offline).

export interface ProjectSummary {
  id: string;
  name: string;
  description?: string | null;
  created_at?: number;
  updated_at?: number;
}

// A discovered property-model artifact addable as a layer. We only need id/property/unit
// to list it; the full meta is fetched lazily on add via fetchMeta.
export interface PropertyModelArtifact {
  id: string;
  property: string;
  canonicalUnit?: string | null;
}

export async function fetchProjects(): Promise<ProjectSummary[]> {
  const r = await fetch(`/projects`);
  if (!r.ok) throw new Error(`projects fetch failed: ${r.status} ${r.statusText}`);
  const body = (await r.json()) as ProjectSummary[];
  return Array.isArray(body) ? body : [];
}

// Discover the property-model artifacts of a project. The catalog response is tolerant of
// a few shapes: an array of rows, or { artifacts: [...] } / { property_models: [...] }.
// Each row is normalized to { id, property, canonicalUnit }.
export async function fetchProjectArtifacts(
  pid: string,
): Promise<PropertyModelArtifact[]> {
  const r = await fetch(`/projects/${encodeURIComponent(pid)}/artifacts`);
  if (!r.ok) throw new Error(`artifacts fetch failed: ${r.status} ${r.statusText}`);
  const body = (await r.json()) as unknown;
  return normalizeArtifacts(body);
}

// Pure normalizer (exported for unit tests): coerce a catalog response into a flat list
// of property-model artifacts, keeping only volume-like property models.
export function normalizeArtifacts(body: unknown): PropertyModelArtifact[] {
  let rows: unknown[] = [];
  if (Array.isArray(body)) rows = body;
  else if (body && typeof body === "object") {
    const o = body as Record<string, unknown>;
    if (Array.isArray(o.artifacts)) rows = o.artifacts;
    else if (Array.isArray(o.property_models)) rows = o.property_models;
    else if (Array.isArray(o.propertyModels)) rows = o.propertyModels;
    else if (Array.isArray(o.items)) rows = o.items;
  }
  const out: PropertyModelArtifact[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const o = row as Record<string, unknown>;
    // A property-model artifact has an id and a property; tolerate kind/type markers.
    const kind = (o.kind ?? o.type ?? o.artifact_type) as string | undefined;
    if (kind != null && !/property[_-]?model|volume/i.test(kind)) continue;
    const id = (o.id ?? o.pm_id ?? o.property_model_id) as string | undefined;
    if (typeof id !== "string") continue;
    const property = (o.property ?? o.name ?? "property") as string;
    const canonicalUnit = (o.canonicalUnit ?? o.canonical_unit ?? null) as
      | string
      | null;
    out.push({ id, property, canonicalUnit });
  }
  return out;
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
