// Feature + 4-D API client (doc 04 §9.2, doc 06 §5.3/§5.4/§9.4). Mirrors the response shapes
// of geosim/api/features.py exactly. The geometry endpoint serves a binary glTF for
// surfaces/faults/solids (loaded by the scene FeatureLayer via three GLTFLoader) and GeoJSON
// for lines / well paths; the points endpoint serves a compact microseismic 4-D cloud; the
// trajectory endpoint serves the resolved well polyline + joined logs; time-extent serves the
// global slider axis union.

import type { WellTrajectory } from "./wells";
import type { PointCloud, RasterTimeSeries } from "./layers";
import { parseEpochMs } from "./time";
import type { SurfaceGrid } from "./terrain";

// Mirrors FeatureSummary (geosim/api/features.py).
export interface FeatureSummary {
  id: string;
  featureKind: string;
  datasetId: string | null;
  storeFormat: string;
  bbox: Record<string, number>;
  hasTime: boolean;
  geometryEndpoint: "gltf" | "geojson" | "points";
  props: Record<string, unknown>;
}

// Mirrors TimeExtent (geosim/api/features.py).
export interface TimeExtentResponse {
  epochs: string[];
  t0: string | null;
  t1: string | null;
  count: number;
  sources: { id: string; kind: string; n: number; [k: string]: unknown }[];
}

// Mirrors the /features/{id}/points payload (compact parallel typed-arrays).
export interface PointCloudResponse {
  featureId: string;
  count: number;
  x: number[];
  y: number[];
  z: number[];
  t: string[];
  magnitude: number[];
  depth_m: number[];
  window: { t0: string | null; t1: string | null; bbox: string | null };
}

export async function fetchProjectFeatures(
  pid: string,
  opts: { featureKind?: string; hasTime?: boolean } = {},
): Promise<FeatureSummary[]> {
  const q = new URLSearchParams();
  if (opts.featureKind) q.set("featureKind", opts.featureKind);
  if (opts.hasTime != null) q.set("has_time", String(opts.hasTime));
  const qs = q.toString();
  const r = await fetch(
    `/projects/${encodeURIComponent(pid)}/features${qs ? `?${qs}` : ""}`,
  );
  if (!r.ok) throw new Error(`features fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as FeatureSummary[];
}

// Fetch a surface/fault/solid feature's binary glTF as an ArrayBuffer (the scene FeatureLayer
// parses it with three GLTFLoader.parse). Throws if the endpoint returns non-glTF.
export async function fetchFeatureGLB(fid: string): Promise<ArrayBuffer> {
  const r = await fetch(`/features/${encodeURIComponent(fid)}/geometry`);
  if (!r.ok) throw new Error(`geometry fetch failed: ${r.status} ${r.statusText}`);
  return r.arrayBuffer();
}

// Fetch a feature's GeoJSON geometry (lines / well paths / point sets).
export async function fetchFeatureGeoJSON(
  fid: string,
): Promise<Record<string, unknown>> {
  const r = await fetch(`/features/${encodeURIComponent(fid)}/geometry`);
  if (!r.ok) throw new Error(`geometry fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as Record<string, unknown>;
}

// Fetch a microseismic 4-D point cloud, optionally pre-filtered by an Engineering bbox + ISO
// time window (doc 06 §5.4). The viewer typically fetches the FULL cloud once (no t0/t1) and
// applies the moving time window on the GPU; the server filter is for very large catalogs.
export async function fetchFeaturePoints(
  fid: string,
  opts: {
    bbox?: [number, number, number, number, number, number];
    t0?: string;
    t1?: string;
  } = {},
): Promise<PointCloudResponse> {
  const q = new URLSearchParams();
  if (opts.bbox) q.set("bbox", opts.bbox.join(","));
  if (opts.t0) q.set("t0", opts.t0);
  if (opts.t1) q.set("t1", opts.t1);
  const qs = q.toString();
  const r = await fetch(
    `/features/${encodeURIComponent(fid)}/points${qs ? `?${qs}` : ""}`,
  );
  if (!r.ok) throw new Error(`points fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as PointCloudResponse;
}

export async function fetchWellTrajectory(fid: string): Promise<WellTrajectory> {
  const r = await fetch(`/wells/${encodeURIComponent(fid)}/trajectory`);
  if (!r.ok) throw new Error(`trajectory fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as WellTrajectory;
}

export async function fetchTimeExtent(pid: string): Promise<TimeExtentResponse> {
  const r = await fetch(`/projects/${encodeURIComponent(pid)}/time-extent`);
  if (!r.ok) throw new Error(`time-extent fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as TimeExtentResponse;
}

// ── Pure response→layer-data builders (exported for unit tests) ───────────────────────

// Pack a microseismic /points response into the GPU-upload-ready PointCloud (doc 06 §5.4):
// interleaved xyz positions, magnitude, and the PARSED epoch-ms per point (so the moving
// time window culls on the GPU without re-parsing ISO strings each frame). A point with no
// time string gets epochMs = NaN (it stays hidden under any finite window).
export function packPointCloud(resp: PointCloudResponse): PointCloud {
  const n = resp.count;
  const positions = new Float32Array(n * 3);
  const magnitude = new Float32Array(n);
  const epochMs = new Float64Array(n);
  const hasDepth = Array.isArray(resp.depth_m) && resp.depth_m.length === n;
  const depth = hasDepth ? new Float32Array(n) : undefined;
  for (let i = 0; i < n; i++) {
    positions[i * 3 + 0] = resp.x[i] ?? 0;
    positions[i * 3 + 1] = resp.y[i] ?? 0;
    positions[i * 3 + 2] = resp.z[i] ?? 0;
    magnitude[i] = resp.magnitude[i] ?? 0;
    const iso = resp.t[i];
    epochMs[i] = iso != null ? parseEpochMs(iso) : NaN;
    if (depth) depth[i] = resp.depth_m[i] ?? 0;
  }
  return { count: n, positions, magnitude, epochMs, depth };
}

// Build an InSAR deformation RasterTimeSeries from a leading-t value stack + a draped surface
// grid (doc 06 §6). `frameValues[f]` is the per-cell deformation for epoch `epochs[f]`,
// co-registered with `surface` (length nx*ny). The colour range is the robust min/max over
// all finite samples. The slider drives `frameIndex` (default 0 = first epoch).
export function buildRasterTimeSeries(
  surface: SurfaceGrid,
  frameValues: Float32Array[],
  epochs: string[],
): RasterTimeSeries {
  let min = Infinity;
  let max = -Infinity;
  for (const frame of frameValues) {
    for (let i = 0; i < frame.length; i++) {
      const v = frame[i];
      if (Number.isNaN(v) || !Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    min = 0;
    max = 1;
  }
  return {
    surface,
    frames: frameValues,
    epochs,
    epochMs: epochs.map(parseEpochMs),
    range: [min, max],
    frameIndex: 0,
  };
}
