// Fusion / analysis API client (doc 07 §3, §6; doc 06 §10.3). The analysis panels
// (cross-plot, histogram, correlation heatmap) and linked brushing are fed by the backend
// fusion router (geosim/api/fusion.py). Response shapes mirror that router's pydantic
// models EXACTLY:
//   POST /fused                         — FusedModelOut (create a fused container grid)
//   POST /fused/{id}/resample           — ResampledLayerOut (resample a PM into the grid)
//   GET  /fused/{id}                     — FusedModelOut (grid geometry + resampled layers)
//   POST /fused/{id}/sample             — SampleOut   (co-located feature matrix + mask)
//   POST /fused/{id}/crossplot          — { n, properties, crossplot?, histogram?, correlation? }
//   POST /fused/{id}/cluster            — { mode, ... } (sync) | { mode:"job", job_id }
//   GET  /projects/{pid}/artifacts      — ArtifactSummary[] (discovery)
//
// When no backend is reachable (the self-contained `npm run dev`) the create/resample/
// sample calls fall back to an in-memory mock (makeMockFusedSample) so the analysis +
// brushing flow is fully exercisable offline. Everything that does NOT touch fetch/DOM is
// kept pure in crossplot.ts / brushing.ts so it is unit-testable headlessly.

// ── wire shapes (mirror geosim/api/fusion.py) ────────────────────────────────────────

export interface FusedLayerOut {
  layer_id: string;
  property: string;
  source_property_model_id: string;
  source_version: string;
  method: string;
  sigma_array: string | null;
  coverage_mask: string | null;
}

export interface FusedModelOut {
  id: string;
  project_id: string;
  grid_type: string;
  origin: [number, number, number]; // [oz, oy, ox]
  spacing: [number, number, number]; // [dz, dy, dx]
  shape: [number, number, number]; // [nz, ny, nx]
  n_cells: number;
  bbox: Record<string, number>;
  layers: FusedLayerOut[];
}

// SampleOut (doc 07 §3.1): the co-located feature matrix + the per-row flat cell index
// (the key to linked brushing — map a selected row back to a voxel) + Engineering coords.
export interface FusedSampleOut {
  properties: string[];
  n: number;
  features: number[][]; // (n, p) native units
  cell_index: number[]; // (n,) flat index into the (nz,ny,nx) grid
  coords: number[][]; // (n, 3) Engineering (z,y,x) metres
  grid_shape: [number, number, number];
  mode: string;
}

// crossplot() payload (doc 07 §3.2): either a downsampled scatter point set OR, for huge N,
// a 2D density grid (histogram2d). color/color_by are optional per-point channels.
export type CrossplotPayload =
  | {
      kind: "scatter";
      axes: string[];
      n: number;
      points: number[][];
      color?: number[];
      color_by?: string;
    }
  | {
      kind: "density";
      axes: string[];
      n: number;
      counts: number[][];
      x_edges: number[];
      y_edges: number[];
    };

export interface HistogramPayload {
  property: string;
  n: number;
  counts: number[];
  bin_edges: number[];
  kde_x?: number[];
  kde_y?: number[];
}

export interface CorrelationPayload {
  properties: string[];
  matrix: (number | null)[][];
}

export interface CrossplotResponse {
  n: number;
  properties: string[];
  crossplot?: CrossplotPayload;
  histogram?: HistogramPayload;
  correlation?: CorrelationPayload;
}

export interface ClusterResponse {
  mode: "sync" | "job";
  job_id?: string;
  // sync payload fields (ClusterResult.to_payload) are passed through opaquely.
  [k: string]: unknown;
}

// ── request bodies ────────────────────────────────────────────────────────────────────

export interface FusedCreateBody {
  project_id: string;
  name?: string;
  source_property_model_ids?: string[] | null;
  bbox?: Record<string, number> | null;
  spacing?: [number, number, number] | null;
}

export interface CrossplotBody {
  axes?: string[] | null; // 2 or 3 properties
  color_by?: string | null; // "depth" | a property name
  properties?: string[] | null;
  bbox?: Record<string, number> | null;
  histogram_property?: string | null;
  kde?: boolean;
  bins?: number;
  correlation?: boolean;
}

// ── fetch wrappers ──────────────────────────────────────────────────────────────────

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

export async function createFused(body: FusedCreateBody): Promise<FusedModelOut> {
  return postJSON<FusedModelOut>(`/fused`, body);
}

export async function resampleLayer(
  gridId: string,
  propertyModelId: string,
  opts: { method?: string; interp_space?: string } = {},
): Promise<unknown> {
  return postJSON(`/fused/${encodeURIComponent(gridId)}/resample`, {
    property_model_id: propertyModelId,
    method: opts.method ?? "auto",
    interp_space: opts.interp_space ?? "auto",
  });
}

export async function getFused(gridId: string): Promise<FusedModelOut> {
  const r = await fetch(`/fused/${encodeURIComponent(gridId)}`);
  if (!r.ok) throw new Error(`fused fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as FusedModelOut;
}

export async function sampleFused(
  gridId: string,
  body: { properties?: string[] | null; mode?: string; bbox?: Record<string, number> | null } = {},
): Promise<FusedSampleOut> {
  return postJSON<FusedSampleOut>(`/fused/${encodeURIComponent(gridId)}/sample`, {
    properties: body.properties ?? null,
    mode: body.mode ?? "all",
    bbox: body.bbox ?? null,
  });
}

export async function crossplotFused(
  gridId: string,
  body: CrossplotBody,
): Promise<CrossplotResponse> {
  return postJSON<CrossplotResponse>(`/fused/${encodeURIComponent(gridId)}/crossplot`, {
    axes: body.axes ?? null,
    color_by: body.color_by ?? null,
    properties: body.properties ?? null,
    bbox: body.bbox ?? null,
    histogram_property: body.histogram_property ?? null,
    kde: body.kde ?? false,
    bins: body.bins ?? 64,
    correlation: body.correlation ?? true,
  });
}

export async function clusterFused(
  gridId: string,
  body: {
    project_id: string;
    algorithm?: string;
    n_clusters?: number;
    properties?: string[] | null;
    bbox?: Record<string, number> | null;
  },
): Promise<ClusterResponse> {
  return postJSON<ClusterResponse>(`/fused/${encodeURIComponent(gridId)}/cluster`, {
    project_id: body.project_id,
    algorithm: body.algorithm ?? "kmeans",
    n_clusters: body.n_clusters ?? 3,
    properties: body.properties ?? null,
    bbox: body.bbox ?? null,
    write_volumes: true,
  });
}

// ── offline mock (self-contained dev / no backend) ────────────────────────────────────
// Synthesize a small co-located sample over a coarse grid with two correlated properties
// (resistivity↓ where density↑, plus a depth trend) so the cross-plot, histogram,
// correlation heatmap AND linked brushing are all exercisable with no backend running.
// Shape mirrors FusedSampleOut exactly so the rest of the pipeline is identical to live.

export function makeMockFusedSample(
  opts: { shape?: [number, number, number]; properties?: string[]; seed?: number } = {},
): { grid: FusedModelOut; sample: FusedSampleOut } {
  const shape = opts.shape ?? [12, 16, 16];
  const properties = opts.properties ?? ["resistivity", "density", "vp"];
  const [nz, ny, nx] = shape;
  const spacing: [number, number, number] = [100, 100, 100];
  const origin: [number, number, number] = [-1200, -800, -800];

  // Deterministic LCG so tests are stable.
  let s = (opts.seed ?? 1) >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };

  const features: number[][] = [];
  const cell_index: number[] = [];
  const coords: number[][] = [];
  for (let k = 0; k < nz; k++) {
    const z = origin[0] + k * spacing[0];
    const depthFrac = nz > 1 ? k / (nz - 1) : 0;
    for (let j = 0; j < ny; j++) {
      const y = origin[1] + j * spacing[1];
      for (let i = 0; i < nx; i++) {
        const x = origin[2] + i * spacing[2];
        // A "conductive anomaly": low resistivity + high density near the centre.
        const cx = (i - nx / 2) / nx;
        const cy = (j - ny / 2) / ny;
        const r = Math.hypot(cx, cy);
        const anomaly = Math.exp(-(r * r) / 0.06);
        const resistivity = 50 - 35 * anomaly + 8 * depthFrac + (rand() - 0.5) * 6;
        const density = 2.2 + 0.5 * anomaly + 0.2 * depthFrac + (rand() - 0.5) * 0.06;
        const vp = 3500 + 900 * anomaly + 400 * depthFrac + (rand() - 0.5) * 120;
        const all = [resistivity, density, vp];
        features.push(properties.map((p) => all[["resistivity", "density", "vp"].indexOf(p)] ?? 0));
        cell_index.push((k * ny + j) * nx + i);
        coords.push([z, y, x]);
      }
    }
  }

  const grid: FusedModelOut = {
    id: "mock-fused",
    project_id: "mock",
    grid_type: "regular",
    origin,
    spacing,
    shape,
    n_cells: nz * ny * nx,
    bbox: {
      xmin: origin[2],
      xmax: origin[2] + (nx - 1) * spacing[2],
      ymin: origin[1],
      ymax: origin[1] + (ny - 1) * spacing[1],
      zmin: origin[0],
      zmax: origin[0] + (nz - 1) * spacing[0],
    },
    layers: properties.map((p, idx) => ({
      layer_id: `mock-${p}`,
      property: p,
      source_property_model_id: `mock-pm-${idx}`,
      source_version: "mock",
      method: "mock",
      sigma_array: null,
      coverage_mask: null,
    })),
  };

  const sample: FusedSampleOut = {
    properties,
    n: features.length,
    features,
    cell_index,
    coords,
    grid_shape: shape,
    mode: "all",
  };
  return { grid, sample };
}
