// Layer model + pure layer-list logic (doc 06 §9.1 Layer interface, §10.1 store shape).
//
// Every dataset becomes a Layer — the toggleable/blendable unit of the viewer (doc 06
// §9.1). M1 had a single resident volume; this evolves the viewer into a MULTI-LAYER
// model: a `Record<id, Layer>` + an explicit `layerOrder` array driving compositing /
// draw order (doc 06 §10.1). All functions here are PURE (no THREE / no DOM / no Zustand)
// so the add/remove/reorder/seed logic is unit-testable headlessly.
//
// Each volume layer carries its OWN decoded volume + meta + transfer function so multiple
// property models (resistivity, density, a fused favorability field) co-render in one
// frame (doc 06 §3.3). The transfer function auto-seeds from the property-type registry
// meta (doc 06 §3.2, §9.1 "registry-seeded defaults") via tfFromMeta below.

import type { PropertyModelMeta } from "./api";
import type { DecodedVolume, AABB } from "./volume";
import { engineeringAABB } from "./volume";
import type { ScalingMode, TransferFnSpec } from "./transferFn";
import {
  type SurfaceGrid,
  type SurfaceModel,
  type XYExtent,
  parseSurfaceModel,
  buildSurfaceGrid,
  elevationRange,
} from "./terrain";
import type { WellTrajectory } from "./wells";

// Compositing / blend operators (doc 06 §3.3). M1 default is `over` (alpha-composite by
// layer order); `additive` / `mip` / `minip` are selectable per layer.
export type BlendMode = "over" | "additive" | "mip" | "minip";
export const BLEND_MODES: readonly BlendMode[] = ["over", "additive", "mip", "minip"];

export function isBlendMode(s: string): s is BlendMode {
  return (BLEND_MODES as readonly string[]).includes(s);
}

// Layer kinds (doc 06 §9.1). M1/this milestone implements `volume`; the other kinds are
// declared for the shared model but not yet rendered.
export type LayerKind =
  | "volume"
  | "slice"
  | "isosurface"
  | "surface"
  | "well"
  | "points"
  | "glyphs"
  | "terrain"
  | "raster";

// The Layer interface (doc 06 §9.1). `transferFn`/`property`/`clip` are per-layer; the
// decoded `volume` + `meta` + `aabb` are carried here (not a separate global) so each
// layer is self-contained and N volumes can co-render (doc 06 §3.3).
export interface Layer {
  id: string;
  datasetId: string; // the source property-model id (or "mock")
  name: string; // display name in the layer manager
  kind: LayerKind;
  visible: boolean;
  opacity: number; // 0..1 — multiplies the transfer-fn opacity gain at composite time
  order: number; // compositing / draw order (mirrors index in layerOrder)
  blend: BlendMode; // doc 06 §3.3
  transferFn: TransferFnSpec; // colormap + opacity + domain + log/linear + invert + band
  property?: string; // which property is mapped (from meta.property)
  clip: boolean; // obey the global clip box (doc 06 §9.1)
  zExplode?: number; // exploded-layers offset (doc 06 §9.3) — reserved

  // Resident data for `volume` layers (doc 06 §1.3 single-resident path, now per-layer).
  meta?: PropertyModelMeta;
  volume?: DecodedVolume;
  aabb?: AABB;

  // Confidence-modulated opacity — the doc 07 §5.3 "honest view" (doc 06 §9.2 opacity
  // modulation). Optional: binds a co-registered confidence/σ volume (same grid as this
  // layer's data) that scales each sample's opacity so low-confidence regions render faint.
  // `invert` handles a σ (uncertainty) volume where HIGH value = LESS confident.
  confidence?: ConfidenceModulation;

  // Resident data for `terrain` layers (doc 06 §6, doc 01 §6). The surface grid is the
  // ground surface in the Engineering Frame; subsurface volumes hang beneath it. `basemap`
  // opts into draped online XYZ tiles (doc 06 §6.2) — default shaded-relief otherwise.
  surface?: SurfaceGrid;
  surfaceModel?: SurfaceModel;
  basemap?: boolean;

  // ── M2/M4/M5 feature + 4-D layer kinds (doc 06 §5, §9.4) ─────────────────────────────
  featureId?: string; // the source GeologicalFeature id (provenance / refetch key)
  featureKind?: string; // "horizon" | "fault" | "solid" | "wellPath" | "pointCloud" | …

  // `surface`/`fault`/`isosurface` layers (doc 06 §5.2): a glTF triangle mesh loaded via
  // three GLTFLoader from GET /features/{id}/geometry, dropped into the Z-up scene. Faults
  // render semi-transparent with an edge highlight; surfaces double-sided with optional
  // per-vertex property draping. The decoded mesh is held by the scene component (THREE
  // objects never live in the store); the layer carries only the load descriptor + style.
  faultStyle?: boolean; // semi-transparent + edge highlight (doc 06 §5.2)

  // `well` layers (doc 06 §5.3): the resolved trajectory (Engineering polyline + MD/TVD +
  // joined LAS logs) the WellLayer turns into a TubeGeometry, plus which log curve colours
  // the tube (null = a neutral solid tube).
  trajectory?: WellTrajectory;
  logProperty?: string | null; // selected LAS curve name (tube colour)
  // Planned-well DLS ceiling (doc 09 §8.1): when set AND no log curve is selected, the tube
  // paints per-segment — segments whose dogleg exceeds this render red (constraint feedback).
  dlsMax_deg30m?: number;

  // `points` layers (microseismic 4-D cloud, doc 06 §5.4): the compact parallel arrays the
  // PointCloudLayer uploads once to a THREE.Points buffer; the time window culls on the GPU
  // via per-point epoch-ms (no per-frame re-upload).
  points?: PointCloud;

  // `raster` layers (InSAR deformation time-series, doc 06 §6): a stack of per-epoch surface
  // grids draped on the ground; the global time slider selects the leading-t frame.
  raster?: RasterTimeSeries;
}

// A microseismic 4-D point cloud ready for GPU upload (doc 06 §5.4). `epochMs` is the parsed
// epoch-millisecond per point (parallel to xyz) so the moving time window culls on the GPU
// without re-parsing ISO strings each frame.
export interface PointCloud {
  count: number;
  positions: Float32Array; // length 3*count, Engineering XYZ (flat x,y,z per point)
  magnitude: Float32Array; // length count
  // Parsed epoch ms (NaN for an undated point). Float64 because ms-since-epoch (~1.6e12)
  // overflows float32 precision; the scene rebases it to a relative float32 at GPU upload so
  // the moving time window's `uTimeWindow` comparison stays exact (doc 06 §5.4, §9.4).
  epochMs: Float64Array;
  depth?: Float32Array; // length count, true depth (m) when present
}

// A draped raster time-series frame (InSAR deformation, doc 06 §6). Each frame is a value
// grid co-registered with `surface` (same nx/ny); the slider selects `frameIndex`.
export interface RasterTimeSeries {
  surface: SurfaceGrid; // the draped ground grid (xy + elevation)
  frames: Float32Array[]; // per-epoch value grids (length nx*ny each)
  epochs: string[]; // ISO-8601 per frame (parallel to frames)
  epochMs: number[]; // parsed ms per frame
  range: [number, number]; // value colour domain
  frameIndex: number; // currently selected frame (slider-driven)
}

// Confidence-modulated-opacity binding (doc 07 §5.3 honest view; doc 06 §9.2). `volume` is a
// co-registered confidence/σ field (same grid as the layer's data); the shader maps its raw
// value over [min,max] → an opacity weight in [floor,1]. `invert` flips it for a σ volume
// (high value = low confidence). `floor` keeps faint regions barely visible rather than
// fully hidden. When `enabled` is false the layer renders unmodulated.
export interface ConfidenceModulation {
  enabled: boolean;
  volume: DecodedVolume; // raw confidence/σ field, co-registered with the layer's data
  min: number; // raw value mapped to weight 0
  max: number; // raw value mapped to weight 1
  invert: boolean; // σ-style: high value = low confidence
  floor: number; // minimum opacity weight (0..1)
  sourceId?: string; // the confidence PropertyModel id (provenance / UI label)
}

const DEFAULT_TF: TransferFnSpec = {
  colormap: "viridis",
  domainMin: 0,
  domainMax: 1,
  scaling: "linear",
  opacity: 0.9,
  invert: false,
};

// Seed a transfer function from the property-type registry meta (doc 06 §3.2, §9.1): the
// registry default colormap, log/linear scaling, and display range — falling back to
// NaN-aware stats (p1/p99 then min/max) when displayRange is absent (doc 04 §9.2 stats
// seed the clamp). Moved here verbatim from store.ts so layer creation can seed defaults.
export function tfFromMeta(meta: PropertyModelMeta): TransferFnSpec {
  let lo = 0;
  let hi = 1;
  if (meta.displayRange && meta.displayRange.length === 2) {
    [lo, hi] = meta.displayRange;
  } else if (meta.stats.p1 != null && meta.stats.p99 != null) {
    lo = meta.stats.p1;
    hi = meta.stats.p99;
  } else if (meta.stats.min != null && meta.stats.max != null) {
    lo = meta.stats.min;
    hi = meta.stats.max;
  }
  if (hi <= lo) hi = lo + 1;
  const scaling: ScalingMode = meta.scaling === "log" ? "log" : "linear";
  return {
    colormap: meta.colormap ?? "viridis",
    domainMin: lo,
    domainMax: hi,
    scaling,
    opacity: 0.9,
    invert: false,
  };
}

let _seq = 0;
// Monotonic unique layer id (stable within a session; not persisted).
export function newLayerId(prefix = "layer"): string {
  _seq += 1;
  return `${prefix}-${_seq}`;
}

// Build a volume Layer from a decoded volume + meta, auto-seeding its transfer function
// from the registry meta (doc 06 §9.1 zero-config). `order` defaults to 0 and is fixed up
// by the store when the layer is appended to layerOrder.
export function makeVolumeLayer(
  meta: PropertyModelMeta,
  volume: DecodedVolume,
  opts: { id?: string; name?: string } = {},
): Layer {
  return {
    id: opts.id ?? newLayerId(),
    datasetId: meta.id,
    name: opts.name ?? meta.property ?? meta.id,
    kind: "volume",
    visible: true,
    opacity: 1,
    order: 0,
    blend: "over",
    transferFn: tfFromMeta(meta),
    property: meta.property,
    clip: true,
    meta,
    volume,
    aabb: engineeringAABB(volume),
  };
}

// Build a terrain Layer (doc 06 §6, §9.1). The surface grid is either supplied directly
// (a backend DEM grid already in Engineering elevation, doc 06 §6.1) or synthesized from
// the project's surfaceModel string ("flat:<z>" / "synthetic:<id>" / "dem:…" fallback,
// doc 01 §6) over an XY extent (typically the project ROI / scene footprint). Terrain has
// no transfer function (it is a shaded-relief surface, not a mapped scalar field), but the
// Layer interface requires one — we give it a neutral default so the field stays present.
export function makeTerrainLayer(
  opts: {
    id?: string;
    name?: string;
    surfaceModelSpec?: string | null;
    extent?: XYExtent;
    res?: number;
    surface?: SurfaceGrid;
    basemap?: boolean;
    datasetId?: string;
  } = {},
): Layer {
  const model = parseSurfaceModel(opts.surfaceModelSpec);
  // Prefer an explicitly supplied grid (backend DEM); else synthesize over the extent.
  const surface =
    opts.surface ??
    (opts.extent
      ? buildSurfaceGrid(model, opts.extent, opts.res ?? 64)
      : undefined);
  // Terrain footprint AABB (Engineering metres) for framing / clip, when we have a grid.
  let aabb: AABB | undefined;
  if (surface) {
    const zr = elevationRange(surface) ?? { min: 0, max: 0 };
    aabb = {
      min: [surface.x0, surface.y0, zr.min],
      max: [
        surface.x0 + (surface.nx - 1) * surface.dx,
        surface.y0 + (surface.ny - 1) * surface.dy,
        zr.max,
      ],
    };
  }
  return {
    id: opts.id ?? newLayerId("terrain"),
    datasetId: opts.datasetId ?? "terrain",
    name: opts.name ?? "Terrain",
    kind: "terrain",
    visible: true,
    opacity: 1,
    order: 0,
    blend: "over",
    transferFn: { ...DEFAULT_TF },
    clip: true,
    surface,
    surfaceModel: model,
    basemap: opts.basemap ?? false,
    aabb,
  };
}

// Build a `surface`/`fault`/`isosurface` feature Layer (doc 06 §5.2). The glTF mesh is loaded
// by the scene FeatureLayer from GET /features/{featureId}/geometry; this carries only the
// descriptor + style. Faults default to the semi-transparent edge-highlight style.
export function makeFeatureLayer(opts: {
  featureId: string;
  featureKind: string;
  id?: string;
  name?: string;
  datasetId?: string;
}): Layer {
  const isFault = opts.featureKind === "fault";
  const kind: LayerKind =
    opts.featureKind === "isosurface" ? "isosurface" : "surface";
  return {
    id: opts.id ?? newLayerId("feature"),
    datasetId: opts.datasetId ?? opts.featureId,
    name: opts.name ?? opts.featureKind,
    kind,
    visible: true,
    opacity: isFault ? 0.6 : 1,
    order: 0,
    blend: "over",
    transferFn: { ...DEFAULT_TF },
    clip: true,
    featureId: opts.featureId,
    featureKind: opts.featureKind,
    faultStyle: isFault,
  };
}

// Build a `well` Layer from a resolved trajectory (doc 06 §5.3). `logProperty` selects the
// LAS curve that colours the tube (defaults to the joined logs' primary property).
export function makeWellLayer(
  trajectory: WellTrajectory,
  opts: {
    id?: string;
    name?: string;
    datasetId?: string;
    logProperty?: string | null;
    dlsMax_deg30m?: number;
  } = {},
): Layer {
  const aabb = polylineAABB(trajectory.polyline);
  const logProperty =
    opts.logProperty !== undefined
      ? opts.logProperty
      : (trajectory.logs?.primaryProperty ?? null);
  return {
    id: opts.id ?? newLayerId("well"),
    datasetId: opts.datasetId ?? trajectory.featureId,
    name: opts.name ?? (trajectory.wellId ? `Well ${trajectory.wellId}` : "Well"),
    kind: "well",
    visible: true,
    opacity: 1,
    order: 0,
    blend: "over",
    transferFn: { ...DEFAULT_TF },
    clip: true,
    featureId: trajectory.featureId,
    featureKind: "wellPath",
    trajectory,
    logProperty,
    dlsMax_deg30m: opts.dlsMax_deg30m,
    aabb,
  };
}

// Build a `points` microseismic-cloud Layer (doc 06 §5.4) from the uploaded buffers.
export function makePointCloudLayer(
  cloud: PointCloud,
  opts: { featureId: string; id?: string; name?: string; datasetId?: string } = { featureId: "" },
): Layer {
  return {
    id: opts.id ?? newLayerId("points"),
    datasetId: opts.datasetId ?? opts.featureId,
    name: opts.name ?? "Microseismic",
    kind: "points",
    visible: true,
    opacity: 1,
    order: 0,
    blend: "over",
    transferFn: { ...DEFAULT_TF, colormap: "inferno" },
    clip: true,
    featureId: opts.featureId,
    featureKind: "pointCloud",
    points: cloud,
    aabb: pointsAABB(cloud),
  };
}

// Build a `raster` InSAR deformation time-series Layer (doc 06 §6). The slider drives
// `frameIndex` (leading-t frame select).
export function makeRasterLayer(
  raster: RasterTimeSeries,
  opts: { featureId?: string; id?: string; name?: string; datasetId?: string } = {},
): Layer {
  return {
    id: opts.id ?? newLayerId("raster"),
    datasetId: opts.datasetId ?? opts.featureId ?? "raster",
    name: opts.name ?? "InSAR deformation",
    kind: "raster",
    visible: true,
    opacity: 1,
    order: 0,
    blend: "over",
    transferFn: { ...DEFAULT_TF, colormap: "turbo", domainMin: raster.range[0], domainMax: raster.range[1] },
    clip: true,
    featureId: opts.featureId,
    featureKind: "deformation",
    raster,
  };
}

// Engineering AABB of a polyline (well-tube framing). Returns undefined for an empty path.
function polylineAABB(poly: readonly [number, number, number][]): AABB | undefined {
  if (poly.length === 0) return undefined;
  const min: [number, number, number] = [...poly[0]];
  const max: [number, number, number] = [...poly[0]];
  for (const p of poly) {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  }
  return { min, max };
}

// Engineering AABB of a point cloud (cloud framing). Returns undefined when empty.
function pointsAABB(cloud: PointCloud): AABB | undefined {
  if (cloud.count === 0) return undefined;
  const p = cloud.positions;
  const min: [number, number, number] = [p[0], p[1], p[2]];
  const max: [number, number, number] = [p[0], p[1], p[2]];
  for (let i = 0; i < cloud.count; i++) {
    for (let k = 0; k < 3; k++) {
      const v = p[i * 3 + k];
      if (v < min[k]) min[k] = v;
      if (v > max[k]) max[k] = v;
    }
  }
  return { min, max };
}

// ── Pure layer-list operations (operate on {layers, layerOrder}) ─────────────────────

export interface LayerCollection {
  layers: Record<string, Layer>;
  layerOrder: string[];
}

// Re-derive each layer's `order` from its index in layerOrder so the field stays in sync
// (doc 06 §9.1 order == compositing/draw order).
function reindex(c: LayerCollection): LayerCollection {
  const layers = { ...c.layers };
  c.layerOrder.forEach((id, i) => {
    if (layers[id]) layers[id] = { ...layers[id], order: i };
  });
  return { layers, layerOrder: c.layerOrder };
}

// Append a layer to the collection (drawn last == on top for `over`). Idempotent on id.
export function addLayer(c: LayerCollection, layer: Layer): LayerCollection {
  if (c.layers[layer.id]) {
    // Replace existing data in place, keep its position.
    return reindex({ ...c, layers: { ...c.layers, [layer.id]: layer } });
  }
  return reindex({
    layers: { ...c.layers, [layer.id]: layer },
    layerOrder: [...c.layerOrder, layer.id],
  });
}

// Prepend a layer to the BOTTOM of the collection (drawn first == beneath everything for
// `over`). Used for terrain so subsurface volumes composite on top of the ground surface
// (doc 06 §6 "subsurface volumes hang beneath it"; §9.1 order == compositing/draw order).
// Idempotent on id (replaces data in place, keeping position).
export function addLayerBottom(c: LayerCollection, layer: Layer): LayerCollection {
  if (c.layers[layer.id]) {
    return reindex({ ...c, layers: { ...c.layers, [layer.id]: layer } });
  }
  return reindex({
    layers: { ...c.layers, [layer.id]: layer },
    layerOrder: [layer.id, ...c.layerOrder],
  });
}

// Remove a layer by id (no-op if absent).
export function removeLayer(c: LayerCollection, id: string): LayerCollection {
  if (!c.layers[id]) return c;
  const layers = { ...c.layers };
  delete layers[id];
  return reindex({ layers, layerOrder: c.layerOrder.filter((x) => x !== id) });
}

// Move a layer up (toward the end / top) or down (toward the start / bottom) by one slot.
export function moveLayer(
  c: LayerCollection,
  id: string,
  dir: "up" | "down",
): LayerCollection {
  const idx = c.layerOrder.indexOf(id);
  if (idx < 0) return c;
  const swap = dir === "up" ? idx + 1 : idx - 1;
  if (swap < 0 || swap >= c.layerOrder.length) return c;
  const order = [...c.layerOrder];
  [order[idx], order[swap]] = [order[swap], order[idx]];
  return reindex({ ...c, layerOrder: order });
}

// Reorder by explicit index (drag-to-reorder, doc 06 §9.1). Clamps to range.
export function reorderLayer(
  c: LayerCollection,
  id: string,
  toIndex: number,
): LayerCollection {
  const from = c.layerOrder.indexOf(id);
  if (from < 0) return c;
  const order = [...c.layerOrder];
  order.splice(from, 1);
  const clamped = Math.max(0, Math.min(toIndex, order.length));
  order.splice(clamped, 0, id);
  return reindex({ ...c, layerOrder: order });
}

// Patch a single layer's fields immutably (no-op if absent).
export function patchLayer(
  c: LayerCollection,
  id: string,
  patch: Partial<Layer>,
): LayerCollection {
  if (!c.layers[id]) return c;
  return { ...c, layers: { ...c.layers, [id]: { ...c.layers[id], ...patch } } };
}

// Patch a single layer's transfer function (the common editor path).
export function patchLayerTF(
  c: LayerCollection,
  id: string,
  patch: Partial<TransferFnSpec>,
): LayerCollection {
  const l = c.layers[id];
  if (!l) return c;
  return patchLayer(c, id, { transferFn: { ...l.transferFn, ...patch } });
}

export { DEFAULT_TF };
