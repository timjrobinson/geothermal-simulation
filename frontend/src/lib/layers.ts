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

  // Resident data for `terrain` layers (doc 06 §6, doc 01 §6). The surface grid is the
  // ground surface in the Engineering Frame; subsurface volumes hang beneath it. `basemap`
  // opts into draped online XYZ tiles (doc 06 §6.2) — default shaded-relief otherwise.
  surface?: SurfaceGrid;
  surfaceModel?: SurfaceModel;
  basemap?: boolean;
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
