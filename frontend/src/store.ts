// Viewer store (doc 06 §10). All stored positions are Engineering metres (Z-up ENU);
// vertical exaggeration and the clip box are render-time transforms, never written back
// into data (doc 06 §10.2).
//
// This is the MULTI-LAYER store (doc 06 §9.1, §10.1): a `layers: Record<id, Layer>` +
// `layerOrder: string[]` model. The M1 single-resident volume becomes ONE volume layer —
// the single-volume path still works, it is just expressed as a one-layer collection.
// Multiple property volumes can co-render, each with its own transfer function + blend
// mode (doc 06 §3.3). The clip box, orthogonal slice, step count and capabilities readout
// stay GLOBAL (one clip gizmo, one slice driven from a chosen layer).

import { create } from "zustand";
import type { PropertyModelMeta } from "./lib/api";
import type { DecodedVolume, AABB } from "./lib/volume";
import { engineeringAABB, aabbCenter, aabbSize } from "./lib/volume";
import type { TransferFnSpec } from "./lib/transferFn";
import {
  type Layer,
  type BlendMode,
  type LayerCollection,
  addLayer as addLayerOp,
  removeLayer as removeLayerOp,
  moveLayer as moveLayerOp,
  patchLayer as patchLayerOp,
  patchLayerTF as patchLayerTFOp,
  addLayerBottom as addLayerBottomOp,
  makeVolumeLayer,
  makeTerrainLayer,
  tfFromMeta,
} from "./lib/layers";
import type { SurfaceGrid, XYExtent } from "./lib/terrain";

// Re-export so existing importers (and tests) keep resolving these from the store.
export { tfFromMeta };
export type { Layer, BlendMode };

// M0 capabilities shape (kept for the minor capabilities readout).
export interface Capabilities {
  api_version: string;
  property_types: { key: string; unit: string; colormap: string; scaling: string }[];
  methods: { id: string; name: string }[];
  plugins: { id: string; version: string }[];
}

export type SliceAxis = "x" | "y" | "z";

export interface ClipBox {
  // Fractions [0,1] of the active AABB along each axis (X, Y, Z). Render-only (doc 06 §2.4).
  min: [number, number, number];
  max: [number, number, number];
}

interface ViewerState extends LayerCollection {
  // ── layers (doc 06 §9.1, §10.1) ──────────────────────────────────────────────────
  layers: Record<string, Layer>;
  layerOrder: string[];
  selectedLayerId: string | null; // the layer the TF editor / slice target

  // ── global load status ──────────────────────────────────────────────────────────
  loading: boolean;
  error: string | null;

  // ── volume render params (doc 06 §3.1) ────────────────────────────────────────────
  steps: number;

  // ── vertical exaggeration (doc 06 §2.3) — render-only Z scale applied at the scene root
  // so terrain, volumes, slices all stretch together and stay registered. Never written
  // back into data (doc 06 §10.2). 1 == true scale.
  verticalExaggeration: number;

  // ── clip box (doc 06 §2.4) ────────────────────────────────────────────────────────
  clip: ClipBox;

  // ── orthogonal slice (doc 06 §4) — global, samples the selected layer ─────────────
  sliceEnabled: boolean;
  sliceAxis: SliceAxis;
  slicePos: number; // fraction [0,1] along the axis
  sliceOpacity: number;

  // ── M0 capabilities (minor) ───────────────────────────────────────────────────────
  capabilities: Capabilities | null;

  // ── derived: union AABB across visible volume layers (camera framing / clip basis) ──
  sceneAABB: AABB | null;

  // ── actions ───────────────────────────────────────────────────────────────────────
  setLoading: (b: boolean) => void;
  setError: (e: string | null) => void;

  // single-volume back-compat: load one volume as a (replacing) primary layer.
  loadData: (meta: PropertyModelMeta, volume: DecodedVolume) => void;

  // layer management (doc 06 §9.1).
  addVolumeLayer: (
    meta: PropertyModelMeta,
    volume: DecodedVolume,
    opts?: { id?: string; name?: string; select?: boolean },
  ) => string;
  // terrain layer (doc 06 §6, §9.1): a ground-surface layer added at the BOTTOM of the
  // stack so subsurface volumes hang beneath it. Either synthesizes a surface from the
  // project's surfaceModel string over an XY extent, or wraps a supplied DEM grid.
  addTerrainLayer: (opts?: {
    id?: string;
    name?: string;
    surfaceModelSpec?: string | null;
    extent?: XYExtent;
    res?: number;
    surface?: SurfaceGrid;
    basemap?: boolean;
    select?: boolean;
  }) => string;
  removeLayer: (id: string) => void;
  moveLayer: (id: string, dir: "up" | "down") => void;
  selectLayer: (id: string | null) => void;
  setLayerVisible: (id: string, visible: boolean) => void;
  setLayerOpacity: (id: string, opacity: number) => void;
  setLayerBlend: (id: string, blend: BlendMode) => void;
  setLayerClip: (id: string, clip: boolean) => void;
  setLayerTF: (id: string, patch: Partial<TransferFnSpec>) => void;

  setSteps: (n: number) => void;
  setVerticalExaggeration: (v: number) => void;
  setClip: (patch: Partial<ClipBox>) => void;
  setSliceEnabled: (b: boolean) => void;
  setSliceAxis: (a: SliceAxis) => void;
  setSlicePos: (p: number) => void;
  setSliceOpacity: (o: number) => void;
  setCapabilities: (c: Capabilities) => void;
}

// Recompute the union AABB over all visible volume layers (doc 06 §2.2 framing basis).
export function unionAABB(
  layers: Record<string, Layer>,
  order: string[],
): AABB | null {
  let min: [number, number, number] | null = null;
  let max: [number, number, number] | null = null;
  for (const id of order) {
    const l = layers[id];
    // Volume + terrain layers carry an Engineering AABB; both frame the camera/clip basis
    // (doc 06 §2.2) so the terrain surface is in view above the subsurface volumes.
    if (!l || !l.aabb || (l.kind !== "volume" && l.kind !== "terrain")) continue;
    if (!min || !max) {
      min = [...l.aabb.min];
      max = [...l.aabb.max];
    } else {
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], l.aabb.min[k]);
        max[k] = Math.max(max[k], l.aabb.max[k]);
      }
    }
  }
  return min && max ? { min, max } : null;
}

// Recompute the derived sceneAABB after any layer mutation.
function withScene(c: LayerCollection): {
  layers: Record<string, Layer>;
  layerOrder: string[];
  sceneAABB: AABB | null;
} {
  return {
    layers: c.layers,
    layerOrder: c.layerOrder,
    sceneAABB: unionAABB(c.layers, c.layerOrder),
  };
}

export const useViewer = create<ViewerState>((set, get) => ({
  layers: {},
  layerOrder: [],
  selectedLayerId: null,
  loading: false,
  error: null,
  steps: 256,
  verticalExaggeration: 1,
  clip: { min: [0, 0, 0], max: [1, 1, 1] },
  sliceEnabled: true,
  sliceAxis: "z",
  slicePos: 0.5,
  sliceOpacity: 1.0,
  capabilities: null,
  sceneAABB: null,

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),

  // Back-compat single-volume load: it REPLACES the "primary" layer (stable id) so the
  // default ?mock / ?id path behaves exactly as M1 did (one layer, framed + selected).
  loadData: (meta, volume) => {
    const layer = makeVolumeLayer(meta, volume, { id: "primary" });
    const next = addLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      layer,
    );
    set({
      ...withScene(next),
      selectedLayerId: "primary",
      loading: false,
      error: null,
    });
  },

  addVolumeLayer: (meta, volume, opts = {}) => {
    const layer = makeVolumeLayer(meta, volume, opts);
    const next = addLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      layer,
    );
    set({
      ...withScene(next),
      ...(opts.select === false ? {} : { selectedLayerId: layer.id }),
      loading: false,
    });
    return layer.id;
  },

  addTerrainLayer: (opts = {}) => {
    const layer = makeTerrainLayer(opts);
    const next = addLayerBottomOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      layer,
    );
    set({
      ...withScene(next),
      ...(opts.select ? { selectedLayerId: layer.id } : {}),
      loading: false,
    });
    return layer.id;
  },

  removeLayer: (id) => {
    const next = removeLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      id,
    );
    const sel = get().selectedLayerId;
    set({
      ...withScene(next),
      selectedLayerId:
        sel === id ? (next.layerOrder[next.layerOrder.length - 1] ?? null) : sel,
    });
  },

  moveLayer: (id, dir) =>
    set(
      withScene(
        moveLayerOp({ layers: get().layers, layerOrder: get().layerOrder }, id, dir),
      ),
    ),

  selectLayer: (selectedLayerId) => set({ selectedLayerId }),

  setLayerVisible: (id, visible) =>
    set(
      withScene(
        patchLayerOp(
          { layers: get().layers, layerOrder: get().layerOrder },
          id,
          { visible },
        ),
      ),
    ),

  setLayerOpacity: (id, opacity) =>
    set(
      patchLayerOp({ layers: get().layers, layerOrder: get().layerOrder }, id, {
        opacity,
      }),
    ),

  setLayerBlend: (id, blend) =>
    set(
      patchLayerOp({ layers: get().layers, layerOrder: get().layerOrder }, id, {
        blend,
      }),
    ),

  setLayerClip: (id, clip) =>
    set(
      patchLayerOp({ layers: get().layers, layerOrder: get().layerOrder }, id, {
        clip,
      }),
    ),

  setLayerTF: (id, patch) =>
    set(
      patchLayerTFOp(
        { layers: get().layers, layerOrder: get().layerOrder },
        id,
        patch,
      ),
    ),

  setSteps: (steps) => set({ steps }),
  setVerticalExaggeration: (verticalExaggeration) => set({ verticalExaggeration }),
  setClip: (patch) => set((s) => ({ clip: { ...s.clip, ...patch } })),
  setSliceEnabled: (sliceEnabled) => set({ sliceEnabled }),
  setSliceAxis: (sliceAxis) => set({ sliceAxis }),
  setSlicePos: (slicePos) => set({ slicePos }),
  setSliceOpacity: (sliceOpacity) => set({ sliceOpacity }),
  setCapabilities: (capabilities) => set({ capabilities }),
}));

// Convenience selectors (avoid re-deriving in components).
export function selectedLayer(s: ViewerState): Layer | null {
  return s.selectedLayerId ? (s.layers[s.selectedLayerId] ?? null) : null;
}

export { engineeringAABB, aabbCenter, aabbSize };
