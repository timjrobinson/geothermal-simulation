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
import type { FusedModelOut, FusedSampleOut } from "./lib/fusion";
import { selectionToVolume, type VoxelReadout } from "./lib/brushing";

// Re-export so existing importers (and tests) keep resolving these from the store.
export { tfFromMeta };
export type { Layer, BlendMode };

// Stable layer id for the cross-plot selection-mask overlay (doc 06 §10.3). One overlay is
// re-used (replaced in place) as the brush changes so it never accumulates layers.
export const SELECTION_LAYER_ID = "selection-mask";

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

  // ── analysis / linked brushing (doc 06 §10.3, doc 07 §3) ──────────────────────────
  // The active fused grid + its co-located sample feed the cross-plot / histogram /
  // correlation panels. `selection` is the set of LOCAL sample-row indices brushed in the
  // cross-plot; it drives the 3D selection-mask overlay layer. `pickedVoxel` is the 3D pick
  // → multi-property inspector readout. `analysisOpen` toggles the panel.
  analysisOpen: boolean;
  fusedGrid: FusedModelOut | null;
  fusedSample: FusedSampleOut | null;
  selection: number[]; // local sample-row indices (brushing key)
  pickedVoxel: VoxelReadout | null;

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

  // analysis / linked brushing actions (doc 06 §10.3).
  setAnalysisOpen: (b: boolean) => void;
  setFusedAnalysis: (grid: FusedModelOut | null, sample: FusedSampleOut | null) => void;
  // Brush a set of cross-plot rows -> store the selection AND (re)build the 3D overlay
  // layer so the brushed voxels highlight in the scene. An empty selection clears both.
  setSelection: (rows: number[]) => void;
  clearSelection: () => void;
  setPickedVoxel: (v: VoxelReadout | null) => void;
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

  analysisOpen: false,
  fusedGrid: null,
  fusedSample: null,
  selection: [],
  pickedVoxel: null,

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

  setAnalysisOpen: (analysisOpen) => set({ analysisOpen }),

  setFusedAnalysis: (fusedGrid, fusedSample) =>
    set({ fusedGrid, fusedSample, selection: [], pickedVoxel: null }),

  setSelection: (rows) => {
    const { fusedGrid, fusedSample } = get();
    // Drop any existing overlay first so we rebuild it fresh (or clear it).
    let next: LayerCollection = removeLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      SELECTION_LAYER_ID,
    );
    if (rows.length > 0 && fusedGrid && fusedSample) {
      // Build a highlight overlay volume: selected cells = 1.0, rest = NaN (doc 06 §10.3).
      const vol = selectionToVolume(fusedSample, rows, {
        origin: fusedGrid.origin,
        spacing: fusedGrid.spacing,
      });
      const overlayMeta: PropertyModelMeta = {
        id: SELECTION_LAYER_ID,
        property: "selection",
        canonicalUnit: "",
        scaling: "linear",
        colormap: "magma",
        displayRange: [0, 1],
        shape: vol.shape as [number, number, number],
        origin: vol.origin as [number, number, number],
        spacing: vol.spacing as [number, number, number],
        levels: 1,
        stats: { min: 1, max: 1, p1: 1, p99: 1 },
        frame: null,
        hasSigma: false,
      };
      const layer = makeVolumeLayer(overlayMeta, vol, {
        id: SELECTION_LAYER_ID,
        name: `Selection (${rows.length} cells)`,
      });
      // A bright, opaque, additive highlight that floats above the source volumes.
      layer.blend = "additive";
      layer.opacity = 1;
      layer.clip = false;
      layer.transferFn = {
        ...layer.transferFn,
        colormap: "magma",
        domainMin: 0,
        domainMax: 1,
        opacity: 1,
      };
      next = addLayerOp(next, layer);
    }
    set({ ...withScene(next), selection: rows });
  },

  clearSelection: () => {
    const next = removeLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      SELECTION_LAYER_ID,
    );
    set({ ...withScene(next), selection: [] });
  },

  setPickedVoxel: (pickedVoxel) => set({ pickedVoxel }),
}));

// Convenience selectors (avoid re-deriving in components).
export function selectedLayer(s: ViewerState): Layer | null {
  return s.selectedLayerId ? (s.layers[s.selectedLayerId] ?? null) : null;
}

export { engineeringAABB, aabbCenter, aabbSize };
