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
  type ConfidenceModulation,
  type LayerCollection,
  addLayer as addLayerOp,
  removeLayer as removeLayerOp,
  moveLayer as moveLayerOp,
  patchLayer as patchLayerOp,
  patchLayerTF as patchLayerTFOp,
  addLayerBottom as addLayerBottomOp,
  makeVolumeLayer,
  makeTerrainLayer,
  makeFeatureLayer,
  makeWellLayer,
  makePointCloudLayer,
  makeRasterLayer,
  tfFromMeta,
  type PointCloud,
  type RasterTimeSeries,
} from "./lib/layers";
import type { WellTrajectory } from "./lib/wells";
import {
  type TimeAxis,
  type TimeWindowMode,
  type TimeWindowMs,
  buildTimeAxis,
  resolveWindowMs,
  nearestEpochIndex,
  fractionToMs,
  playheadFraction,
} from "./lib/time";
import type { SurfaceGrid, XYExtent } from "./lib/terrain";
import type { FusedModelOut, FusedSampleOut } from "./lib/fusion";
import { selectionToVolume, type VoxelReadout } from "./lib/brushing";
import type { WellReadout } from "./lib/wells";
import {
  type DesignParams,
  type RiskWeights,
  type PredictedLog,
  defaultDesignParams,
  defaultRiskWeights,
} from "./lib/planning";
import type { DrillTargetOut } from "./lib/planningApi";

// Re-export so existing importers (and tests) keep resolving these from the store.
export { tfFromMeta };
export type { Layer, BlendMode, ConfidenceModulation };

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

// A well-tube hover readout (doc 06 §5.3): the nearest-station true depths plus the well
// identity so panels/tooltips can sync. Extends the pure WellReadout with provenance.
export interface WellHoverReadout extends WellReadout {
  wellId: string | null;
  featureId: string;
  layerId: string;
}

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

  // ── well hover readout (doc 06 §5.3) — the MD/TVD/elevation of the last-hovered well tube
  // station + which well it belongs to, so a tooltip + the log-track panel sync to it. Null
  // when the pointer is off any well. Depths are TRUE (not vertically-exaggerated, doc 06 §2.3).
  wellReadout: WellHoverReadout | null;

  // ── well-planning workflow (doc 09 §8) ────────────────────────────────────────────
  // The planning panel drives a target-pick → design-solve → predict loop. `planningOpen`
  // toggles the panel. `pickTargetMode` arms the 3D scene to convert the next pick into an
  // Engineering XYZ → POST target. `planningProjectId`/`planningFusedModelId` scope the
  // backend calls (a planning session is bound to one project + fused model). `planTarget`
  // is the enriched DrillTarget (temperature/favorability/lithology readout). `designParams`
  // is the live trajectory form. `activeWellId`/`activeLayerId` link the solved planned-well
  // layer. `predictedLog` is the last POST predict response (drives tracks + tube colours +
  // outputs). `scenarios` accumulates named alternatives for the comparison table. The risk
  // weights are the §7.4 glass-box composite, editable in the panel.
  planningOpen: boolean;
  pickTargetMode: boolean;
  // A transient picked Engineering XYZ from the 3D scene (PickTargetLayer) the panel consumes
  // to POST a target. Cleared once the panel has handled it.
  pendingPickXYZ: [number, number, number] | null;
  planningProjectId: string | null;
  planningFusedModelId: string | null;
  planTarget: DrillTargetOut | null;
  designParams: DesignParams;
  riskWeights: RiskWeights;
  activeWellId: string | null;
  activeLayerId: string | null;
  predictedLog: PredictedLog | null;
  scenarios: PlanScenario[];

  // ── global 4-D time slider (doc 06 §9.4) ──────────────────────────────────────────
  // The time axis is the UNION of every time-bearing layer's epochs (microseismic, InSAR,
  // repeat surveys). The playhead is a single instant (ms); `timeWindowMode` expands it to
  // the inclusive [t0,t1] each layer filters against (instant / cumulative / rolling). The
  // window drives uniforms (microseismic uTimeWindow, InSAR frame select) — NO per-tick
  // geometry rebuild. `timePlaying`/`timeSpeed` drive the play loop (the Scene advances the
  // playhead each frame). `timeRollingWidthMs` is the rolling-window span.
  timeAxis: TimeAxis;
  timePlayheadMs: number;
  timeWindowMode: TimeWindowMode;
  timeRollingWidthMs: number;
  timePlaying: boolean;
  timeSpeed: number; // axis-ms advanced per real second when playing

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
  // Bind / update / clear a layer's confidence-modulated opacity (doc 07 §5.3 honest view).
  // Pass null to remove the binding (layer renders unmodulated).
  setLayerConfidence: (id: string, conf: ConfidenceModulation | null) => void;

  // feature + 4-D layer adders (doc 06 §5, §9.4). Each appends a layer and (optionally)
  // contributes its epochs to the global time axis.
  addFeatureLayer: (opts: {
    featureId: string;
    featureKind: string;
    id?: string;
    name?: string;
    datasetId?: string;
    select?: boolean;
  }) => string;
  addWellLayer: (
    trajectory: WellTrajectory,
    opts?: { id?: string; name?: string; datasetId?: string; logProperty?: string | null; dlsMax_deg30m?: number; select?: boolean },
  ) => string;
  addPointCloudLayer: (
    cloud: PointCloud,
    opts: { featureId: string; epochs?: string[]; id?: string; name?: string; datasetId?: string; select?: boolean },
  ) => string;
  addRasterLayer: (
    raster: RasterTimeSeries,
    opts?: { featureId?: string; id?: string; name?: string; datasetId?: string; select?: boolean },
  ) => string;
  setLayerLogProperty: (id: string, property: string | null) => void;

  // time-slider actions (doc 06 §9.4). `setTimeAxis` replaces the axis (e.g. from
  // GET /projects/{pid}/time-extent); `mergeTimeEpochs` unions in a newly-added layer's
  // epochs. The playhead is set in ms or as a [0,1] fraction (slider scrub).
  setTimeAxis: (epochLists: string[][]) => void;
  mergeTimeEpochs: (epochs: string[]) => void;
  setTimePlayhead: (ms: number) => void;
  setTimePlayheadFraction: (fraction: number) => void;
  setTimeWindowMode: (mode: TimeWindowMode) => void;
  setTimeRollingWidthMs: (ms: number) => void;
  setTimePlaying: (playing: boolean) => void;
  setTimeSpeed: (speed: number) => void;
  // Advance the playhead by `dtMs` of axis time (the play loop tick); wraps to the axis
  // start at the end. Also re-snaps each raster layer's frameIndex to the new playhead.
  tickTime: (dtMs: number) => void;

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
  setWellReadout: (r: WellHoverReadout | null) => void;

  // ── planning actions (doc 09 §8) ──────────────────────────────────────────────────
  setPlanningOpen: (b: boolean) => void;
  setPickTargetMode: (b: boolean) => void;
  setPendingPickXYZ: (xyz: [number, number, number] | null) => void;
  setPlanningContext: (projectId: string | null, fusedModelId: string | null) => void;
  setPlanTarget: (t: DrillTargetOut | null) => void;
  setDesignParams: (patch: Partial<DesignParams>) => void;
  setRiskWeights: (patch: Partial<RiskWeights>) => void;
  setActiveWell: (wellId: string | null, layerId: string | null) => void;
  setPredictedLog: (log: PredictedLog | null) => void;
  // Snapshot the active well + its predicted log as a named comparison scenario (doc 09 §8.2).
  saveScenario: (name: string, maxDLS_deg30m: number) => void;
  removeScenario: (wellId: string) => void;
}

// A saved alternative-scenario snapshot for the comparison table (doc 09 §8.2). Holds the
// metrics-bearing predicted log + the design params so a scenario can be reloaded/compared.
export interface PlanScenario {
  wellId: string;
  layerId: string | null;
  name: string;
  design: DesignParams;
  log: PredictedLog;
  maxDLS_deg30m: number;
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

// Re-snap every raster (InSAR) layer's `frameIndex` to the nearest axis epoch for the new
// playhead (doc 06 §9.4 frame select). Returns a fresh `layers` map only when something
// changed, so unrelated layers keep their identity (no spurious re-renders).
function snapRasters(
  layers: Record<string, Layer>,
  order: string[],
  playheadMs: number,
): { layers: Record<string, Layer> } | Record<string, never> {
  let next: Record<string, Layer> | null = null;
  for (const id of order) {
    const l = layers[id];
    if (!l || l.kind !== "raster" || !l.raster) continue;
    const idx = nearestEpochIndex(
      { epochs: l.raster.epochs, epochMs: l.raster.epochMs, t0Ms: null, t1Ms: null },
      playheadMs,
    );
    if (idx < 0 || idx === l.raster.frameIndex) continue;
    next ??= { ...layers };
    next[id] = { ...l, raster: { ...l.raster, frameIndex: idx } };
  }
  return next ? { layers: next } : {};
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

  timeAxis: buildTimeAxis(),
  timePlayheadMs: 0,
  timeWindowMode: "cumulative",
  timeRollingWidthMs: 0,
  timePlaying: false,
  timeSpeed: 0,

  analysisOpen: false,
  fusedGrid: null,
  fusedSample: null,
  selection: [],
  pickedVoxel: null,
  wellReadout: null,

  planningOpen: false,
  pickTargetMode: false,
  pendingPickXYZ: null,
  planningProjectId: null,
  planningFusedModelId: null,
  planTarget: null,
  designParams: defaultDesignParams(),
  riskWeights: defaultRiskWeights(),
  activeWellId: null,
  activeLayerId: null,
  predictedLog: null,
  scenarios: [],

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

  addFeatureLayer: (opts) => {
    const layer = makeFeatureLayer(opts);
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

  addWellLayer: (trajectory, opts = {}) => {
    const layer = makeWellLayer(trajectory, opts);
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

  addPointCloudLayer: (cloud, opts) => {
    const layer = makePointCloudLayer(cloud, opts);
    const next = addLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      layer,
    );
    set({
      ...withScene(next),
      ...(opts.select === false ? {} : { selectedLayerId: layer.id }),
      loading: false,
    });
    if (opts.epochs && opts.epochs.length) get().mergeTimeEpochs(opts.epochs);
    return layer.id;
  },

  addRasterLayer: (raster, opts = {}) => {
    const layer = makeRasterLayer(raster, opts);
    const next = addLayerOp(
      { layers: get().layers, layerOrder: get().layerOrder },
      layer,
    );
    set({
      ...withScene(next),
      ...(opts.select === false ? {} : { selectedLayerId: layer.id }),
      loading: false,
    });
    if (raster.epochs.length) get().mergeTimeEpochs(raster.epochs);
    return layer.id;
  },

  setLayerLogProperty: (id, logProperty) =>
    set(
      patchLayerOp({ layers: get().layers, layerOrder: get().layerOrder }, id, {
        logProperty,
      }),
    ),

  setTimeAxis: (epochLists) => {
    const axis = buildTimeAxis(...epochLists);
    set({
      timeAxis: axis,
      timePlayheadMs: axis.t1Ms ?? 0,
      ...snapRasters(get().layers, get().layerOrder, axis.t1Ms ?? 0),
    });
  },

  mergeTimeEpochs: (epochs) => {
    const axis = buildTimeAxis(get().timeAxis.epochs, epochs);
    // Keep the playhead where it is unless the axis was previously empty (then jump to end).
    const ph = get().timeAxis.t1Ms == null ? (axis.t1Ms ?? 0) : get().timePlayheadMs;
    set({
      timeAxis: axis,
      timePlayheadMs: ph,
      ...snapRasters(get().layers, get().layerOrder, ph),
    });
  },

  setTimePlayhead: (ms) =>
    set({
      timePlayheadMs: ms,
      ...snapRasters(get().layers, get().layerOrder, ms),
    }),

  setTimePlayheadFraction: (fraction) => {
    const ms = fractionToMs(get().timeAxis, fraction);
    set({ timePlayheadMs: ms, ...snapRasters(get().layers, get().layerOrder, ms) });
  },

  setTimeWindowMode: (timeWindowMode) => set({ timeWindowMode }),
  setTimeRollingWidthMs: (timeRollingWidthMs) => set({ timeRollingWidthMs }),
  setTimePlaying: (timePlaying) => set({ timePlaying }),
  setTimeSpeed: (timeSpeed) => set({ timeSpeed }),

  tickTime: (dtMs) => {
    const { timeAxis, timePlayheadMs } = get();
    if (timeAxis.t0Ms == null || timeAxis.t1Ms == null) return;
    let ms = timePlayheadMs + dtMs;
    if (ms > timeAxis.t1Ms) ms = timeAxis.t0Ms; // loop
    if (ms < timeAxis.t0Ms) ms = timeAxis.t0Ms;
    set({ timePlayheadMs: ms, ...snapRasters(get().layers, get().layerOrder, ms) });
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

  setLayerConfidence: (id, conf) =>
    set(
      patchLayerOp({ layers: get().layers, layerOrder: get().layerOrder }, id, {
        confidence: conf ?? undefined,
      }),
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
  setWellReadout: (wellReadout) => set({ wellReadout }),

  // ── planning actions (doc 09 §8) ──────────────────────────────────────────────────
  setPlanningOpen: (planningOpen) => set({ planningOpen }),
  setPickTargetMode: (pickTargetMode) => set({ pickTargetMode }),
  setPendingPickXYZ: (pendingPickXYZ) => set({ pendingPickXYZ }),
  setPlanningContext: (planningProjectId, planningFusedModelId) =>
    set({ planningProjectId, planningFusedModelId }),
  setPlanTarget: (planTarget) => set({ planTarget }),
  setDesignParams: (patch) =>
    set((s) => ({ designParams: { ...s.designParams, ...patch } })),
  setRiskWeights: (patch) =>
    set((s) => ({ riskWeights: { ...s.riskWeights, ...patch } })),
  setActiveWell: (activeWellId, activeLayerId) => set({ activeWellId, activeLayerId }),
  setPredictedLog: (predictedLog) => set({ predictedLog }),

  saveScenario: (name, maxDLS_deg30m) => {
    const { activeWellId, activeLayerId, designParams, predictedLog } = get();
    if (!activeWellId || !predictedLog) return;
    const scenario: PlanScenario = {
      wellId: activeWellId,
      layerId: activeLayerId,
      name,
      design: { ...designParams },
      log: predictedLog,
      maxDLS_deg30m,
    };
    // Replace an existing snapshot for the same well, else append.
    const rest = get().scenarios.filter((sc) => sc.wellId !== activeWellId);
    set({ scenarios: [...rest, scenario] });
  },

  removeScenario: (wellId) =>
    set((s) => ({ scenarios: s.scenarios.filter((sc) => sc.wellId !== wellId) })),
}));

// Convenience selectors (avoid re-deriving in components).
export function selectedLayer(s: ViewerState): Layer | null {
  return s.selectedLayerId ? (s.layers[s.selectedLayerId] ?? null) : null;
}

// The resolved active time window in epoch-ms (doc 06 §9.4) — what the microseismic cloud's
// uTimeWindow uniform and any CPU parity check consume. Derived from the playhead + mode.
export function currentTimeWindow(s: ViewerState): TimeWindowMs {
  return resolveWindowMs(
    s.timeWindowMode,
    s.timePlayheadMs,
    s.timeAxis.t0Ms,
    s.timeRollingWidthMs,
  );
}

// Playhead position as a [0,1] fraction along the axis (slider rendering).
export function timePlayheadFraction(s: ViewerState): number {
  return playheadFraction(s.timeAxis, s.timePlayheadMs);
}

export { engineeringAABB, aabbCenter, aabbSize };
