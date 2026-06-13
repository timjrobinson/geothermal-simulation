// Viewer store (doc 06 §10). All stored positions are Engineering metres (Z-up ENU);
// vertical exaggeration and the clip box are render-time transforms, never written back
// into data (doc 06 §10.2). M1 scope: a single resident volume, one transfer function,
// one orthogonal slice, one clip box, plus the M0 capabilities fetch (kept minor).

import { create } from "zustand";
import type { PropertyModelMeta } from "./lib/api";
import type { DecodedVolume, AABB } from "./lib/volume";
import { engineeringAABB } from "./lib/volume";
import type { ScalingMode, TransferFnSpec } from "./lib/transferFn";

// M0 capabilities shape (kept for the minor capabilities readout).
export interface Capabilities {
  api_version: string;
  property_types: { key: string; unit: string; colormap: string; scaling: string }[];
  methods: { id: string; name: string }[];
  plugins: { id: string; version: string }[];
}

export type SliceAxis = "x" | "y" | "z";

export interface ClipBox {
  // Fractions [0,1] of the volume AABB along each axis (X, Y, Z). Render-only (doc 06 §2.4).
  min: [number, number, number];
  max: [number, number, number];
}

interface ViewerState {
  // ── data ────────────────────────────────────────────────────────────────────────
  meta: PropertyModelMeta | null;
  volume: DecodedVolume | null;
  aabb: AABB | null;
  loading: boolean;
  error: string | null;

  // ── transfer function (doc 06 §3.2) ───────────────────────────────────────────────
  tf: TransferFnSpec;

  // ── volume render params (doc 06 §3.1) ────────────────────────────────────────────
  steps: number;

  // ── clip box (doc 06 §2.4) ────────────────────────────────────────────────────────
  clip: ClipBox;

  // ── orthogonal slice (doc 06 §4) ──────────────────────────────────────────────────
  sliceEnabled: boolean;
  sliceAxis: SliceAxis;
  slicePos: number; // fraction [0,1] along the axis
  sliceOpacity: number;
  volumeVisible: boolean;

  // ── M0 capabilities (minor) ───────────────────────────────────────────────────────
  capabilities: Capabilities | null;

  // ── actions ───────────────────────────────────────────────────────────────────────
  setLoading: (b: boolean) => void;
  setError: (e: string | null) => void;
  loadData: (meta: PropertyModelMeta, volume: DecodedVolume) => void;
  setTF: (patch: Partial<TransferFnSpec>) => void;
  setSteps: (n: number) => void;
  setClip: (patch: Partial<ClipBox>) => void;
  setSliceEnabled: (b: boolean) => void;
  setSliceAxis: (a: SliceAxis) => void;
  setSlicePos: (p: number) => void;
  setSliceOpacity: (o: number) => void;
  setVolumeVisible: (b: boolean) => void;
  setCapabilities: (c: Capabilities) => void;
}

const DEFAULT_TF: TransferFnSpec = {
  colormap: "viridis",
  domainMin: 0,
  domainMax: 1,
  scaling: "linear",
  opacity: 0.9,
  invert: false,
};

// Seed the transfer function from the property-type registry meta (doc 06 §3.2): default
// colormap, log/linear scaling, and display range — falling back to NaN-aware stats
// (p1/p99 then min/max) when displayRange is absent (doc 04 §9.2 stats seed the clamp).
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

export const useViewer = create<ViewerState>((set) => ({
  meta: null,
  volume: null,
  aabb: null,
  loading: false,
  error: null,
  tf: DEFAULT_TF,
  steps: 256,
  clip: { min: [0, 0, 0], max: [1, 1, 1] },
  sliceEnabled: true,
  sliceAxis: "z",
  slicePos: 0.5,
  sliceOpacity: 1.0,
  volumeVisible: true,
  capabilities: null,

  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  loadData: (meta, volume) =>
    set({
      meta,
      volume,
      aabb: engineeringAABB(volume),
      tf: tfFromMeta(meta),
      loading: false,
      error: null,
    }),
  setTF: (patch) => set((s) => ({ tf: { ...s.tf, ...patch } })),
  setSteps: (steps) => set({ steps }),
  setClip: (patch) => set((s) => ({ clip: { ...s.clip, ...patch } })),
  setSliceEnabled: (sliceEnabled) => set({ sliceEnabled }),
  setSliceAxis: (sliceAxis) => set({ sliceAxis }),
  setSlicePos: (slicePos) => set({ slicePos }),
  setSliceOpacity: (sliceOpacity) => set({ sliceOpacity }),
  setVolumeVisible: (volumeVisible) => set({ volumeVisible }),
  setCapabilities: (capabilities) => set({ capabilities }),
}));
