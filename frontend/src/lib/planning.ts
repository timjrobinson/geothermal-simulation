// Well-planning data shaping (doc 09 §4, §5, §7, §8). PURE — no THREE / no DOM / no fetch —
// so the design-param→intent payload, the predicted-log→tube/track shaping, and the
// DLS-segment colour-threshold logic are unit-testable headlessly (npm test). The planning
// UI (ui/PlanningPanel) and the scene (WellLayer per-segment colours) feed these arrays
// straight into the backend POST bodies / THREE geometry colours.
//
// Backend contract mirrors geosim/api/planning.py exactly:
//   POST /projects/{pid}/wells   body: { name, wellhead:[x,y], kb_elev_m, target_ids?,
//                                        design{...} | survey[[MD,inc,azi]], max_dls_deg30m }
//   POST /wells/{wid}/solve      body: { design{...}, max_dls_deg30m, max_inc_deg }
//   POST /wells/{wid}/predict    body: { fused_model_id, md_step_m, target_id?, risk_weights? }
// and the response payloads (camelCase) from _solve_payload / PredictedLog.to_payload /
// DrillTarget.to_payload.

import type { Vec3, WellTrajectory, WellLogs } from "./wells";
import { resolveColormap, sampleColormap } from "./colormaps";

// ── Design params (the planning panel form state) → intent payload (doc 09 §4.4) ──────────

// Trajectory design method (doc 09 §4.2). Matches the backend DesignSpec.method strings.
export type DesignMethod = "vertical" | "build-hold-land" | "s-curve";

export const DESIGN_METHODS: readonly DesignMethod[] = [
  "vertical",
  "build-hold-land",
  "s-curve",
];

// The planning-panel form state (camelCase UI-side). `target` is the picked/entered
// Engineering XYZ the solver lands on (null for a pure-vertical well to TD).
export interface DesignParams {
  method: DesignMethod;
  target: Vec3 | null;
  kopMD_m: number; // kick-off-point measured depth
  buildRate_deg30m: number; // build rate (DLS) in the build section
  dropRate_deg30m?: number; // S-curve drop rate
  holdInc_deg?: number | null; // hold-section inclination (build-hold-land)
  landingInc_deg?: number | null; // inclination at the landing point
  stationStep_m?: number; // survey densification step
  maxDLS_deg30m: number; // the DLS ceiling (constraint + per-segment red threshold)
  maxInc_deg?: number; // inclination ceiling (constraint)
}

// Default planning params (a vertical well, DLS ceiling 5°/30m canonical, doc 09 §4.4).
export function defaultDesignParams(): DesignParams {
  return {
    method: "build-hold-land",
    target: null,
    kopMD_m: 500,
    buildRate_deg30m: 3,
    dropRate_deg30m: 3,
    holdInc_deg: null,
    landingInc_deg: 90,
    stationStep_m: 30,
    maxDLS_deg30m: 5,
    maxInc_deg: 92,
  };
}

// The `design` sub-object of the POST body (snake_case keys the backend _design_from_dict
// reads — it accepts both snake and camel; we emit snake to be canonical). `target` is a
// flat [x,y,z] list (or omitted for a vertical well). Numeric-only — no nulls leak through
// except landing/hold which the backend treats as "unset".
export interface DesignIntent {
  method: DesignMethod;
  target?: [number, number, number];
  kop_md_m: number;
  build_rate_deg30m: number;
  drop_rate_deg30m: number;
  hold_inc_deg?: number;
  landing_inc_deg?: number;
  station_step_m: number;
}

// Shape the planning-panel form state into the `design` intent the solver consumes (doc 09
// §4.4). Drops null/undefined optionals (the backend defaults them) and only carries a
// target when one is set. A vertical method needs no target/landing — they are dropped.
export function designToIntent(p: DesignParams): DesignIntent {
  const out: DesignIntent = {
    method: p.method,
    kop_md_m: p.kopMD_m,
    build_rate_deg30m: p.buildRate_deg30m,
    drop_rate_deg30m: p.dropRate_deg30m ?? p.buildRate_deg30m,
    station_step_m: p.stationStep_m ?? 30,
  };
  if (p.method !== "vertical" && p.target) {
    out.target = [p.target[0], p.target[1], p.target[2]];
  }
  if (p.holdInc_deg != null) out.hold_inc_deg = p.holdInc_deg;
  if (p.landingInc_deg != null && p.method !== "vertical") {
    out.landing_inc_deg = p.landingInc_deg;
  }
  return out;
}

// The full POST /projects/{pid}/wells body (intent mode). The wellhead is the surface [x,y];
// `target_ids` links the created DrillTargets so predict can score target intersection.
export interface WellCreateBody {
  name: string;
  wellhead: [number, number];
  kb_elev_m: number;
  target_ids?: string[];
  design: DesignIntent;
  max_dls_deg30m: number;
  max_inc_deg: number;
}

export function wellCreateBody(
  name: string,
  wellhead: [number, number],
  p: DesignParams,
  opts: { kbElev_m?: number; targetIds?: string[] } = {},
): WellCreateBody {
  return {
    name,
    wellhead,
    kb_elev_m: opts.kbElev_m ?? 0,
    ...(opts.targetIds && opts.targetIds.length ? { target_ids: opts.targetIds } : {}),
    design: designToIntent(p),
    max_dls_deg30m: p.maxDLS_deg30m,
    max_inc_deg: p.maxInc_deg ?? 92,
  };
}

// The POST /wells/{wid}/solve body (re-solve on a target move / param edit, doc 09 §8.1).
export interface SolveBody {
  design: DesignIntent;
  max_dls_deg30m: number;
  max_inc_deg: number;
}

export function solveBody(p: DesignParams): SolveBody {
  return {
    design: designToIntent(p),
    max_dls_deg30m: p.maxDLS_deg30m,
    max_inc_deg: p.maxInc_deg ?? 92,
  };
}

// The POST /wells/{wid}/predict body (doc 09 §5–§7). Risk weights are the §7.4 glass-box
// blend; omitted → backend defaults (0.40/0.30/0.10/0.20).
export interface RiskWeights {
  tempConfidence: number;
  hazard: number;
  dlsExceedance: number;
  structuralUncertainty: number;
}

export function defaultRiskWeights(): RiskWeights {
  return {
    tempConfidence: 0.4,
    hazard: 0.3,
    dlsExceedance: 0.1,
    structuralUncertainty: 0.2,
  };
}

export interface PredictBody {
  fused_model_id: string;
  md_step_m: number;
  target_id?: string;
  risk_weights?: RiskWeights;
  favorability_threshold: number;
  fracture_threshold: number;
}

export function predictBody(
  fusedModelId: string,
  opts: {
    mdStep_m?: number;
    targetId?: string | null;
    riskWeights?: RiskWeights;
    favorabilityThreshold?: number;
    fractureThreshold?: number;
  } = {},
): PredictBody {
  return {
    fused_model_id: fusedModelId,
    md_step_m: opts.mdStep_m ?? 5,
    ...(opts.targetId ? { target_id: opts.targetId } : {}),
    ...(opts.riskWeights ? { risk_weights: opts.riskWeights } : {}),
    favorability_threshold: opts.favorabilityThreshold ?? 0.7,
    fracture_threshold: opts.fractureThreshold ?? 0.5,
  };
}

// ── Solve response → trajectory (the planned well is a deviation survey, doc 09 §4.1) ─────

// The _solve_payload response shape (geosim/api/planning.py).
export interface SolveResult {
  survey: number[][]; // [[MD, inc°, azi°], …]
  maxDLS_deg30m: number;
  dlsExceeded: boolean;
  maxInc_deg: number;
  incExceeded: boolean;
  landingError_m: number;
  method: string;
}

// The GET /wells/{wid}/positions response shape (geosim/api/planning.py well_positions_route).
export interface WellPositions {
  wellId: string;
  md: number[];
  tvd: number[];
  enu: number[][]; // Engineering XYZ per station [[x,y,z], …]
  dls: number[]; // per-station DLS (°/30m)
  drillability?: DrillabilityFlag;
}

// Build a renderable WellTrajectory (lib/wells.ts) from a positions payload (doc 06 §5.3 +
// doc 09 §4.3). The predicted log is joined in separately via predictedLogToLogs.
export function positionsToTrajectory(
  pos: WellPositions,
  opts: { featureId: string; wellhead?: number[]; logs?: WellLogs } = { featureId: "" },
): WellTrajectory {
  const polyline: Vec3[] = pos.enu.map((p) => [p[0], p[1], p[2]] as Vec3);
  return {
    featureId: opts.featureId,
    wellId: pos.wellId,
    wellhead: opts.wellhead ?? (polyline[0] ? [polyline[0][0], polyline[0][1]] : [0, 0]),
    polyline,
    md: pos.md,
    tvd: pos.tvd,
    dls: pos.dls,
    logs: opts.logs ?? emptyLogs(pos.wellId),
  };
}

function emptyLogs(wellId: string | null): WellLogs {
  return { wellId, md: [], curves: {}, primaryProperty: null };
}

// ── DLS-segment colouring (doc 09 §8.1 — "tube segments exceeding maxDLS render red") ─────

// Per-segment colour decision for the well tube. A segment spans station i→i+1; its DLS is
// the dogleg at the *end* station (the backend reports per-station DLS, the rate to reach it).
// Segments at/over the ceiling render red (constraint violation), others neutral. Exposed as
// a flat array so the scene can paint per-segment without re-deriving the threshold logic.

export type Vec3Color = [number, number, number];

// Default tube colours (Catppuccin): neutral amber spine, red over-DLS, dim out-of-window.
export const DLS_OK_COLOR: Vec3Color = [0.976, 0.886, 0.686]; // #f9e2af amber
export const DLS_OVER_COLOR: Vec3Color = [0.953, 0.545, 0.659]; // #f38ba8 red

// Classify each inter-station segment as over/under the DLS ceiling. `dls[i]` is the dogleg
// severity to reach station i; a segment i→i+1 is "over" when the dogleg at its terminating
// station (i+1) exceeds the ceiling. Returns one boolean per segment (length = stations−1).
export function dlsSegmentExceeded(
  dls: readonly number[],
  maxDLS_deg30m: number,
): boolean[] {
  const segs: boolean[] = [];
  for (let i = 0; i + 1 < dls.length; i++) {
    // The dogleg accrues over the segment ending at i+1; flag the segment by that value.
    segs.push(dls[i + 1] > maxDLS_deg30m);
  }
  return segs;
}

// Per-segment RGB colours for the tube spine (one colour per segment). Over-DLS segments
// take `overColor`; the rest take `okColor`. This is the per-segment override doc 09 §8.1
// asks doc 06 for ("per-segment color override for DLS flags").
export function dlsSegmentColors(
  dls: readonly number[],
  maxDLS_deg30m: number,
  okColor: Vec3Color = DLS_OK_COLOR,
  overColor: Vec3Color = DLS_OVER_COLOR,
): Vec3Color[] {
  return dlsSegmentExceeded(dls, maxDLS_deg30m).map((over) =>
    over ? overColor : okColor,
  );
}

// Expand per-segment colours to a per-tube-vertex Float32 RGB array (length 3*ringCount),
// for a TubeGeometry laid out ring-by-ring along the spine (doc 06 §5.3). `ringCount` is the
// number of rings (tubularSegments+1). Ring r maps to the segment containing fraction
// r/(ringCount-1) of the spine. Falls back to the OK colour when there are no segments.
export function dlsRingColors(
  dls: readonly number[],
  maxDLS_deg30m: number,
  ringCount: number,
  okColor: Vec3Color = DLS_OK_COLOR,
  overColor: Vec3Color = DLS_OVER_COLOR,
): Float32Array {
  const segColors = dlsSegmentColors(dls, maxDLS_deg30m, okColor, overColor);
  const out = new Float32Array(ringCount * 3);
  const nSeg = segColors.length;
  for (let r = 0; r < ringCount; r++) {
    const f = ringCount > 1 ? r / (ringCount - 1) : 0;
    // Ring fraction → segment index (clamp the last ring into the final segment).
    let seg = nSeg > 0 ? Math.min(nSeg - 1, Math.floor(f * nSeg)) : -1;
    if (seg < 0) {
      out[r * 3] = okColor[0];
      out[r * 3 + 1] = okColor[1];
      out[r * 3 + 2] = okColor[2];
      continue;
    }
    const c = segColors[seg];
    out[r * 3] = c[0];
    out[r * 3 + 1] = c[1];
    out[r * 3 + 2] = c[2];
  }
  return out;
}

// ── Predicted log → 2D tracks + tube logs (doc 09 §5.2, doc 06 §5.3/§10.3) ────────────────

// One station of the predicted log (PredictedLog.to_payload → stations[i]). Values are
// {value, sigma, confidence} per property; lithology is categorical.
export interface SampledValue {
  value: number | null;
  sigma?: number | null;
  confidence?: number | null;
}

export interface PredictedStation {
  md: number;
  tvd: number;
  z: number;
  x: number;
  y: number;
  values: Record<string, SampledValue>;
  lithology: string | null;
  hazards: Record<string, number>;
  distToNearestFault_m: number | null;
  risk: number;
  riskDrivers: Record<string, number>;
}

export interface DrillabilityCheck {
  name: string;
  verdict: "ok" | "warn";
  value: number;
  limit: number;
  mdInterval_m?: [number, number];
}

export interface DrillabilityFlag {
  verdict: "ok" | "warn";
  checks: DrillabilityCheck[];
}

export interface GeothermalSummary {
  bhtC: number | null;
  bhtSigmaC: number | null;
  bhtConfidence: number | null;
  maxTempC: number | null;
  maxTempMD_m: number | null;
  maxTempTVD_m: number | null;
  targetIntersectionLength_m: number;
  reservoirIntersectionLength_m: number;
  productiveFractureIntersections: number;
  fractureIntersectionMDs_m: number[];
  inWindowFraction: number;
  meanRisk: number;
  peakRisk: number;
}

export interface PredictedLog {
  wellId: string;
  modelVersion: string;
  mdStep_m: number;
  stations: PredictedStation[];
  summary: GeothermalSummary;
  riskWeights: RiskWeights;
  drillability?: DrillabilityFlag;
}

// A single 2D log-track series (md vs value, with an optional ±σ uncertainty band, doc 09
// §5.2). `min`/`max` are the auto-scaled value domain. NaN samples are dropped (gaps).
export interface LogTrack {
  property: string;
  unit: string;
  md: number[];
  value: number[];
  lower?: number[]; // value − σ (uncertainty band lower edge), parallel to value
  upper?: number[]; // value + σ
  min: number;
  max: number;
  hasBand: boolean;
}

// Extract a numeric per-station series for `property` from the predicted log, building a 2D
// track with a ±σ uncertainty band when σ is present (doc 09 §5.2 shaded bands). Stations
// whose value is null are skipped so the track only spans measured depths.
export function predictedTrack(
  log: PredictedLog,
  property: string,
  unit = "",
): LogTrack {
  const md: number[] = [];
  const value: number[] = [];
  const lower: number[] = [];
  const upper: number[] = [];
  let hasBand = false;
  let min = Infinity;
  let max = -Infinity;
  for (const s of log.stations) {
    const sv = s.values[property];
    if (!sv || sv.value == null || !Number.isFinite(sv.value)) continue;
    const v = sv.value;
    md.push(s.md);
    value.push(v);
    const sig = sv.sigma != null && Number.isFinite(sv.sigma) ? Math.abs(sv.sigma) : 0;
    if (sig > 0) hasBand = true;
    lower.push(v - sig);
    upper.push(v + sig);
    if (v - sig < min) min = v - sig;
    if (v + sig > max) max = v + sig;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0;
    max = 1;
  } else if (max <= min) {
    max = min + 1;
  }
  return {
    property,
    unit,
    md,
    value,
    ...(hasBand ? { lower, upper } : {}),
    min,
    max,
    hasBand,
  };
}

// The per-station risk track (doc 09 §7.4). Risk is always in [0,1]; we fix the domain so the
// track reads comparably across wells (a glass-box scale, not auto-stretched).
export function riskTrack(log: PredictedLog): LogTrack {
  const md = log.stations.map((s) => s.md);
  const value = log.stations.map((s) => s.risk);
  return { property: "risk", unit: "", md, value, min: 0, max: 1, hasBand: false };
}

// A categorical lithology fill track (doc 09 §5.2 "lithology fill"). Returns the contiguous
// MD intervals per lithology class + a stable colour per class, for a filled column.
export interface LithInterval {
  lithology: string;
  mdTop: number;
  mdBottom: number;
  color: Vec3Color;
}

// A small categorical palette (Catppuccin accents) cycled per distinct lithology class.
const LITH_PALETTE: Vec3Color[] = [
  [0.537, 0.706, 0.98], // blue
  [0.651, 0.89, 0.631], // green
  [0.98, 0.702, 0.529], // peach
  [0.796, 0.651, 0.969], // mauve
  [0.976, 0.886, 0.686], // yellow
  [0.576, 0.886, 0.831], // teal
  [0.953, 0.545, 0.659], // red
];

export function lithologyColor(_lith: string, index: number): Vec3Color {
  return LITH_PALETTE[index % LITH_PALETTE.length];
}

// Collapse the per-station lithology classes into contiguous fill intervals (doc 09 §5.2).
// Adjacent stations with the same class merge; each distinct class gets a stable palette
// colour (assigned in first-seen order). Null lithology stations break a run.
export function lithologyIntervals(log: PredictedLog): LithInterval[] {
  const classIndex = new Map<string, number>();
  const out: LithInterval[] = [];
  let runLith: string | null = null;
  let runTop = 0;
  let prevMd = 0;
  for (const s of log.stations) {
    const lith = s.lithology;
    if (lith !== runLith) {
      if (runLith != null) {
        out.push(makeInterval(runLith, runTop, prevMd, classIndex));
      }
      runLith = lith;
      runTop = s.md;
    }
    prevMd = s.md;
  }
  if (runLith != null) out.push(makeInterval(runLith, runTop, prevMd, classIndex));
  return out;
}

function makeInterval(
  lith: string,
  top: number,
  bottom: number,
  classIndex: Map<string, number>,
): LithInterval {
  let idx = classIndex.get(lith);
  if (idx === undefined) {
    idx = classIndex.size;
    classIndex.set(lith, idx);
  }
  return { lithology: lith, mdTop: top, mdBottom: bottom, color: lithologyColor(lith, idx) };
}

// Build the joined WellLogs (lib/wells.ts) from the predicted log so the existing tube
// colour-by-curve path (resampleCurveToStations) can paint the predicted temperature /
// favorability / risk straight onto the tube (doc 06 §5.3). One curve per numeric property
// found on the stations, plus a synthetic "risk" curve.
export function predictedLogToLogs(
  log: PredictedLog,
  primaryProperty = "temperatureC",
): WellLogs {
  const md = log.stations.map((s) => s.md);
  const props = new Set<string>();
  for (const s of log.stations) for (const k of Object.keys(s.values)) props.add(k);
  const curves: Record<string, number[]> = {};
  for (const p of props) {
    curves[p] = log.stations.map((s) => {
      const sv = s.values[p];
      return sv && sv.value != null ? sv.value : NaN;
    });
  }
  curves.risk = log.stations.map((s) => s.risk);
  const primary = curves[primaryProperty]
    ? primaryProperty
    : (Object.keys(curves)[0] ?? null);
  return { wellId: log.wellId, md, curves, primaryProperty: primary };
}

// ── Glass-box risk driver breakdown (doc 09 §7.4 "driver breakdown always shown") ─────────

export interface RiskDriver {
  name: string;
  contribution: number; // weight·term contribution to the composite
  fraction: number; // share of the total risk (0..1)
}

// Aggregate the per-station riskDrivers into a well-level breakdown (mean contribution per
// driver), sorted by contribution descending — "what's driving risk" (doc 09 §7.4). Each
// station's riskDrivers already carry the weighted contributions; we average over stations.
export function riskDriverBreakdown(log: PredictedLog): RiskDriver[] {
  const sums = new Map<string, number>();
  const n = log.stations.length || 1;
  for (const s of log.stations) {
    for (const [k, v] of Object.entries(s.riskDrivers)) {
      sums.set(k, (sums.get(k) ?? 0) + (Number.isFinite(v) ? v : 0));
    }
  }
  let total = 0;
  const means: { name: string; contribution: number }[] = [];
  for (const [k, sum] of sums) {
    const mean = sum / n;
    means.push({ name: k, contribution: mean });
    total += mean;
  }
  means.sort((a, b) => b.contribution - a.contribution);
  return means.map((m) => ({
    name: m.name,
    contribution: m.contribution,
    fraction: total > 0 ? m.contribution / total : 0,
  }));
}

// ── Alternative-scenario comparison (doc 09 §8.2) ─────────────────────────────────────────

// One comparison-table row per scenario (a named PlannedWell). Best-in-column is decided by
// the panel (higher-is-better for pay/BHT/fractures; lower for risk/DLS/MD).
export interface ScenarioRow {
  wellId: string;
  name: string;
  bhtC: number | null;
  payLength_m: number;
  reservoirLength_m: number;
  fractureIntersections: number;
  meanRisk: number;
  peakRisk: number;
  maxDLS_deg30m: number;
  totalMD_m: number;
  inWindowFraction: number;
}

export function scenarioRow(
  wellId: string,
  name: string,
  log: PredictedLog,
  opts: { maxDLS_deg30m?: number } = {},
): ScenarioRow {
  const s = log.summary;
  const lastMd = log.stations.length ? log.stations[log.stations.length - 1].md : 0;
  return {
    wellId,
    name,
    bhtC: s.bhtC,
    payLength_m: s.targetIntersectionLength_m,
    reservoirLength_m: s.reservoirIntersectionLength_m,
    fractureIntersections: s.productiveFractureIntersections,
    meanRisk: s.meanRisk,
    peakRisk: s.peakRisk,
    maxDLS_deg30m: opts.maxDLS_deg30m ?? 0,
    totalMD_m: lastMd,
    inWindowFraction: s.inWindowFraction,
  };
}

// Which row index wins each comparison column (best-in-column highlighting, doc 09 §8.2).
// `higher` columns prefer the max; the rest prefer the min. Returns a map column→rowIndex.
export type ScenarioColumn =
  | "bhtC"
  | "payLength_m"
  | "reservoirLength_m"
  | "fractureIntersections"
  | "meanRisk"
  | "peakRisk"
  | "maxDLS_deg30m"
  | "totalMD_m"
  | "inWindowFraction";

const HIGHER_IS_BETTER: Record<ScenarioColumn, boolean> = {
  bhtC: true,
  payLength_m: true,
  reservoirLength_m: true,
  fractureIntersections: true,
  meanRisk: false,
  peakRisk: false,
  maxDLS_deg30m: false,
  totalMD_m: false,
  inWindowFraction: true,
};

export function bestInColumn(rows: ScenarioRow[]): Partial<Record<ScenarioColumn, number>> {
  const out: Partial<Record<ScenarioColumn, number>> = {};
  if (rows.length === 0) return out;
  for (const col of Object.keys(HIGHER_IS_BETTER) as ScenarioColumn[]) {
    const higher = HIGHER_IS_BETTER[col];
    let bestIdx = -1;
    let bestVal = higher ? -Infinity : Infinity;
    for (let i = 0; i < rows.length; i++) {
      const raw = rows[i][col];
      if (raw == null || !Number.isFinite(raw as number)) continue;
      const v = raw as number;
      if ((higher && v > bestVal) || (!higher && v < bestVal)) {
        bestVal = v;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) out[col] = bestIdx;
  }
  return out;
}

// ── A 2D track → SVG polyline points helper (panel rendering; pure for testability) ───────

// Map a track's (md, value) series into SVG path points over a [W,H] box (MD increases
// downward). Returns the polyline points string + the band polygon points (when present).
export function trackToSvg(
  track: LogTrack,
  W: number,
  H: number,
  mdMin: number,
  mdMax: number,
): { line: string; band: string | null } {
  const mdSpan = mdMax - mdMin || 1;
  const vSpan = track.max - track.min || 1;
  const xOf = (v: number) => ((v - track.min) / vSpan) * (W - 4) + 2;
  const yOf = (md: number) => ((md - mdMin) / mdSpan) * (H - 4) + 2;
  const line = track.md
    .map((md, i) => `${xOf(track.value[i]).toFixed(1)},${yOf(md).toFixed(1)}`)
    .join(" ");
  let band: string | null = null;
  if (track.hasBand && track.lower && track.upper) {
    const up = track.md.map(
      (md, i) => `${xOf(track.upper![i]).toFixed(1)},${yOf(md).toFixed(1)}`,
    );
    const down = track.md
      .map((md, i) => `${xOf(track.lower![i]).toFixed(1)},${yOf(md).toFixed(1)}`)
      .reverse();
    band = [...up, ...down].join(" ");
  }
  return { line, band };
}

// ── colour helper for risk/temperature swatches (reuses the shared colormaps) ─────────────

export function valueToCss(t: number, colormapName: string): string {
  const cm = resolveColormap(colormapName);
  const [r, g, b] = sampleColormap(cm, Math.min(1, Math.max(0, t)));
  return `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
}
