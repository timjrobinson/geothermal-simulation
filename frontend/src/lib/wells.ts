// Well-trajectory tube + log-colour math (doc 06 §5.3). PURE — no THREE / no DOM — so the
// deviation-survey→tube vertex/MD math and the log→vertex-colour resampling are unit-testable
// headlessly (npm test). The scene WellLayer feeds these arrays straight into a THREE
// TubeGeometry / BufferAttribute.
//
// The backend (GET /wells/{id}/trajectory, geosim/api/features.py) already resolves the
// min-curvature Engineering polyline + per-station MD/TVD/DLS and the joined LAS log curves
// (samples vs MD). This module turns that into renderable arrays:
//   - a per-station Engineering polyline + cumulative MD (for the tube spine + hover),
//   - per-vertex colours by resampling a chosen LAS curve (defined on its own MD grid) onto
//     the trajectory station MDs and looking the value up through a transfer function,
//   - MD/TVD/elevation lookup for hover readout (depths reported TRUE, doc 06 §2.3).

import { resolveColormap, sampleColormap, type Colormap } from "./colormaps";

export type Vec3 = [number, number, number];

// The trajectory payload shape from GET /wells/{id}/trajectory (geosim/api/features.py).
export interface WellTrajectory {
  featureId: string;
  wellId: string | null;
  wellhead: number[];
  polyline: Vec3[]; // Engineering XYZ per station
  md: number[]; // measured depth per station (m)
  tvd: number[]; // true vertical depth (m, +down)
  dls?: number[];
  logs: WellLogs;
}

// Joined LAS curves vs MD (geosim/api/features.py _well_logs).
export interface WellLogs {
  wellId: string | null;
  md: number[]; // the MD grid the curve samples are defined on
  curves: Record<string, number[]>; // {property: [samples vs MD]}
  primaryProperty: string | null;
}

// ── Polyline / MD math (deviation survey already integrated server-side) ──────────────

// Cumulative arc-length (chord MD) along an Engineering polyline. Used as a fallback MD when
// the backend did not persist a station MD array, and as the parity check in unit tests that
// the min-curvature MD is monotonic and matches the chord length within survey tolerance.
export function cumulativeArcLength(polyline: readonly Vec3[]): number[] {
  const out: number[] = [];
  let acc = 0;
  for (let i = 0; i < polyline.length; i++) {
    if (i > 0) {
      const a = polyline[i - 1];
      const b = polyline[i];
      acc += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    out.push(acc);
  }
  return out;
}

// Resolve the per-station MD array for a trajectory: prefer the backend min-curvature MD,
// else derive it from the polyline arc length. Guards a length mismatch (truncates / pads).
export function stationMD(traj: Pick<WellTrajectory, "polyline" | "md">): number[] {
  if (traj.md && traj.md.length === traj.polyline.length) return traj.md;
  return cumulativeArcLength(traj.polyline);
}

// ── Log resampling → per-vertex colours (doc 06 §5.3 tube colouring) ──────────────────

// Linearly interpolate a curve defined on a (monotonic-ascending) MD grid at an arbitrary
// MD. Returns NaN outside the curve's MD coverage so the caller can render those vertices
// neutral (no extrapolation — a log only paints where it was measured). `curveMD` and
// `values` are parallel; a NaN sample stays NaN (logs carry null gaps).
export function sampleCurveAtMD(
  curveMD: readonly number[],
  values: readonly number[],
  md: number,
): number {
  const n = Math.min(curveMD.length, values.length);
  if (n === 0) return NaN;
  if (md < curveMD[0] || md > curveMD[n - 1]) return NaN;
  // Binary search for the bracketing interval [i-1, i].
  let lo = 0;
  let hi = n - 1;
  if (md <= curveMD[0]) return values[0];
  if (md >= curveMD[n - 1]) return values[n - 1];
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (curveMD[mid] < md) lo = mid + 1;
    else hi = mid;
  }
  const i1 = lo;
  const i0 = lo - 1;
  const v0 = values[i0];
  const v1 = values[i1];
  if (Number.isNaN(v0) || Number.isNaN(v1)) return NaN;
  const span = curveMD[i1] - curveMD[i0];
  const f = span > 0 ? (md - curveMD[i0]) / span : 0;
  return v0 + (v1 - v0) * f;
}

// Resample a chosen LAS curve onto the trajectory station MDs (doc 06 §5.3). Returns one
// value per station (NaN where the log doesn't cover that depth).
export function resampleCurveToStations(
  logs: WellLogs,
  property: string,
  stationMd: readonly number[],
): number[] {
  const values = logs.curves[property];
  if (!values) return stationMd.map(() => NaN);
  return stationMd.map((md) => sampleCurveAtMD(logs.md, values, md));
}

export interface LogColorRange {
  min: number;
  max: number;
}

// Robust [min,max] of a curve's finite samples (the default colour domain). Falls back to
// [0,1] when the curve is all-NaN/empty so the transfer function stays well-defined.
export function curveRange(values: readonly number[]): LogColorRange {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (Number.isNaN(v) || !Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 }; // all-NaN / empty curve
  }
  if (max <= min) {
    return { min, max: min + 1 }; // constant curve -> non-degenerate domain
  }
  return { min, max };
}

// Normalize a value to t∈[0,1] over [min,max] (clamped). NaN → NaN (caller renders neutral).
export function normalize(value: number, range: LogColorRange): number {
  if (Number.isNaN(value)) return NaN;
  const span = range.max - range.min;
  if (span <= 0) return 0;
  return Math.min(1, Math.max(0, (value - range.min) / span));
}

// Map resampled per-station curve values to a flat RGB Float32 array (length 3*N), looking
// each value up through a named colormap over [min,max] (doc 06 §5.3). NaN (uncovered/gap)
// vertices get the supplied `neutral` colour so the tube stays continuous but visibly unlogged.
export function curveToVertexColors(
  values: readonly number[],
  range: LogColorRange,
  colormapName: string,
  neutral: Vec3 = [0.5, 0.5, 0.5],
): Float32Array {
  const cm: Colormap = resolveColormap(colormapName);
  const out = new Float32Array(values.length * 3);
  for (let i = 0; i < values.length; i++) {
    const t = normalize(values[i], range);
    const [r, g, b] = Number.isNaN(t) ? neutral : sampleColormap(cm, t);
    out[i * 3 + 0] = r;
    out[i * 3 + 1] = g;
    out[i * 3 + 2] = b;
  }
  return out;
}

// ── Hover readout (depths reported TRUE, doc 06 §2.3) ─────────────────────────────────

export interface WellReadout {
  md: number; // measured depth (m)
  tvd: number; // true vertical depth (m, +down)
  elevation: number; // Engineering Z (m) — true, NOT vertically-exaggerated
  stationIndex: number;
}

// Nearest-station readout for a picked Engineering point on the tube spine (doc 06 §5.3
// "hover shows MD/TVD/elevation"). Picks the closest polyline station by 3-D distance and
// reports its true depths. `tvd` falls back to (kbElev - z) when no TVD array is present.
export function readoutAtPoint(
  traj: Pick<WellTrajectory, "polyline" | "md" | "tvd">,
  point: Vec3,
): WellReadout | null {
  const poly = traj.polyline;
  if (poly.length === 0) return null;
  const md = stationMD(traj);
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const d =
      (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2 + (p[2] - point[2]) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  const z = poly[best][2];
  const tvd =
    traj.tvd && traj.tvd.length === poly.length
      ? traj.tvd[best]
      : (poly[0]?.[2] ?? 0) - z;
  return { md: md[best], tvd, elevation: z, stationIndex: best };
}
