// Pure cross-plot / histogram / correlation data-shaping (doc 06 §10.3, doc 07 §3.2).
//
// The analysis panels render hand-rolled SVG/canvas (no heavy charting dep — keeps the
// build self-contained, doc 06 §1 phasing). This module owns ALL the math that turns a
// FusedSampleOut (the backend co-located feature matrix) + a pixel viewport into screen
// geometry, and — critically for LINKED BRUSHING — turns a pixel brush rectangle back into
// the set of selected *sample rows* (local indices). It is intentionally free of React /
// DOM / fetch so the brushing + data-shaping logic is unit-testable headlessly (npm test).
//
// Coordinate conventions: data space is property value; screen space is pixels with y DOWN
// (SVG/canvas convention) so larger data values sit HIGHER on screen (smaller pixel y).

import type { FusedSampleOut } from "./fusion";

export interface Bounds {
  min: number;
  max: number;
}

export interface Viewport {
  width: number;
  height: number;
  pad: number; // inner padding (axis gutter) in px, applied on all sides
}

// Finite [min,max] of one property column across the sample (NaN-aware). Falls back to a
// unit range if the column is empty/constant so scales never divide by zero.
export function columnBounds(sample: FusedSampleOut, prop: string): Bounds {
  const col = sample.properties.indexOf(prop);
  if (col < 0) return { min: 0, max: 1 };
  let min = Infinity;
  let max = -Infinity;
  for (const row of sample.features) {
    const v = row[col];
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!Number.isFinite(min)) return { min: 0, max: 1 };
  if (max <= min) max = min + 1;
  return { min, max };
}

// Map a data value to a pixel coordinate within [pad, size-pad]. `flip` (default true for
// the Y axis) puts larger values at the top (smaller pixel y).
export function toPixel(
  v: number,
  bounds: Bounds,
  size: number,
  pad: number,
  flip = false,
): number {
  const span = bounds.max - bounds.min || 1;
  const t = (v - bounds.min) / span;
  const inner = size - 2 * pad;
  return flip ? pad + (1 - t) * inner : pad + t * inner;
}

// Inverse of toPixel: pixel coordinate back to a data value (for brush-rect → data range).
export function fromPixel(
  px: number,
  bounds: Bounds,
  size: number,
  pad: number,
  flip = false,
): number {
  const span = bounds.max - bounds.min || 1;
  const inner = size - 2 * pad || 1;
  const t = flip ? 1 - (px - pad) / inner : (px - pad) / inner;
  return bounds.min + t * span;
}

export interface ScatterPoint {
  i: number; // local sample row index (== position in sample.features) — the brushing key
  px: number;
  py: number;
  c?: number; // optional colour channel value (depth / 3rd property)
}

// Project a 2-property scatter into screen pixels. `xProp`/`yProp` name two of the sample's
// properties; `colorBy` optionally names a 3rd property or "depth" (the z coord) for the
// per-point colour channel. Rows with a non-finite X or Y are dropped (they cannot plot).
export function projectScatter(
  sample: FusedSampleOut,
  xProp: string,
  yProp: string,
  vp: Viewport,
  xb: Bounds,
  yb: Bounds,
  colorBy?: string | null,
): ScatterPoint[] {
  const xc = sample.properties.indexOf(xProp);
  const yc = sample.properties.indexOf(yProp);
  if (xc < 0 || yc < 0) return [];
  const cc = colorBy && colorBy !== "depth" ? sample.properties.indexOf(colorBy) : -1;
  const out: ScatterPoint[] = [];
  for (let i = 0; i < sample.features.length; i++) {
    const row = sample.features[i];
    const xv = row[xc];
    const yv = row[yc];
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) continue;
    let c: number | undefined;
    if (colorBy === "depth") c = sample.coords[i]?.[0];
    else if (cc >= 0) c = row[cc];
    out.push({
      i,
      px: toPixel(xv, xb, vp.width, vp.pad, false),
      py: toPixel(yv, yb, vp.height, vp.pad, true),
      c,
    });
  }
  return out;
}

// A pixel-space brush rectangle (drag on the cross-plot). Normalized so x0<=x1, y0<=y1.
export interface BrushRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function normalizeRect(a: { x: number; y: number }, b: { x: number; y: number }): BrushRect {
  return {
    x0: Math.min(a.x, b.x),
    x1: Math.max(a.x, b.x),
    y0: Math.min(a.y, b.y),
    y1: Math.max(a.y, b.y),
  };
}

export function rectIsDegenerate(r: BrushRect, minPx = 2): boolean {
  return r.x1 - r.x0 < minPx || r.y1 - r.y0 < minPx;
}

// LINKED BRUSHING core (doc 06 §10.3): given a pixel brush rectangle over a projected
// scatter, return the LOCAL sample-row indices whose points fall inside it. These indices
// index into sample.features / sample.cell_index — brushing.ts maps them on to a voxel mask
// for the 3D highlight. Pure, so it is the unit under test.
export function pointsInRect(points: ScatterPoint[], rect: BrushRect): number[] {
  const out: number[] = [];
  for (const p of points) {
    if (p.px >= rect.x0 && p.px <= rect.x1 && p.py >= rect.y0 && p.py <= rect.y1) {
      out.push(p.i);
    }
  }
  return out;
}

// Build a histogram (counts per bin) of one property over the sample, NaN-aware. Mirrors
// the backend histogram() bin convention (uniform bins over [min,max]) so the offline mock
// path matches the live endpoint.
export interface HistResult {
  counts: number[];
  edges: number[]; // length bins+1
  bounds: Bounds;
}

export function histogramOf(sample: FusedSampleOut, prop: string, bins = 32): HistResult {
  const col = sample.properties.indexOf(prop);
  const bounds = columnBounds(sample, prop);
  const counts = new Array<number>(bins).fill(0);
  const edges = new Array<number>(bins + 1);
  const span = bounds.max - bounds.min || 1;
  for (let b = 0; b <= bins; b++) edges[b] = bounds.min + (b / bins) * span;
  if (col < 0) return { counts, edges, bounds };
  const scale = bins / span;
  for (const row of sample.features) {
    const v = row[col];
    if (!Number.isFinite(v)) continue;
    let b = Math.floor((v - bounds.min) * scale);
    if (b < 0) b = 0;
    else if (b >= bins) b = bins - 1;
    counts[b] += 1;
  }
  return { counts, edges, bounds };
}

// Pearson correlation matrix across all sample properties (NaN-aware, listwise within each
// pair). Mirrors the backend correlation_matrix() shape ({ properties, matrix }), so the UI
// renders the same heatmap whether the data came from the endpoint or the offline mock.
// `null` entries mean undefined (too few finite pairs / zero variance).
export function correlationMatrix(sample: FusedSampleOut): {
  properties: string[];
  matrix: (number | null)[][];
} {
  const props = sample.properties;
  const p = props.length;
  const matrix: (number | null)[][] = Array.from({ length: p }, () =>
    new Array<number | null>(p).fill(null),
  );
  for (let a = 0; a < p; a++) {
    for (let b = a; b < p; b++) {
      const r = pearson(sample, a, b);
      matrix[a][b] = r;
      matrix[b][a] = r;
    }
  }
  return { properties: props, matrix };
}

function pearson(sample: FusedSampleOut, a: number, b: number): number | null {
  let n = 0;
  let sa = 0;
  let sb = 0;
  let saa = 0;
  let sbb = 0;
  let sab = 0;
  for (const row of sample.features) {
    const x = row[a];
    const y = row[b];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    n += 1;
    sa += x;
    sb += y;
    saa += x * x;
    sbb += y * y;
    sab += x * y;
  }
  if (n < 2) return null;
  const cov = sab - (sa * sb) / n;
  const va = saa - (sa * sa) / n;
  const vb = sbb - (sb * sb) / n;
  const denom = Math.sqrt(va * vb);
  if (!(denom > 0)) return null;
  return cov / denom;
}

// Diverging blue↔white↔red colour for a correlation cell in [-1, 1] (heatmap). Returns a
// CSS rgb() string. r=+1 → red, r=-1 → blue, r=0 → near-white. `null` → neutral grey.
export function correlationColor(r: number | null): string {
  if (r == null || !Number.isFinite(r)) return "rgb(70,76,92)";
  const t = Math.max(-1, Math.min(1, r));
  // Interpolate white(255) → channel at the extremes.
  const mix = (lo: number, hi: number, x: number) => Math.round(lo + (hi - lo) * x);
  let rr: number;
  let gg: number;
  let bb: number;
  if (t >= 0) {
    rr = 255;
    gg = mix(255, 90, t);
    bb = mix(255, 90, t);
  } else {
    rr = mix(255, 90, -t);
    gg = mix(255, 90, -t);
    bb = 255;
  }
  return `rgb(${rr},${gg},${bb})`;
}
