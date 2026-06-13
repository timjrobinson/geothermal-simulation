// Global 4-D time axis (doc 06 §9.4). PURE — no THREE / no DOM / no Zustand — so the
// epoch-union, sort, and time-window-membership logic is unit-testable headlessly (npm test
// esbuild-bundles src/lib/*.test.ts then `node --test`).
//
// The viewer's global time slider is built from the UNION of every time-bearing layer's
// epochs (microseismic clouds, InSAR deformation rasters, repeat surveys) — fetched from
// GET /projects/{pid}/time-extent (doc 02 §8). The playhead is a single instant; a window
// MODE expands it into the inclusive [t0,t1] interval each layer filters against:
//
//   instant     — only the current playhead epoch (a single frame)
//   cumulative   — everything from the axis start up to the playhead (history accretes)
//   rolling      — a fixed-width trailing window ending at the playhead (decays out behind)
//
// Microseismic uses this window for its `uTimeWindow` uniform (GPU fade, no re-upload);
// InSAR raster snaps the playhead to the nearest frame epoch (frame select). All epochs are
// ISO-8601 UTC strings; we compare on parsed epoch-milliseconds so heterogeneous cadences
// reconcile onto one timeline (doc 06 §9.4).

export type TimeWindowMode = "instant" | "cumulative" | "rolling";
export const TIME_WINDOW_MODES: readonly TimeWindowMode[] = [
  "instant",
  "cumulative",
  "rolling",
];

export function isTimeWindowMode(s: string): s is TimeWindowMode {
  return (TIME_WINDOW_MODES as readonly string[]).includes(s);
}

// Parse an ISO-8601 UTC epoch to milliseconds since the Unix epoch. Returns NaN for an
// unparseable string (callers filter those out before building the axis).
export function parseEpochMs(iso: string): number {
  const t = Date.parse(iso);
  return Number.isNaN(t) ? NaN : t;
}

// The reconciled global time axis: the sorted-unique union of all contributing epochs, with
// their parsed milliseconds cached alongside so membership tests never re-parse (doc 06
// §9.4). `t0Ms`/`t1Ms` are the axis bounds (the slider's scrub extent).
export interface TimeAxis {
  epochs: string[]; // sorted unique ISO-8601 UTC
  epochMs: number[]; // parallel parsed ms (sorted ascending, matches epochs)
  t0Ms: number | null; // earliest epoch ms (null when empty)
  t1Ms: number | null; // latest epoch ms (null when empty)
}

// Build the global time axis from one-or-more epoch lists (each list = one layer/dataset's
// own sample times, doc 06 §9.4 "union of all time-bearing layers"). Dedupes by parsed ms
// (so equivalent ISO spellings of the same instant collapse), drops unparseable strings,
// and sorts ascending. Returns the canonical ISO spelling first-seen for each ms.
export function buildTimeAxis(...epochLists: (readonly string[])[]): TimeAxis {
  // ms -> first-seen ISO spelling (canonical for that instant).
  const byMs = new Map<number, string>();
  for (const list of epochLists) {
    for (const iso of list) {
      const ms = parseEpochMs(iso);
      if (Number.isNaN(ms)) continue;
      if (!byMs.has(ms)) byMs.set(ms, iso);
    }
  }
  const sortedMs = [...byMs.keys()].sort((a, b) => a - b);
  const epochs = sortedMs.map((ms) => byMs.get(ms)!);
  return {
    epochs,
    epochMs: sortedMs,
    t0Ms: sortedMs.length ? sortedMs[0] : null,
    t1Ms: sortedMs.length ? sortedMs[sortedMs.length - 1] : null,
  };
}

// A resolved time window in epoch-milliseconds (inclusive). `t0Ms === t1Ms` for the instant
// mode (a single frame). Layers test membership against this (doc 06 §9.4).
export interface TimeWindowMs {
  t0Ms: number;
  t1Ms: number;
  mode: TimeWindowMode;
}

// Resolve the active window in milliseconds for a playhead at `playheadMs`, given the axis
// bounds and (for rolling mode) a window width in milliseconds (doc 06 §9.4):
//   instant     → [playhead, playhead]
//   cumulative   → [axisStart, playhead]
//   rolling      → [playhead - width, playhead]
// `axisT0Ms` anchors the cumulative window's start; when null (empty axis) it falls back to
// the playhead so the window degenerates to a single instant.
export function resolveWindowMs(
  mode: TimeWindowMode,
  playheadMs: number,
  axisT0Ms: number | null,
  rollingWidthMs: number,
): TimeWindowMs {
  switch (mode) {
    case "instant":
      return { t0Ms: playheadMs, t1Ms: playheadMs, mode };
    case "cumulative":
      return { t0Ms: axisT0Ms ?? playheadMs, t1Ms: playheadMs, mode };
    case "rolling":
      return {
        t0Ms: playheadMs - Math.max(0, rollingWidthMs),
        t1Ms: playheadMs,
        mode,
      };
  }
}

// Is an event at `eventMs` inside the resolved window (inclusive bounds, doc 06 §9.4)?
// Used by the microseismic cloud's CPU-side culling parity check and unit tests; the GPU
// path applies the identical [t0Ms,t1Ms] comparison in the shader.
export function inWindowMs(eventMs: number, win: TimeWindowMs): boolean {
  return eventMs >= win.t0Ms && eventMs <= win.t1Ms;
}

// Snap a playhead ms to the NEAREST axis epoch index (doc 06 §9.4 InSAR frame select). The
// playhead scrubs continuously but a raster only has discrete frames, so we pick the closest
// epoch. Returns -1 for an empty axis. Ties resolve to the earlier frame.
export function nearestEpochIndex(axis: TimeAxis, playheadMs: number): number {
  const ms = axis.epochMs;
  if (ms.length === 0) return -1;
  // Binary search for the insertion point, then compare the two flanking candidates.
  let lo = 0;
  let hi = ms.length - 1;
  if (playheadMs <= ms[0]) return 0;
  if (playheadMs >= ms[hi]) return hi;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ms[mid] < playheadMs) lo = mid + 1;
    else hi = mid;
  }
  // lo is the first index with ms[lo] >= playheadMs; compare it with lo-1.
  const before = lo - 1;
  return playheadMs - ms[before] <= ms[lo] - playheadMs ? before : lo;
}

// Fraction [0,1] of the playhead along the axis (for slider rendering). 0 for a degenerate
// (single-epoch / empty) axis.
export function playheadFraction(axis: TimeAxis, playheadMs: number): number {
  if (axis.t0Ms == null || axis.t1Ms == null || axis.t1Ms <= axis.t0Ms) return 0;
  const f = (playheadMs - axis.t0Ms) / (axis.t1Ms - axis.t0Ms);
  return Math.min(1, Math.max(0, f));
}

// Map a slider fraction [0,1] back to a playhead ms (the scrub inverse of playheadFraction).
export function fractionToMs(axis: TimeAxis, fraction: number): number {
  if (axis.t0Ms == null || axis.t1Ms == null) return 0;
  const f = Math.min(1, Math.max(0, fraction));
  return axis.t0Ms + f * (axis.t1Ms - axis.t0Ms);
}
