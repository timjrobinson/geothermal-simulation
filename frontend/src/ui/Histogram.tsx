// Histogram panel (doc 07 §3.2). Hand-rolled SVG bars over the co-located fused sample —
// no charting dep. Bins highlight the brushed sub-population so the 1D distribution stays
// linked with the cross-plot brush + the 3D selection (doc 06 §10.3): each bin shows the
// full count (faint) with the brushed-subset count overlaid (bright).

import { useMemo } from "react";
import { useViewer } from "../store";
import { histogramOf } from "../lib/crossplot";
import type { FusedSampleOut } from "../lib/fusion";

const W = 268;
const H = 120;
const PAD = 4;

// Per-bin counts restricted to the brushed rows (same binning as the full histogram).
function brushedCounts(
  sample: FusedSampleOut,
  prop: string,
  bins: number,
  edges: number[],
  selection: number[],
): number[] {
  const col = sample.properties.indexOf(prop);
  const counts = new Array<number>(bins).fill(0);
  if (col < 0 || edges.length < 2) return counts;
  const lo = edges[0];
  const span = edges[edges.length - 1] - lo || 1;
  const scale = bins / span;
  for (const row of selection) {
    const v = sample.features[row]?.[col];
    if (!Number.isFinite(v)) continue;
    let b = Math.floor((v - lo) * scale);
    if (b < 0) b = 0;
    else if (b >= bins) b = bins - 1;
    counts[b] += 1;
  }
  return counts;
}

export function Histogram({ prop, bins = 32 }: { prop: string; bins?: number }) {
  const sample = useViewer((s) => s.fusedSample);
  const selection = useViewer((s) => s.selection);

  const { counts, edges, brushed, maxC } = useMemo(() => {
    if (!sample) return { counts: [], edges: [], brushed: [], maxC: 1 };
    const h = histogramOf(sample, prop, bins);
    const brushed = brushedCounts(sample, prop, bins, h.edges, selection);
    return { counts: h.counts, edges: h.edges, brushed, maxC: Math.max(1, ...h.counts) };
  }, [sample, prop, bins, selection]);

  if (!sample) return null;

  const bw = (W - 2 * PAD) / Math.max(1, counts.length);
  const barH = (c: number) => ((H - 2 * PAD) * c) / maxC;

  return (
    <div>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 2 }}>{prop}</div>
      <svg width={W} height={H} style={{ display: "block" }}>
        {counts.map((c, i) => {
          const x = PAD + i * bw;
          return (
            <g key={i}>
              <rect
                x={x}
                y={H - PAD - barH(c)}
                width={Math.max(0.5, bw - 0.5)}
                height={barH(c)}
                fill="#585b70"
              />
              {brushed[i] > 0 && (
                <rect
                  x={x}
                  y={H - PAD - barH(brushed[i])}
                  width={Math.max(0.5, bw - 0.5)}
                  height={barH(brushed[i])}
                  fill="#f9e2af"
                />
              )}
            </g>
          );
        })}
      </svg>
      {edges.length >= 2 && (
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, opacity: 0.5 }}>
          <span>{edges[0].toPrecision(3)}</span>
          <span>{edges[edges.length - 1].toPrecision(3)}</span>
        </div>
      )}
    </div>
  );
}
