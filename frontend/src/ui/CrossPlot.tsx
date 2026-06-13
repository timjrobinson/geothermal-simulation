// 2D cross-plot panel with LINKED BRUSHING (doc 06 §10.3, doc 07 §3.2). Hand-rolled
// canvas scatter (fast for tens of thousands of points) + an SVG brush-rectangle overlay —
// no charting dependency, so the build stays self-contained (doc 06 §1 phasing). Points are
// coloured by a 3rd channel (depth or a third property). Dragging a rectangle brushes the
// enclosed points; the enclosed LOCAL sample rows are pushed to the store, which rebuilds
// the 3D selection-mask overlay so the same voxels light up in the scene.
//
// All the math (projection, brush-rect → selected rows) lives in lib/crossplot.ts and is
// unit-tested headlessly; this component is the thin canvas/SVG/event shell.

import { useEffect, useMemo, useRef, useState } from "react";
import { useViewer } from "../store";
import {
  columnBounds,
  projectScatter,
  normalizeRect,
  rectIsDegenerate,
  pointsInRect,
  type Viewport,
  type ScatterPoint,
  type BrushRect,
} from "../lib/crossplot";
import { resolveColormap, sampleColormap } from "../lib/colormaps";

const VIRIDIS = resolveColormap("viridis");

const W = 268;
const H = 220;
const PAD = 34;
const VP: Viewport = { width: W, height: H, pad: PAD };

// Colour-channel bounds for the per-point colour ramp (viridis), NaN-aware.
function channelBounds(pts: ScatterPoint[]): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (const p of pts) {
    if (p.c != null && Number.isFinite(p.c)) {
      if (p.c < min) min = p.c;
      if (p.c > max) max = p.c;
    }
  }
  return min === Infinity ? null : { min, max: max <= min ? min + 1 : max };
}

export function CrossPlot({
  xProp,
  yProp,
  colorBy,
}: {
  xProp: string;
  yProp: string;
  colorBy: string | null;
}) {
  const sample = useViewer((s) => s.fusedSample);
  const selection = useViewer((s) => s.selection);
  const setSelection = useViewer((s) => s.setSelection);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rect, setRect] = useState<BrushRect | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  const { points, xb, yb } = useMemo(() => {
    if (!sample) return { points: [] as ScatterPoint[], xb: null, yb: null };
    const xb = columnBounds(sample, xProp);
    const yb = columnBounds(sample, yProp);
    return { points: projectScatter(sample, xProp, yProp, VP, xb, yb, colorBy), xb, yb };
  }, [sample, xProp, yProp, colorBy]);

  const selSet = useMemo(() => new Set(selection), [selection]);
  const cbounds = useMemo(() => channelBounds(points), [points]);

  // Paint the scatter (selected points brightened; unselected dimmed when a brush is active).
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    // axis gutter
    ctx.strokeStyle = "#45475a";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);

    const hasSel = selSet.size > 0;
    for (const p of points) {
      const isSel = selSet.has(p.i);
      let color = "#89b4fa";
      if (colorBy && cbounds && p.c != null && Number.isFinite(p.c)) {
        const t = (p.c - cbounds.min) / (cbounds.max - cbounds.min);
        const [r, g, b] = sampleColormap(VIRIDIS, t);
        color = `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;
      }
      ctx.globalAlpha = hasSel ? (isSel ? 1 : 0.12) : 0.7;
      ctx.fillStyle = isSel ? "#f9e2af" : color;
      ctx.beginPath();
      ctx.arc(p.px, p.py, isSel ? 2.4 : 1.7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, [points, selSet, colorBy, cbounds]);

  const localXY = (e: React.PointerEvent) => {
    const r = (e.target as HTMLElement).getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onDown = (e: React.PointerEvent) => {
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    dragRef.current = localXY(e);
    setRect(null);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!dragRef.current) return;
    setRect(normalizeRect(dragRef.current, localXY(e)));
  };
  const onUp = (e: React.PointerEvent) => {
    const start = dragRef.current;
    dragRef.current = null;
    if (!start) return;
    const r = normalizeRect(start, localXY(e));
    if (rectIsDegenerate(r)) {
      // A click (not a drag) clears the brush.
      setRect(null);
      setSelection([]);
      return;
    }
    setRect(r);
    setSelection(pointsInRect(points, r));
  };

  if (!sample || !xb || !yb) {
    return <div style={{ fontSize: 12, opacity: 0.6 }}>No fused sample loaded.</div>;
  }

  return (
    <div>
      <div style={{ position: "relative", width: W, height: H }}>
        <canvas ref={canvasRef} width={W} height={H} style={{ display: "block" }} />
        {/* SVG brush overlay + axis labels (events captured here, drawn over the canvas). */}
        <svg
          width={W}
          height={H}
          style={{ position: "absolute", inset: 0, cursor: "crosshair", touchAction: "none" }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
        >
          {rect && (
            <rect
              x={rect.x0}
              y={rect.y0}
              width={rect.x1 - rect.x0}
              height={rect.y1 - rect.y0}
              fill="rgba(249,226,175,0.12)"
              stroke="#f9e2af"
              strokeDasharray="3 2"
            />
          )}
          <text x={W / 2} y={H - 6} fill="#cdd6f4" fontSize={10} textAnchor="middle">
            {xProp}
          </text>
          <text
            x={10}
            y={H / 2}
            fill="#cdd6f4"
            fontSize={10}
            textAnchor="middle"
            transform={`rotate(-90 10 ${H / 2})`}
          >
            {yProp}
          </text>
        </svg>
      </div>
      <div style={{ fontSize: 11, opacity: 0.65, marginTop: 4, display: "flex", gap: 10 }}>
        <span>n = {points.length}</span>
        <span>brushed = {selection.length}</span>
        {colorBy && <span>colour: {colorBy}</span>}
        <span style={{ marginLeft: "auto", opacity: 0.5 }}>drag to brush · click to clear</span>
      </div>
    </div>
  );
}
