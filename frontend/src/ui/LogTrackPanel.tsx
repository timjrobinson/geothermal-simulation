// Well log-track 2D panel (doc 06 §5.3, §10.3). A vertical track plotting the selected well's
// chosen LAS curve vs measured depth, synced to the well selected in the layer manager (or the
// last-hovered well). A horizontal marker tracks the hovered MD so the 3D tube hover and the
// 2D track stay in lockstep (the natural pick target syncing the log-track panel, doc 06
// §5.3). A curve switcher picks which log colours the tube + draws here. Reuses the
// analysis-panel docked-panel pattern.

import { useMemo } from "react";
import { useViewer } from "../store";
import type { Layer } from "../lib/layers";
import { curveRange } from "../lib/wells";
import { resolveColormap, sampleColormap } from "../lib/colormaps";

const panel: React.CSSProperties = {
  position: "absolute",
  top: 64,
  right: 12,
  width: 200,
  maxHeight: "calc(100vh - 160px)",
  overflowY: "auto",
  padding: 12,
  background: "rgba(17,22,33,0.94)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 12,
  zIndex: 11,
};
const sel: React.CSSProperties = {
  fontSize: 12,
  background: "#1e2230",
  color: "#cdd6f4",
  border: "1px solid #313244",
  borderRadius: 4,
  padding: "2px 4px",
  width: "100%",
};

// Pick the well layer to display: the selected layer if it is a well, else the last-hovered
// well's layer, else the first well layer present.
function activeWell(
  layers: Record<string, Layer>,
  order: string[],
  selectedId: string | null,
  hoverLayerId: string | null,
): Layer | null {
  if (selectedId && layers[selectedId]?.kind === "well") return layers[selectedId];
  if (hoverLayerId && layers[hoverLayerId]?.kind === "well") return layers[hoverLayerId];
  for (const id of order) if (layers[id]?.kind === "well") return layers[id];
  return null;
}

export function LogTrackPanel() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  const selectedId = useViewer((s) => s.selectedLayerId);
  const readout = useViewer((s) => s.wellReadout);
  const setLayerLogProperty = useViewer((s) => s.setLayerLogProperty);

  const well = activeWell(layers, layerOrder, selectedId, readout?.layerId ?? null);

  const traj = well?.trajectory;
  const property = well?.logProperty ?? null;
  const curves = traj?.logs?.curves ?? {};
  const curveNames = Object.keys(curves);

  // The (md, value) series for the chosen curve + its colour domain.
  const series = useMemo(() => {
    if (!traj || !property || !curves[property]) return null;
    const md = traj.logs.md;
    const values = curves[property];
    const n = Math.min(md.length, values.length);
    const pts: { md: number; v: number }[] = [];
    for (let i = 0; i < n; i++) {
      if (!Number.isNaN(values[i])) pts.push({ md: md[i], v: values[i] });
    }
    if (pts.length === 0) return null;
    const range = curveRange(values);
    const mdMin = md[0];
    const mdMax = md[n - 1];
    return { pts, range, mdMin, mdMax };
  }, [traj, property, curves]);

  if (!well || !traj) return null;

  const W = 160;
  const H = 260;
  const cm = resolveColormap(well.transferFn.colormap);

  // Marker Y for the hovered MD (only when the readout belongs to THIS well).
  const markerY =
    series && readout && readout.layerId === well.id
      ? ((readout.md - series.mdMin) / (series.mdMax - series.mdMin || 1)) * H
      : null;

  return (
    <div style={panel}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>
        Log track — {traj.wellId ?? well.name}
      </div>

      <label style={{ display: "block", marginBottom: 8 }}>
        Curve
        <select
          style={sel}
          value={property ?? ""}
          onChange={(e) => setLayerLogProperty(well.id, e.target.value || null)}
        >
          <option value="">(none — solid tube)</option>
          {curveNames.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>

      {series ? (
        <svg width={W} height={H} style={{ background: "#11131c", borderRadius: 4 }}>
          {/* curve polyline (MD increases downward) */}
          <polyline
            fill="none"
            stroke="#89b4fa"
            strokeWidth={1.5}
            points={series.pts
              .map((p) => {
                const x =
                  ((p.v - series.range.min) / (series.range.max - series.range.min || 1)) *
                  (W - 8) +
                  4;
                const y =
                  ((p.md - series.mdMin) / (series.mdMax - series.mdMin || 1)) * (H - 4) + 2;
                return `${x.toFixed(1)},${y.toFixed(1)}`;
              })
              .join(" ")}
          />
          {/* colour-domain swatch strip (matches the tube transfer function) */}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const [r, g, b] = sampleColormap(cm, t);
            return (
              <rect
                key={t}
                x={(W - 8) * t + 4}
                y={H - 6}
                width={Math.max(2, (W - 8) / 4)}
                height={4}
                fill={`rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`}
              />
            );
          })}
          {/* hovered-MD marker */}
          {markerY != null && (
            <line x1={0} x2={W} y1={markerY} y2={markerY} stroke="#f9e2af" strokeWidth={1} />
          )}
        </svg>
      ) : (
        <div style={{ opacity: 0.6, fontSize: 11 }}>
          {property ? "no samples for this curve" : "select a curve to plot"}
        </div>
      )}

      {readout && readout.layerId === well.id && (
        <div style={{ marginTop: 8, fontFamily: "ui-monospace, monospace", fontSize: 11 }}>
          MD {readout.md.toFixed(1)} m · TVD {readout.tvd.toFixed(1)} m
          <br />
          elev {readout.elevation.toFixed(1)} m
        </div>
      )}
    </div>
  );
}
