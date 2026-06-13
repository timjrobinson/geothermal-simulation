// Analysis panel (doc 06 §10.3, doc 07 §3.2) — the multivariate-analysis + linked-brushing
// surface. It hosts the "build fused grid / cross-plot these layers" control flow, then the
// cross-plot (with brushing), histogram, correlation heatmap, and the 3D-pick inspector.
//
// Build flow (doc 07 §6): collect the volume layers' source property-model ids, POST /fused
// to make a container grid, resample each into it, then POST /fused/{id}/sample to pull the
// co-located feature matrix that feeds every panel. If no backend is reachable it falls back
// to makeMockFusedSample so the whole brushing pipeline is exercisable offline.
//
// Linked brushing (the key R&D feature): the cross-plot brush writes a row selection to the
// store, which rebuilds a 3D selection-mask overlay; a 3D voxel pick writes pickedVoxel,
// shown in the inspector here. Both directions flow through the shared Zustand store.

import { useMemo, useState } from "react";
import { useViewer } from "../store";
import { CrossPlot } from "./CrossPlot";
import { Histogram } from "./Histogram";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import {
  createFused,
  resampleLayer,
  sampleFused,
  makeMockFusedSample,
} from "../lib/fusion";

const panel: React.CSSProperties = {
  position: "absolute",
  top: 64,
  left: 12,
  width: 320,
  maxHeight: "calc(100vh - 76px)",
  overflowY: "auto",
  padding: 14,
  background: "rgba(17,22,33,0.94)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 13,
  zIndex: 11,
};
const btn: React.CSSProperties = {
  background: "#313244",
  color: "#cdd6f4",
  border: "none",
  borderRadius: 4,
  padding: "3px 10px",
  cursor: "pointer",
  fontSize: 12,
};
const hr: React.CSSProperties = { border: "none", borderTop: "1px solid #313244", margin: "10px 0" };
const sel: React.CSSProperties = { fontSize: 12, background: "#1e2230", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 4 };

export function AnalysisPanel() {
  const setAnalysisOpen = useViewer((s) => s.setAnalysisOpen);
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  const fusedGrid = useViewer((s) => s.fusedGrid);
  const sample = useViewer((s) => s.fusedSample);
  const setFusedAnalysis = useViewer((s) => s.setFusedAnalysis);
  const clearSelection = useViewer((s) => s.clearSelection);
  const pickedVoxel = useViewer((s) => s.pickedVoxel);

  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  // The candidate source property-model ids: every distinct volume layer's source dataset
  // (excluding the selection-mask overlay + the mock layer, which has no backend id).
  const volumeLayers = useMemo(
    () =>
      layerOrder
        .map((id) => layers[id])
        .filter((l) => l && l.kind === "volume" && l.id !== "selection-mask"),
    [layers, layerOrder],
  );

  const props = sample?.properties ?? [];
  const [xProp, setXProp] = useState("");
  const [yProp, setYProp] = useState("");
  const [colorBy, setColorBy] = useState<string>("depth");
  const [histProp, setHistProp] = useState("");

  // Seed the axis selectors once a sample arrives.
  const ensureAxes = (p: string[]) => {
    setXProp((x) => (p.includes(x) ? x : (p[0] ?? "")));
    setYProp((y) => (p.includes(y) ? y : (p[1] ?? p[0] ?? "")));
    setHistProp((h) => (p.includes(h) ? h : (p[0] ?? "")));
  };

  // BUILD FUSED GRID + SAMPLE (doc 07 §6). Tries the backend; falls back to the mock.
  const buildAndSample = async () => {
    setBusy(true);
    setNote("");
    try {
      const ids = Array.from(
        new Set(volumeLayers.map((l) => l!.datasetId).filter((d) => d && d !== "mock")),
      );
      const projectId =
        (volumeLayers[0]?.meta?.frame as Record<string, unknown> | undefined)?.[
          "projectId"
        ] as string | undefined;
      if (ids.length >= 2 && projectId) {
        const grid = await createFused({
          project_id: projectId,
          name: "analysis",
          source_property_model_ids: ids,
        });
        for (const id of ids) await resampleLayer(grid.id, id);
        const s = await sampleFused(grid.id, { mode: "all" });
        setFusedAnalysis(grid, s);
        ensureAxes(s.properties);
        setNote(`fused ${grid.id} · ${s.n} co-located cells`);
        return;
      }
      throw new Error("need ≥2 backend volume layers + project id");
    } catch {
      // Offline / not enough backend layers → synthesize a co-located mock sample so the
      // cross-plot + brushing + 3D overlay are still fully demonstrable.
      const { grid, sample: s } = makeMockFusedSample();
      setFusedAnalysis(grid, s);
      ensureAxes(s.properties);
      setNote(`offline mock · ${s.n} co-located cells (resistivity/density/vp)`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={panel}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontWeight: 600 }}>Analysis — cross-plot &amp; brushing</span>
        <button style={btn} onClick={() => setAnalysisOpen(false)}>
          close
        </button>
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 6 }}>
        <button style={btn} onClick={buildAndSample} disabled={busy}>
          {sample ? "Rebuild fused sample" : "Build fused grid + cross-plot"}
        </button>
        {sample && (
          <button style={btn} onClick={clearSelection}>
            clear brush
          </button>
        )}
      </div>
      <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 6 }}>
        {volumeLayers.length} volume layer{volumeLayers.length === 1 ? "" : "s"} ·{" "}
        {fusedGrid ? `grid ${fusedGrid.shape.join("×")}` : "no fused grid yet"}
      </div>
      {note && <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 6 }}>{note}</div>}

      {sample && props.length >= 2 && (
        <>
          <hr style={hr} />
          {/* axis + colour controls */}
          <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8, flexWrap: "wrap" }}>
            <label style={{ fontSize: 11 }}>
              X{" "}
              <select style={sel} value={xProp} onChange={(e) => setXProp(e.target.value)}>
                {props.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ fontSize: 11 }}>
              Y{" "}
              <select style={sel} value={yProp} onChange={(e) => setYProp(e.target.value)}>
                {props.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label style={{ fontSize: 11 }}>
              colour{" "}
              <select style={sel} value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
                <option value="">none</option>
                <option value="depth">depth</option>
                {props.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <CrossPlot xProp={xProp} yProp={yProp} colorBy={colorBy || null} />

          <hr style={hr} />
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
            <span style={{ fontWeight: 600, fontSize: 12 }}>Histogram</span>
            <select style={sel} value={histProp} onChange={(e) => setHistProp(e.target.value)}>
              {props.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>
          <Histogram prop={histProp || props[0]} />

          <hr style={hr} />
          <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>Correlation</div>
          <CorrelationHeatmap />
        </>
      )}

      {/* 3D pick → multi-property inspector (doc 06 §10.3) */}
      {pickedVoxel && pickedVoxel.row >= 0 && (
        <>
          <hr style={hr} />
          <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>
            Picked voxel
          </div>
          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>
            (z,y,x) = {pickedVoxel.coords.map((c) => c.toFixed(0)).join(", ")} m
          </div>
          <table style={{ fontSize: 11, width: "100%" }}>
            <tbody>
              {pickedVoxel.values.map((v) => (
                <tr key={v.property}>
                  <td style={{ opacity: 0.7 }}>{v.property}</td>
                  <td style={{ textAlign: "right" }}>
                    {Number.isFinite(v.value) ? v.value.toPrecision(4) : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
