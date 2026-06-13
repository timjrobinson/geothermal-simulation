// Control panel (doc 06 §9, §10). DOM overlay (not WebGL) driving the Zustand store. It
// hosts three sections:
//   1. Layer manager (doc 06 §9.1): add/remove/reorder/toggle/opacity/blend per layer.
//   2. Per-layer transfer-function editor (doc 06 §9.2) for the selected layer.
//   3. Global orthogonal slice (§4) + clip box (§2.4) controls.
// The "open project" discovery flow (doc 06 §9.1) is a togglable sub-panel. All edits push
// store state which the scene shaders mirror live (no volume refetch — doc 06 §9.2).

import { useState } from "react";
import { useViewer, selectedLayer } from "../store";
import { BLEND_MODES, type BlendMode } from "../lib/layers";
import { TransferFnEditor } from "./TransferFnEditor";
import { DiscoveryPanel } from "./DiscoveryPanel";

const row: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  marginBottom: 8,
};
const label: React.CSSProperties = { width: 84, fontSize: 12, opacity: 0.8 };
const panel: React.CSSProperties = {
  position: "absolute",
  top: 12,
  right: 12,
  width: 300,
  maxHeight: "calc(100vh - 24px)",
  overflowY: "auto",
  padding: 14,
  background: "rgba(17,22,33,0.92)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 13,
  zIndex: 10,
};
const iconBtn: React.CSSProperties = {
  background: "#313244",
  color: "#cdd6f4",
  border: "none",
  borderRadius: 4,
  padding: "1px 6px",
  cursor: "pointer",
  fontSize: 12,
  lineHeight: "16px",
};
const hr: React.CSSProperties = {
  border: "none",
  borderTop: "1px solid #313244",
  margin: "10px 0",
};

function LayerManager() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  const selectedLayerId = useViewer((s) => s.selectedLayerId);
  const selectLayer = useViewer((s) => s.selectLayer);
  const removeLayer = useViewer((s) => s.removeLayer);
  const moveLayer = useViewer((s) => s.moveLayer);
  const setLayerVisible = useViewer((s) => s.setLayerVisible);
  const setLayerOpacity = useViewer((s) => s.setLayerOpacity);
  const setLayerBlend = useViewer((s) => s.setLayerBlend);

  // Top of the list == top of the composite (drawn last). layerOrder is bottom→top.
  const ordered = [...layerOrder].reverse();

  if (ordered.length === 0) {
    return (
      <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 8 }}>
        No layers — open a project or add the mock layer.
      </div>
    );
  }

  return (
    <div style={{ marginBottom: 4 }}>
      {ordered.map((id) => {
        const l = layers[id];
        if (!l) return null;
        const sel = id === selectedLayerId;
        return (
          <div
            key={id}
            style={{
              border: sel ? "1px solid #89b4fa" : "1px solid #313244",
              borderRadius: 6,
              padding: 6,
              marginBottom: 6,
              background: sel ? "rgba(137,180,250,0.08)" : "transparent",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <input
                type="checkbox"
                checked={l.visible}
                onChange={(e) => setLayerVisible(id, e.target.checked)}
                title="visible"
              />
              <button
                onClick={() => selectLayer(id)}
                style={{
                  ...iconBtn,
                  flex: 1,
                  textAlign: "left",
                  background: "transparent",
                  fontWeight: sel ? 600 : 400,
                }}
              >
                {l.name}
              </button>
              <button style={iconBtn} title="move up" onClick={() => moveLayer(id, "up")}>
                ↑
              </button>
              <button
                style={iconBtn}
                title="move down"
                onClick={() => moveLayer(id, "down")}
              >
                ↓
              </button>
              <button style={iconBtn} title="remove" onClick={() => removeLayer(id)}>
                ✕
              </button>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 4 }}>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={l.opacity}
                onChange={(e) => setLayerOpacity(id, parseFloat(e.target.value))}
                style={{ flex: 1 }}
                title="layer opacity"
              />
              <select
                value={l.blend}
                onChange={(e) => setLayerBlend(id, e.target.value as BlendMode)}
                title="blend mode"
                style={{ fontSize: 11 }}
              >
                {BLEND_MODES.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Default terrain extent (Engineering metres) when no scene footprint is available yet —
// the doc 01 §2 default ROI (±5 km). When a scene AABB exists we use its XY footprint so
// the surface spans the loaded data.
const DEFAULT_TERRAIN_EXTENT = {
  xmin: -5000,
  xmax: 5000,
  ymin: -5000,
  ymax: 5000,
};

export function ControlPanel() {
  const layer = useViewer(selectedLayer);
  const steps = useViewer((s) => s.steps);
  const setSteps = useViewer((s) => s.setSteps);
  const vex = useViewer((s) => s.verticalExaggeration);
  const setVex = useViewer((s) => s.setVerticalExaggeration);
  const addTerrainLayer = useViewer((s) => s.addTerrainLayer);
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  const sceneAABB = useViewer((s) => s.sceneAABB);

  // The surfaceModel spec from the selected (or any) layer's frame meta, when present
  // (doc 01 §2). Falls back to flat:0 (local mode default).
  const hasTerrain = layerOrder.some((id) => layers[id]?.kind === "terrain");
  const onAddTerrain = () => {
    const frame =
      (layer?.meta?.frame as Record<string, unknown> | undefined) ?? undefined;
    const surfaceModelSpec =
      (frame?.surfaceModel as string | null | undefined) ?? "flat:0";
    const extent = sceneAABB
      ? {
          xmin: sceneAABB.min[0],
          xmax: sceneAABB.max[0],
          ymin: sceneAABB.min[1],
          ymax: sceneAABB.max[1],
        }
      : DEFAULT_TERRAIN_EXTENT;
    addTerrainLayer({ surfaceModelSpec, extent });
  };
  const clip = useViewer((s) => s.clip);
  const setClip = useViewer((s) => s.setClip);
  const sliceEnabled = useViewer((s) => s.sliceEnabled);
  const setSliceEnabled = useViewer((s) => s.setSliceEnabled);
  const sliceAxis = useViewer((s) => s.sliceAxis);
  const setSliceAxis = useViewer((s) => s.setSliceAxis);
  const slicePos = useViewer((s) => s.slicePos);
  const setSlicePos = useViewer((s) => s.setSlicePos);
  const sliceOpacity = useViewer((s) => s.sliceOpacity);
  const setSliceOpacity = useViewer((s) => s.setSliceOpacity);

  const [showDiscovery, setShowDiscovery] = useState(false);

  return (
    <div style={panel}>
      {showDiscovery ? (
        <DiscoveryPanel onClose={() => setShowDiscovery(false)} />
      ) : (
        <>
          <div
            style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}
          >
            <span style={{ fontWeight: 600 }}>Layers</span>
            <span style={{ display: "flex", gap: 6 }}>
              <button
                style={iconBtn}
                title="add a ground-surface terrain layer (doc 06 §6)"
                disabled={hasTerrain}
                onClick={onAddTerrain}
              >
                + terrain
              </button>
              <button style={iconBtn} onClick={() => setShowDiscovery(true)}>
                + open project
              </button>
            </span>
          </div>

          <LayerManager />

          {layer && layer.kind === "volume" && (
            <>
              <hr style={hr} />
              <TransferFnEditor layer={layer} />
            </>
          )}

          <hr style={hr} />
          <div style={row}>
            <span style={label}>Steps</span>
            <input
              type="range"
              min={64}
              max={512}
              step={8}
              value={steps}
              onChange={(e) => setSteps(parseInt(e.target.value, 10))}
              style={{ flex: 1 }}
            />
            <span style={{ width: 48, textAlign: "right" }}>{steps}</span>
          </div>

          <div style={row}>
            <span style={label}>Vert. exag</span>
            <input
              type="range"
              min={1}
              max={10}
              step={0.5}
              value={vex}
              onChange={(e) => setVex(parseFloat(e.target.value))}
              style={{ flex: 1 }}
              title="vertical exaggeration (render-only, doc 06 §2.3)"
            />
            <span style={{ width: 48, textAlign: "right" }}>{vex.toFixed(1)}×</span>
          </div>

          <hr style={hr} />
          <div style={{ fontWeight: 600, marginBottom: 10 }}>
            Orthogonal slice
            <span style={{ opacity: 0.6, fontWeight: 400 }}>
              {" "}
              — {layer ? layer.name : "no layer"}
            </span>
          </div>

          <div style={row}>
            <span style={label}>Enabled</span>
            <input
              type="checkbox"
              checked={sliceEnabled}
              onChange={(e) => setSliceEnabled(e.target.checked)}
            />
            <span style={{ ...label, marginLeft: 12, width: 36 }}>Axis</span>
            {(["x", "y", "z"] as const).map((a) => (
              <button
                key={a}
                onClick={() => setSliceAxis(a)}
                style={{
                  background: sliceAxis === a ? "#89b4fa" : "#313244",
                  color: sliceAxis === a ? "#11161f" : "#cdd6f4",
                  border: "none",
                  borderRadius: 4,
                  padding: "2px 8px",
                  cursor: "pointer",
                }}
              >
                {a.toUpperCase()}
              </button>
            ))}
          </div>

          <div style={row}>
            <span style={label}>Position</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.005}
              value={slicePos}
              onChange={(e) => setSlicePos(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{ width: 48, textAlign: "right" }}>{slicePos.toFixed(2)}</span>
          </div>

          <div style={row}>
            <span style={label}>Slice α</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={sliceOpacity}
              onChange={(e) => setSliceOpacity(parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{ width: 48, textAlign: "right" }}>
              {sliceOpacity.toFixed(2)}
            </span>
          </div>

          <hr style={hr} />
          <div style={row}>
            <span style={label}>Clip box</span>
            <button
              onClick={() => setClip({ min: [0, 0, 0], max: [1, 1, 1] })}
              style={{
                background: "#313244",
                color: "#cdd6f4",
                border: "none",
                borderRadius: 4,
                padding: "3px 10px",
                cursor: "pointer",
              }}
            >
              Reset
            </button>
            <span style={{ fontSize: 11, opacity: 0.6, marginLeft: 8 }}>
              drag the colored handles
            </span>
          </div>
          <div style={{ fontSize: 11, opacity: 0.6 }}>
            clip [{clip.min.map((v) => v.toFixed(2)).join(", ")}] – [
            {clip.max.map((v) => v.toFixed(2)).join(", ")}]
          </div>
        </>
      )}
    </div>
  );
}
