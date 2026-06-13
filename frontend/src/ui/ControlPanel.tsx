// Transfer-function / control panel (doc 06 §9.2, §10). DOM overlay (not WebGL) driving
// the Zustand store: colormap pick, domain min/max, log toggle, opacity, slice axis+pos,
// clip box reset, step count. All edits push store state which the scene shaders mirror
// live (no volume refetch — doc 06 §9.2).

import { useViewer } from "../store";
import { COLORMAP_NAMES } from "../lib/colormaps";

const row: React.CSSProperties = { display: "flex", alignItems: "center", gap: 8, marginBottom: 8 };
const label: React.CSSProperties = { width: 92, fontSize: 12, opacity: 0.8 };
const panel: React.CSSProperties = {
  position: "absolute",
  top: 12,
  right: 12,
  width: 280,
  padding: 14,
  background: "rgba(17,22,33,0.92)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 13,
  zIndex: 10,
};

export function ControlPanel() {
  const meta = useViewer((s) => s.meta);
  const tf = useViewer((s) => s.tf);
  const setTF = useViewer((s) => s.setTF);
  const steps = useViewer((s) => s.steps);
  const setSteps = useViewer((s) => s.setSteps);
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
  const volumeVisible = useViewer((s) => s.volumeVisible);
  const setVolumeVisible = useViewer((s) => s.setVolumeVisible);

  // Domain slider span (a little beyond stats for headroom).
  const lo = meta?.stats.min ?? tf.domainMin - 1;
  const hi = meta?.stats.max ?? tf.domainMax + 1;
  const step = Math.max((hi - lo) / 200, 1e-6);

  return (
    <div style={panel}>
      <div style={{ fontWeight: 600, marginBottom: 10 }}>
        Transfer function
        {meta && (
          <span style={{ opacity: 0.6, fontWeight: 400 }}>
            {" "}— {meta.property} ({meta.canonicalUnit})
          </span>
        )}
      </div>

      <div style={row}>
        <span style={label}>Colormap</span>
        <select
          value={tf.colormap}
          onChange={(e) => setTF({ colormap: e.target.value })}
          style={{ flex: 1 }}
        >
          {COLORMAP_NAMES.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      <div style={row}>
        <span style={label}>Domain min</span>
        <input
          type="range"
          min={lo}
          max={hi}
          step={step}
          value={tf.domainMin}
          onChange={(e) => setTF({ domainMin: parseFloat(e.target.value) })}
          style={{ flex: 1 }}
        />
        <span style={{ width: 48, textAlign: "right" }}>{tf.domainMin.toFixed(2)}</span>
      </div>

      <div style={row}>
        <span style={label}>Domain max</span>
        <input
          type="range"
          min={lo}
          max={hi}
          step={step}
          value={tf.domainMax}
          onChange={(e) => setTF({ domainMax: parseFloat(e.target.value) })}
          style={{ flex: 1 }}
        />
        <span style={{ width: 48, textAlign: "right" }}>{tf.domainMax.toFixed(2)}</span>
      </div>

      <div style={row}>
        <span style={label}>Log scale</span>
        <input
          type="checkbox"
          checked={tf.scaling === "log"}
          onChange={(e) => setTF({ scaling: e.target.checked ? "log" : "linear" })}
        />
        <span style={{ ...label, marginLeft: 16 }}>Invert</span>
        <input
          type="checkbox"
          checked={tf.invert}
          onChange={(e) => setTF({ invert: e.target.checked })}
        />
      </div>

      <div style={row}>
        <span style={label}>Opacity</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={tf.opacity}
          onChange={(e) => setTF({ opacity: parseFloat(e.target.value) })}
          style={{ flex: 1 }}
        />
        <span style={{ width: 48, textAlign: "right" }}>{tf.opacity.toFixed(2)}</span>
      </div>

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
        <span style={label}>Volume</span>
        <input
          type="checkbox"
          checked={volumeVisible}
          onChange={(e) => setVolumeVisible(e.target.checked)}
        />
      </div>

      <hr style={{ border: "none", borderTop: "1px solid #313244", margin: "10px 0" }} />
      <div style={{ fontWeight: 600, marginBottom: 10 }}>Orthogonal slice</div>

      <div style={row}>
        <span style={label}>Enabled</span>
        <input
          type="checkbox"
          checked={sliceEnabled}
          onChange={(e) => setSliceEnabled(e.target.checked)}
        />
        <span style={{ ...label, marginLeft: 12 }}>Axis</span>
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
        <span style={{ width: 48, textAlign: "right" }}>{sliceOpacity.toFixed(2)}</span>
      </div>

      <hr style={{ border: "none", borderTop: "1px solid #313244", margin: "10px 0" }} />
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
        clip [{clip.min.map((v) => v.toFixed(2)).join(", ")}] –
        [{clip.max.map((v) => v.toFixed(2)).join(", ")}]
      </div>
    </div>
  );
}
