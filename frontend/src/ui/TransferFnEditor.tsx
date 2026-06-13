// Per-layer transfer-function editor (doc 06 §9.2). For the selected layer: a colormap
// gallery (clickable swatches), the value histogram (a simple bar chart from the resident
// volume / meta stats, doc 04 §9.2), domain min/max sliders, log/linear toggle, invert,
// opacity, and an "isolate band" window. All edits push the layer's transfer function into
// the store, which the scene re-bakes into a new LUT texture with NO volume refetch
// (doc 06 §9.2 "edits push a new LUT texture").

import { useMemo } from "react";
import { useViewer, type Layer } from "../store";
import { COLORMAPS, COLORMAP_NAMES, sampleColormap } from "../lib/colormaps";
import { histogram } from "../lib/volume";

const row: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  marginBottom: 8,
};
const label: React.CSSProperties = { width: 84, fontSize: 12, opacity: 0.8 };

// A CSS linear-gradient string previewing a colormap (gallery swatch).
function colormapGradient(name: string): string {
  const cm = COLORMAPS[name];
  if (!cm) return "#444";
  const stops: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    const [r, g, b] = sampleColormap(cm, t);
    stops.push(
      `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)}) ${(
        t * 100
      ).toFixed(0)}%`,
    );
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

export function TransferFnEditor({ layer }: { layer: Layer }) {
  const setLayerTF = useViewer((s) => s.setLayerTF);
  const tf = layer.transferFn;
  const meta = layer.meta;

  // Domain slider span (a little beyond stats for headroom).
  const statLo = meta?.stats.min ?? tf.domainMin - 1;
  const statHi = meta?.stats.max ?? tf.domainMax + 1;
  const lo = Math.min(statLo, tf.domainMin);
  const hi = Math.max(statHi, tf.domainMax);
  const step = Math.max((hi - lo) / 200, 1e-6);

  // Histogram from the resident volume (preferred) or a flat placeholder from stats.
  const bins = useMemo(() => {
    if (layer.volume) return histogram(layer.volume.data, lo, hi, 48);
    return new Array<number>(48).fill(1); // flat placeholder when no resident data
  }, [layer.volume, lo, hi]);
  const maxBin = Math.max(1, ...bins);

  const set = (patch: Partial<typeof tf>) => setLayerTF(layer.id, patch);

  // Map domain min/max to normalized [0,1] positions for the band/histogram overlay.
  const span = Math.max(hi - lo, 1e-9);
  const domLoT = (tf.domainMin - lo) / span;
  const domHiT = (tf.domainMax - lo) / span;

  return (
    <div>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>
        Transfer function
        {meta && (
          <span style={{ opacity: 0.6, fontWeight: 400 }}>
            {" "}
            — {meta.property} ({meta.canonicalUnit})
          </span>
        )}
      </div>

      {/* Histogram (doc 06 §9.2). Bars over the value range; band window shaded. */}
      <div
        style={{
          position: "relative",
          height: 46,
          display: "flex",
          alignItems: "flex-end",
          gap: 1,
          marginBottom: 6,
          background: "#11161f",
          borderRadius: 4,
          padding: 2,
          overflow: "hidden",
        }}
      >
        {tf.bandEnabled && (
          <div
            style={{
              position: "absolute",
              top: 0,
              bottom: 0,
              left: `${(tf.bandMin ?? 0) * 100}%`,
              width: `${((tf.bandMax ?? 1) - (tf.bandMin ?? 0)) * 100}%`,
              background: "rgba(137,180,250,0.18)",
              pointerEvents: "none",
            }}
          />
        )}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `${Math.max(0, domLoT) * 100}%`,
            width: `${Math.max(0, Math.min(1, domHiT) - Math.max(0, domLoT)) * 100}%`,
            background: colormapGradient(tf.colormap),
            opacity: 0.12,
            pointerEvents: "none",
          }}
        />
        {bins.map((v, i) => (
          <div
            key={i}
            style={{
              flex: 1,
              height: `${(v / maxBin) * 100}%`,
              background: "#89b4fa",
              opacity: 0.7,
              minHeight: 1,
            }}
          />
        ))}
      </div>

      {/* Colormap gallery */}
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 4 }}>Colormap</div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 8 }}>
        {COLORMAP_NAMES.map((c) => (
          <button
            key={c}
            title={c}
            onClick={() => set({ colormap: c })}
            style={{
              width: 42,
              height: 18,
              borderRadius: 3,
              border:
                tf.colormap === c ? "2px solid #cdd6f4" : "1px solid #313244",
              background: colormapGradient(c),
              cursor: "pointer",
              padding: 0,
            }}
          />
        ))}
      </div>

      <div style={row}>
        <span style={label}>Domain min</span>
        <input
          type="range"
          min={lo}
          max={hi}
          step={step}
          value={tf.domainMin}
          onChange={(e) => set({ domainMin: parseFloat(e.target.value) })}
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
          onChange={(e) => set({ domainMax: parseFloat(e.target.value) })}
          style={{ flex: 1 }}
        />
        <span style={{ width: 48, textAlign: "right" }}>{tf.domainMax.toFixed(2)}</span>
      </div>

      <div style={row}>
        <span style={label}>Log scale</span>
        <input
          type="checkbox"
          checked={tf.scaling === "log"}
          onChange={(e) => set({ scaling: e.target.checked ? "log" : "linear" })}
        />
        <span style={{ ...label, marginLeft: 12, width: 50 }}>Invert</span>
        <input
          type="checkbox"
          checked={tf.invert}
          onChange={(e) => set({ invert: e.target.checked })}
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
          onChange={(e) => set({ opacity: parseFloat(e.target.value) })}
          style={{ flex: 1 }}
        />
        <span style={{ width: 48, textAlign: "right" }}>{tf.opacity.toFixed(2)}</span>
      </div>

      {/* Isolate band (doc 06 §9.2) */}
      <div style={row}>
        <span style={label}>Isolate band</span>
        <input
          type="checkbox"
          checked={tf.bandEnabled ?? false}
          onChange={(e) =>
            set({
              bandEnabled: e.target.checked,
              bandMin: tf.bandMin ?? 0.4,
              bandMax: tf.bandMax ?? 0.7,
            })
          }
        />
      </div>
      {tf.bandEnabled && (
        <>
          <div style={row}>
            <span style={label}>Band lo</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={tf.bandMin ?? 0}
              onChange={(e) => set({ bandMin: parseFloat(e.target.value) })}
              style={{ flex: 1 }}
            />
            <span style={{ width: 48, textAlign: "right" }}>
              {(tf.bandMin ?? 0).toFixed(2)}
            </span>
          </div>
          <div style={row}>
            <span style={label}>Band hi</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={tf.bandMax ?? 1}
              onChange={(e) => set({ bandMax: parseFloat(e.target.value) })}
              style={{ flex: 1 }}
            />
            <span style={{ width: 48, textAlign: "right" }}>
              {(tf.bandMax ?? 1).toFixed(2)}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
