// Per-evidence membership-curve editor (doc 07 §4.6 transferFn; doc 06 §3.2 transfer
// functions). Picks the curve TYPE (ramp / sigmoid / gaussian-band), exposes its parameters
// in the evidence's canonical unit, and draws a live SVG preview sampled by the SAME pure
// membership() the backend uses — so what the user shapes is what the favorability volume
// computes. Stateless: it renders the supplied TransferFnSpec and reports edits up via
// onChange (the panel owns the evidence state).

import { useMemo } from "react";
import {
  type TransferFnSpec,
  type TransferType,
  TRANSFER_TYPES,
  sampleMembershipCurve,
  defaultTransferFn,
} from "../lib/favorability";

const lbl: React.CSSProperties = { fontSize: 10, opacity: 0.7, display: "block" };
const num: React.CSSProperties = {
  width: 64,
  fontSize: 11,
  background: "#1e2230",
  color: "#cdd6f4",
  border: "1px solid #313244",
  borderRadius: 3,
  padding: "1px 4px",
};
const sel: React.CSSProperties = {
  fontSize: 11,
  background: "#1e2230",
  color: "#cdd6f4",
  border: "1px solid #313244",
  borderRadius: 3,
};

// A small number input that tolerates the empty string while typing.
function NumberField({
  value,
  onChange,
  label,
}: {
  value: number | null | undefined;
  onChange: (v: number) => void;
  label: string;
}) {
  return (
    <label style={{ marginRight: 8 }}>
      <span style={lbl}>{label}</span>
      <input
        type="number"
        style={num}
        value={value ?? ""}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </label>
  );
}

export function MembershipCurveEditor({
  tf,
  unit,
  range,
  onChange,
}: {
  tf: TransferFnSpec;
  unit?: string | null;
  range: [number, number]; // observed value range, drives the preview x-extent
  onChange: (tf: TransferFnSpec) => void;
}) {
  const [lo, hi] = range;
  // Preview x-extent: pad a touch beyond the observed range so ramp shoulders are visible.
  const pad = hi > lo ? (hi - lo) * 0.1 : 1;
  const x0 = lo - pad;
  const x1 = hi + pad;

  const curve = useMemo(
    () => sampleMembershipCurve(tf, x0, x1, 80),
    [tf, x0, x1],
  );

  const W = 220;
  const H = 64;
  const path = useMemo(() => {
    if (curve.length === 0) return "";
    return curve
      .map(([x, m], i) => {
        const px = ((x - x0) / (x1 - x0)) * W;
        const py = H - m * H;
        return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(" ");
  }, [curve, x0, x1]);

  const setType = (type: TransferType) => onChange(defaultTransferFn(type, lo, hi));
  const patch = (p: Partial<TransferFnSpec>) => onChange({ ...tf, ...p });

  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
        <span style={lbl}>curve</span>
        <select
          style={sel}
          value={tf.type}
          onChange={(e) => setType(e.target.value as TransferType)}
        >
          {TRANSFER_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        {unit && <span style={{ fontSize: 10, opacity: 0.5 }}>({unit})</span>}
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", marginBottom: 4 }}>
        {tf.type === "ramp" && (
          <>
            <NumberField label="lo (m=0)" value={tf.lo} onChange={(v) => patch({ lo: v })} />
            <NumberField label="hi (m=1)" value={tf.hi} onChange={(v) => patch({ hi: v })} />
          </>
        )}
        {tf.type === "sigmoid" && (
          <>
            <NumberField label="center" value={tf.center} onChange={(v) => patch({ center: v })} />
            <NumberField label="k (slope)" value={tf.k} onChange={(v) => patch({ k: v })} />
          </>
        )}
        {tf.type === "gaussian-band" && (
          <>
            <NumberField label="center" value={tf.center} onChange={(v) => patch({ center: v })} />
            <NumberField label="width" value={tf.width} onChange={(v) => patch({ width: v })} />
          </>
        )}
      </div>

      {/* live membership preview (0→1 on Y) */}
      <svg
        width={W}
        height={H}
        style={{ background: "#11131c", border: "1px solid #313244", borderRadius: 3 }}
      >
        <line x1={0} y1={H} x2={W} y2={H} stroke="#45475a" strokeWidth={0.5} />
        <line x1={0} y1={0} x2={0} y2={H} stroke="#45475a" strokeWidth={0.5} />
        {path && <path d={path} fill="none" stroke="#fab387" strokeWidth={1.5} />}
      </svg>
    </div>
  );
}
