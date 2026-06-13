// Geothermal outputs + glass-box risk readout (doc 09 §6, §7.4). The numbers a geothermal
// engineer judges a plan by — BHT (±σ), pay length, fracture intersections, in-window %,
// mean/peak risk — plus the risk-driver breakdown ("what's driving risk", always shown
// alongside the number, doc 09 §7.4) and the editable §7.4 weights (the composite is a glass
// box, not a black box). Pure shaping (riskDriverBreakdown) lives in lib/planning.

import { useViewer } from "../store";
import {
  riskDriverBreakdown,
  valueToCss,
  type PredictedLog,
  type RiskWeights,
} from "../lib/planning";

const row: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  fontFamily: "ui-monospace, monospace",
  fontSize: 11,
  lineHeight: 1.6,
};
const bar: React.CSSProperties = {
  height: 8,
  borderRadius: 2,
  background: "#1e2230",
  overflow: "hidden",
  flex: 1,
  marginLeft: 6,
};

function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div style={row}>
      <span style={{ opacity: 0.75 }}>{k}</span>
      <span>{v}</span>
    </div>
  );
}

const WEIGHT_KEYS: { key: keyof RiskWeights; label: string }[] = [
  { key: "tempConfidence", label: "temp conf" },
  { key: "hazard", label: "hazard" },
  { key: "dlsExceedance", label: "DLS exc" },
  { key: "structuralUncertainty", label: "struct unc" },
];

export function RiskReadout({ log }: { log: PredictedLog }) {
  const weights = useViewer((s) => s.riskWeights);
  const setRiskWeights = useViewer((s) => s.setRiskWeights);
  const drillability = log.drillability;

  const s = log.summary;
  const drivers = riskDriverBreakdown(log);

  const fmt = (v: number | null, suffix = "", digits = 1) =>
    v == null ? "—" : `${v.toFixed(digits)}${suffix}`;

  return (
    <div>
      {/* geothermal outputs (doc 09 §6) */}
      <Stat
        k="BHT"
        v={
          s.bhtC == null
            ? "—"
            : `${s.bhtC.toFixed(1)} ± ${fmt(s.bhtSigmaC, "", 1)} °C`
        }
      />
      <Stat
        k="max temp"
        v={s.maxTempC == null ? "—" : `${s.maxTempC.toFixed(1)} °C @ ${fmt(s.maxTempMD_m, " m MD", 0)}`}
      />
      <Stat k="pay length" v={fmt(s.targetIntersectionLength_m, " m", 0)} />
      <Stat k="reservoir" v={fmt(s.reservoirIntersectionLength_m, " m", 0)} />
      <Stat k="fractures" v={`${s.productiveFractureIntersections}`} />
      <Stat k="in-window" v={`${(s.inWindowFraction * 100).toFixed(0)}%`} />

      {/* risk (doc 09 §7.4) */}
      <div style={{ ...row, marginTop: 6 }}>
        <span style={{ opacity: 0.75 }}>mean / peak risk</span>
        <span>
          {s.meanRisk.toFixed(2)} / {s.peakRisk.toFixed(2)}
        </span>
      </div>

      {/* glass-box driver breakdown */}
      <div style={{ fontSize: 10, opacity: 0.7, margin: "6px 0 2px" }}>
        risk drivers (mean contribution)
      </div>
      {drivers.map((dr) => (
        <div key={dr.name} style={{ display: "flex", alignItems: "center", marginBottom: 2 }}>
          <span style={{ fontSize: 10, width: 80, fontFamily: "ui-monospace, monospace" }}>
            {dr.name}
          </span>
          <div style={bar}>
            <div
              style={{
                width: `${Math.min(100, dr.fraction * 100)}%`,
                height: "100%",
                background: valueToCss(dr.fraction, "magma"),
              }}
            />
          </div>
          <span style={{ fontSize: 10, width: 36, textAlign: "right" }}>
            {(dr.fraction * 100).toFixed(0)}%
          </span>
        </div>
      ))}

      {/* editable §7.4 weights (the glass box is tunable) */}
      <div style={{ fontSize: 10, opacity: 0.7, margin: "8px 0 2px" }}>
        weights (re-Predict to apply)
      </div>
      {WEIGHT_KEYS.map(({ key, label }) => (
        <div key={key} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ fontSize: 10, width: 70 }}>{label}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={weights[key]}
            onChange={(e) => setRiskWeights({ [key]: parseFloat(e.target.value) })}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 10, width: 28, textAlign: "right" }}>
            {weights[key].toFixed(2)}
          </span>
        </div>
      ))}

      {/* crude drillability flag (doc 09 §4.6) */}
      {drillability && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 10, opacity: 0.7, marginBottom: 2 }}>
            drillability:{" "}
            <span style={{ color: drillability.verdict === "warn" ? "#f9e2af" : "#a6e3a1" }}>
              {drillability.verdict}
            </span>
          </div>
          {drillability.checks.map((c) => (
            <div key={c.name} style={{ ...row, opacity: c.verdict === "warn" ? 1 : 0.6 }}>
              <span>
                {c.verdict === "warn" ? "⚠ " : ""}
                {c.name}
              </span>
              <span>
                {c.value.toFixed(1)} / {c.limit.toFixed(1)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
