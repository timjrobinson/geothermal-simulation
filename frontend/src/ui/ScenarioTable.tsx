// Alternative-scenario comparison table (doc 09 §8.2). One row per saved PlannedWell snapshot
// (a named scenario); best-in-column highlighted (higher-is-better for BHT/pay/fractures,
// lower for risk/DLS/MD). Pure shaping (scenarioRow / bestInColumn) lives in lib/planning.

import { useMemo } from "react";
import { useViewer } from "../store";
import {
  scenarioRow,
  bestInColumn,
  type ScenarioColumn,
  type ScenarioRow,
} from "../lib/planning";

const COLUMNS: { key: ScenarioColumn; label: string; fmt: (v: number | null) => string }[] = [
  { key: "bhtC", label: "BHT°C", fmt: (v) => (v == null ? "—" : v.toFixed(0)) },
  { key: "payLength_m", label: "pay m", fmt: (v) => (v == null ? "—" : v.toFixed(0)) },
  { key: "fractureIntersections", label: "frac", fmt: (v) => (v == null ? "—" : String(v)) },
  { key: "inWindowFraction", label: "win%", fmt: (v) => (v == null ? "—" : (v * 100).toFixed(0)) },
  { key: "meanRisk", label: "risk", fmt: (v) => (v == null ? "—" : v.toFixed(2)) },
  { key: "maxDLS_deg30m", label: "DLS", fmt: (v) => (v == null ? "—" : v.toFixed(1)) },
  { key: "totalMD_m", label: "MD m", fmt: (v) => (v == null ? "—" : v.toFixed(0)) },
];

const cell: React.CSSProperties = {
  padding: "2px 4px",
  fontFamily: "ui-monospace, monospace",
  fontSize: 10,
  whiteSpace: "nowrap",
};

export function ScenarioTable() {
  const scenarios = useViewer((s) => s.scenarios);
  const removeScenario = useViewer((s) => s.removeScenario);
  const activeWellId = useViewer((s) => s.activeWellId);

  const { rows, best } = useMemo(() => {
    const rows: ScenarioRow[] = scenarios.map((sc) =>
      scenarioRow(sc.wellId, sc.name, sc.log, { maxDLS_deg30m: sc.maxDLS_deg30m }),
    );
    return { rows, best: bestInColumn(rows) };
  }, [scenarios]);

  if (rows.length === 0) {
    return (
      <div style={{ opacity: 0.6, fontSize: 11 }}>
        no scenarios saved — Predict then Save to compare
      </div>
    );
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ borderCollapse: "collapse", fontSize: 10 }}>
        <thead>
          <tr style={{ opacity: 0.7 }}>
            <th style={cell}>scenario</th>
            {COLUMNS.map((c) => (
              <th key={c.key} style={cell}>
                {c.label}
              </th>
            ))}
            <th style={cell}></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={r.wellId}
              style={{
                background: r.wellId === activeWellId ? "rgba(137,180,250,0.12)" : undefined,
              }}
            >
              <td style={{ ...cell, fontFamily: "ui-sans-serif" }}>{r.name}</td>
              {COLUMNS.map((c) => {
                const isBest = best[c.key] === i;
                return (
                  <td
                    key={c.key}
                    style={{
                      ...cell,
                      color: isBest ? "#a6e3a1" : "#cdd6f4",
                      fontWeight: isBest ? 700 : 400,
                    }}
                  >
                    {c.fmt(r[c.key] as number | null)}
                  </td>
                );
              })}
              <td style={cell}>
                <button
                  onClick={() => removeScenario(r.wellId)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#f38ba8",
                    cursor: "pointer",
                    fontSize: 11,
                  }}
                  title="remove scenario"
                >
                  ✕
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
