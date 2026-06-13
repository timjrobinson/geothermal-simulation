// Well-planning workflow panel (doc 09 §8). The full plan loop in one docked panel:
//   1. Context — bind the planning session to a project + fused model (scopes the backend).
//   2. Target — a 'pick target' mode (click the 3D scene → Engineering XYZ → POST target) OR
//      manual XYZ entry; shows the enriched temperature/favorability/lithology readout.
//   3. Trajectory — a design form (method / KOP / build rate / landing inc / maxDLS) → POST
//      solve → render the planned well as a WellLayer tube (over-DLS segments red, §8.1).
//      A debounced 'target-pull' re-solves when the target moves.
//   4. Predict — POST predict → predicted-log tracks (PredictedLogTracks) + tube colours +
//      the geothermal outputs / glass-box risk readout (RiskReadout).
//   5. Scenarios — save named alternatives → comparison table (best-in-column).
//   6. Export — CSV survey / CSV log / WITSML via the export endpoint.
//
// All request-body shaping is in lib/planning (pure, unit-tested); this component is the
// wiring + form state via the Zustand store.

import { useEffect, useRef, useState } from "react";
import { useViewer } from "../store";
import {
  DESIGN_METHODS,
  positionsToTrajectory,
  predictedLogToLogs,
  type DesignParams,
  type PredictedLog,
} from "../lib/planning";
import {
  createTarget,
  createWell,
  solveWell,
  fetchWellPositions,
  predictWell,
  exportUrl,
  type DrillTargetOut,
} from "../lib/planningApi";
import { RiskReadout } from "./RiskReadout";
import { ScenarioTable } from "./ScenarioTable";
import { PredictedLogTracks } from "./PredictedLogTracks";

const panel: React.CSSProperties = {
  position: "absolute",
  top: 64,
  left: 12,
  width: 320,
  maxHeight: "calc(100vh - 96px)",
  overflowY: "auto",
  padding: 12,
  background: "rgba(17,22,33,0.96)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 12,
  zIndex: 12,
};
const input: React.CSSProperties = {
  fontSize: 12,
  background: "#1e2230",
  color: "#cdd6f4",
  border: "1px solid #313244",
  borderRadius: 4,
  padding: "2px 4px",
  width: "100%",
  boxSizing: "border-box",
};
const btn: React.CSSProperties = {
  fontSize: 12,
  background: "#313244",
  color: "#cdd6f4",
  border: "1px solid #45475a",
  borderRadius: 4,
  padding: "4px 8px",
  cursor: "pointer",
};
const btnPrimary: React.CSSProperties = {
  ...btn,
  background: "#89b4fa",
  color: "#11131c",
  borderColor: "#89b4fa",
  fontWeight: 600,
};
const section: React.CSSProperties = {
  borderTop: "1px solid #313244",
  marginTop: 10,
  paddingTop: 8,
};
const label: React.CSSProperties = { fontSize: 11, opacity: 0.8, marginBottom: 2 };

function NumField({
  value,
  onChange,
  step = 1,
}: {
  value: number | null | undefined;
  onChange: (v: number) => void;
  step?: number;
}) {
  return (
    <input
      style={input}
      type="number"
      step={step}
      value={value ?? ""}
      onChange={(e) => onChange(parseFloat(e.target.value))}
    />
  );
}

export function PlanningPanel() {
  const projectId = useViewer((s) => s.planningProjectId);
  const fusedModelId = useViewer((s) => s.planningFusedModelId);
  const setPlanningContext = useViewer((s) => s.setPlanningContext);
  const pickMode = useViewer((s) => s.pickTargetMode);
  const setPickTargetMode = useViewer((s) => s.setPickTargetMode);
  const pendingPickXYZ = useViewer((s) => s.pendingPickXYZ);
  const setPendingPickXYZ = useViewer((s) => s.setPendingPickXYZ);
  const planTarget = useViewer((s) => s.planTarget);
  const setPlanTarget = useViewer((s) => s.setPlanTarget);
  const design = useViewer((s) => s.designParams);
  const setDesignParams = useViewer((s) => s.setDesignParams);
  const activeWellId = useViewer((s) => s.activeWellId);
  const setActiveWell = useViewer((s) => s.setActiveWell);
  const predictedLog = useViewer((s) => s.predictedLog);
  const setPredictedLog = useViewer((s) => s.setPredictedLog);
  const setPlanningOpen = useViewer((s) => s.setPlanningOpen);
  const addWellLayer = useViewer((s) => s.addWellLayer);
  const removeLayer = useViewer((s) => s.removeLayer);
  const saveScenario = useViewer((s) => s.saveScenario);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [name, setName] = useState("W-01");
  const [scenarioName, setScenarioName] = useState("scenario-1");

  // Render the solved well as a WellLayer tube. Replaces the active layer in place (stable id
  // per well) so re-solves update the tube without stacking layers. `joinLog` (the freshly
  // -fetched predicted log) joins the predicted curves onto the tube so a curve can colour it;
  // when absent the tube paints by the DLS constraint (over-ceiling segments red, §8.1).
  async function renderWell(wid: string, joinLog?: PredictedLog | null) {
    const pos = await fetchWellPositions(wid);
    const log = joinLog ?? useViewer.getState().predictedLog;
    const logs = log ? predictedLogToLogs(log) : undefined;
    const traj = positionsToTrajectory(pos, { featureId: wid, logs });
    // Read the live layer id (the debounced re-solve closure can be stale).
    const prevLayer = useViewer.getState().activeLayerId;
    if (prevLayer) removeLayer(prevLayer);
    const layerId = addWellLayer(traj, {
      id: `plan-${wid}`,
      name,
      logProperty: logs ? "temperatureC" : null,
      dlsMax_deg30m: useViewer.getState().designParams.maxDLS_deg30m,
    });
    setActiveWell(wid, layerId);
  }

  // ── target: pick / manual / POST ─────────────────────────────────────────────────────
  async function postTarget(xyz: [number, number, number]) {
    if (!projectId || !fusedModelId) {
      setErr("set a project + fused-model id first");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const t = await createTarget(projectId, fusedModelId, xyz, { name: `${name}-target` });
      setPlanTarget(t);
      setDesignParams({ target: [t.location.x, t.location.y, t.location.z] });
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  // Consume a 3D pick (PickTargetLayer wrote it to the store) → POST a target.
  useEffect(() => {
    if (!pendingPickXYZ) return;
    const xyz = pendingPickXYZ;
    setPendingPickXYZ(null);
    void postTarget(xyz);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingPickXYZ]);

  // ── trajectory: solve / target-pull ──────────────────────────────────────────────────
  async function solve() {
    if (!projectId) {
      setErr("set a project id first");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      let wid = activeWellId;
      if (!wid) {
        const well = await createWell(
          projectId,
          name,
          [design.target?.[0] ?? 0, design.target?.[1] ?? 0],
          design,
          { targetIds: planTarget ? [planTarget.id] : undefined },
        );
        wid = well.id;
      } else {
        await solveWell(wid, design);
      }
      await renderWell(wid);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  // Debounced target-pull: when the design target moves (panel edit OR a 3D drag), re-solve
  // an existing well live (doc 09 §8.1). Skips the very first render + when no well exists.
  const targetKey = design.target ? design.target.join(",") : "";
  const firstPull = useRef(true);
  useEffect(() => {
    if (firstPull.current) {
      firstPull.current = false;
      return;
    }
    if (!activeWellId || !targetKey) return;
    const h = setTimeout(() => {
      void solve();
    }, 400);
    return () => clearTimeout(h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetKey]);

  // ── predict ──────────────────────────────────────────────────────────────────────────
  async function predict() {
    if (!activeWellId || !fusedModelId) {
      setErr("solve a well + set a fused-model id first");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const log: PredictedLog = await predictWell(activeWellId, fusedModelId, {
        targetId: planTarget?.id ?? null,
        riskWeights: useViewer.getState().riskWeights,
      });
      setPredictedLog(log);
      // Re-join the predicted log onto the tube so the temperature curve colours it.
      await renderWell(activeWellId, log);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const d = design;
  const setD = (patch: Partial<DesignParams>) => setDesignParams(patch);

  return (
    <div style={panel}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 6,
        }}
      >
        <div style={{ fontWeight: 600 }}>Well planning</div>
        <button style={btn} onClick={() => setPlanningOpen(false)} title="close">
          ✕
        </button>
      </div>

      {/* ── context ── */}
      <div style={{ display: "flex", gap: 4 }}>
        <div style={{ flex: 1 }}>
          <div style={label}>project id</div>
          <input
            style={input}
            value={projectId ?? ""}
            placeholder="proj_…"
            onChange={(e) => setPlanningContext(e.target.value || null, fusedModelId)}
          />
        </div>
        <div style={{ flex: 1 }}>
          <div style={label}>fused model id</div>
          <input
            style={input}
            value={fusedModelId ?? ""}
            placeholder="fem_…"
            onChange={(e) => setPlanningContext(projectId, e.target.value || null)}
          />
        </div>
      </div>
      <div style={{ marginTop: 6 }}>
        <div style={label}>well name</div>
        <input style={input} value={name} onChange={(e) => setName(e.target.value)} />
      </div>

      {/* ── target ── */}
      <div style={section}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>1 · Target</div>
        <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
          <button
            style={pickMode ? btnPrimary : btn}
            onClick={() => setPickTargetMode(!pickMode)}
            title="click in the 3D scene to drop a target"
          >
            {pickMode ? "click scene…" : "Pick target"}
          </button>
          <button
            style={btn}
            disabled={!d.target}
            onClick={() => d.target && void postTarget(d.target)}
            title="enrich the manually-entered XYZ"
          >
            Enrich XYZ
          </button>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {(["x", "y", "z"] as const).map((axis, i) => (
            <div key={axis} style={{ flex: 1 }}>
              <div style={label}>{axis} (m)</div>
              <NumField
                value={d.target?.[i] ?? null}
                onChange={(v) => {
                  const t: [number, number, number] = [
                    d.target?.[0] ?? 0,
                    d.target?.[1] ?? 0,
                    d.target?.[2] ?? 0,
                  ];
                  t[i] = v;
                  setD({ target: t });
                }}
              />
            </div>
          ))}
        </div>
        {planTarget && <TargetReadout target={planTarget} />}
      </div>

      {/* ── trajectory ── */}
      <div style={section}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>2 · Trajectory</div>
        <div style={label}>method</div>
        <select
          style={input}
          value={d.method}
          onChange={(e) => setD({ method: e.target.value as DesignParams["method"] })}
        >
          {DESIGN_METHODS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
        <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
          <div style={{ flex: 1 }}>
            <div style={label}>KOP MD (m)</div>
            <NumField value={d.kopMD_m} step={10} onChange={(v) => setD({ kopMD_m: v })} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={label}>build °/30m</div>
            <NumField
              value={d.buildRate_deg30m}
              step={0.5}
              onChange={(v) => setD({ buildRate_deg30m: v })}
            />
          </div>
        </div>
        <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
          <div style={{ flex: 1 }}>
            <div style={label}>landing inc °</div>
            <NumField
              value={d.landingInc_deg ?? null}
              step={1}
              onChange={(v) => setD({ landingInc_deg: v })}
            />
          </div>
          <div style={{ flex: 1 }}>
            <div style={label}>max DLS °/30m</div>
            <NumField
              value={d.maxDLS_deg30m}
              step={0.5}
              onChange={(v) => setD({ maxDLS_deg30m: v })}
            />
          </div>
        </div>
        <button style={{ ...btnPrimary, marginTop: 8, width: "100%" }} disabled={busy} onClick={() => void solve()}>
          {activeWellId ? "Re-solve" : "Solve trajectory"}
        </button>
        {activeWellId && (
          <div style={{ fontSize: 10, opacity: 0.6, marginTop: 4 }}>
            well {activeWellId} — over-DLS segments render red. Target-pull re-solves on move.
          </div>
        )}
      </div>

      {/* ── predict ── */}
      <div style={section}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>3 · Predicted log</div>
        <button
          style={{ ...btnPrimary, width: "100%" }}
          disabled={busy || !activeWellId}
          onClick={() => void predict()}
        >
          Predict along path
        </button>
        {predictedLog && (
          <div style={{ marginTop: 8 }}>
            <PredictedLogTracks />
          </div>
        )}
      </div>

      {/* ── outputs + risk ── */}
      {predictedLog && (
        <div style={section}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>4 · Outputs & risk</div>
          <RiskReadout log={predictedLog} />
        </div>
      )}

      {/* ── scenarios ── */}
      <div style={section}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>5 · Scenarios</div>
        <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
          <input
            style={input}
            value={scenarioName}
            onChange={(e) => setScenarioName(e.target.value)}
          />
          <button
            style={btn}
            disabled={!activeWellId || !predictedLog}
            onClick={() => saveScenario(scenarioName, design.maxDLS_deg30m)}
            title="snapshot the current plan as a comparison scenario"
          >
            Save
          </button>
        </div>
        <ScenarioTable />
      </div>

      {/* ── export ── */}
      <div style={section}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>6 · Export</div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          <a
            href={activeWellId ? exportUrl(activeWellId, "csv-survey") : undefined}
            style={{
              ...btn,
              textDecoration: "none",
              pointerEvents: activeWellId ? "auto" : "none",
              opacity: activeWellId ? 1 : 0.5,
            }}
          >
            CSV survey
          </a>
          <a
            href={
              activeWellId && fusedModelId
                ? exportUrl(activeWellId, "csv-log", {
                    fusedModelId,
                    targetId: planTarget?.id ?? null,
                  })
                : undefined
            }
            style={{
              ...btn,
              textDecoration: "none",
              pointerEvents: activeWellId && fusedModelId ? "auto" : "none",
              opacity: activeWellId && fusedModelId ? 1 : 0.5,
            }}
          >
            CSV log
          </a>
          <a
            href={
              activeWellId
                ? exportUrl(activeWellId, "witsml", { fusedModelId: fusedModelId ?? undefined })
                : undefined
            }
            style={{
              ...btn,
              textDecoration: "none",
              pointerEvents: activeWellId ? "auto" : "none",
              opacity: activeWellId ? 1 : 0.5,
            }}
          >
            WITSML
          </a>
        </div>
      </div>

      {err && (
        <div style={{ color: "#f38ba8", fontSize: 11, marginTop: 8 }}>error: {err}</div>
      )}
      {busy && <div style={{ opacity: 0.6, fontSize: 11, marginTop: 6 }}>working…</div>}
    </div>
  );
}

function TargetReadout({ target }: { target: DrillTargetOut }) {
  const s = target.sampled;
  return (
    <div
      style={{
        marginTop: 6,
        fontSize: 11,
        fontFamily: "ui-monospace, monospace",
        background: "#11131c",
        borderRadius: 4,
        padding: 6,
        lineHeight: 1.5,
      }}
    >
      <div>target {target.id}</div>
      {s ? (
        <>
          {s.temperatureC?.value != null && (
            <div>
              T {s.temperatureC.value.toFixed(1)} °C
              {s.temperatureC.confidence != null &&
                ` · conf ${(s.temperatureC.confidence * 100).toFixed(0)}%`}
            </div>
          )}
          {s.favorability?.value != null && (
            <div>fav {s.favorability.value.toFixed(2)}</div>
          )}
          {s.lithology && <div>lith {s.lithology}</div>}
          {s.depthTVD_m != null && <div>TVD {s.depthTVD_m.toFixed(0)} m</div>}
        </>
      ) : (
        <div style={{ opacity: 0.6 }}>not enriched</div>
      )}
    </div>
  );
}
