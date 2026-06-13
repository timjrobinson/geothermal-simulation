// Favorability panel — the R&D favorability instrument (doc 07 §4.6; doc 06 §3.2/§9.2).
//
// Flow: pick evidence volume layers → per-evidence membership-curve editor (ramp / sigmoid /
// gaussian-band) + weight slider + role (required ⇒ fuzzy-AND / supporting ⇒ fuzzy-OR) →
// method selector (fuzzy DEFAULT / weighted exploratory) + fuzzy-AND op + missingPolicy →
// "Compute favorability" builds (or reuses) a fused grid, resamples the evidence sources into
// it, and POSTs /fused/{id}/favorability. The returned favorability PropertyModel renders as
// an inferno layer (reusing the existing layer system); its three honesty companions
// (confidence, evidence-overlap, assumption-burden, doc 07 §4.6) load as TOGGLEABLE
// diagnostic layers. The confidence companion is then BOUND to the favorability layer to
// drive confidence-modulated opacity (the doc 07 §5.3 "honest view" — low-confidence regions
// render faint). Re-running on a weight/curve change recomputes + replaces the volume.
//
// All spec building + validation lives in lib/favorability.ts (unit-tested); this component
// is the wiring + the controls. Backend shapes mirror geosim/api/fusion.py.

import { useEffect, useMemo, useState } from "react";
import { useViewer } from "../store";
import { fetchMeta, fetchVolume } from "../lib/api";
import { createFused, resampleLayer } from "../lib/fusion";
import {
  buildFavorabilitySpec,
  computeFavorability,
  defaultTransferFn,
  listTransforms,
  type EvidenceSpec,
  type FavorabilityMethod,
  type FuzzyAnd,
  type MissingPolicy,
  type FavorabilityResult,
  FAVORABILITY_METHODS,
  FUZZY_AND_OPS,
  MISSING_POLICIES,
} from "../lib/favorability";
import { MembershipCurveEditor } from "./MembershipCurveEditor";
import { finiteMinMax } from "../lib/volume";

const panel: React.CSSProperties = {
  position: "absolute",
  top: 64,
  right: 12,
  width: 340,
  maxHeight: "calc(100vh - 88px)",
  overflowY: "auto",
  padding: 14,
  background: "rgba(17,22,33,0.94)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 13,
  zIndex: 12,
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
const primaryBtn: React.CSSProperties = { ...btn, background: "#fab387", color: "#11131c", fontWeight: 600 };
const hr: React.CSSProperties = { border: "none", borderTop: "1px solid #313244", margin: "10px 0" };
const sel: React.CSSProperties = { fontSize: 12, background: "#1e2230", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 4 };
const lbl: React.CSSProperties = { fontSize: 11, opacity: 0.75 };

// Stable layer ids for the favorability product + its honesty diagnostics so a re-run
// REPLACES them in place (never accumulating duplicate layers).
const FAVOR_LAYER_ID = "favorability";
const CONF_LAYER_ID = "favorability-confidence";
const OVERLAP_LAYER_ID = "favorability-overlap";
const BURDEN_LAYER_ID = "favorability-burden";

interface EvidenceRow extends EvidenceSpec {
  // the layer this evidence came from (for the observed range → curve preview + label).
  layerId: string;
  range: [number, number];
  unit?: string | null;
}

export function FavorabilityPanel({ onClose }: { onClose: () => void }) {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  const addVolumeLayer = useViewer((s) => s.addVolumeLayer);
  const setLayerConfidence = useViewer((s) => s.setLayerConfidence);
  const setLayerTF = useViewer((s) => s.setLayerTF);

  // Candidate evidence layers: distinct backend-sourced volume layers (skip the mock + the
  // selection overlay + the favorability products themselves).
  const candidates = useMemo(
    () =>
      layerOrder
        .map((id) => layers[id])
        .filter(
          (l) =>
            l &&
            l.kind === "volume" &&
            l.datasetId &&
            l.datasetId !== "mock" &&
            ![FAVOR_LAYER_ID, CONF_LAYER_ID, OVERLAP_LAYER_ID, BURDEN_LAYER_ID, "selection-mask"].includes(
              l.id,
            ),
        ),
    [layers, layerOrder],
  );

  const projectId = useMemo(() => {
    for (const l of candidates) {
      const pid = (l?.meta?.frame as Record<string, unknown> | undefined)?.["projectId"];
      if (typeof pid === "string") return pid;
    }
    return "";
  }, [candidates]);

  const [evidence, setEvidence] = useState<EvidenceRow[]>([]);
  const [method, setMethod] = useState<FavorabilityMethod>("fuzzy");
  const [fuzzyAnd, setFuzzyAnd] = useState<FuzzyAnd>("min");
  const [missingPolicy, setMissingPolicy] = useState<MissingPolicy>("nodata");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const [result, setResult] = useState<FavorabilityResult | null>(null);
  // Cache the built fused grid so weight/curve re-runs reuse it (resample is idempotent).
  const [gridId, setGridId] = useState<string | null>(null);

  // Diagnostic-layer visibility toggles + the honest-view binding toggle.
  const [showOverlap, setShowOverlap] = useState(false);
  const [showBurden, setShowBurden] = useState(false);
  const [honestView, setHonestView] = useState(true);

  useEffect(() => {
    // Best-effort: warm the transform palette (doc 07 §6) so target hints could seed evidence
    // defaults later. Failure is non-fatal (offline); we do not block the panel on it.
    listTransforms().catch(() => {});
  }, []);

  const addEvidence = (layerId: string) => {
    const l = layers[layerId];
    if (!l || !l.volume) return;
    const mm = finiteMinMax(l.volume.data);
    const range: [number, number] = mm ? [mm.min, mm.max] : [0, 1];
    const target = l.property ?? l.meta?.property ?? "property";
    setEvidence((prev) => [
      ...prev,
      {
        layerId,
        source: l.datasetId,
        target,
        transferFn: defaultTransferFn("ramp", range[0], range[1]),
        weight: 1,
        role: "required",
        range,
        unit: l.meta?.canonicalUnit ?? null,
      },
    ]);
  };

  const patchEvidence = (i: number, p: Partial<EvidenceRow>) =>
    setEvidence((prev) => prev.map((e, j) => (j === i ? { ...e, ...p } : e)));
  const removeEvidence = (i: number) =>
    setEvidence((prev) => prev.filter((_, j) => j !== i));

  const notYetAdded = candidates.filter((c) => !evidence.some((e) => e.layerId === c!.id));

  // Load a favorability product PropertyModel as a (replacing) layer with a fixed id +
  // colormap; returns the decoded volume so the caller can also use it as confidence.
  const loadProductLayer = async (
    modelId: string,
    layerId: string,
    name: string,
    colormap: string,
    visible: boolean,
  ) => {
    const meta = await fetchMeta(modelId);
    const vol = await fetchVolume(modelId, 0, meta);
    addVolumeLayer({ ...meta, colormap }, vol, { id: layerId, name, select: false });
    if (!visible) useViewer.getState().setLayerVisible(layerId, false);
    return { meta, vol };
  };

  const compute = async () => {
    setBusy(true);
    setErr("");
    setNote("");
    try {
      const body = buildFavorabilitySpec({
        projectId,
        method,
        fuzzyAnd,
        missingPolicy,
        evidence: evidence.map((e) => ({
          source: e.source,
          target: e.target,
          transferFn: e.transferFn,
          weight: e.weight,
          role: e.role,
        })),
      });
      if (!projectId) throw new Error("no backend project id on the evidence layers");

      // 1) build (or reuse) a fused grid over the evidence sources + resample each in.
      let gid = gridId;
      const sourceIds = Array.from(new Set(evidence.map((e) => e.source)));
      if (!gid) {
        const grid = await createFused({
          project_id: projectId,
          name: "favorability",
          source_property_model_ids: sourceIds,
        });
        gid = grid.id;
        setGridId(gid);
      }
      for (const id of sourceIds) await resampleLayer(gid, id);

      // 2) compute favorability + honesty diagnostics.
      const res = await computeFavorability(gid, body);
      if (res.mode === "job") {
        setNote(`favorability queued as job ${res.job_id} (large grid)`);
        setResult(res);
        return;
      }
      setResult(res);

      // 3) render the favorability volume (inferno) + the three diagnostics as layers.
      if (res.model_id) {
        await loadProductLayer(res.model_id, FAVOR_LAYER_ID, "Favorability", "inferno", true);
        setLayerTF(FAVOR_LAYER_ID, { domainMin: 0, domainMax: 1, colormap: "inferno" });
      }
      if (res.confidence_model_id) {
        const r = await loadProductLayer(
          res.confidence_model_id,
          CONF_LAYER_ID,
          "Confidence (diagnostic)",
          "viridis",
          false,
        );
        // 4) bind confidence → favorability opacity (doc 07 §5.3 honest view).
        if (r.vol) {
          setLayerConfidence(FAVOR_LAYER_ID, {
            enabled: honestView,
            volume: r.vol,
            min: 0,
            max: 1,
            invert: false, // confidence: high = MORE confident (not σ)
            floor: 0.05,
            sourceId: res.confidence_model_id,
          });
        }
      }
      if (res.overlap_model_id) {
        await loadProductLayer(res.overlap_model_id, OVERLAP_LAYER_ID, "Evidence overlap (diagnostic)", "viridis", showOverlap);
      }
      if (res.burden_model_id) {
        await loadProductLayer(res.burden_model_id, BURDEN_LAYER_ID, "Assumption burden (diagnostic)", "inferno", showBurden);
      }

      setNote(
        `favorability computed · ${res.n_valid ?? 0} scored cells` +
          (res.n_missing_required ? ` · ${res.n_missing_required} missing-required` : ""),
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Toggle a diagnostic layer's visibility (it is loaded hidden by default).
  const toggleLayer = (id: string, show: boolean, setShow: (b: boolean) => void) => {
    setShow(show);
    if (useViewer.getState().layers[id]) useViewer.getState().setLayerVisible(id, show);
  };

  // Toggle the honest-view confidence binding on the favorability layer live (re-uses the
  // already-bound confidence volume; just flips `enabled`).
  const toggleHonest = (on: boolean) => {
    setHonestView(on);
    const cur = useViewer.getState().layers[FAVOR_LAYER_ID]?.confidence;
    if (cur) setLayerConfidence(FAVOR_LAYER_ID, { ...cur, enabled: on });
  };

  const canCompute = evidence.length > 0 && method !== "bayesian";

  return (
    <div style={panel}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontWeight: 600 }}>Favorability — R&amp;D instrument</span>
        <button style={btn} onClick={onClose}>
          close
        </button>
      </div>
      <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 6 }}>
        {candidates.length} candidate evidence layer{candidates.length === 1 ? "" : "s"}
        {projectId ? ` · project ${projectId}` : " · no backend project (load layers first)"}
      </div>

      {/* add evidence */}
      <div style={{ display: "flex", gap: 6, alignItems: "center", marginBottom: 8 }}>
        <span style={lbl}>add evidence</span>
        <select
          style={sel}
          value=""
          onChange={(e) => {
            if (e.target.value) addEvidence(e.target.value);
          }}
        >
          <option value="">+ layer…</option>
          {notYetAdded.map((c) => (
            <option key={c!.id} value={c!.id}>
              {c!.name}
            </option>
          ))}
        </select>
      </div>

      {/* per-evidence editors */}
      {evidence.map((e, i) => (
        <div
          key={e.layerId}
          style={{ border: "1px solid #313244", borderRadius: 5, padding: 8, marginBottom: 8 }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 12, fontWeight: 600 }}>
              {layers[e.layerId]?.name ?? e.target}
            </span>
            <button style={btn} onClick={() => removeEvidence(i)}>
              remove
            </button>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
            <label style={lbl}>
              role{" "}
              <select
                style={sel}
                value={e.role}
                onChange={(ev) => patchEvidence(i, { role: ev.target.value as EvidenceRow["role"] })}
              >
                <option value="required">required (AND)</option>
                <option value="supporting">supporting (OR)</option>
              </select>
            </label>
            <label style={lbl} title="weight (used by the weighted-linear method)">
              w {e.weight.toFixed(2)}
              <input
                type="range"
                min={0}
                max={3}
                step={0.05}
                value={e.weight}
                onChange={(ev) => patchEvidence(i, { weight: parseFloat(ev.target.value) })}
                style={{ width: 90, verticalAlign: "middle" }}
              />
            </label>
          </div>
          <MembershipCurveEditor
            tf={e.transferFn}
            unit={e.unit}
            range={e.range}
            onChange={(tf) => patchEvidence(i, { transferFn: tf })}
          />
        </div>
      ))}

      <hr style={hr} />

      {/* method + combination controls */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 6 }}>
        <label style={lbl}>
          method{" "}
          <select style={sel} value={method} onChange={(e) => setMethod(e.target.value as FavorabilityMethod)}>
            {FAVORABILITY_METHODS.map((m) => (
              <option key={m} value={m} disabled={m === "bayesian"}>
                {m === "fuzzy" ? "fuzzy (default)" : m === "weighted" ? "weighted (exploratory)" : "bayesian (deferred)"}
              </option>
            ))}
          </select>
        </label>
        {method === "fuzzy" && (
          <label style={lbl}>
            AND op{" "}
            <select style={sel} value={fuzzyAnd} onChange={(e) => setFuzzyAnd(e.target.value as FuzzyAnd)}>
              {FUZZY_AND_OPS.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
          </label>
        )}
        <label style={lbl}>
          missing{" "}
          <select style={sel} value={missingPolicy} onChange={(e) => setMissingPolicy(e.target.value as MissingPolicy)}>
            {MISSING_POLICIES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button style={primaryBtn} onClick={compute} disabled={busy || !canCompute}>
        {busy ? "computing…" : result ? "Re-compute favorability" : "Compute favorability"}
      </button>

      {note && <div style={{ fontSize: 11, opacity: 0.7, marginTop: 6 }}>{note}</div>}
      {err && <div style={{ fontSize: 11, color: "#f38ba8", marginTop: 6 }}>{err}</div>}

      {/* honesty controls — only once a result exists */}
      {result && result.mode === "sync" && (
        <>
          <hr style={hr} />
          <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 4 }}>Honesty diagnostics</div>
          <label style={{ display: "block", fontSize: 12, marginBottom: 3 }}>
            <input
              type="checkbox"
              checked={honestView}
              onChange={(e) => toggleHonest(e.target.checked)}
            />{" "}
            confidence-modulated opacity (honest view)
          </label>
          <label style={{ display: "block", fontSize: 12, marginBottom: 3 }}>
            <input
              type="checkbox"
              checked={showOverlap}
              onChange={(e) => toggleLayer(OVERLAP_LAYER_ID, e.target.checked, setShowOverlap)}
            />{" "}
            evidence-overlap layer
          </label>
          <label style={{ display: "block", fontSize: 12 }}>
            <input
              type="checkbox"
              checked={showBurden}
              onChange={(e) => toggleLayer(BURDEN_LAYER_ID, e.target.checked, setShowBurden)}
            />{" "}
            assumption-burden layer
          </label>
          <div style={{ fontSize: 10, opacity: 0.55, marginTop: 6 }}>
            {result.n_required ?? 0} required · {result.n_supporting ?? 0} supporting · method {result.method}
          </div>
        </>
      )}
    </div>
  );
}
