// Unit tests for the well-planning data shaping (doc 09 §4, §5, §7, §8). Pure — no THREE /
// no DOM / no fetch — runnable headlessly via `npm test`. Covers: design-param→intent payload
// (the solver POST body), DLS per-segment colour thresholds (red over the ceiling), predicted
// -log→2D track shaping (uncertainty bands, risk, lithology fill), the glass-box risk-driver
// breakdown, and the alternative-scenario comparison (best-in-column).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  defaultDesignParams,
  designToIntent,
  wellCreateBody,
  solveBody,
  predictBody,
  defaultRiskWeights,
  positionsToTrajectory,
  dlsSegmentExceeded,
  dlsSegmentColors,
  dlsRingColors,
  DLS_OK_COLOR,
  DLS_OVER_COLOR,
  predictedTrack,
  riskTrack,
  lithologyIntervals,
  predictedLogToLogs,
  riskDriverBreakdown,
  scenarioRow,
  bestInColumn,
  trackToSvg,
  type DesignParams,
  type PredictedLog,
  type WellPositions,
} from "./planning";

// ── design-param → intent payload ────────────────────────────────────────────────────────

test("designToIntent emits snake_case keys and carries the target for a deviated well", () => {
  const p: DesignParams = {
    ...defaultDesignParams(),
    method: "build-hold-land",
    target: [100, 200, -1500],
    kopMD_m: 600,
    buildRate_deg30m: 4,
    landingInc_deg: 88,
  };
  const i = designToIntent(p);
  assert.equal(i.method, "build-hold-land");
  assert.deepEqual(i.target, [100, 200, -1500]);
  assert.equal(i.kop_md_m, 600);
  assert.equal(i.build_rate_deg30m, 4);
  assert.equal(i.landing_inc_deg, 88);
  // drop_rate defaults to build_rate when unset.
  assert.equal(i.drop_rate_deg30m, p.dropRate_deg30m ?? p.buildRate_deg30m);
});

test("designToIntent drops the target + landing for a vertical well", () => {
  const p: DesignParams = {
    ...defaultDesignParams(),
    method: "vertical",
    target: [1, 2, 3],
    landingInc_deg: 90,
  };
  const i = designToIntent(p);
  assert.equal(i.target, undefined);
  assert.equal(i.landing_inc_deg, undefined);
  assert.equal(i.method, "vertical");
});

test("designToIntent drops null hold/landing optionals", () => {
  const p: DesignParams = {
    ...defaultDesignParams(),
    method: "build-hold-land",
    target: [0, 0, -100],
    holdInc_deg: null,
    landingInc_deg: null,
  };
  const i = designToIntent(p);
  assert.equal(i.hold_inc_deg, undefined);
  assert.equal(i.landing_inc_deg, undefined);
});

test("wellCreateBody assembles the full POST body and only adds target_ids when present", () => {
  const p = defaultDesignParams();
  p.target = [10, 20, -900];
  const body = wellCreateBody("W-01", [0, 0], p, { kbElev_m: 5, targetIds: ["t1"] });
  assert.equal(body.name, "W-01");
  assert.deepEqual(body.wellhead, [0, 0]);
  assert.equal(body.kb_elev_m, 5);
  assert.deepEqual(body.target_ids, ["t1"]);
  assert.equal(body.max_dls_deg30m, p.maxDLS_deg30m);
  assert.equal(body.max_inc_deg, p.maxInc_deg);
  // no target_ids key when none supplied
  const body2 = wellCreateBody("W-02", [0, 0], p);
  assert.equal("target_ids" in body2, false);
});

test("solveBody carries the design + constraints", () => {
  const p = defaultDesignParams();
  p.maxDLS_deg30m = 6.5;
  const b = solveBody(p);
  assert.equal(b.max_dls_deg30m, 6.5);
  assert.equal(b.design.method, p.method);
});

test("predictBody defaults md step + thresholds, includes target/weights only when given", () => {
  const b1 = predictBody("fem_1");
  assert.equal(b1.fused_model_id, "fem_1");
  assert.equal(b1.md_step_m, 5);
  assert.equal(b1.favorability_threshold, 0.7);
  assert.equal("target_id" in b1, false);
  assert.equal("risk_weights" in b1, false);
  const b2 = predictBody("fem_1", {
    mdStep_m: 2,
    targetId: "t1",
    riskWeights: defaultRiskWeights(),
  });
  assert.equal(b2.md_step_m, 2);
  assert.equal(b2.target_id, "t1");
  assert.deepEqual(b2.risk_weights, defaultRiskWeights());
});

test("defaultRiskWeights sum to 1 (the glass-box composite)", () => {
  const w = defaultRiskWeights();
  const sum =
    w.tempConfidence + w.hazard + w.dlsExceedance + w.structuralUncertainty;
  assert.ok(Math.abs(sum - 1) < 1e-9);
});

// ── positions → trajectory ───────────────────────────────────────────────────────────────

test("positionsToTrajectory builds an Engineering polyline + MD/TVD/DLS", () => {
  const pos: WellPositions = {
    wellId: "w1",
    md: [0, 30, 60],
    tvd: [0, 30, 58],
    enu: [
      [0, 0, 0],
      [0, 0, -30],
      [5, 0, -58],
    ],
    dls: [0, 0, 3],
  };
  const traj = positionsToTrajectory(pos, { featureId: "f1" });
  assert.equal(traj.polyline.length, 3);
  assert.deepEqual(traj.polyline[2], [5, 0, -58]);
  assert.deepEqual(traj.md, [0, 30, 60]);
  assert.deepEqual(traj.dls, [0, 0, 3]);
  assert.deepEqual(traj.wellhead, [0, 0]);
});

// ── DLS per-segment colour thresholds ────────────────────────────────────────────────────

test("dlsSegmentExceeded flags a segment whose terminating-station dogleg is over the ceiling", () => {
  // stations: dls per station (dogleg to reach it). 4 stations -> 3 segments.
  const dls = [0, 2, 6, 3];
  // segment 0->1 ends at dls[1]=2 (ok), 1->2 ends at dls[2]=6 (>5 over), 2->3 ends at dls[3]=3 (ok)
  assert.deepEqual(dlsSegmentExceeded(dls, 5), [false, true, false]);
});

test("dlsSegmentExceeded is empty for a single station", () => {
  assert.deepEqual(dlsSegmentExceeded([0], 5), []);
});

test("dlsSegmentColors paints over-ceiling segments red and others neutral", () => {
  const dls = [0, 1, 7];
  const colors = dlsSegmentColors(dls, 5);
  assert.equal(colors.length, 2);
  assert.deepEqual(colors[0], DLS_OK_COLOR); // 0->1 (dls 1) ok
  assert.deepEqual(colors[1], DLS_OVER_COLOR); // 1->2 (dls 7) over
});

test("dlsRingColors expands per-segment colours to per-ring vertices", () => {
  // one over-ceiling segment at the bottom -> later rings turn red.
  const dls = [0, 1, 1, 8]; // 3 segments; last (->dls 8) is over
  const ringCount = 9;
  const colors = dlsRingColors(dls, 5, ringCount);
  assert.equal(colors.length, ringCount * 3);
  // first ring is OK colour
  assert.deepEqual(
    [colors[0], colors[1], colors[2]],
    DLS_OK_COLOR.map((c) => Math.fround(c)),
  );
  // last ring maps into the final (over) segment -> red
  const last = (ringCount - 1) * 3;
  assert.deepEqual(
    [colors[last], colors[last + 1], colors[last + 2]],
    DLS_OVER_COLOR.map((c) => Math.fround(c)),
  );
});

test("dlsRingColors falls back to OK colour when there are no segments", () => {
  const colors = dlsRingColors([3], 5, 4);
  assert.equal(colors.length, 12);
  assert.deepEqual(
    [colors[0], colors[1], colors[2]],
    DLS_OK_COLOR.map((c) => Math.fround(c)),
  );
});

// ── predicted-log → 2D tracks ────────────────────────────────────────────────────────────

function mkLog(): PredictedLog {
  return {
    wellId: "w1",
    modelVersion: "fused_v1",
    mdStep_m: 5,
    stations: [
      {
        md: 100, tvd: 100, z: 0, x: 0, y: 0,
        values: {
          temperatureC: { value: 120, sigma: 10, confidence: 0.7 },
          favorability: { value: 0.4, sigma: 0.05 },
        },
        lithology: "granite", hazards: {}, distToNearestFault_m: 500,
        risk: 0.2, riskDrivers: { tempConfidence: 0.1, hazard: 0.1 },
      },
      {
        md: 105, tvd: 105, z: -5, x: 1, y: 0,
        values: {
          temperatureC: { value: 130, sigma: 12, confidence: 0.6 },
          favorability: { value: 0.6, sigma: 0.05 },
        },
        lithology: "granite", hazards: {}, distToNearestFault_m: 480,
        risk: 0.5, riskDrivers: { tempConfidence: 0.2, hazard: 0.3 },
      },
      {
        md: 110, tvd: 110, z: -10, x: 2, y: 0,
        values: {
          temperatureC: { value: null, sigma: null }, // a gap
          favorability: { value: 0.8, sigma: 0.05 },
        },
        lithology: "basalt", hazards: {}, distToNearestFault_m: 100,
        risk: 0.3, riskDrivers: { tempConfidence: 0.1, hazard: 0.2 },
      },
    ],
    summary: {
      bhtC: 130, bhtSigmaC: 12, bhtConfidence: 0.6,
      maxTempC: 130, maxTempMD_m: 105, maxTempTVD_m: 105,
      targetIntersectionLength_m: 250, reservoirIntersectionLength_m: 300,
      productiveFractureIntersections: 4, fractureIntersectionMDs_m: [102, 108],
      inWindowFraction: 0.8, meanRisk: 0.33, peakRisk: 0.5,
    },
    riskWeights: defaultRiskWeights(),
  };
}

test("predictedTrack builds a band track skipping null-value stations", () => {
  const log = mkLog();
  const t = predictedTrack(log, "temperatureC", "degC");
  // only the two finite temperature stations
  assert.deepEqual(t.md, [100, 105]);
  assert.deepEqual(t.value, [120, 130]);
  assert.equal(t.hasBand, true);
  assert.deepEqual(t.lower, [110, 118]);
  assert.deepEqual(t.upper, [130, 142]);
  // domain spans the band
  assert.equal(t.min, 110);
  assert.equal(t.max, 142);
});

test("predictedTrack has no band when no sigma present", () => {
  const log = mkLog();
  // favorability has sigma; build a property with no sigma
  log.stations.forEach((s) => (s.values.poro = { value: 0.1 }));
  const t = predictedTrack(log, "poro");
  assert.equal(t.hasBand, false);
  assert.equal(t.lower, undefined);
});

test("riskTrack fixes the [0,1] domain", () => {
  const t = riskTrack(mkLog());
  assert.equal(t.min, 0);
  assert.equal(t.max, 1);
  assert.deepEqual(t.value, [0.2, 0.5, 0.3]);
});

test("lithologyIntervals merges contiguous classes and assigns stable colours", () => {
  const ivals = lithologyIntervals(mkLog());
  assert.equal(ivals.length, 2);
  assert.equal(ivals[0].lithology, "granite");
  assert.equal(ivals[0].mdTop, 100);
  assert.equal(ivals[0].mdBottom, 105);
  assert.equal(ivals[1].lithology, "basalt");
  assert.equal(ivals[1].mdTop, 110);
  // distinct classes get distinct colours
  assert.notDeepEqual(ivals[0].color, ivals[1].color);
});

test("predictedLogToLogs yields one curve per property + a risk curve, NaN gaps", () => {
  const logs = predictedLogToLogs(mkLog());
  assert.deepEqual(logs.md, [100, 105, 110]);
  assert.ok(logs.curves.temperatureC);
  assert.ok(Number.isNaN(logs.curves.temperatureC[2])); // the null gap
  assert.ok(logs.curves.favorability);
  assert.deepEqual(logs.curves.risk, [0.2, 0.5, 0.3]);
  assert.equal(logs.primaryProperty, "temperatureC");
});

// ── glass-box risk driver breakdown ──────────────────────────────────────────────────────

test("riskDriverBreakdown averages contributions and sorts descending", () => {
  const b = riskDriverBreakdown(mkLog());
  // hazard means: (0.1+0.3+0.2)/3 = 0.2 ; tempConfidence: (0.1+0.2+0.1)/3 ≈ 0.133
  assert.equal(b[0].name, "hazard");
  assert.ok(Math.abs(b[0].contribution - 0.2) < 1e-9);
  assert.equal(b[1].name, "tempConfidence");
  // fractions sum to ~1
  const fracSum = b.reduce((a, d) => a + d.fraction, 0);
  assert.ok(Math.abs(fracSum - 1) < 1e-9);
});

// ── alternative-scenario comparison ──────────────────────────────────────────────────────

test("scenarioRow extracts the comparison metrics from the summary", () => {
  const row = scenarioRow("w1", "vertical", mkLog(), { maxDLS_deg30m: 4.2 });
  assert.equal(row.bhtC, 130);
  assert.equal(row.payLength_m, 250);
  assert.equal(row.reservoirLength_m, 300);
  assert.equal(row.fractureIntersections, 4);
  assert.equal(row.meanRisk, 0.33);
  assert.equal(row.maxDLS_deg30m, 4.2);
  assert.equal(row.totalMD_m, 110); // last station MD
});

test("bestInColumn picks max for pay/BHT and min for risk/DLS", () => {
  const a = scenarioRow("w1", "A", mkLog(), { maxDLS_deg30m: 6 });
  const log2 = mkLog();
  log2.summary.bhtC = 150;
  log2.summary.targetIntersectionLength_m = 100;
  log2.summary.meanRisk = 0.1;
  const b = scenarioRow("w2", "B", log2, { maxDLS_deg30m: 3 });
  const best = bestInColumn([a, b]);
  assert.equal(best.bhtC, 1); // B hotter
  assert.equal(best.payLength_m, 0); // A more pay
  assert.equal(best.meanRisk, 1); // B lower risk
  assert.equal(best.maxDLS_deg30m, 1); // B lower DLS
});

test("bestInColumn ignores null metrics", () => {
  const a = scenarioRow("w1", "A", mkLog());
  const log2 = mkLog();
  log2.summary.bhtC = null;
  const b = scenarioRow("w2", "B", log2);
  const best = bestInColumn([a, b]);
  assert.equal(best.bhtC, 0); // A is the only finite BHT
});

// ── 2D track → SVG ───────────────────────────────────────────────────────────────────────

test("trackToSvg maps md (down) + value (across) into the box and builds a band polygon", () => {
  const log = mkLog();
  const t = predictedTrack(log, "temperatureC");
  const { line, band } = trackToSvg(t, 100, 200, 100, 110);
  // two points
  assert.equal(line.split(" ").length, 2);
  // a band polygon exists (sigma present) with up+down points
  assert.ok(band);
  assert.equal(band!.split(" ").length, 4);
});

test("trackToSvg has no band for a band-less track", () => {
  const log = mkLog();
  log.stations.forEach((s) => (s.values.poro = { value: 0.1 }));
  const t = predictedTrack(log, "poro");
  const { band } = trackToSvg(t, 100, 200, 100, 110);
  assert.equal(band, null);
});
