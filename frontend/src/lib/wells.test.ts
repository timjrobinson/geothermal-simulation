// Unit tests for the well-tube + log-colour math (doc 06 §5.3). Pure — no THREE / no DOM —
// runnable headlessly via `npm test`. Covers: deviation-survey→tube vertex/MD math (arc
// length + station MD), log→vertex-colour resampling (curve resample onto stations, range,
// normalize, NaN gaps), and the MD/TVD/elevation hover readout.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  cumulativeArcLength,
  stationMD,
  sampleCurveAtMD,
  resampleCurveToStations,
  curveRange,
  normalize,
  curveToVertexColors,
  readoutAtPoint,
  type WellTrajectory,
  type Vec3,
} from "./wells";

test("cumulativeArcLength accumulates 3-D chord length per station", () => {
  const poly: Vec3[] = [
    [0, 0, 0],
    [3, 4, 0], // +5
    [3, 4, 12], // +12
  ];
  assert.deepEqual(cumulativeArcLength(poly), [0, 5, 17]);
});

test("stationMD prefers the backend MD, else derives arc length", () => {
  const poly: Vec3[] = [
    [0, 0, 0],
    [0, 0, -10],
    [0, 0, -20],
  ];
  // matching-length backend MD wins verbatim
  assert.deepEqual(stationMD({ polyline: poly, md: [0, 10, 20] }), [0, 10, 20]);
  // mismatched / empty MD -> arc length fallback (monotonic)
  const derived = stationMD({ polyline: poly, md: [] });
  assert.deepEqual(derived, [0, 10, 20]);
});

test("sampleCurveAtMD linearly interpolates within coverage, NaN outside", () => {
  const md = [100, 200, 300];
  const v = [10, 20, 40];
  assert.equal(sampleCurveAtMD(md, v, 100), 10); // exact start
  assert.equal(sampleCurveAtMD(md, v, 300), 40); // exact end
  assert.equal(sampleCurveAtMD(md, v, 150), 15); // half between 10..20
  assert.equal(sampleCurveAtMD(md, v, 250), 30); // half between 20..40
  assert.ok(Number.isNaN(sampleCurveAtMD(md, v, 50))); // below coverage
  assert.ok(Number.isNaN(sampleCurveAtMD(md, v, 400))); // above coverage
});

test("sampleCurveAtMD propagates a NaN log gap (no interpolation across nulls)", () => {
  const md = [0, 10, 20];
  const v = [1, NaN, 3];
  assert.ok(Number.isNaN(sampleCurveAtMD(md, v, 5))); // bracket touches the NaN sample
  assert.equal(sampleCurveAtMD(md, v, 20), 3); // clean sample still resolves
});

test("resampleCurveToStations maps a curve onto trajectory station MDs", () => {
  const logs = {
    wellId: "W1",
    md: [0, 100, 200],
    curves: { GR: [50, 100, 150] },
    primaryProperty: "GR",
  };
  const stations = [0, 50, 100, 250]; // last station beyond log coverage
  const out = resampleCurveToStations(logs, "GR", stations);
  assert.equal(out[0], 50);
  assert.equal(out[1], 75); // halfway 50..100
  assert.equal(out[2], 100);
  assert.ok(Number.isNaN(out[3])); // uncovered
  // unknown property -> all NaN
  assert.ok(resampleCurveToStations(logs, "NOPE", stations).every(Number.isNaN));
});

test("curveRange ignores NaN, guards a degenerate range", () => {
  assert.deepEqual(curveRange([3, 1, NaN, 5, 2]), { min: 1, max: 5 });
  // all NaN -> [0,1] fallback
  assert.deepEqual(curveRange([NaN, NaN]), { min: 0, max: 1 });
  // constant -> non-degenerate
  const r = curveRange([7, 7, 7]);
  assert.ok(r.max > r.min);
});

test("normalize clamps to [0,1] and keeps NaN", () => {
  const r = { min: 0, max: 10 };
  assert.equal(normalize(5, r), 0.5);
  assert.equal(normalize(-5, r), 0); // clamp low
  assert.equal(normalize(50, r), 1); // clamp high
  assert.ok(Number.isNaN(normalize(NaN, r)));
});

test("curveToVertexColors emits 3 floats/vertex and a neutral colour for NaN gaps", () => {
  const range = { min: 0, max: 1 };
  const neutral: Vec3 = [0.1, 0.2, 0.3];
  const colors = curveToVertexColors([0, NaN, 1], range, "gray", neutral);
  assert.equal(colors.length, 9);
  // gray ramp at t=0 -> black, t=1 -> white
  assert.ok(colors[0] < 0.01 && colors[2] < 0.01);
  assert.ok(colors[6] > 0.99 && colors[8] > 0.99);
  // the NaN vertex got the neutral colour (Float32 stored -> compare within tolerance)
  assert.ok(Math.abs(colors[3] - neutral[0]) < 1e-6);
  assert.ok(Math.abs(colors[4] - neutral[1]) < 1e-6);
  assert.ok(Math.abs(colors[5] - neutral[2]) < 1e-6);
});

test("readoutAtPoint reports the nearest station's true MD/TVD/elevation", () => {
  const traj: Pick<WellTrajectory, "polyline" | "md" | "tvd"> = {
    polyline: [
      [0, 0, 1000],
      [0, 0, 900],
      [10, 0, 800],
    ],
    md: [0, 100, 200],
    tvd: [0, 100, 200],
  };
  const r = readoutAtPoint(traj, [9, 1, 805]);
  assert.ok(r);
  assert.equal(r!.stationIndex, 2);
  assert.equal(r!.md, 200);
  assert.equal(r!.tvd, 200);
  assert.equal(r!.elevation, 800); // true Z, NOT exaggerated
});
