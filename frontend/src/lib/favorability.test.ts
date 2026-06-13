// Unit tests for the favorability PURE logic (doc 07 §4.6): the membership-curve evaluator
// (raw → [0,1]) and the FavorabilitySpec builder (the wire payload). Pure functions only —
// no THREE / no DOM — runnable headlessly via `npm test` (esbuild-bundles each *.test.ts,
// then `node --test`). The membership math is checked against the SAME cases the backend
// geosim.fusion.favorability.membership uses so the curve preview matches the result volume.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  membership,
  sampleMembershipCurve,
  defaultTransferFn,
  buildFavorabilitySpec,
  TRANSFER_TYPES,
  FAVORABILITY_METHODS,
  type TransferFnSpec,
  type FavorabilityFormState,
  type EvidenceSpec,
} from "./favorability";

function approx(a: number, b: number, eps = 1e-9): void {
  assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b}`);
}

// ── membership: ramp ─────────────────────────────────────────────────────────────────────
test("ramp membership maps lo→0, hi→1, midpoint→0.5 and clamps", () => {
  const tf: TransferFnSpec = { type: "ramp", lo: 300, hi: 500 };
  approx(membership(300, tf), 0);
  approx(membership(500, tf), 1);
  approx(membership(400, tf), 0.5);
  approx(membership(250, tf), 0); // below lo clamps to 0
  approx(membership(600, tf), 1); // above hi clamps to 1
});

test("descending ramp (hi<lo) makes low values favorable", () => {
  const tf: TransferFnSpec = { type: "ramp", lo: 500, hi: 300 };
  approx(membership(300, tf), 1);
  approx(membership(500, tf), 0);
  approx(membership(400, tf), 0.5);
});

test("ramp with missing/equal bounds throws", () => {
  assert.throws(() => membership(1, { type: "ramp", lo: 0 }));
  assert.throws(() => membership(1, { type: "ramp", lo: 5, hi: 5 }));
});

// ── membership: sigmoid ──────────────────────────────────────────────────────────────────
test("sigmoid is 0.5 at center, monotone increasing for k>0", () => {
  const tf: TransferFnSpec = { type: "sigmoid", center: 400, k: 0.1 };
  approx(membership(400, tf), 0.5);
  assert.ok(membership(450, tf) > 0.5);
  assert.ok(membership(350, tf) < 0.5);
});

test("sigmoid with k<0 descends", () => {
  const tf: TransferFnSpec = { type: "sigmoid", center: 400, k: -0.1 };
  assert.ok(membership(450, tf) < 0.5);
});

// ── membership: gaussian-band ────────────────────────────────────────────────────────────
test("gaussian-band peaks (==1) at center and falls off symmetrically", () => {
  const tf: TransferFnSpec = { type: "gaussian-band", center: 100, width: 20 };
  approx(membership(100, tf), 1);
  approx(membership(80, tf), membership(120, tf)); // symmetric
  assert.ok(membership(80, tf) < 1);
  approx(membership(120, tf), Math.exp(-0.5)); // one width out ⇒ e^-0.5
});

test("gaussian-band requires positive width", () => {
  assert.throws(() => membership(1, { type: "gaussian-band", center: 0, width: 0 }));
});

// ── membership: NaN coverage passthrough (doc 07 §2.3) ───────────────────────────────────
test("non-finite input stays NaN — never invents evidence", () => {
  assert.ok(Number.isNaN(membership(NaN, { type: "ramp", lo: 0, hi: 1 })));
  assert.ok(Number.isNaN(membership(Infinity, { type: "sigmoid", center: 0, k: 1 })));
});

// ── curve sampler + defaults ─────────────────────────────────────────────────────────────
test("sampleMembershipCurve returns n monotone-x in-range points", () => {
  const pts = sampleMembershipCurve({ type: "ramp", lo: 0, hi: 10 }, 0, 10, 11);
  assert.equal(pts.length, 11);
  assert.equal(pts[0][0], 0);
  assert.equal(pts[10][0], 10);
  for (const [, m] of pts) assert.ok(m >= 0 && m <= 1);
  approx(pts[5][1], 0.5);
});

test("sampleMembershipCurve returns [] for invalid range or incomplete curve", () => {
  assert.deepEqual(sampleMembershipCurve({ type: "ramp", lo: 0, hi: 10 }, 5, 5), []);
  assert.deepEqual(sampleMembershipCurve({ type: "ramp", lo: 0 }, 0, 10), []);
});

test("defaultTransferFn seeds valid params for every transfer type", () => {
  for (const t of TRANSFER_TYPES) {
    const tf = defaultTransferFn(t, 300, 500);
    assert.equal(tf.type, t);
    // The seed must be a curve membership() accepts.
    assert.ok(Number.isFinite(membership(400, tf)));
  }
});

// ── buildFavorabilitySpec: happy path ────────────────────────────────────────────────────
function ev(over: Partial<EvidenceSpec> = {}): EvidenceSpec {
  return {
    source: "pm-temp",
    target: "temperature",
    transferFn: { type: "ramp", lo: 300, hi: 500 },
    weight: 1,
    role: "required",
    ...over,
  };
}

function form(over: Partial<FavorabilityFormState> = {}): FavorabilityFormState {
  return {
    projectId: "proj-1",
    method: "fuzzy",
    fuzzyAnd: "min",
    missingPolicy: "nodata",
    evidence: [ev()],
    ...over,
  };
}

test("buildFavorabilitySpec emits the exact wire body with camelCase transferFn", () => {
  const body = buildFavorabilitySpec(form());
  assert.equal(body.project_id, "proj-1");
  assert.equal(body.method, "fuzzy");
  assert.equal(body.fuzzy_and, "min");
  assert.equal(body.missing_policy, "nodata");
  assert.equal(body.evidence.length, 1);
  const e = body.evidence[0];
  assert.equal(e.source, "pm-temp");
  assert.equal(e.target, "temperature");
  assert.equal(e.role, "required");
  assert.deepEqual(e.transferFn, { type: "ramp", lo: 300, hi: 500 });
});

test("buildFavorabilitySpec normalizes each transfer type to only its own params", () => {
  const sig = buildFavorabilitySpec(
    form({ evidence: [ev({ transferFn: { type: "sigmoid", center: 400, k: 0.1, lo: 9, hi: 9 } })] }),
  );
  assert.deepEqual(sig.evidence[0].transferFn, { type: "sigmoid", center: 400, k: 0.1 });
  const band = buildFavorabilitySpec(
    form({ evidence: [ev({ transferFn: { type: "gaussian-band", center: 5, width: 2 } })] }),
  );
  assert.deepEqual(band.evidence[0].transferFn, { type: "gaussian-band", center: 5, width: 2 });
});

test("buildFavorabilitySpec passes force_job only when set", () => {
  assert.equal(buildFavorabilitySpec(form()).force_job, undefined);
  assert.equal(buildFavorabilitySpec(form({ forceJob: true })).force_job, true);
});

// ── buildFavorabilitySpec: validation guards (mirror the backend __post_init__) ──────────
test("rejects empty evidence", () => {
  assert.throws(() => buildFavorabilitySpec(form({ evidence: [] })), /at least one evidence/);
});

test("rejects the deferred bayesian method", () => {
  assert.throws(() => buildFavorabilitySpec(form({ method: "bayesian" })), /deferred/);
});

test("rejects a negative weight", () => {
  assert.throws(
    () => buildFavorabilitySpec(form({ evidence: [ev({ weight: -1 })] })),
    /weight must be >= 0/,
  );
});

test("rejects an evidence with no source / target", () => {
  assert.throws(() => buildFavorabilitySpec(form({ evidence: [ev({ source: "" })] })), /source/);
  assert.throws(() => buildFavorabilitySpec(form({ evidence: [ev({ target: "" })] })), /target/);
});

test("rejects a ramp evidence with lo==hi", () => {
  assert.throws(
    () => buildFavorabilitySpec(form({ evidence: [ev({ transferFn: { type: "ramp", lo: 5, hi: 5 } })] })),
    /lo != hi/,
  );
});

test("rejects a gaussian-band with non-positive width", () => {
  assert.throws(
    () =>
      buildFavorabilitySpec(
        form({ evidence: [ev({ transferFn: { type: "gaussian-band", center: 0, width: 0 } })] }),
      ),
    /width > 0/,
  );
});

test("fuzzy is the default headline method", () => {
  assert.equal(FAVORABILITY_METHODS[0], "fuzzy");
});
