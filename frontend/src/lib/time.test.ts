// Unit tests for the global 4-D time axis (doc 06 §9.4). Pure — no THREE / no DOM —
// runnable headlessly via `npm test` (esbuild-bundles each *.test.ts, then `node --test`).
// Covers: epoch union/dedupe/sort, time-window membership (instant / cumulative / rolling),
// nearest-epoch frame select, and the playhead↔fraction inverse.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildTimeAxis,
  parseEpochMs,
  resolveWindowMs,
  inWindowMs,
  nearestEpochIndex,
  playheadFraction,
  fractionToMs,
} from "./time";

const D = (iso: string) => parseEpochMs(iso);

test("buildTimeAxis unions, dedupes by instant, drops junk, and sorts ascending", () => {
  const axis = buildTimeAxis(
    ["2020-01-03T00:00:00Z", "2020-01-01T00:00:00Z"],
    ["2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z", "not-a-date"],
  );
  assert.deepEqual(axis.epochs, [
    "2020-01-01T00:00:00Z",
    "2020-01-02T00:00:00Z",
    "2020-01-03T00:00:00Z",
  ]);
  assert.equal(axis.epochMs.length, 3);
  // monotonic ascending
  assert.ok(axis.epochMs[0] < axis.epochMs[1] && axis.epochMs[1] < axis.epochMs[2]);
  assert.equal(axis.t0Ms, D("2020-01-01T00:00:00Z"));
  assert.equal(axis.t1Ms, D("2020-01-03T00:00:00Z"));
});

test("buildTimeAxis collapses equivalent ISO spellings of the same instant", () => {
  // Same instant, two spellings (Z vs +00:00) -> one axis epoch.
  const axis = buildTimeAxis(["2020-06-01T12:00:00Z", "2020-06-01T12:00:00+00:00"]);
  assert.equal(axis.epochs.length, 1);
});

test("empty axis has null bounds", () => {
  const axis = buildTimeAxis();
  assert.equal(axis.t0Ms, null);
  assert.equal(axis.t1Ms, null);
  assert.deepEqual(axis.epochs, []);
});

test("instant window contains only the playhead epoch", () => {
  const ph = D("2020-01-02T00:00:00Z");
  const w = resolveWindowMs("instant", ph, D("2020-01-01T00:00:00Z"), 0);
  assert.equal(w.t0Ms, ph);
  assert.equal(w.t1Ms, ph);
  assert.equal(inWindowMs(ph, w), true);
  assert.equal(inWindowMs(D("2020-01-01T00:00:00Z"), w), false);
  assert.equal(inWindowMs(D("2020-01-03T00:00:00Z"), w), false);
});

test("cumulative window spans axis start through the playhead (history accretes)", () => {
  const start = D("2020-01-01T00:00:00Z");
  const ph = D("2020-01-03T00:00:00Z");
  const w = resolveWindowMs("cumulative", ph, start, 0);
  assert.equal(w.t0Ms, start);
  assert.equal(w.t1Ms, ph);
  // everything up to and including the playhead is in
  assert.equal(inWindowMs(start, w), true);
  assert.equal(inWindowMs(D("2020-01-02T00:00:00Z"), w), true);
  assert.equal(inWindowMs(ph, w), true);
  // a future event (after the playhead) is out
  assert.equal(inWindowMs(D("2020-01-04T00:00:00Z"), w), false);
});

test("rolling window is a fixed trailing span ending at the playhead", () => {
  const ph = D("2020-01-10T00:00:00Z");
  const dayMs = 24 * 3600 * 1000;
  const w = resolveWindowMs("rolling", ph, D("2020-01-01T00:00:00Z"), 2 * dayMs);
  assert.equal(w.t1Ms, ph);
  assert.equal(w.t0Ms, ph - 2 * dayMs);
  // inside the trailing 2-day window
  assert.equal(inWindowMs(D("2020-01-09T00:00:00Z"), w), true);
  assert.equal(inWindowMs(ph, w), true);
  // older than the window start -> decayed out
  assert.equal(inWindowMs(D("2020-01-07T00:00:00Z"), w), false);
  // future -> out
  assert.equal(inWindowMs(D("2020-01-11T00:00:00Z"), w), false);
});

test("cumulative falls back to the playhead when the axis is empty", () => {
  const ph = D("2020-01-05T00:00:00Z");
  const w = resolveWindowMs("cumulative", ph, null, 0);
  assert.equal(w.t0Ms, ph);
  assert.equal(w.t1Ms, ph);
});

test("nearestEpochIndex snaps the playhead to the closest frame (ties -> earlier)", () => {
  const axis = buildTimeAxis([
    "2020-01-01T00:00:00Z",
    "2020-01-03T00:00:00Z",
    "2020-01-05T00:00:00Z",
  ]);
  // before start -> 0
  assert.equal(nearestEpochIndex(axis, D("2019-12-01T00:00:00Z")), 0);
  // after end -> last
  assert.equal(nearestEpochIndex(axis, D("2021-01-01T00:00:00Z")), 2);
  // closer to frame 1
  assert.equal(nearestEpochIndex(axis, D("2020-01-03T06:00:00Z")), 1);
  // exact midpoint between 0 and 1 -> earlier (0)
  assert.equal(nearestEpochIndex(axis, D("2020-01-02T00:00:00Z")), 0);
  // empty axis -> -1
  assert.equal(nearestEpochIndex(buildTimeAxis(), 0), -1);
});

test("playheadFraction and fractionToMs are inverse over the axis span", () => {
  const axis = buildTimeAxis([
    "2020-01-01T00:00:00Z",
    "2020-01-11T00:00:00Z",
  ]);
  const mid = D("2020-01-06T00:00:00Z");
  assert.ok(Math.abs(playheadFraction(axis, mid) - 0.5) < 1e-9);
  assert.equal(fractionToMs(axis, 0), axis.t0Ms);
  assert.equal(fractionToMs(axis, 1), axis.t1Ms);
  // clamps out of range
  assert.equal(playheadFraction(axis, D("2019-01-01T00:00:00Z")), 0);
  assert.equal(playheadFraction(axis, D("2099-01-01T00:00:00Z")), 1);
});
