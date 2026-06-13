// Unit tests for the pure feature response→layer-data builders (doc 06 §5.4, §6). Pure — no
// THREE / no DOM — runnable headlessly via `npm test`. Covers: microseismic point-cloud
// packing (interleaved xyz + parsed epoch-ms, NaN for undated points) and the InSAR raster
// time-series build (frame stack + robust colour range + parsed frame epochs).

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  packPointCloud,
  buildRasterTimeSeries,
  type PointCloudResponse,
} from "./features";
import { parseEpochMs } from "./time";
import type { SurfaceGrid } from "./terrain";

test("packPointCloud interleaves xyz and parses epoch-ms per point", () => {
  const resp: PointCloudResponse = {
    featureId: "f1",
    count: 2,
    x: [1, 4],
    y: [2, 5],
    z: [3, 6],
    t: ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"],
    magnitude: [1.5, 2.5],
    depth_m: [100, 200],
    window: { t0: null, t1: null, bbox: null },
  };
  const c = packPointCloud(resp);
  assert.equal(c.count, 2);
  assert.deepEqual([...c.positions], [1, 2, 3, 4, 5, 6]);
  assert.deepEqual([...c.magnitude], [1.5, 2.5]);
  assert.equal(c.epochMs[0], parseEpochMs("2020-01-01T00:00:00Z"));
  assert.equal(c.epochMs[1], parseEpochMs("2020-01-02T00:00:00Z"));
  assert.ok(c.depth);
  assert.deepEqual([...c.depth!], [100, 200]);
});

test("packPointCloud yields NaN epoch-ms for an undated point (stays hidden)", () => {
  const resp: PointCloudResponse = {
    featureId: "f1",
    count: 1,
    x: [0],
    y: [0],
    z: [0],
    t: [], // no time supplied
    magnitude: [1],
    depth_m: [],
    window: { t0: null, t1: null, bbox: null },
  };
  const c = packPointCloud(resp);
  assert.ok(Number.isNaN(c.epochMs[0]));
  assert.equal(c.depth, undefined); // no depth array
});

function grid(nx: number, ny: number): SurfaceGrid {
  return {
    nx,
    ny,
    x0: 0,
    y0: 0,
    dx: 1,
    dy: 1,
    elevation: new Float32Array(nx * ny),
  };
}

test("buildRasterTimeSeries stacks frames, derives a robust range, parses epochs", () => {
  const g = grid(2, 2);
  const frames = [
    new Float32Array([0, 1, 2, 3]),
    new Float32Array([NaN, 5, -2, 4]),
  ];
  const epochs = ["2021-01-01T00:00:00Z", "2021-02-01T00:00:00Z"];
  const ts = buildRasterTimeSeries(g, frames, epochs);
  assert.equal(ts.frames.length, 2);
  assert.equal(ts.frameIndex, 0);
  // range = robust min/max over finite samples across all frames (NaN ignored)
  assert.deepEqual(ts.range, [-2, 5]);
  assert.equal(ts.epochMs[0], parseEpochMs("2021-01-01T00:00:00Z"));
  assert.equal(ts.epochMs[1], parseEpochMs("2021-02-01T00:00:00Z"));
});

test("buildRasterTimeSeries falls back to [0,1] for an all-NaN/empty stack", () => {
  const g = grid(1, 1);
  const ts = buildRasterTimeSeries(g, [new Float32Array([NaN])], ["2021-01-01T00:00:00Z"]);
  assert.deepEqual(ts.range, [0, 1]);
});
