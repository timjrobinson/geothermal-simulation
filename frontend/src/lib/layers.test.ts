// Unit tests for the multi-layer store logic (doc 06 §9.1, §10.1) and transfer-fn seeding
// (doc 06 §3.2). Pure functions only — no THREE render / no DOM — runnable headlessly via
//   npm test    (esbuild-bundles each *.test.ts, then `node --test`).
// Covers: layer add/remove/reorder/move/patch, the order reindex invariant, the blend
// enum, tfFromMeta registry seeding, and the isolate-band LUT bake.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  addLayer,
  removeLayer,
  moveLayer,
  reorderLayer,
  patchLayer,
  patchLayerTF,
  makeVolumeLayer,
  tfFromMeta,
  isBlendMode,
  BLEND_MODES,
  type LayerCollection,
  type Layer,
} from "./layers";
import { bakeTransferFnRGBA } from "./transferFn";
import type { PropertyModelMeta } from "./api";
import type { DecodedVolume } from "./volume";

function emptyCollection(): LayerCollection {
  return { layers: {}, layerOrder: [] };
}

function fakeMeta(over: Partial<PropertyModelMeta> = {}): PropertyModelMeta {
  return {
    id: "pm1",
    property: "resistivity",
    canonicalUnit: "ohm.m",
    scaling: "linear",
    colormap: "turbo",
    displayRange: [1, 3],
    shape: [2, 3, 4],
    origin: [0, 0, 0],
    spacing: [1, 1, 1],
    levels: 1,
    stats: { min: 0, max: 5, p1: 0.5, p99: 4.5 },
    frame: null,
    hasSigma: false,
    ...over,
  };
}

function fakeVolume(): DecodedVolume {
  return {
    shape: [2, 3, 4],
    origin: [0, 0, 0],
    spacing: [1, 1, 1],
    data: new Float32Array(2 * 3 * 4),
  };
}

function layer(id: string): Layer {
  return makeVolumeLayer(fakeMeta(), fakeVolume(), { id, name: id });
}

test("addLayer appends and sets order from index", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  c = addLayer(c, layer("b"));
  c = addLayer(c, layer("c"));
  assert.deepEqual(c.layerOrder, ["a", "b", "c"]);
  assert.equal(c.layers["a"].order, 0);
  assert.equal(c.layers["b"].order, 1);
  assert.equal(c.layers["c"].order, 2);
});

test("addLayer with an existing id replaces in place (no duplicate)", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  c = addLayer(c, layer("b"));
  const replacement = { ...layer("a"), name: "renamed" };
  c = addLayer(c, replacement);
  assert.deepEqual(c.layerOrder, ["a", "b"]);
  assert.equal(c.layers["a"].name, "renamed");
});

test("removeLayer drops the layer and reindexes order", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  c = addLayer(c, layer("b"));
  c = addLayer(c, layer("c"));
  c = removeLayer(c, "b");
  assert.deepEqual(c.layerOrder, ["a", "c"]);
  assert.equal(c.layers["c"].order, 1);
  assert.equal(c.layers["b"], undefined);
});

test("removeLayer is a no-op for an unknown id", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  const before = c;
  c = removeLayer(c, "zzz");
  assert.equal(c, before);
});

test("moveLayer up/down swaps neighbours and clamps at the ends", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  c = addLayer(c, layer("b"));
  c = addLayer(c, layer("c"));
  c = moveLayer(c, "a", "up"); // a <-> b
  assert.deepEqual(c.layerOrder, ["b", "a", "c"]);
  c = moveLayer(c, "b", "down"); // already at bottom -> no-op
  assert.deepEqual(c.layerOrder, ["b", "a", "c"]);
  c = moveLayer(c, "c", "up"); // already at top -> no-op
  assert.deepEqual(c.layerOrder, ["b", "a", "c"]);
  assert.equal(c.layers["a"].order, 1);
});

test("reorderLayer moves to an explicit index and clamps", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  c = addLayer(c, layer("b"));
  c = addLayer(c, layer("c"));
  c = reorderLayer(c, "c", 0);
  assert.deepEqual(c.layerOrder, ["c", "a", "b"]);
  c = reorderLayer(c, "c", 99); // clamps to end
  assert.deepEqual(c.layerOrder, ["a", "b", "c"]);
});

test("patchLayer and patchLayerTF update immutably", () => {
  let c = emptyCollection();
  c = addLayer(c, layer("a"));
  c = patchLayer(c, "a", { visible: false, opacity: 0.5 });
  assert.equal(c.layers["a"].visible, false);
  assert.equal(c.layers["a"].opacity, 0.5);
  c = patchLayerTF(c, "a", { colormap: "inferno", invert: true });
  assert.equal(c.layers["a"].transferFn.colormap, "inferno");
  assert.equal(c.layers["a"].transferFn.invert, true);
  // unrelated tf fields preserved
  assert.equal(typeof c.layers["a"].transferFn.domainMin, "number");
});

test("blend enum guard accepts the four modes, rejects others", () => {
  assert.deepEqual([...BLEND_MODES], ["over", "additive", "mip", "minip"]);
  for (const m of BLEND_MODES) assert.equal(isBlendMode(m), true);
  assert.equal(isBlendMode("over"), true);
  assert.equal(isBlendMode("screen"), false);
  assert.equal(isBlendMode(""), false);
});

test("makeVolumeLayer seeds a volume layer with registry defaults", () => {
  const l = makeVolumeLayer(fakeMeta(), fakeVolume(), { id: "x" });
  assert.equal(l.kind, "volume");
  assert.equal(l.blend, "over");
  assert.equal(l.visible, true);
  assert.equal(l.clip, true);
  assert.equal(l.property, "resistivity");
  // tf seeded from displayRange + registry colormap
  assert.equal(l.transferFn.colormap, "turbo");
  assert.deepEqual([l.transferFn.domainMin, l.transferFn.domainMax], [1, 3]);
  // aabb computed
  assert.ok(l.aabb);
});

test("tfFromMeta: displayRange wins, then p1/p99, then min/max", () => {
  assert.deepEqual(
    [tfFromMeta(fakeMeta()).domainMin, tfFromMeta(fakeMeta()).domainMax],
    [1, 3],
  );
  const noRange = fakeMeta({ displayRange: null });
  assert.deepEqual(
    [tfFromMeta(noRange).domainMin, tfFromMeta(noRange).domainMax],
    [0.5, 4.5],
  );
  const onlyMinMax = fakeMeta({
    displayRange: null,
    stats: { min: 10, max: 20, p1: null, p99: null },
  });
  assert.deepEqual(
    [tfFromMeta(onlyMinMax).domainMin, tfFromMeta(onlyMinMax).domainMax],
    [10, 20],
  );
});

test("tfFromMeta honours log scaling and degenerate-range guard", () => {
  const logMeta = fakeMeta({ scaling: "log" });
  assert.equal(tfFromMeta(logMeta).scaling, "log");
  const degenerate = fakeMeta({ displayRange: [5, 5] });
  const tf = tfFromMeta(degenerate);
  assert.ok(tf.domainMax > tf.domainMin); // hi bumped to lo+1
});

test("isolate band zeroes alpha outside [bandMin,bandMax]", () => {
  const base = {
    colormap: "gray",
    domainMin: 0,
    domainMax: 1,
    scaling: "linear" as const,
    opacity: 1,
    invert: false,
  };
  const noBand = bakeTransferFnRGBA(base);
  // alpha rises across the LUT without a band
  assert.ok(noBand[255 * 4 + 3] > 200);

  const banded = bakeTransferFnRGBA({
    ...base,
    bandEnabled: true,
    bandMin: 0.4,
    bandMax: 0.6,
  });
  // outside the band (t=0 and t=1) -> transparent
  assert.equal(banded[0 * 4 + 3], 0);
  assert.equal(banded[255 * 4 + 3], 0);
  // inside the band (t~0.5, index 128) -> opaque
  assert.ok(banded[128 * 4 + 3] > 0);
});
