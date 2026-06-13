// Unit tests for the dataset-discovery normalizer (doc 06 §9.1 datasets→layers). Pure —
// no fetch / no DOM. Covers the tolerant artifact-catalog shapes the "open project" flow
// must accept (array, {artifacts}, {property_models}, snake/camel field names) and the
// filtering of non-property-model rows.

import { test } from "node:test";
import assert from "node:assert/strict";
import { normalizeArtifacts } from "./api";

test("normalizeArtifacts accepts a bare array of rows", () => {
  const out = normalizeArtifacts([
    { id: "a", property: "resistivity", canonical_unit: "ohm.m" },
    { id: "b", property: "density" },
  ]);
  assert.equal(out.length, 2);
  assert.deepEqual(out[0], {
    id: "a",
    property: "resistivity",
    canonicalUnit: "ohm.m",
  });
  assert.equal(out[1].canonicalUnit, null);
});

test("normalizeArtifacts unwraps {artifacts} / {property_models} / {items}", () => {
  assert.equal(
    normalizeArtifacts({ artifacts: [{ id: "a", property: "p" }] }).length,
    1,
  );
  assert.equal(
    normalizeArtifacts({ property_models: [{ id: "a", property: "p" }] }).length,
    1,
  );
  assert.equal(
    normalizeArtifacts({ propertyModels: [{ id: "a", property: "p" }] }).length,
    1,
  );
  assert.equal(normalizeArtifacts({ items: [{ id: "a", property: "p" }] }).length, 1);
});

test("normalizeArtifacts tolerates id aliases and camelCase unit", () => {
  const out = normalizeArtifacts([
    { pm_id: "x", property: "vel", canonicalUnit: "m/s" },
    { property_model_id: "y", name: "named" },
  ]);
  assert.equal(out[0].id, "x");
  assert.equal(out[0].canonicalUnit, "m/s");
  assert.equal(out[1].id, "y");
  assert.equal(out[1].property, "named"); // falls back to name
});

test("normalizeArtifacts filters out non-property-model kinds", () => {
  const out = normalizeArtifacts([
    { id: "a", property: "p", kind: "property_model" },
    { id: "b", property: "p", kind: "property-model" },
    { id: "c", property: "p", kind: "volume" },
    { id: "w", property: "p", kind: "well" },
    { id: "f", property: "p", type: "feature_set" },
  ]);
  assert.deepEqual(
    out.map((o) => o.id),
    ["a", "b", "c"],
  );
});

test("normalizeArtifacts skips rows without a string id and handles junk", () => {
  const out = normalizeArtifacts([
    { property: "no id" },
    null,
    42,
    { id: 7, property: "numeric id" },
    { id: "ok", property: "p" },
  ]);
  assert.deepEqual(
    out.map((o) => o.id),
    ["ok"],
  );
});

test("normalizeArtifacts returns [] for unrecognized bodies", () => {
  assert.deepEqual(normalizeArtifacts(null), []);
  assert.deepEqual(normalizeArtifacts("nope"), []);
  assert.deepEqual(normalizeArtifacts({ foo: 1 }), []);
});
