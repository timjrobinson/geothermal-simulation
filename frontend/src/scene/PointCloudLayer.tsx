// Microseismic 4-D point cloud (doc 06 §5.4). One THREE.Points cloud, uploaded ONCE — size
// ∝ magnitude, colour by time/depth/magnitude via a transfer-function LUT, and the global
// time slider drives a `uTimeWindow` uniform that fades/reveals events inside the moving
// window ENTIRELY on the GPU (no per-frame re-upload), so tens–hundreds of thousands of
// events animate cheaply (doc 06 §5.4, §9.4).
//
// Epoch-ms overflow float32, so the cloud is uploaded with a per-point epoch REBASED to the
// axis t0 (a relative float32 the shader compares against the rebased window bounds). The TF
// LUT and the colour-by mode are uniforms; recolouring is a uniform swap, not a re-upload.

import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { useViewer, currentTimeWindow } from "../store";
import type { Layer } from "../lib/layers";
import { makeTransferFnTexture, updateTransferFnTexture } from "../lib/transferFn";

const POINT_VERT = /* glsl */ `
precision highp float;
attribute float aMag;     // magnitude (size + optional colour)
attribute float aEpoch;   // epoch ms REBASED to axis t0 (relative; float32-safe)
attribute float aDepth;   // true depth (m) for colour-by-depth

uniform vec2 uTimeWindow; // [t0,t1] rebased ms (inclusive) — the moving window
uniform float uPointScale;
uniform float uVex;       // vertical exaggeration (render-only Z scale)
uniform int uColorBy;     // 0=time 1=depth 2=magnitude
uniform vec2 uColorRange; // [min,max] of the selected colour scalar

varying float vT;         // colour parameter in [0,1]
varying float vVisible;   // 1 inside the window, 0 outside (discarded in frag)

void main() {
  // Time-window membership (doc 06 §9.4): an undated point (aEpoch huge sentinel) stays out.
  vVisible = (aEpoch >= uTimeWindow.x && aEpoch <= uTimeWindow.y) ? 1.0 : 0.0;

  float c;
  if (uColorBy == 1) c = aDepth;
  else if (uColorBy == 2) c = aMag;
  else c = aEpoch;
  float span = max(uColorRange.y - uColorRange.x, 1e-6);
  vT = clamp((c - uColorRange.x) / span, 0.0, 1.0);

  vec3 p = position;
  p.z *= uVex;
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  // Size ∝ magnitude, attenuated by distance (doc 06 §5.4).
  float size = uPointScale * (1.0 + max(aMag, 0.0));
  gl_PointSize = size * (300.0 / max(-mv.z, 1.0));
  gl_Position = projectionMatrix * mv;
}
`;

const POINT_FRAG = /* glsl */ `
precision highp float;
uniform sampler2D uLut; // 256x1 RGBA transfer-function LUT
uniform float uOpacity;
varying float vT;
varying float vVisible;

void main() {
  if (vVisible < 0.5) discard; // outside the time window
  // round sprite
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  vec4 c = texture2D(uLut, vec2(vT, 0.5));
  gl_FragColor = vec4(c.rgb, uOpacity);
}
`;

const COLOR_BY: Record<string, number> = { time: 0, depth: 1, magnitude: 2 };

export function PointCloudLayer({ layer }: { layer: Layer }) {
  const win = useViewer(currentTimeWindow);
  const axisT0 = useViewer((s) => s.timeAxis.t0Ms);
  const vex = useViewer((s) => s.verticalExaggeration);
  const matRef = useRef<THREE.ShaderMaterial | null>(null);
  const lutRef = useRef<THREE.DataTexture | null>(null);

  const cloud = layer.points;
  // Rebase epoch-ms to the axis t0 so the shader compares float32-safe relative values. An
  // undated (NaN) epoch becomes a huge sentinel so it never enters any finite window.
  const base = axisT0 ?? 0;
  const colorBy = layer.transferFn.colorBy ?? "time";

  const geometry = useMemo(() => {
    if (!cloud || cloud.count === 0) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
    g.setAttribute("aMag", new THREE.BufferAttribute(cloud.magnitude, 1));
    const epochRel = new Float32Array(cloud.count);
    for (let i = 0; i < cloud.count; i++) {
      epochRel[i] = Number.isNaN(cloud.epochMs[i]) ? 1e30 : cloud.epochMs[i] - base;
    }
    g.setAttribute("aEpoch", new THREE.BufferAttribute(epochRel, 1));
    g.setAttribute(
      "aDepth",
      new THREE.BufferAttribute(cloud.depth ?? new Float32Array(cloud.count), 1),
    );
    return g;
    // base is intentionally excluded: rebasing on axis-t0 change is handled below by patching
    // the uniform window (both attribute + window share the same base reference once built).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cloud]);

  // Colour-scalar range for the colour-by mode (rebased epoch for time).
  const colorRange = useMemo<[number, number]>(() => {
    if (!cloud || cloud.count === 0) return [0, 1];
    let lo = Infinity;
    let hi = -Infinity;
    const pick = (i: number) =>
      colorBy === "depth"
        ? (cloud.depth?.[i] ?? 0)
        : colorBy === "magnitude"
          ? cloud.magnitude[i]
          : Number.isNaN(cloud.epochMs[i])
            ? NaN
            : cloud.epochMs[i] - base;
    for (let i = 0; i < cloud.count; i++) {
      const v = pick(i);
      if (Number.isNaN(v) || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return [0, 1];
    return [lo, hi];
  }, [cloud, colorBy, base]);

  const material = useMemo(() => {
    const lut = makeTransferFnTexture(layer.transferFn);
    lutRef.current = lut;
    const m = new THREE.ShaderMaterial({
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      transparent: true,
      depthWrite: false,
      uniforms: {
        uLut: { value: lut },
        uTimeWindow: { value: new THREE.Vector2(0, 0) },
        uPointScale: { value: 6 },
        uVex: { value: vex },
        uColorBy: { value: COLOR_BY[colorBy] ?? 0 },
        uColorRange: { value: new THREE.Vector2(colorRange[0], colorRange[1]) },
        uOpacity: { value: layer.opacity },
      },
    });
    matRef.current = m;
    return m;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Drive uniforms (no re-upload): window, colour mapping, opacity, vex, LUT (doc 06 §9.4).
  useEffect(() => {
    const m = matRef.current;
    if (!m) return;
    (m.uniforms.uTimeWindow.value as THREE.Vector2).set(
      win.t0Ms - base,
      win.t1Ms - base,
    );
    (m.uniforms.uColorRange.value as THREE.Vector2).set(colorRange[0], colorRange[1]);
    m.uniforms.uColorBy.value = COLOR_BY[colorBy] ?? 0;
    m.uniforms.uVex.value = vex;
    m.uniforms.uOpacity.value = layer.opacity;
  }, [win, base, colorRange, colorBy, vex, layer.opacity]);

  // Re-bake the LUT in place on transfer-fn edits (no GPU realloc, doc 06 §9.2).
  useEffect(() => {
    if (lutRef.current) updateTransferFnTexture(lutRef.current, layer.transferFn);
  }, [layer.transferFn]);

  useEffect(
    () => () => {
      geometry?.dispose();
      material.dispose();
      lutRef.current?.dispose();
    },
    [geometry, material],
  );

  if (!geometry || !layer.visible) return null;
  return <points geometry={geometry} material={material} renderOrder={3} />;
}

// Render all visible microseismic point-cloud layers (doc 06 §5.4, §9.1).
export function PointCloudLayers() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  return (
    <>
      {layerOrder.map((id) => {
        const l = layers[id];
        if (!l || l.kind !== "points") return null;
        return <PointCloudLayer key={id} layer={l} />;
      })}
    </>
  );
}
