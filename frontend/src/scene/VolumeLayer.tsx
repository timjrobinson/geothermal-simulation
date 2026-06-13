// Ray-marched volume layer (doc 06 §3.1, §3.3). One layer === one box mesh spanning that
// layer's Engineering AABB; a single-pass WebGL2 ShaderMaterial marches the ray through
// the layer's own Data3DTexture + transfer-function LUT, clipping in-shader against both
// the layer AABB and the global clip box. N of these co-render, alpha-composited by layer
// order (doc 06 §3.3 default); the per-layer blend mode (over/additive/MIP/minIP) selects
// both the in-shader accumulation and the GL blend equation.

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { useViewer, SELECTION_LAYER_ID } from "../store";
import { pickNearestVoxel } from "../lib/brushing";
import type { Layer, BlendMode } from "../lib/layers";
import { makeData3DTexture } from "../lib/data3d";
import { makeTransferFnTexture, updateTransferFnTexture } from "../lib/transferFn";
import { VOLUME_VERT, VOLUME_FRAG } from "../lib/shaders";
import { aabbCenter, aabbSize, type AABB } from "../lib/volume";
import { StreamingVolumeLayer } from "./StreamingVolumeLayer";
import { shouldStream, type PyramidSpec } from "../lib/bricks";
import { BRICK_SIZE } from "../lib/bricks";

// Route a volume layer to the M2+ brick-streaming path (doc 06 §3.4) vs the M1 single-
// resident fast path (doc 06 §1.3). A layer streams when it has NO resident `volume` buffer
// (the loader chose not to fully resident-load it) but carries pyramid `meta`, OR when its
// level-0 size exceeds the single-resident budget and a pyramid exists to stream from
// (shouldStream). Small volumes with a resident buffer stay on the proven single-resident
// shader so the M1 fast path is never regressed.
export function isStreamingLayer(layer: Layer): boolean {
  if (layer.kind !== "volume") return false;
  if (!layer.meta) return false;
  if (!layer.volume) return true; // no resident buffer -> must stream from the pyramid
  const spec: PyramidSpec = {
    shape0: layer.meta.shape,
    origin: layer.meta.origin,
    spacing0: layer.meta.spacing,
    levels: Math.max(1, layer.meta.levels),
    brick: BRICK_SIZE,
  };
  return shouldStream(spec);
}

// A 1×1×1 unit-confidence Data3DTexture (value 1.0) — the default `uConfidence` sampler for
// layers with no confidence binding, so modulation is a no-op until one is bound (doc 07
// §5.3). Created once and shared (it is immutable / read-only).
let _unitConfTex: THREE.Data3DTexture | null = null;
function unitConfidenceTexture(): THREE.Data3DTexture {
  if (_unitConfTex) return _unitConfTex;
  const tex = new THREE.Data3DTexture(new Float32Array([1]) as unknown as BufferSource, 1, 1, 1);
  tex.format = THREE.RedFormat;
  tex.type = THREE.FloatType;
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  _unitConfTex = tex;
  return tex;
}

const BLEND_INDEX: Record<BlendMode, number> = {
  over: 0,
  additive: 1,
  mip: 2,
  minip: 3,
};

// Apply the GL blend equation for a blend mode. `over` uses standard premultiplied-ish
// alpha; `additive`/`mip`/`minIP` use additive / max / min hardware blending so multiple
// layer meshes composite correctly without an offscreen pass.
function applyGLBlend(mat: THREE.ShaderMaterial, blend: BlendMode): void {
  mat.transparent = true;
  mat.depthWrite = false;
  mat.depthTest = blend === "over"; // additive/MIP/minIP read every layer regardless of depth
  if (blend === "additive") {
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.AddEquation;
    mat.blendSrc = THREE.SrcAlphaFactor;
    mat.blendDst = THREE.OneFactor;
  } else if (blend === "mip") {
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.MaxEquation;
    mat.blendSrc = THREE.OneFactor;
    mat.blendDst = THREE.OneFactor;
  } else if (blend === "minip") {
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.MinEquation;
    mat.blendSrc = THREE.OneFactor;
    mat.blendDst = THREE.OneFactor;
  } else {
    mat.blending = THREE.NormalBlending;
  }
  mat.needsUpdate = true;
}

export function VolumeLayer({ layer, order }: { layer: Layer; order: number }) {
  const steps = useViewer((s) => s.steps);
  const clip = useViewer((s) => s.clip);
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const camera = useThree((s) => s.camera);

  const tf = layer.transferFn;
  const volume = layer.volume ?? null;
  const aabb = layer.aabb ?? null;

  const materialRef = useRef<THREE.ShaderMaterial | null>(null);

  // 3D texture — rebuilt only when this layer's volume data changes.
  const volumeTex = useMemo(
    () => (volume ? makeData3DTexture(volume) : null),
    [volume],
  );

  // Confidence/σ texture for opacity modulation (doc 07 §5.3 honest view). Rebuilt only when
  // the bound confidence volume changes; a 1×1×1 unit texture stands in when none is bound so
  // the sampler3D uniform is always satisfied (an unbound layer renders unmodulated).
  const conf = layer.confidence ?? null;
  const confTex = useMemo(
    () => (conf?.volume ? makeData3DTexture(conf.volume) : unitConfidenceTexture()),
    [conf?.volume],
  );

  // Transfer-function LUT texture — created once per layer, re-baked in place on edits.
  const tfTex = useMemo(() => makeTransferFnTexture(tf), []); // eslint-disable-line react-hooks/exhaustive-deps

  const material = useMemo(() => {
    if (!volumeTex || !aabb) return null;
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VOLUME_VERT,
      fragmentShader: VOLUME_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide, // march from back faces so we enter the box even from inside
      uniforms: {
        uVolume: { value: volumeTex },
        uTransferFn: { value: tfTex },
        uBoxMin: { value: new THREE.Vector3(...aabb.min) },
        uBoxMax: { value: new THREE.Vector3(...aabb.max) },
        uClipMin: { value: new THREE.Vector3(...aabb.min) },
        uClipMax: { value: new THREE.Vector3(...aabb.max) },
        uCameraPos: { value: new THREE.Vector3() },
        uDomainMin: { value: tf.domainMin },
        uDomainMax: { value: tf.domainMax },
        uLog: { value: tf.scaling === "log" ? 1 : 0 },
        uOpacityGain: { value: tf.opacity * layer.opacity },
        uSteps: { value: steps },
        uRefStep: { value: 1.0 },
        uBlend: { value: BLEND_INDEX[layer.blend] },
        // Confidence-modulated opacity (doc 07 §5.3). Seeded off here; pushed each frame.
        uConfidence: { value: confTex },
        uConfidenceOn: { value: 0 },
        uConfMin: { value: 0 },
        uConfMax: { value: 1 },
        uConfInvert: { value: 0 },
        uConfFloor: { value: 0.05 },
      },
    });
    applyGLBlend(mat, layer.blend);
    materialRef.current = mat;
    return mat;
  }, [volumeTex, tfTex, confTex, aabb]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-apply the GL blend equation when the layer's blend mode changes.
  useEffect(() => {
    if (materialRef.current) applyGLBlend(materialRef.current, layer.blend);
  }, [layer.blend]);

  // Box proxy transform from this layer's AABB.
  const { center, size } = useMemo(() => {
    if (!aabb) return { center: [0, 0, 0] as const, size: [1, 1, 1] as const };
    return { center: aabbCenter(aabb), size: aabbSize(aabb) };
  }, [aabb]);

  // The clip box fractions are relative to the union sceneAABB; convert to Engineering m.
  const clipBasis: AABB = sceneAABB ?? aabb ?? { min: [0, 0, 0], max: [1, 1, 1] };

  // Push live uniforms each frame (camera moves; store edits are cheap to mirror).
  useFrame(() => {
    const mat = materialRef.current;
    if (!mat || !aabb) return;
    const u = mat.uniforms;
    (u.uCameraPos.value as THREE.Vector3).copy(camera.position);
    u.uDomainMin.value = tf.domainMin;
    u.uDomainMax.value = tf.domainMax;
    u.uLog.value = tf.scaling === "log" ? 1 : 0;
    u.uOpacityGain.value = tf.opacity * layer.opacity;
    u.uSteps.value = steps;
    u.uBlend.value = BLEND_INDEX[layer.blend];
    // Confidence-modulated opacity (doc 07 §5.3): off unless a binding is enabled.
    if (conf && conf.enabled) {
      if (u.uConfidence.value !== confTex) u.uConfidence.value = confTex;
      u.uConfidenceOn.value = 1;
      u.uConfMin.value = conf.min;
      u.uConfMax.value = conf.max;
      u.uConfInvert.value = conf.invert ? 1 : 0;
      u.uConfFloor.value = conf.floor;
    } else {
      u.uConfidenceOn.value = 0;
    }
    const span = aabbSize(aabb);
    u.uRefStep.value = Math.max(Math.min(span[0], span[1], span[2]) / steps, 1e-3);
    // Clip box: scene-AABB fractions -> Engineering metres (or disabled => layer AABB).
    if (layer.clip) {
      const cspan = aabbSize(clipBasis);
      (u.uClipMin.value as THREE.Vector3).set(
        clipBasis.min[0] + clip.min[0] * cspan[0],
        clipBasis.min[1] + clip.min[1] * cspan[1],
        clipBasis.min[2] + clip.min[2] * cspan[2],
      );
      (u.uClipMax.value as THREE.Vector3).set(
        clipBasis.min[0] + clip.max[0] * cspan[0],
        clipBasis.min[1] + clip.max[1] * cspan[1],
        clipBasis.min[2] + clip.max[2] * cspan[2],
      );
    } else {
      (u.uClipMin.value as THREE.Vector3).set(...aabb.min);
      (u.uClipMax.value as THREE.Vector3).set(...aabb.max);
    }
  });

  // Re-bake the LUT when this layer's transfer function changes (no refetch — doc 06 §9.2).
  useMemo(() => {
    if (tfTex) updateTransferFnTexture(tfTex, tf);
  }, [tf, tfTex]);

  // Linked brushing — 3D pick → cross-plot inspector (doc 06 §10.3). Clicking a source
  // volume resolves the front-face hit point (Engineering metres) to the nearest sampled
  // fused cell and pushes its multi-property values to the store. The selection-mask overlay
  // itself is not pickable (it would shadow the source volumes). Pointer-down + small-move
  // gating keeps orbit drags from registering as picks.
  const setPickedVoxel = useViewer((s) => s.setPickedVoxel);
  const downRef = useRef<{ x: number; y: number } | null>(null);
  const pickable = layer.id !== SELECTION_LAYER_ID;

  const onPointerDown = (e: ThreeEvent<PointerEvent>) => {
    downRef.current = { x: e.clientX, y: e.clientY };
  };
  const onPointerUp = (e: ThreeEvent<PointerEvent>) => {
    const d = downRef.current;
    downRef.current = null;
    if (!d) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return; // a drag (orbit), not a pick
    const sample = useViewer.getState().fusedSample;
    if (!sample) return;
    e.stopPropagation();
    const p = e.point; // Engineering XYZ metres of the hit on the box face
    const readout = pickNearestVoxel(sample, [p.x, p.y, p.z]);
    setPickedVoxel(readout);
  };

  if (!material || !aabb || !layer.visible) return null;

  return (
    <mesh
      position={center as unknown as [number, number, number]}
      renderOrder={1 + order}
      onPointerDown={pickable ? onPointerDown : undefined}
      onPointerUp={pickable ? onPointerUp : undefined}
    >
      <boxGeometry args={[size[0], size[1], size[2]]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}

// Pick the single-resident M1 renderer or the M2+ brick streamer per layer (doc 06 §1.3 vs
// §3.4). Both render one box-proxy mesh + transfer fn; only the texture backing differs.
function VolumeLayerRouter({ layer, order }: { layer: Layer; order: number }) {
  if (isStreamingLayer(layer)) {
    return <StreamingVolumeLayer layer={layer} order={order} />;
  }
  return <VolumeLayer layer={layer} order={order} />;
}

// Render all visible volume layers in compositing order (doc 06 §3.3 default: separate
// ray-marches alpha-composited by layer order). renderOrder follows layerOrder so the
// transparent passes draw bottom→top.
export function VolumeLayers() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  return (
    <>
      {layerOrder.map((id, i) => {
        const l = layers[id];
        if (!l || l.kind !== "volume") return null;
        return <VolumeLayerRouter key={id} layer={l} order={i} />;
      })}
    </>
  );
}
