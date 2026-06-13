// Orthogonal slice plane (doc 06 §4). A quad placed in the Engineering scene whose
// fragment shader samples the SAME Data3DTexture + same transfer-function LUT as the
// volume (doc 06 §4.1) — zero extra fetch, perfectly registered. The plane is draggable
// along its axis via the control panel (slicePos fraction). M1 supports X/Y/Z planes.

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useViewer, selectedLayer } from "../store";
import { makeData3DTexture } from "../lib/data3d";
import { makeTransferFnTexture, updateTransferFnTexture } from "../lib/transferFn";
import { SLICE_VERT, SLICE_FRAG } from "../lib/shaders";

// Build a quad geometry for the slice plane at axis/fraction within the Engineering AABB,
// carrying per-vertex volume texcoords (uvw in [0,1]^3) so the shader looks up directly.
function buildSliceGeometry(
  axis: "x" | "y" | "z",
  frac: number,
  min: [number, number, number],
  max: [number, number, number],
): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  const f = Math.min(1, Math.max(0, frac));
  // Four corners (a,b,c,d) in world XYZ + their uvw.
  let pos: number[];
  let uvw: number[];
  if (axis === "z") {
    const z = min[2] + f * (max[2] - min[2]);
    pos = [
      min[0], min[1], z,
      max[0], min[1], z,
      max[0], max[1], z,
      min[0], max[1], z,
    ];
    uvw = [0, 0, f, 1, 0, f, 1, 1, f, 0, 1, f];
  } else if (axis === "y") {
    const y = min[1] + f * (max[1] - min[1]);
    pos = [
      min[0], y, min[2],
      max[0], y, min[2],
      max[0], y, max[2],
      min[0], y, max[2],
    ];
    uvw = [0, f, 0, 1, f, 0, 1, f, 1, 0, f, 1];
  } else {
    const x = min[0] + f * (max[0] - min[0]);
    pos = [
      x, min[1], min[2],
      x, max[1], min[2],
      x, max[1], max[2],
      x, min[1], max[2],
    ];
    uvw = [f, 0, 0, f, 1, 0, f, 1, 1, f, 0, 1];
  }
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uvw", new THREE.Float32BufferAttribute(uvw, 3));
  g.setIndex([0, 1, 2, 0, 2, 3]);
  g.computeVertexNormals();
  return g;
}

export function SliceLayer() {
  // The slice samples the SELECTED layer's volume + transfer function (doc 06 §4.1).
  const layer = useViewer(selectedLayer);
  const volume = layer?.volume ?? null;
  const aabb = layer?.aabb ?? null;
  const tf = layer?.transferFn ?? null;
  const clip = useViewer((s) => s.clip);
  const enabled = useViewer((s) => s.sliceEnabled);
  const axis = useViewer((s) => s.sliceAxis);
  const pos = useViewer((s) => s.slicePos);
  const sliceOpacity = useViewer((s) => s.sliceOpacity);

  const matRef = useRef<THREE.ShaderMaterial | null>(null);

  // Rebuild the slice's 3D texture when the selected layer's volume changes.
  const volumeTex = useMemo(
    () => (volume ? makeData3DTexture(volume) : null),
    [volume],
  );
  // LUT texture: created lazily once a layer is selected, re-baked in place on TF edits.
  // Keyed off the volume so it is re-created when the selected layer changes.
  const tfTex = useMemo(
    () => (tf ? makeTransferFnTexture(tf) : null),
    [volume], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const geometry = useMemo(() => {
    if (!aabb) return null;
    return buildSliceGeometry(axis, pos, aabb.min, aabb.max);
  }, [aabb, axis, pos]);

  const material = useMemo(() => {
    if (!volumeTex || !tfTex || !tf) return null;
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: SLICE_VERT,
      fragmentShader: SLICE_FRAG,
      side: THREE.DoubleSide,
      transparent: true,
      uniforms: {
        uVolume: { value: volumeTex },
        uTransferFn: { value: tfTex },
        uClipMin: { value: new THREE.Vector3(0, 0, 0) },
        uClipMax: { value: new THREE.Vector3(1, 1, 1) },
        uDomainMin: { value: tf.domainMin },
        uDomainMax: { value: tf.domainMax },
        uLog: { value: tf.scaling === "log" ? 1 : 0 },
        uSliceOpacity: { value: sliceOpacity },
      },
    });
    matRef.current = mat;
    return mat;
  }, [volumeTex, tfTex]); // eslint-disable-line react-hooks/exhaustive-deps

  useMemo(() => {
    if (tfTex && tf) updateTransferFnTexture(tfTex, tf);
  }, [tf, tfTex]);

  useFrame(() => {
    const mat = matRef.current;
    if (!mat || !tf) return;
    const u = mat.uniforms;
    u.uDomainMin.value = tf.domainMin;
    u.uDomainMax.value = tf.domainMax;
    u.uLog.value = tf.scaling === "log" ? 1 : 0;
    u.uSliceOpacity.value = sliceOpacity;
    (u.uClipMin.value as THREE.Vector3).set(clip.min[0], clip.min[1], clip.min[2]);
    (u.uClipMax.value as THREE.Vector3).set(clip.max[0], clip.max[1], clip.max[2]);
  });

  if (!enabled || !material || !geometry) return null;

  return (
    <mesh renderOrder={2}>
      <primitive object={geometry} attach="geometry" />
      <primitive object={material} attach="material" />
    </mesh>
  );
}
