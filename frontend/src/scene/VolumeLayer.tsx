// Ray-marched volume layer (doc 06 §3.1). A box mesh spanning the volume's Engineering
// AABB is the proxy geometry; a single-pass WebGL2 ShaderMaterial marches the ray from
// the box front face to its exit, sampling the shared Data3DTexture through the transfer
// function LUT, clipping in-shader against both the AABB and the user clip box.

import { useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useViewer } from "../store";
import { makeData3DTexture } from "../lib/data3d";
import { makeTransferFnTexture, updateTransferFnTexture } from "../lib/transferFn";
import { VOLUME_VERT, VOLUME_FRAG } from "../lib/shaders";
import { aabbCenter, aabbSize } from "../lib/volume";

export function VolumeLayer() {
  const volume = useViewer((s) => s.volume);
  const aabb = useViewer((s) => s.aabb);
  const tf = useViewer((s) => s.tf);
  const steps = useViewer((s) => s.steps);
  const clip = useViewer((s) => s.clip);
  const visible = useViewer((s) => s.volumeVisible);
  const camera = useThree((s) => s.camera);

  const materialRef = useRef<THREE.ShaderMaterial | null>(null);

  // 3D texture — rebuilt only when the volume data changes.
  const volumeTex = useMemo(
    () => (volume ? makeData3DTexture(volume) : null),
    [volume],
  );

  // Transfer-function LUT texture — created once, re-baked in place on edits.
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
        uOpacityGain: { value: tf.opacity },
        uSteps: { value: steps },
        uRefStep: { value: 1.0 },
      },
    });
    materialRef.current = mat;
    return mat;
  }, [volumeTex, tfTex, aabb]); // eslint-disable-line react-hooks/exhaustive-deps

  // Box proxy transform from the AABB.
  const { center, size } = useMemo(() => {
    if (!aabb) return { center: [0, 0, 0] as const, size: [1, 1, 1] as const };
    return { center: aabbCenter(aabb), size: aabbSize(aabb) };
  }, [aabb]);

  // Push live uniforms each frame (camera moves; store edits are cheap to mirror).
  useFrame(() => {
    const mat = materialRef.current;
    if (!mat || !aabb) return;
    const u = mat.uniforms;
    (u.uCameraPos.value as THREE.Vector3).copy(camera.position);
    u.uDomainMin.value = tf.domainMin;
    u.uDomainMax.value = tf.domainMax;
    u.uLog.value = tf.scaling === "log" ? 1 : 0;
    u.uOpacityGain.value = tf.opacity;
    u.uSteps.value = steps;
    // Reference step = one voxel-ish along the smallest axis for opacity correction.
    const span = aabbSize(aabb);
    u.uRefStep.value = Math.max(Math.min(span[0], span[1], span[2]) / steps, 1e-3);
    // Clip box: fractions -> Engineering metres.
    (u.uClipMin.value as THREE.Vector3).set(
      aabb.min[0] + clip.min[0] * span[0],
      aabb.min[1] + clip.min[1] * span[1],
      aabb.min[2] + clip.min[2] * span[2],
    );
    (u.uClipMax.value as THREE.Vector3).set(
      aabb.min[0] + clip.max[0] * span[0],
      aabb.min[1] + clip.max[1] * span[1],
      aabb.min[2] + clip.max[2] * span[2],
    );
  });

  // Re-bake the LUT when the transfer function changes (no refetch — doc 06 §9.2).
  useMemo(() => {
    if (tfTex) updateTransferFnTexture(tfTex, tf);
  }, [tf, tfTex]);

  if (!material || !aabb || !visible) return null;

  return (
    <mesh position={center as unknown as [number, number, number]} renderOrder={1}>
      <boxGeometry args={[size[0], size[1], size[2]]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}
