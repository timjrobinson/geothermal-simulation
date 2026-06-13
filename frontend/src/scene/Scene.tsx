// The M1 R3F scene (doc 06 §2). Z-up Engineering Frame root (DEFAULT_UP=(0,0,1)), drei
// CameraControls (target-centric orbit), hemispheric + key lights, the ray-marched
// VolumeLayer, one orthogonal SliceLayer, and the draggable ClipBox. The scene world
// space IS the Engineering Frame: positions are Engineering metres straight to the GPU
// (doc 06 §2.1) — no CRS ever reaches the GPU, float32-safe via the floating origin.

import { useEffect, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { CameraControls } from "@react-three/drei";
import * as THREE from "three";
import { useViewer } from "../store";
import { VolumeLayers } from "./VolumeLayer";
import { TerrainLayers } from "./TerrainLayer";
import { SliceLayer } from "./SliceLayer";
import { ClipBox } from "./ClipBox";
import { aabbCenter, aabbSize } from "../lib/volume";

// One-time Z-up: ENU Z is up (doc 06 §2.1). Set before any Object3D is created.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

// Frame the camera obliquely down into the volume AABB when data first loads (doc 06 §2.2).
function CameraFramer() {
  const controls = useThree((s) => s.controls) as CameraControls | null;
  const aabb = useViewer((s) => s.sceneAABB);
  const framed = useRef(false);

  useEffect(() => {
    if (!controls || !aabb || framed.current) return;
    const c = aabbCenter(aabb);
    const s = aabbSize(aabb);
    const diag = Math.hypot(s[0], s[1], s[2]);
    // Look obliquely down into the volume (subsurface-aware framing, doc 06 §2.2).
    const eye = new THREE.Vector3(
      c[0] + diag * 0.9,
      c[1] - diag * 0.9,
      c[2] + diag * 0.7,
    );
    controls.setLookAt(eye.x, eye.y, eye.z, c[0], c[1], c[2], false);
    framed.current = true;
  }, [controls, aabb]);

  return null;
}

export function Scene() {
  return (
    <Canvas
      gl={{ antialias: true, alpha: false, logarithmicDepthBuffer: true }}
      camera={{ up: [0, 0, 1], near: 0.1, far: 1e6, position: [3000, -3000, 2000] }}
      style={{ position: "absolute", inset: 0 }}
      // Per-material clippingPlanes (terrain mesh clip, doc 06 §2.4) require local clipping.
      onCreated={({ gl }) => {
        gl.localClippingEnabled = true;
      }}
    >
      <color attach="background" args={["#0a0e14"]} />
      <hemisphereLight args={["#cdd6f4", "#1e1e2e", 0.9]} />
      <directionalLight position={[1, -1, 1]} intensity={0.6} />
      <ambientLight intensity={0.25} />

      <TerrainLayers />
      <SliceLayer />
      <VolumeLayers />
      <ClipBox />

      <axesHelper args={[1000]} />
      <CameraControls makeDefault />
      <CameraFramer />
    </Canvas>
  );
}
