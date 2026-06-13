// Shared clip-box → Engineering-metre THREE.Plane[] helper (doc 06 §2.4). The global clip
// box is stored as [0,1] fractions of the scene AABB; every mesh-based layer (terrain,
// feature surfaces/faults, well tubes, draped rasters) carves itself with the SAME six
// planes so one box cuts terrain + subsurface together. Planes live in world (Engineering)
// space; the per-layer mesh Z scale (vertical exaggeration) is folded into the Z bound so
// the clip tracks the exaggerated geometry. Each THREE.Plane keeps the half-space on its
// +normal side, so the normals point INTO the box.

import * as THREE from "three";
import { aabbSize, type AABB } from "../lib/volume";

export function clipPlanesFor(
  basis: AABB,
  clip: { min: [number, number, number]; max: [number, number, number] },
  vex: number,
): THREE.Plane[] {
  const s = aabbSize(basis);
  const lo: [number, number, number] = [
    basis.min[0] + clip.min[0] * s[0],
    basis.min[1] + clip.min[1] * s[1],
    (basis.min[2] + clip.min[2] * s[2]) * vex,
  ];
  const hi: [number, number, number] = [
    basis.min[0] + clip.max[0] * s[0],
    basis.min[1] + clip.max[1] * s[1],
    (basis.min[2] + clip.max[2] * s[2]) * vex,
  ];
  return [
    new THREE.Plane(new THREE.Vector3(1, 0, 0), -lo[0]),
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), hi[0]),
    new THREE.Plane(new THREE.Vector3(0, 1, 0), -lo[1]),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), hi[1]),
    new THREE.Plane(new THREE.Vector3(0, 0, 1), -lo[2]),
    new THREE.Plane(new THREE.Vector3(0, 0, -1), hi[2]),
  ];
}
