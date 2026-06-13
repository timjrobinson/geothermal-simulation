// Draggable axis-aligned clip box (doc 06 §2.4). Render-only: it stores six fractions
// (min/max along X/Y/Z of the volume AABB) which the volume ray-marcher clips against
// in-shader and the slice shader respects (doc 06 §2.4 — "volume ray-marcher clips
// in-shader"). A wireframe shows the current box; six draggable handle spheres (one per
// face) move that face along its axis by dragging in screen space.

import { useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useViewer } from "../store";
import { aabbSize } from "../lib/volume";

type Face = "minX" | "maxX" | "minY" | "maxY" | "minZ" | "maxZ";
const AXIS: Record<Face, 0 | 1 | 2> = {
  minX: 0, maxX: 0, minY: 1, maxY: 1, minZ: 2, maxZ: 2,
};
const IS_MIN: Record<Face, boolean> = {
  minX: true, maxX: false, minY: true, maxY: false, minZ: true, maxZ: false,
};

export function ClipBox() {
  const aabb = useViewer((s) => s.aabb);
  const clip = useViewer((s) => s.clip);
  const setClip = useViewer((s) => s.setClip);
  const { gl } = useThree();

  const drag = useRef<{ face: Face } | null>(null);

  // World-space box corners from the clip fractions.
  const { lo, hi, size } = useMemo(() => {
    if (!aabb)
      return {
        lo: new THREE.Vector3(),
        hi: new THREE.Vector3(),
        size: [1, 1, 1] as [number, number, number],
      };
    const s = aabbSize(aabb);
    const lo = new THREE.Vector3(
      aabb.min[0] + clip.min[0] * s[0],
      aabb.min[1] + clip.min[1] * s[1],
      aabb.min[2] + clip.min[2] * s[2],
    );
    const hi = new THREE.Vector3(
      aabb.min[0] + clip.max[0] * s[0],
      aabb.min[1] + clip.max[1] * s[1],
      aabb.min[2] + clip.max[2] * s[2],
    );
    return { lo, hi, size: s };
  }, [aabb, clip]);

  if (!aabb) return null;

  const center = new THREE.Vector3()
    .addVectors(lo, hi)
    .multiplyScalar(0.5);
  const dims = new THREE.Vector3().subVectors(hi, lo);

  // Handle world positions (centre of each face).
  const handlePos: Record<Face, THREE.Vector3> = {
    minX: new THREE.Vector3(lo.x, center.y, center.z),
    maxX: new THREE.Vector3(hi.x, center.y, center.z),
    minY: new THREE.Vector3(center.x, lo.y, center.z),
    maxY: new THREE.Vector3(center.x, hi.y, center.z),
    minZ: new THREE.Vector3(center.x, center.y, lo.z),
    maxZ: new THREE.Vector3(center.x, center.y, hi.z),
  };

  const handleSize = Math.max(...size) * 0.025;

  const onDown = (face: Face) => (e: { stopPropagation: () => void }) => {
    e.stopPropagation();
    drag.current = { face };
    (gl.domElement as HTMLElement).style.cursor = "grabbing";
  };

  const onMove = (e: { ray: THREE.Ray; stopPropagation: () => void }) => {
    const d = drag.current;
    if (!d) return;
    e.stopPropagation();
    const axis = AXIS[d.face];
    // Project the pointer ray onto the dragged axis line through the box centre.
    const axisDir = new THREE.Vector3(0, 0, 0);
    axisDir.setComponent(axis, 1);
    const linePoint = center.clone();
    // Closest point between ray and the axis line → take its axis coordinate.
    const t = closestOnLine(e.ray, linePoint, axisDir);
    const world = linePoint.clone().add(axisDir.clone().multiplyScalar(t));
    const frac =
      (world.getComponent(axis) - aabb.min[axis]) / Math.max(size[axis], 1e-6);
    const clamped = Math.min(1, Math.max(0, frac));
    if (IS_MIN[d.face]) {
      const newMin = [...clip.min] as [number, number, number];
      newMin[axis] = Math.min(clamped, clip.max[axis] - 0.01);
      setClip({ min: newMin });
    } else {
      const newMax = [...clip.max] as [number, number, number];
      newMax[axis] = Math.max(clamped, clip.min[axis] + 0.01);
      setClip({ max: newMax });
    }
  };

  const onUp = () => {
    drag.current = null;
    (gl.domElement as HTMLElement).style.cursor = "auto";
  };

  // Invisible large plane to capture pointer-move/up while dragging.
  return (
    <group>
      {/* Wireframe of the current clip box */}
      <lineSegments position={center.toArray()}>
        <edgesGeometry args={[new THREE.BoxGeometry(dims.x, dims.y, dims.z)]} />
        <lineBasicMaterial color="#89b4fa" />
      </lineSegments>

      {/* Capture plane (huge, invisible) active during a drag */}
      <mesh
        visible={false}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        position={center.toArray()}
      >
        <boxGeometry args={[size[0] * 6, size[1] * 6, size[2] * 6]} />
        <meshBasicMaterial transparent opacity={0} depthWrite={false} side={THREE.DoubleSide} />
      </mesh>

      {/* Face handles */}
      {(Object.keys(handlePos) as Face[]).map((face) => (
        <mesh
          key={face}
          position={handlePos[face].toArray()}
          onPointerDown={onDown(face)}
          onPointerMove={onMove}
          onPointerUp={onUp}
        >
          <sphereGeometry args={[handleSize, 16, 16]} />
          <meshBasicMaterial color={AXIS[face] === 0 ? "#f38ba8" : AXIS[face] === 1 ? "#a6e3a1" : "#89b4fa"} />
        </mesh>
      ))}
    </group>
  );
}

// Signed distance along (point, dir) of the closest point to a ray (line-line closest).
function closestOnLine(ray: THREE.Ray, point: THREE.Vector3, dir: THREE.Vector3): number {
  const p0 = ray.origin;
  const u = ray.direction; // assumed normalized
  const q0 = point;
  const v = dir; // normalized
  const w0 = new THREE.Vector3().subVectors(p0, q0);
  const a = u.dot(u);
  const b = u.dot(v);
  const c = v.dot(v);
  const d = u.dot(w0);
  const e = v.dot(w0);
  const denom = a * c - b * b;
  if (Math.abs(denom) < 1e-9) return -e / Math.max(c, 1e-9);
  // tc parameter along the axis line v.
  const tc = (a * e - b * d) / denom;
  return tc;
}
