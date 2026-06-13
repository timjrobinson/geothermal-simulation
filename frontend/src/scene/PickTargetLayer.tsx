// Pick-target catcher (doc 09 §8.1 "pick target mode"). When the planning panel arms
// `pickTargetMode`, a near-invisible box spanning the scene AABB captures the next pointer
// click and converts the hit point → Engineering XYZ (dividing out the render-only vertical
// exaggeration, doc 06 §2.3) → the panel's onPick callback (which POSTs the target). The
// catcher only renders while armed so it never intercepts normal scene interaction. This is
// the doc 06 substrate doc 09 §8.1 asks for: "ray-pick returning Engineering XYZ".

import { useMemo } from "react";
import { type ThreeEvent } from "@react-three/fiber";
import { useViewer } from "../store";
import { aabbCenter, aabbSize } from "../lib/volume";

export function PickTargetLayer() {
  const armed = useViewer((s) => s.pickTargetMode);
  const aabb = useViewer((s) => s.sceneAABB);
  const vex = useViewer((s) => s.verticalExaggeration);
  const setPendingPickXYZ = useViewer((s) => s.setPendingPickXYZ);
  const setPickTargetMode = useViewer((s) => s.setPickTargetMode);

  // Box geometry args + centre in render space (Z exaggerated). The hit point is converted
  // back to true Engineering Z before being handed to the panel.
  const box = useMemo(() => {
    if (!aabb) return null;
    const c = aabbCenter(aabb);
    const s = aabbSize(aabb);
    // Pad slightly so isosurfaces/terrain at the AABB faces stay clickable.
    return {
      center: [c[0], c[1], c[2] * vex] as [number, number, number],
      size: [s[0] * 1.05, s[1] * 1.05, s[2] * 1.05 * vex] as [number, number, number],
    };
  }, [aabb, vex]);

  if (!armed || !box) return null;

  const onClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    const p = e.point;
    setPendingPickXYZ([p.x, p.y, p.z / (vex || 1)]);
    setPickTargetMode(false); // one-shot pick; the panel re-arms for the next
  };

  return (
    <mesh position={box.center} onClick={onClick} renderOrder={999}>
      <boxGeometry args={box.size} />
      {/* Faintly tinted so the user sees pick mode is armed; transparent + depthWrite off so
          it never occludes the data underneath. */}
      <meshBasicMaterial color="#89b4fa" transparent opacity={0.06} depthWrite={false} />
    </mesh>
  );
}
