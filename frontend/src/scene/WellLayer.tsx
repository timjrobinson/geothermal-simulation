// Well-path layer (doc 06 §5.3). Builds a THREE.TubeGeometry along the resolved Engineering
// trajectory polyline (from GET /wells/{id}/trajectory) and colours the tube by a selected
// LAS log curve via a transfer function resampled to the station MDs (lib/wells.ts — pure,
// unit-tested). Hover reports MD/TVD/elevation (true depths, NOT vertically-exaggerated, doc
// 06 §2.3) into the store so the log-track panel + a tooltip can sync to the picked well.
//
// Vertical exaggeration is a render-only Z scale on the group (the geometry stays true so the
// hover readout reports true elevation). The clip box carves the tube via shared
// clippingPlanes (doc 06 §2.4).

import { useEffect, useMemo } from "react";
import { type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { useViewer } from "../store";
import type { Layer } from "../lib/layers";
import {
  stationMD,
  resampleCurveToStations,
  curveRange,
  curveToVertexColors,
  readoutAtPoint,
  type Vec3,
} from "../lib/wells";
import { clipPlanesFor } from "./clipPlanes";
import type { AABB } from "../lib/volume";

// Approximate tube radius from the trajectory extent so the well reads at any zoom.
function tubeRadius(aabb: AABB | undefined): number {
  if (!aabb) return 10;
  const d = Math.hypot(
    aabb.max[0] - aabb.min[0],
    aabb.max[1] - aabb.min[1],
    aabb.max[2] - aabb.min[2],
  );
  return Math.max(2, d * 0.004);
}

export function WellLayer({ layer }: { layer: Layer }) {
  const clip = useViewer((s) => s.clip);
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const vex = useViewer((s) => s.verticalExaggeration);
  const setWellReadout = useViewer((s) => s.setWellReadout);

  const traj = layer.trajectory;
  const radius = tubeRadius(layer.aabb);

  // Build the tube geometry from the polyline (rebuilt only when the path changes).
  const geometry = useMemo(() => {
    if (!traj || traj.polyline.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(
      traj.polyline.map((p) => new THREE.Vector3(p[0], p[1], p[2])),
    );
    const segs = Math.max(8, traj.polyline.length * 4);
    const g = new THREE.TubeGeometry(curve, segs, radius, 8, false);
    return g;
  }, [traj, radius]);

  // Per-vertex log colours: resample the selected curve onto the tube's vertex MDs. The tube
  // has (segs+1)*(radial+1) vertices laid out ring-by-ring along the spine, so each ring maps
  // to a fractional position along the polyline -> an interpolated station MD -> a colour.
  const vertexColors = useMemo(() => {
    if (!geometry || !traj) return null;
    const property = layer.logProperty;
    if (!property || !traj.logs?.curves?.[property]) return null;
    const md = stationMD(traj);
    const stationValues = resampleCurveToStations(traj.logs, property, md);
    const range = curveRange(stationValues);

    const pos = geometry.getAttribute("position");
    const tubeSegs = (geometry.parameters as { tubularSegments: number }).tubularSegments;
    const radial = (geometry.parameters as { radialSegments: number }).radialSegments;
    const ringCount = tubeSegs + 1;
    const perRing = radial + 1;
    // Value per ring (interp along the polyline by ring fraction), then expand to vertices.
    const ringValues = new Array<number>(ringCount);
    const mdMin = md[0];
    const mdMax = md[md.length - 1];
    for (let r = 0; r < ringCount; r++) {
      const f = ringCount > 1 ? r / (ringCount - 1) : 0;
      const ringMd = mdMin + f * (mdMax - mdMin);
      // nearest station value (the resample already interpolated logs onto stations)
      let nearest = 0;
      let bestD = Infinity;
      for (let s = 0; s < md.length; s++) {
        const d = Math.abs(md[s] - ringMd);
        if (d < bestD) {
          bestD = d;
          nearest = s;
        }
      }
      ringValues[r] = stationValues[nearest];
    }
    const perVertex = new Array<number>(pos.count);
    for (let r = 0; r < ringCount; r++) {
      for (let k = 0; k < perRing; k++) perVertex[r * perRing + k] = ringValues[r];
    }
    return curveToVertexColors(perVertex, range, layer.transferFn.colormap);
  }, [geometry, traj, layer.logProperty, layer.transferFn.colormap]);

  // Attach / detach the colour attribute when it changes.
  useEffect(() => {
    if (!geometry) return;
    if (vertexColors) {
      geometry.setAttribute("color", new THREE.BufferAttribute(vertexColors, 3));
    } else {
      geometry.deleteAttribute("color");
    }
    geometry.attributes.color && (geometry.attributes.color.needsUpdate = true);
  }, [geometry, vertexColors]);

  useEffect(() => () => geometry?.dispose(), [geometry]);

  const basis: AABB | null = sceneAABB ?? layer.aabb ?? null;
  const planes = useMemo(
    () => (layer.clip && basis ? clipPlanesFor(basis, clip, vex) : []),
    [layer.clip, basis, clip, vex],
  );

  if (!geometry || !traj || !layer.visible) return null;

  const hasColors = vertexColors != null;

  // Hover -> MD/TVD/elevation readout (true depths) into the store (doc 06 §5.3). The picked
  // point is in the EXAGGERATED group space, so divide Z back out before the readout.
  const onMove = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    const p = e.point;
    const truePoint: Vec3 = [p.x, p.y, p.z / (vex || 1)];
    const r = readoutAtPoint(traj, truePoint);
    if (r) setWellReadout({ ...r, wellId: traj.wellId, featureId: traj.featureId, layerId: layer.id });
  };

  return (
    <group scale={[1, 1, vex]} renderOrder={2}>
      <mesh
        geometry={geometry}
        onPointerMove={onMove}
        onPointerOut={() => setWellReadout(null)}
      >
        <meshStandardMaterial
          vertexColors={hasColors}
          color={hasColors ? "#ffffff" : "#f9e2af"}
          roughness={0.5}
          metalness={0.1}
          side={THREE.DoubleSide}
          transparent={layer.opacity < 1}
          opacity={layer.opacity}
          clippingPlanes={planes.length ? planes : null}
        />
      </mesh>
    </group>
  );
}

// Render all visible well layers (doc 06 §5.3, §9.1).
export function WellLayers() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  return (
    <>
      {layerOrder.map((id) => {
        const l = layers[id];
        if (!l || l.kind !== "well") return null;
        return <WellLayer key={id} layer={l} />;
      })}
    </>
  );
}
