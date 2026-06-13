// Surface / fault / solid feature layer (doc 06 §5.2). Loads a binary glTF triangle mesh
// (server-converted from a GeoJSON grid/polygon, geosim/api/features.py) via three's
// GLTFLoader into the Z-up Engineering scene — the mesh vertices are ALREADY Engineering
// metres so it drops straight in with no reprojection (doc 06 §5.2 "already in Engineering
// coordinates"). Surfaces render double-sided; faults render semi-transparent with an edge
// highlight (a wireframe overlay). Picking uses three-mesh-bvh's accelerated raycast so
// hover/identify stays fast on big horizons (doc 06 §5.2).
//
// THREE objects never live in the Zustand store; this component owns the decoded mesh and
// disposes it on unmount / featureId change. The clip box is honoured via material
// clippingPlanes (the same Engineering-metre planes the terrain mesh uses, doc 06 §2.4).

import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import {
  computeBoundsTree,
  disposeBoundsTree,
  acceleratedRaycast,
} from "three-mesh-bvh";
import { useViewer } from "../store";
import type { Layer } from "../lib/layers";
import { clipPlanesFor } from "./clipPlanes";
import type { AABB } from "../lib/volume";

// Wire three-mesh-bvh into BufferGeometry/Mesh once (accelerated raycast for picking).
type BVHGeometry = THREE.BufferGeometry & {
  computeBoundsTree?: typeof computeBoundsTree;
  disposeBoundsTree?: typeof disposeBoundsTree;
};
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

// Parse a .glb ArrayBuffer to its first triangle mesh geometry, building a BVH for picking.
function parseGLB(buffer: ArrayBuffer): Promise<THREE.BufferGeometry | null> {
  const loader = new GLTFLoader();
  return new Promise((resolve) => {
    loader.parse(
      buffer,
      "",
      (gltf) => {
        let geom: THREE.BufferGeometry | null = null;
        gltf.scene.traverse((o) => {
          if (!geom && (o as THREE.Mesh).isMesh) {
            geom = (o as THREE.Mesh).geometry as THREE.BufferGeometry;
          }
        });
        if (geom) {
          const g = geom as BVHGeometry;
          g.computeVertexNormals?.();
          g.computeBoundsTree?.(); // BVH for accelerated picking (doc 06 §5.2)
        }
        resolve(geom);
      },
      () => resolve(null),
    );
  });
}

export function FeatureLayer({ layer }: { layer: Layer }) {
  const clip = useViewer((s) => s.clip);
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const vex = useViewer((s) => s.verticalExaggeration);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);

  // Load + decode the glTF mesh once per featureId (server-converted to Engineering glTF).
  useEffect(() => {
    let alive = true;
    let current: THREE.BufferGeometry | null = null;
    if (!layer.featureId) return;
    fetch(`/features/${encodeURIComponent(layer.featureId)}/geometry`)
      .then((r) => {
        if (!r.ok) throw new Error(`geometry ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => parseGLB(buf))
      .then((g) => {
        if (!alive) {
          (g as BVHGeometry | null)?.disposeBoundsTree?.();
          g?.dispose();
          return;
        }
        current = g;
        setGeometry(g);
      })
      .catch(() => alive && setGeometry(null));
    return () => {
      alive = false;
      (current as BVHGeometry | null)?.disposeBoundsTree?.();
      current?.dispose();
    };
  }, [layer.featureId]);

  const basis: AABB | null = sceneAABB ?? layer.aabb ?? null;
  const planes = useMemo(
    () => (layer.clip && basis ? clipPlanesFor(basis, clip, vex) : []),
    [layer.clip, basis, clip, vex],
  );

  if (!geometry || !layer.visible) return null;

  const isFault = layer.faultStyle === true;
  return (
    <group scale={[1, 1, vex]} renderOrder={1}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          color={isFault ? "#f38ba8" : "#89b4fa"}
          roughness={0.7}
          metalness={0.0}
          side={THREE.DoubleSide}
          transparent={isFault || layer.opacity < 1}
          opacity={isFault ? Math.min(layer.opacity, 0.6) : layer.opacity}
          depthWrite={!isFault}
          clippingPlanes={planes.length ? planes : null}
          clipIntersection={false}
        />
      </mesh>
      {/* Fault edge highlight (doc 06 §5.2): a thin wireframe overlay of the same mesh. */}
      {isFault && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color="#f5c2e7"
            wireframe
            transparent
            opacity={0.35}
            clippingPlanes={planes.length ? planes : null}
          />
        </mesh>
      )}
    </group>
  );
}

// Render all visible surface/fault/isosurface feature layers (doc 06 §5.2, §9.1).
export function FeatureLayers() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  return (
    <>
      {layerOrder.map((id) => {
        const l = layers[id];
        if (!l || (l.kind !== "surface" && l.kind !== "isosurface")) return null;
        return <FeatureLayer key={id} layer={l} />;
      })}
    </>
  );
}
