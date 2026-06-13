// Terrain ground-surface layer (doc 06 §6, doc 01 §6). Renders a terrain `Layer`'s
// SurfaceGrid as a PlaneGeometry-style mesh DISPLACED IN Z in the Engineering Frame: each
// grid sample is a vertex at (X, Y, elevation) in Engineering metres, so the surface lands
// exactly in the ENU scene with NO reprojection at render time (doc 06 §6.1) and the
// subsurface volume layers hang BENEATH it because they share the frame and Z is elevation.
//
// Vertical exaggeration (doc 06 §2.3) is a RENDER-ONLY transform: applied here as a mesh Z
// scale (not baked into the grid data) so picking/readouts can divide it back out. The clip
// box (doc 06 §2.4) is honoured via three.js material clippingPlanes (the surface is a
// regular mesh — hardware clip planes carve it, unlike the in-shader volume clip).
//
// Shading: shaded-relief by default (a lit MeshStandardMaterial with computed normals,
// doc 06 §6.2). Optional online XYZ basemap tiles draped via render-time
// engineering→CRS→lat/lon UVs are structured for (the per-vertex grid UVs are emitted by
// gridToMesh) but the tile fetch/drape itself is deferred (doc 06 §6.2 — offline-safe
// default is the shaded-relief DEM).

import { useMemo } from "react";
import * as THREE from "three";
import { useViewer } from "../store";
import type { Layer } from "../lib/layers";
import { gridToMesh, type SurfaceGrid } from "../lib/terrain";
import { aabbSize, type AABB } from "../lib/volume";

// Build a THREE.BufferGeometry from a SurfaceGrid (vertex/Z math lives in lib/terrain.ts —
// pure + unit-tested). vex=1 here: vertical exaggeration is applied as a mesh scale below
// so it stays a render-only transform that the geometry cache doesn't have to rebuild on.
function buildGeometry(grid: SurfaceGrid): THREE.BufferGeometry {
  const { positions, uvs, indices } = gridToMesh(grid, 1);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  g.computeVertexNormals(); // shaded-relief lighting (doc 06 §6.2)
  return g;
}

// Map the global clip-box fractions (relative to the scene AABB) to Engineering-metre
// THREE.Plane[] that carve the terrain mesh. Mirrors the volume layer's clip basis so a
// single box cuts terrain + subsurface together (doc 06 §2.4 "cut a box that slices terrain
// + subsurface"). Planes are in the mesh's parent (world/Engineering) space; the mesh Z
// scale (vex) is applied to the geometry vertices, so the clip Z bound is scaled to match.
function clipPlanes(
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
  // Each THREE.Plane keeps the half-space on its +normal side: normal points INTO the box.
  return [
    new THREE.Plane(new THREE.Vector3(1, 0, 0), -lo[0]),
    new THREE.Plane(new THREE.Vector3(-1, 0, 0), hi[0]),
    new THREE.Plane(new THREE.Vector3(0, 1, 0), -lo[1]),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), hi[1]),
    new THREE.Plane(new THREE.Vector3(0, 0, 1), -lo[2]),
    new THREE.Plane(new THREE.Vector3(0, 0, -1), hi[2]),
  ];
}

export function TerrainLayer({ layer }: { layer: Layer }) {
  const clip = useViewer((s) => s.clip);
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const vex = useViewer((s) => s.verticalExaggeration);

  const grid = layer.surface ?? null;

  // Rebuild geometry only when the surface grid changes (not on vex / clip edits).
  const geometry = useMemo(() => (grid ? buildGeometry(grid) : null), [grid]);

  // The clip basis is the union scene AABB (same basis the ClipBox gizmo + volumes use), so
  // the terrain is carved by the very same box. Falls back to the terrain's own AABB.
  const basis: AABB | null = sceneAABB ?? layer.aabb ?? null;
  const planes = useMemo(
    () => (layer.clip && basis ? clipPlanes(basis, clip, vex) : []),
    [layer.clip, basis, clip, vex],
  );

  if (!geometry || !layer.visible) return null;

  return (
    <mesh scale={[1, 1, vex]} renderOrder={0}>
      <primitive object={geometry} attach="geometry" />
      <meshStandardMaterial
        color="#8a7f6d"
        roughness={0.95}
        metalness={0.0}
        side={THREE.DoubleSide}
        transparent={layer.opacity < 1}
        opacity={layer.opacity}
        clippingPlanes={planes.length ? planes : null}
        clipIntersection={false}
        flatShading={false}
      />
    </mesh>
  );
}

// Render all visible terrain layers (doc 06 §9.1). Drawn beneath the volume layers because
// terrain is prepended to the bottom of layerOrder (store.addTerrainLayer) and renderOrder
// 0 < the volumes' renderOrder.
export function TerrainLayers() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  return (
    <>
      {layerOrder.map((id) => {
        const l = layers[id];
        if (!l || l.kind !== "terrain") return null;
        return <TerrainLayer key={id} layer={l} />;
      })}
    </>
  );
}
