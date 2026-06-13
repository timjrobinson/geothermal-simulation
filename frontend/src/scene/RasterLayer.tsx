// InSAR deformation raster time-series layer (doc 06 §6). A surface grid draped on the ground
// (Engineering XY + elevation Z), coloured per-vertex by the deformation value of the
// time-slider-selected frame (leading-t frame select, doc 06 §9.4). The slider snaps
// `raster.frameIndex` in the store (snapRasters); this component only re-colours the existing
// geometry when the frame changes — NO geometry rebuild per tick (doc 06 §9.4).
//
// The grid geometry is built once from the surface; the per-vertex colours are swapped on
// frame change via the transfer-fn colormap over the value range. Clip + vex match the
// terrain/feature layers (doc 06 §2.4).

import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { useViewer } from "../store";
import type { Layer } from "../lib/layers";
import { gridToMesh } from "../lib/terrain";
import { resolveColormap, sampleColormap } from "../lib/colormaps";
import { clipPlanesFor } from "./clipPlanes";
import type { AABB } from "../lib/volume";

export function RasterLayer({ layer }: { layer: Layer }) {
  const clip = useViewer((s) => s.clip);
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const vex = useViewer((s) => s.verticalExaggeration);
  const raster = layer.raster;

  // Build the draped grid geometry once (vertex/Z math from lib/terrain.gridToMesh — pure).
  const geometry = useMemo(() => {
    if (!raster) return null;
    const { positions, uvs, indices } = gridToMesh(raster.surface, 1);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    g.setIndex(new THREE.BufferAttribute(indices, 1));
    g.computeVertexNormals();
    return g;
  }, [raster?.surface]);

  // Re-colour on frame / colormap / range change (no geometry rebuild, doc 06 §9.4).
  useEffect(() => {
    if (!geometry || !raster) return;
    const frame = raster.frames[raster.frameIndex];
    if (!frame) return;
    const cm = resolveColormap(layer.transferFn.colormap);
    const lo = layer.transferFn.domainMin;
    const hi = layer.transferFn.domainMax;
    const span = hi - lo || 1;
    const n = frame.length;
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const v = frame[i];
      if (Number.isNaN(v)) {
        colors[i * 3] = 0.3;
        colors[i * 3 + 1] = 0.3;
        colors[i * 3 + 2] = 0.3;
        continue;
      }
      const t = Math.min(1, Math.max(0, (v - lo) / span));
      const [r, g, b] = sampleColormap(cm, t);
      colors[i * 3] = r;
      colors[i * 3 + 1] = g;
      colors[i * 3 + 2] = b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.attributes.color.needsUpdate = true;
  }, [
    geometry,
    raster,
    raster?.frameIndex,
    layer.transferFn.colormap,
    layer.transferFn.domainMin,
    layer.transferFn.domainMax,
  ]);

  useEffect(() => () => geometry?.dispose(), [geometry]);

  const basis: AABB | null = sceneAABB ?? layer.aabb ?? null;
  const planes = useMemo(
    () => (layer.clip && basis ? clipPlanesFor(basis, clip, vex) : []),
    [layer.clip, basis, clip, vex],
  );

  if (!geometry || !raster || !layer.visible) return null;

  return (
    <mesh geometry={geometry} scale={[1, 1, vex]} renderOrder={1}>
      <meshStandardMaterial
        vertexColors
        roughness={0.85}
        metalness={0}
        side={THREE.DoubleSide}
        transparent={layer.opacity < 1}
        opacity={layer.opacity}
        clippingPlanes={planes.length ? planes : null}
      />
    </mesh>
  );
}

// Render all visible raster (InSAR) layers (doc 06 §6, §9.1).
export function RasterLayers() {
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  return (
    <>
      {layerOrder.map((id) => {
        const l = layers[id];
        if (!l || l.kind !== "raster") return null;
        return <RasterLayer key={id} layer={l} />;
      })}
    </>
  );
}
