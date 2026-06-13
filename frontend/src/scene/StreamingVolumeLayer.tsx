// M2+ brick-streaming volume layer (doc 06 §3.4, §7). The LARGE-volume path: instead of one
// resident Data3DTexture (the M1 single-resident fast path, kept in VolumeLayer.tsx for small
// volumes), this maintains a FIXED-VRAM brick-pool atlas + a page-table texture, streams
// bricks by screen-space-error LOD selection (lib/lod.ts) into the atlas via a Web Worker
// (lib/brick.worker.ts), and ray-marches with the page-table-walking shader (lib/brickShaders
// .ts) that falls back to coarser resident levels on a miss (no holes).
//
// All the HARD pure logic lives in unit-tested modules (lib/bricks, lib/lod, lib/brickPool,
// lib/brickDecode); this component is the THREE/R3F wiring: create the atlas + page-table
// textures, drive selection each frame, dispatch worker decodes, upload bricks, and push
// uniforms. The GPU VISUAL result is unverifiable headlessly (see blockers[]).

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useViewer } from "../store";
import type { Layer, BlendMode } from "../lib/layers";
import { makeTransferFnTexture, updateTransferFnTexture } from "../lib/transferFn";
import { BRICK_VOLUME_VERT, BRICK_VOLUME_FRAG, MAX_LEVELS } from "../lib/brickShaders";
import { aabbCenter, aabbSize, type AABB } from "../lib/volume";
import {
  type PyramidSpec,
  brickEdge as edgeOf,
  levelShape,
  levelBrickGrid,
  coarsestLevel,
  volumeAABB,
  brickKey,
  BRICK_SIZE,
} from "../lib/bricks";
import {
  BrickPool,
  type AtlasLayout,
  chooseAtlasGrid,
  atlasDims,
  slotVoxelOrigin,
  makePageTable,
  fillPageTable,
} from "../lib/brickPool";
import { selectBricks, type ViewDesc } from "../lib/lod";

const BLEND_INDEX: Record<BlendMode, number> = { over: 0, additive: 1, mip: 2, minip: 3 };

// VRAM budget: number of brick slots in the atlas (doc 06 §7.2 ~256-512 MB; at 64³ f32 ≈ 1 MB
// per brick, 256 slots ≈ 256 MB). Coarsest level is pinned within this budget.
const ATLAS_SLOTS = 256;

function applyGLBlend(mat: THREE.ShaderMaterial, blend: BlendMode): void {
  mat.transparent = true;
  mat.depthWrite = false;
  mat.depthTest = blend === "over";
  if (blend === "additive") {
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.AddEquation;
    mat.blendSrc = THREE.SrcAlphaFactor;
    mat.blendDst = THREE.OneFactor;
  } else if (blend === "mip") {
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.MaxEquation;
    mat.blendSrc = THREE.OneFactor;
    mat.blendDst = THREE.OneFactor;
  } else if (blend === "minip") {
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.MinEquation;
    mat.blendSrc = THREE.OneFactor;
    mat.blendDst = THREE.OneFactor;
  } else {
    mat.blending = THREE.NormalBlending;
  }
  mat.needsUpdate = true;
}

// Build the PyramidSpec from the layer's meta (level-0 shape/origin/spacing + level count).
function specOf(layer: Layer): PyramidSpec | null {
  const m = layer.meta;
  if (!m) return null;
  return {
    shape0: m.shape,
    origin: m.origin,
    spacing0: m.spacing,
    levels: Math.max(1, m.levels),
    brick: BRICK_SIZE,
  };
}

export function StreamingVolumeLayer({ layer, order }: { layer: Layer; order: number }) {
  const steps = useViewer((s) => s.steps);
  const clip = useViewer((s) => s.clip);
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);

  const tf = layer.transferFn;
  const spec = useMemo(() => specOf(layer), [layer.meta]); // eslint-disable-line react-hooks/exhaustive-deps
  const aabb = useMemo<AABB | null>(() => (spec ? volumeAABB(spec) : null), [spec]);

  const materialRef = useRef<THREE.ShaderMaterial | null>(null);
  const tfTex = useMemo(() => makeTransferFnTexture(tf), []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Atlas + page-table textures + pool (one set per layer, fixed VRAM) ──────────────────
  const rig = useMemo(() => {
    if (!spec) return null;
    const edge = edgeOf(spec);
    const grid = chooseAtlasGrid(ATLAS_SLOTS);
    const layout: AtlasLayout = { brickEdge: edge, grid };
    const [aw, ah, ad] = atlasDims(layout);

    // Atlas 3D texture: f32 RedFormat, NaN-initialised (empty == no-data). NearestFilter on
    // mag avoids inter-brick bleeding (we inset in the shader too); Linear would need padding.
    const atlasData = new Float32Array(aw * ah * ad).fill(NaN);
    const atlas = new THREE.Data3DTexture(
      atlasData as unknown as BufferSource,
      aw,
      ah,
      ad,
    );
    atlas.format = THREE.RedFormat;
    atlas.type = THREE.FloatType;
    atlas.minFilter = THREE.LinearFilter;
    atlas.magFilter = THREE.LinearFilter;
    atlas.wrapS = atlas.wrapT = atlas.wrapR = THREE.ClampToEdgeWrapping;
    atlas.unpackAlignment = 1;
    atlas.needsUpdate = true;

    // Per-level brick grids + voxel extents (index == level).
    const levels = spec.levels;
    const levelGrids: Array<readonly [number, number, number]> = [];
    const levelVoxels: Array<readonly [number, number, number]> = [];
    for (let l = 0; l < levels; l++) {
      levelGrids.push(levelBrickGrid(spec, l)); // [gz,gy,gx]
      const [nz, ny, nx] = levelShape(spec, l);
      levelVoxels.push([nz, ny, nx]);
    }
    const pageTable = makePageTable(levelGrids);

    // Page-table 2D texture: width = power-of-two >= sqrt(total), R32F (slot index or -1).
    const total = pageTable.data.length;
    const pageW = Math.max(1, 1 << Math.ceil(Math.log2(Math.ceil(Math.sqrt(total)) || 1)));
    const pageH = Math.max(1, Math.ceil(total / pageW));
    const pageData = new Float32Array(pageW * pageH).fill(-1);
    const pageTex = new THREE.DataTexture(
      pageData,
      pageW,
      pageH,
      THREE.RedFormat,
      THREE.FloatType,
    );
    pageTex.magFilter = THREE.NearestFilter;
    pageTex.minFilter = THREE.NearestFilter;
    pageTex.needsUpdate = true;

    const pool = new BrickPool(layout);
    return {
      layout,
      atlas,
      atlasData,
      atlasDims: [aw, ah, ad] as [number, number, number],
      pageTable,
      pageTex,
      pageData,
      pageW,
      pageH,
      pool,
      levelGrids,
      levelVoxels,
    };
  }, [spec]);

  // ── Worker (one per layer) ──────────────────────────────────────────────────────────────
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef<Set<string>>(new Set());
  const reqSeq = useRef(0);
  const reqMap = useRef<Map<number, string>>(new Map());

  useEffect(() => {
    if (!rig || !spec || !layer.meta) return;
    let worker: Worker;
    try {
      worker = new Worker(new URL("../lib/brick.worker.ts", import.meta.url), {
        type: "module",
      });
    } catch {
      // Worker construction can fail in non-browser/test contexts; streaming then no-ops and
      // the coarsest-level fallback (uploaded eagerly below) keeps the volume visible.
      return;
    }
    workerRef.current = worker;
    worker.postMessage({ type: "config", maxCachedLevels: 4 });

    const edge = rig.layout.brickEdge;
    const [aw, ah] = rig.atlasDims;
    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data as {
        type: string;
        reqId: number;
        bz: number;
        by: number;
        bx: number;
        level: number;
        empty?: boolean;
        data?: Float32Array | null;
      };
      if (msg.type === "error") {
        const k = reqMap.current.get(msg.reqId);
        if (k) {
          pendingRef.current.delete(k);
          reqMap.current.delete(msg.reqId);
        }
        return;
      }
      if (msg.type !== "brick") return;
      const k = reqMap.current.get(msg.reqId);
      reqMap.current.delete(msg.reqId);
      if (k) pendingRef.current.delete(k);
      const key = `${msg.level}/0/${msg.bz}/${msg.by}/${msg.bx}`;
      const pinned = msg.level === coarsestLevel(spec);
      if (msg.empty || !msg.data) {
        // empty brick: still admit so we don't re-request, but no atlas upload (page -1).
        return;
      }
      const entry = rig.pool.admit(key, pinned);
      // Upload the brick into its atlas slot via a sub-image copy into the CPU mirror, then
      // flag the whole atlas for re-upload (THREE re-uploads the full Data3DTexture image).
      const [ox, oy, oz] = slotVoxelOrigin(rig.layout, entry.slot);
      const src = msg.data;
      for (let z = 0; z < edge; z++)
        for (let y = 0; y < edge; y++) {
          const srcRow = (z * edge + y) * edge;
          const dstRow = ((oz + z) * ah + (oy + y)) * aw + ox;
          rig.atlasData.set(src.subarray(srcRow, srcRow + edge), dstRow);
        }
      rig.atlas.needsUpdate = true;
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
      pendingRef.current.clear();
      reqMap.current.clear();
    };
  }, [rig, spec, layer.meta]);

  // ── Material ───────────────────────────────────────────────────────────────────────────
  const material = useMemo(() => {
    if (!rig || !spec || !aabb) return null;
    const [aw, ah, ad] = rig.atlasDims;
    const levelOffset = new Array<number>(MAX_LEVELS).fill(0);
    rig.pageTable.levelOffset.forEach((v, i) => {
      if (i < MAX_LEVELS) levelOffset[i] = v;
    });
    const levelGrid = new Array(MAX_LEVELS)
      .fill(0)
      .map(() => new THREE.Vector3(1, 1, 1));
    rig.levelGrids.forEach((g, i) => {
      if (i < MAX_LEVELS) levelGrid[i].set(g[0], g[1], g[2]); // [gz,gy,gx]
    });
    const levelVoxels = new Array(MAX_LEVELS)
      .fill(0)
      .map(() => new THREE.Vector3(1, 1, 1));
    rig.levelVoxels.forEach((v, i) => {
      if (i < MAX_LEVELS) levelVoxels[i].set(v[2], v[1], v[0]); // shader wants [nx,ny,nz]
    });

    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: BRICK_VOLUME_VERT,
      fragmentShader: BRICK_VOLUME_FRAG,
      transparent: true,
      depthWrite: false,
      side: THREE.BackSide,
      uniforms: {
        uAtlas: { value: rig.atlas },
        uPageTable: { value: rig.pageTex },
        uTransferFn: { value: tfTex },
        uBoxMin: { value: new THREE.Vector3(...aabb.min) },
        uBoxMax: { value: new THREE.Vector3(...aabb.max) },
        uClipMin: { value: new THREE.Vector3(...aabb.min) },
        uClipMax: { value: new THREE.Vector3(...aabb.max) },
        uCameraPos: { value: new THREE.Vector3() },
        uDomainMin: { value: tf.domainMin },
        uDomainMax: { value: tf.domainMax },
        uLog: { value: tf.scaling === "log" ? 1 : 0 },
        uOpacityGain: { value: tf.opacity * layer.opacity },
        uSteps: { value: steps },
        uRefStep: { value: 1.0 },
        uBlend: { value: BLEND_INDEX[layer.blend] },
        uBrickEdge: { value: rig.layout.brickEdge },
        uAtlasGrid: { value: new THREE.Vector3(...rig.layout.grid) },
        uAtlasDim: { value: new THREE.Vector3(aw, ah, ad) },
        uPageW: { value: rig.pageW },
        uPageH: { value: rig.pageH },
        uMaxLevel: { value: coarsestLevel(spec) },
        uLevelOffset: { value: levelOffset },
        uLevelGrid: { value: levelGrid },
        uLevelVoxels: { value: levelVoxels },
      },
    });
    applyGLBlend(mat, layer.blend);
    materialRef.current = mat;
    return mat;
  }, [rig, spec, aabb, tfTex]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (materialRef.current) applyGLBlend(materialRef.current, layer.blend);
  }, [layer.blend]);

  // Re-bake LUT on TF edits (no refetch — doc 06 §9.2).
  useMemo(() => {
    if (tfTex) updateTransferFnTexture(tfTex, tf);
  }, [tf, tfTex]);

  // Dispose GPU resources on unmount (doc 06 §7.5 — pooled atlas isn't auto-disposed).
  useEffect(() => {
    return () => {
      rig?.atlas.dispose();
      rig?.pageTex.dispose();
      tfTex?.dispose();
    };
  }, [rig, tfTex]);

  const { center, boxSize } = useMemo(() => {
    if (!aabb) return { center: [0, 0, 0] as const, boxSize: [1, 1, 1] as const };
    return { center: aabbCenter(aabb), boxSize: aabbSize(aabb) };
  }, [aabb]);

  const clipBasis: AABB = sceneAABB ?? aabb ?? { min: [0, 0, 0], max: [1, 1, 1] };

  // Throttle LOD selection to a few times/sec (selection is cheap but uploads are not).
  const lastSelect = useRef(0);

  useFrame(() => {
    const mat = materialRef.current;
    if (!mat || !rig || !spec || !aabb) return;
    const u = mat.uniforms;
    (u.uCameraPos.value as THREE.Vector3).copy(camera.position);
    u.uDomainMin.value = tf.domainMin;
    u.uDomainMax.value = tf.domainMax;
    u.uLog.value = tf.scaling === "log" ? 1 : 0;
    u.uOpacityGain.value = tf.opacity * layer.opacity;
    u.uSteps.value = steps;
    u.uBlend.value = BLEND_INDEX[layer.blend];
    const span = aabbSize(aabb);
    u.uRefStep.value = Math.max(Math.min(span[0], span[1], span[2]) / steps, 1e-3);

    // Resolve the clip box (scene-AABB fractions -> Engineering metres) for both the shader
    // clip and the LOD clip-culling.
    let clipBox: AABB;
    if (layer.clip) {
      const cspan = aabbSize(clipBasis);
      clipBox = {
        min: [
          clipBasis.min[0] + clip.min[0] * cspan[0],
          clipBasis.min[1] + clip.min[1] * cspan[1],
          clipBasis.min[2] + clip.min[2] * cspan[2],
        ],
        max: [
          clipBasis.min[0] + clip.max[0] * cspan[0],
          clipBasis.min[1] + clip.max[1] * cspan[1],
          clipBasis.min[2] + clip.max[2] * cspan[2],
        ],
      };
    } else {
      clipBox = aabb;
    }
    (u.uClipMin.value as THREE.Vector3).set(...clipBox.min);
    (u.uClipMax.value as THREE.Vector3).set(...clipBox.max);

    // LOD selection a few times/sec.
    const now = performance.now();
    if (now - lastSelect.current < 200) return;
    lastSelect.current = now;

    // View descriptor (Engineering metres). Frustum planes from the camera; FOV/viewport for
    // the screen-space-error projection (doc 06 §7.3).
    const planes: Array<readonly [number, number, number, number]> = [];
    const projScreen = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse,
    );
    const frustum = new THREE.Frustum().setFromProjectionMatrix(projScreen);
    for (const pl of frustum.planes) {
      planes.push([pl.normal.x, pl.normal.y, pl.normal.z, pl.constant]);
    }
    const persp = camera as THREE.PerspectiveCamera;
    const view: ViewDesc = {
      eye: [camera.position.x, camera.position.y, camera.position.z],
      planes,
      fovYRad: persp.isPerspectiveCamera ? (persp.fov * Math.PI) / 180 : undefined,
      viewportH: size.height,
    };

    const sel = selectBricks(spec, view, {
      targetVoxelPx: 1.5,
      maxBricks: Math.max(0, rig.pool.capacity - 8),
      clip: layer.clip ? clipBox : undefined,
    });

    // Touch wanted resident bricks; dispatch decodes for missing ones (coarsest pinned).
    rig.pool.tick();
    const wanted = new Set<string>();
    const worker = workerRef.current;
    for (const s of sel) {
      const key = brickKey(s.addr);
      wanted.add(key);
      if (rig.pool.has(key)) {
        rig.pool.touch(key);
        continue;
      }
      if (pendingRef.current.has(key)) continue;
      if (!worker) continue;
      // Bound concurrent in-flight requests (doc 06 §7.5 throttle).
      if (pendingRef.current.size >= 24) break;
      pendingRef.current.add(key);
      const reqId = ++reqSeq.current;
      reqMap.current.set(reqId, key);
      worker.postMessage({
        type: "decode",
        reqId,
        id: layer.datasetId,
        property: layer.property ?? layer.meta!.property,
        level: s.addr.level,
        t: s.addr.t,
        bz: s.addr.bz,
        by: s.addr.by,
        bx: s.addr.bx,
        brickEdge: rig.layout.brickEdge,
      });
    }
    // Evict unwanted, unpinned bricks proactively (frees slots; coarsest stays pinned).
    rig.pool.evictExcept(wanted);

    // Rebuild the page-table texture from the resident set.
    fillPageTable(rig.pageTable, rig.pool.list());
    rig.pageData.fill(-1);
    rig.pageData.set(rig.pageTable.data as unknown as ArrayLike<number>);
    // Int32 slot indices -> float texels (exact for slot counts < 2^24).
    for (let i = 0; i < rig.pageTable.data.length; i++) {
      rig.pageData[i] = rig.pageTable.data[i];
    }
    rig.pageTex.needsUpdate = true;
  });

  if (!material || !aabb || !layer.visible || !spec) return null;

  return (
    <mesh
      position={center as unknown as [number, number, number]}
      renderOrder={1 + order}
    >
      <boxGeometry args={[boxSize[0], boxSize[1], boxSize[2]]} />
      <primitive object={material} attach="material" />
    </mesh>
  );
}
