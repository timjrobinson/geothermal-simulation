# 06 — Visualization Engine (the 3D viewer)

> Parent: `OVERVIEW.md` §7, §10 row 6. This doc specifies the browser 3D engine —
> scene graph, GPU volume ray-marching, slicing, isosurfaces, terrain bridge,
> well paths, microseismic, the time slider, and the performance/LOD strategy.
>
> **Binds to (does not redefine):**
> - **Doc 01 (locked)** — the Engineering Frame (ENU, metres, Z-up, floating origin),
>   `SpatialFrame`, terrain/basemap rules. The viewer is a *consumer* of this frame.
> - **Doc 04 (parallel)** — Zarr brick/pyramid storage and the `tile / slice / sample /
>   isosurface` HTTP API. This doc *calls* those endpoints; it assumes a contract
>   (§12) and flags what it needs.
> - **Doc 02 (parallel)** — property-model & feature schemas, the property-type
>   registry (canonical unit, default colourmap, default scaling, display range).
>   The viewer reads that registry to seed transfer functions; it does not define it.

---

## 0. Scope & non-goals

**In scope:** everything that runs in the browser to turn streamed bricks, meshes,
point clouds and rasters into an interactive, georeferenced 3D scene; the GPU
pipelines (volume, slice, isosurface, terrain); the layer/transfer-function/time UX
model; client state (Zustand); and the performance budget that decides client-side
vs server-side rendering.

**Out of scope (owned elsewhere):** how bricks are chunked/pyramided and served
(doc 04); how properties are typed/normalized (doc 01 §5, doc 02); fusion/cross-plot
math (doc 07). We render what those produce.

---

## 1. Rendering tech choice

### 1.1 Framework — confirmed

**Three.js via react-three-fiber (R3F)**, per OVERVIEW §5. R3F gives us a declarative
React reconciler over the Three.js scene graph, so layers map cleanly to components
and Zustand state drives the scene without imperative glue. Supporting libs:

| Concern | Library |
|---|---|
| Scene graph / reconciler | three + @react-three/fiber |
| Camera controls, gizmos, helpers | @react-three/drei (`CameraControls`, `Bvh`) |
| Raycast acceleration (picking large meshes) | three-mesh-bvh |
| glTF load (horizons, faults, solids) | three `GLTFLoader` (+ `meshopt`/Draco decode) |
| Point clouds (microseismic, optional Potree) | three `Points` MVP → `3D Tiles`/Potree later |
| Analysis panels (log tracks, cross-plots) | Observable Plot / D3 (DOM, not WebGL) |

### 1.2 WebGL2 vs WebGPU — **recommendation: WebGPU-capable abstraction, WebGL2 as the shipping default**

This is the highest-leverage rendering decision, because **volume rendering is the
centerpiece** and the two backends differ most exactly there.

| | **WebGL2** | **WebGPU** |
|---|---|---|
| 3D textures (`Data3DTexture`) | yes (sampler3D) | yes, plus storage textures |
| Compute shaders | **no** (fake via fragment passes) | **yes** — marching cubes, brick transforms, histogram, slice resampling on-GPU |
| Volume ray-march | works (fragment shader) | works, cleaner; can split work across compute+render |
| Multi-volume compositing | doable but shader-juggling | bind-groups make N-volume binding far cleaner |
| Browser support (mid-2026) | universal | Chrome/Edge stable, Safari 18+ stable, Firefox shipping; still ~not-everywhere |
| Three.js path | `WebGLRenderer` (mature, all examples) | `WebGPURenderer` + TSL (node materials), maturing fast |

**Stance:**

1. **Ship on WebGL2 for the MVP.** It is universal, every Three.js volume example
   targets it, and a fragment-shader ray-marcher fully covers Phase 1–4. The
   subsurface, single-user, R&D audience is on modern desktop GPUs, but we do not
   want a hard WebGPU gate on day one.
2. **Architect the renderer behind a thin `RenderBackend` seam** so the volume,
   slice and isosurface passes are expressed once and can target either backend.
   In practice: write materials in Three's **TSL (node) system** where feasible so
   they compile to GLSL *and* WGSL, and keep a hand-written GLSL ray-marcher as the
   guaranteed WebGL2 path.
3. **Enable WebGPU when present** (feature-detect `navigator.gpu`) to unlock
   **compute** for the things WebGL2 forces onto the server or onto fragment hacks:
   client-side **marching-cubes isosurfaces**, **on-GPU slice resampling** from
   bricks, brick→3D-texture transforms, and histogram/transfer-function previews.
   Same scene, faster/heavier client path.
4. **Fallback ladder** (never a blank screen):
   `WebGPU compute path → WebGL2 fragment path → server-side render (doc 04 slice/
   isosurface/image endpoints) → 2D-only basemap + slices`.

> **Net:** WebGL2 is the floor and the default; WebGPU is a detected accelerator that
> moves isosurfacing and resampling from server/fragment-hacks to client compute.
> **Resolved (DECISIONS.md):** render backend = **WebGL2 floor + WebGPU progressive
> enhancement**. We may require a WebGL2-class GPU; WebGPU is never a hard gate. WebGPU
> compute-path detail lives in the **Appendix (§13, later phase)** so it does not
> dominate the first build.

### 1.3 Renderer staging — build order (resolves critique #17)

The renderer ships in stages; the **first viewer milestone is deliberately simple** so
we prove the end-to-end path before building the streaming machinery.

| Stage | Renderer scope | Volume backing |
|---|---|---|
| **M1 (first proof)** | **one resident volume** ray-marched + **orthogonal slice** + **clip box** + **transfer function**. WebGL2 only. No streaming, no LOD, no atlas. | a **single resident `Data3DTexture`** at a modest size (≤ a level that fits one upload, e.g. the coarse pyramid level or a ≤256³ crop) |
| **M2+ (later)** | **brick-pool / page-table virtual-texturing + octree-LOD streaming** (§3.4): coarse-first refinement, fixed-VRAM atlas, screen-space-error LOD. | streamed bricks into a fixed brick-pool atlas |

- **M1 is the proof:** load one property volume into a single `Data3DTexture`, march it,
  cut it with one orthogonal slice that samples the *same* texture, clip with the box,
  and drive a transfer function. This validates the Z-up frame, the shader, and the
  Zarr→GPU path with the least moving parts.
- The **brick-pool / page-table / virtual-texturing / octree-LOD** design in §3.4 is
  **fully kept but explicitly LATER** — it is *not* part of the first proof. Treat §3.4
  as the M2+ design, not an M1 requirement.
- **Early-validation SPIKE (critique #5/#17):** the M1 milestone must **prove browser
  Zarr v3 + Blosc/zstd decode maturity** (doc 02 §10.3 flags the same spike). If the JS
  Zarr v3 + Blosc/zstd decoder is not ready, **fall back to server-side decode-to-raw**
  via the doc 04 brick endpoint (`GET /property-models/{id}/bricks/...`) using the **same
  chunk addressing** — so the on-disk layout (doc 02 §10.2) is unaffected; only *who
  decodes* changes. The single-resident-volume M1 works either way.

---

## 2. Scene graph & the Engineering-Frame camera

### 2.1 Consuming the floating-origin ENU frame (doc 01)

The viewer's world space **is the Engineering Frame**: right-handed ENU, metres,
Z-up, origin at the project anchor. This is the single most important bridge to doc 01
and it costs us one adjustment, because **Three.js is Y-up by default**.

We adopt **Z-up at the scene root** rather than rotating every dataset:

```ts
// one-time, at the <Canvas> root
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);   // ENU: Z is up
// camera.up = (0,0,1); controls polar axis = Z
```

Then **every position fed to the GPU is already Engineering XYZ in metres** — no
per-object transform, no CRS coordinates ever reach the GPU. Because doc 01 guarantees
a floating origin (coords in the ±tens-of-km range), **float32 is safe**: vertices,
3D-texture box corners, and well vertices all sit comfortably inside ~mm precision.

```
SceneRoot (Z-up, Engineering metres)
├── <CameraRig>            EngineeringFrameCamera + CameraControls
├── <Lights>              hemispheric + key (subsurface look-dev)
├── <ClippingBox>         6 user planes, drives renderer.clippingPlanes
├── <TerrainLayer>        DEM mesh + draped basemap (georef mode only)
├── <VolumeLayers>        N × <VolumeLayer> (ray-marched bricks)
├── <SliceLayers>         orthogonal planes + fence/cross-section meshes
├── <IsosurfaceLayers>    marching-cubes meshes (client or server)
├── <FeatureLayers>       horizons / faults / solids (glTF)
├── <WellLayers>          well tubes + log color-mapping + markers
├── <PointCloudLayers>    microseismic (time-animated)
├── <GlyphLayers>         vector glyphs / streamlines (EM/MT/flow)
└── <Overlays>            axes, scale bar, north arrow, ROI box, depth ruler
```

R3F renders this tree declaratively from Zustand state (§10): a layer's visibility,
opacity, transfer function and time are props; toggling a layer mounts/unmounts a
subtree.

### 2.2 The Engineering-Frame camera & subsurface navigation

- **Controls:** drei `CameraControls` (orbit + truck + dolly with smooth damping),
  **target-centric** so users orbit a point of interest (a well, an anomaly), not the
  origin. Perspective camera default; **orthographic toggle** for measurement/section
  work (no perspective foreshortening when reading a cross-section).
- **Subsurface-aware framing:** because Z is depth-positive-up and the ROI extends
  *down* to `depthRange.zmin` (e.g. −8000 m, doc 01), default framing tilts the camera
  to look obliquely down into the volume, with near/far planes derived from the ROI
  diagonal. "Frame ROI", "Frame selection", "Top/Front/Side/Section" preset views.
- **No gimbal lock at vertical:** allow polar angle through nadir for map-down views.

### 2.3 Depth (vertical) exaggeration

A scene-graph scale on Z applied **above the camera target math**, not baked into data:

```ts
sceneRoot.scale.set(1, 1, verticalExaggeration);  // e.g. 1.0 … 5.0
```

- Exaggeration is a **render-only** transform. Picking, measurements and readouts
  divide Z back out so reported depths/coordinates stay true (a 2× exaggerated horizon
  still reports its real elevation).
- Applied at the root so terrain, volumes, slices, wells and seismic all stretch
  together and stay registered. UI: a slider (1×–10×) plus "reset to 1×".

### 2.4 Clipping box

A user-draggable **axis-aligned (and optionally rotated) clipping box** that maps to
`THREE.Plane[]` on `renderer.clippingPlanes` (global) and per-material
`clippingPlanes`:

- Drives **all** layers uniformly: meshes are hardware-clipped; the **volume
  ray-marcher clips in-shader** by intersecting the ray with the box (it must, since
  hardware clip planes don't carve a fragment ray's samples).
- Modes: **clip** (hide outside) and **section/cap** (show the cut face, optionally
  cap volumes with a colored slice so the box face reads as a real cross-section).
- "Exploded layers" mode (OVERVIEW §7) = per-layer Z offset stack, a sibling of the
  clip box, for pulling stacked volumes/horizons apart vertically.

---

## 3. Volume rendering (the centerpiece)

### 3.1 Pipeline overview

GPU **single-pass ray-marching** of a property volume held in a `Data3DTexture`,
shaded through a per-property **transfer function** (colormap + opacity), composited
front-to-back. A box mesh (the volume's Engineering-frame AABB) is the proxy geometry;
the fragment shader marches the ray from the box's front face to its exit.

```glsl
// fragment, per pixel — WebGL2 GLSL sketch (WGSL equivalent under WebGPU)
vec3  ro = cameraPosLocal;                 // ray origin in volume-local [0,1]^3
vec3  rd = normalize(vDirLocal);           // ray dir
vec2  t  = intersectBox(ro, rd, clipMin, clipMax);   // box ∩ user clip box
float t0 = max(t.x, 0.0), t1 = t.y;
vec4  acc = vec4(0.0);
float dt  = stepSize / float(MAX_STEPS);   // adaptive vs quality budget
for (int i = 0; i < MAX_STEPS; ++i) {
    float s = t0 + float(i) * dt; if (s > t1) break;
    vec3  p   = ro + rd * s;                       // sample position
    float raw = texture(uVolume, p).r;             // scalar property
    if (raw == uNoData) continue;                  // masked voxels
    float v   = applyScaling(raw, uLogScale, uRange);   // log/linear → [0,1]
    vec4  c   = texture(uTransferFn, vec2(v, 0.5)); // 1D LUT: rgb + a
    c.a      *= uOpacityGain * dt * uReferenceStep; // opacity-correct for step size
    acc.rgb  += (1.0 - acc.a) * c.a * c.rgb;        // front-to-back compositing
    acc.a    += (1.0 - acc.a) * c.a;
    if (acc.a > 0.98) break;                        // early ray termination
}
gl_FragColor = acc;
```

Key correctness details:

- **No-data / masking:** volumes carry a sentinel or a companion mask channel (doc 02);
  masked samples are skipped so empty ROI regions stay transparent.
- **Opacity correction** for variable step size keeps appearance stable across LOD.
- **Pre-integrated transfer functions** (optional optimization) reduce step count for
  sharp colormaps without banding.
- **Gradient-based shading** (optional) — on-the-fly central differences for a lit
  isosurface-like look on a chosen iso value, toggle per layer.

### 3.2 Transfer functions (per property)

A transfer function = **1D colormap LUT × opacity curve**, baked to a small
`DataTexture` (e.g. 256×1 RGBA) the shader samples.

- **Seeded from the property-type registry** (doc 01 §5 / doc 02): canonical unit,
  default colormap, default log/linear scaling, default display range. So resistivity
  arrives log-scaled with a sensible Ω·m range and an appropriate colormap *without
  the user touching anything*.
- **Editable** in the layer panel (§9): colormap picker, opacity curve editor
  (control points over the property histogram), domain min/max with log toggle,
  invert, and "isolate band" (window the opacity to a value range — pulls out, e.g.,
  the conductive geothermal anomaly).
- The **value histogram** behind the editor comes from doc 04's `sample`/stats endpoint
  (or a client compute pass under WebGPU).

### 3.3 Multi-volume compositing / blending

Multiple property volumes (resistivity, density, velocity, a fused favorability field)
must co-render in one frame. Options, selectable per scene:

| Mode | How | Use |
|---|---|---|
| **Separate ray-marches, alpha over** | render each VolumeLayer; alpha-composite by layer order | default; cheap; works on WebGL2 |
| **Multi-volume single march** | one shader binds K 3D textures + K transfer fns, blends per-sample (over / max-intensity / additive) | sharper co-registration; cleaner under WebGPU bind groups |
| **Derived/fused volume** | doc 07 produces one volume server-side; we render it as a single layer | favorability, uncertainty, clusters |

Blend operators exposed: **over, additive, max-intensity-projection (MIP), min-IP**.
MIP is the standard "show me the hottest/most-conductive surface" view.

### 3.4 Volumes larger than GPU memory — brick/LOD streaming (binds doc 04) — **M2+ (later stage)**

> **Staging (critique #17):** everything in §3.4 is the **M2+** renderer (see §1.3). The
> **M1 first proof uses a single resident `Data3DTexture`** — no brick pool, no page
> table, no streaming. The advanced design below is kept but is explicitly *not* part of
> the first milestone.

A full ROI at fine resolution easily exceeds GPU 3D-texture limits (WebGL2 guarantees
only 256³ `MAX_3D_TEXTURE_SIZE`; 512³–1024³ on real desktop GPUs) and VRAM. We rely on
doc 04's **octree-compatible multiresolution pyramid** and do **virtual-texture / brick
streaming** client-side:

```
View change ──▶ pick LOD per octree node (screen-space error / dist) 
            ──▶ request needed bricks (doc 04 §6 addressing):
                  GET /property-models/{id}/bricks/{level}/{t}/{bz}/{by}/{bx}
                  (alias of the Zarr chunk key <property>/<level>/c/<bz>/<by>/<bx>, doc 02 §10.2)
            ──▶ decode → upload into a brick-pool 3D texture atlas (fixed VRAM)
            ──▶ update page-table 3D texture (node → atlas slot)
   ray-march ──▶ shader walks page table, samples resident bricks,
                 falls back to coarser resident level on miss (no holes)
```

- **Fixed VRAM budget:** a brick **pool atlas** (e.g. one `Data3DTexture` of fixed size,
  say 16–32 bricks of 64³–128³) + a **page-table** texture; LRU eviction. We never
  allocate per-volume textures unbounded.
- **Multiresolution / coarse-first:** always keep the coarsest level fully resident so
  the volume is *never* blank; refine visible bricks progressively (a clear "loading
  detail" feel, not pop-in holes).
- **LOD selection:** screen-space-error driven (project brick size to pixels) plus
  distance and clip-box culling — only bricks intersecting the view frustum *and* the
  clip box, near enough to matter, are requested.
- **Brick contract (now resolved — doc 04 §6 is authoritative):** brick = Zarr chunk,
  **64³ cubic** (doc 04 §4.2), addressed `(id, level, t, bz, by, bx)` == the Zarr chunk
  path `<property>/<level>/c/<bz>/<by>/<bx>` (doc 02 §10.2). Encoding = **Blosc+zstd**,
  float32 canonical; **NaN fill** is the no-data convention (doc 02 §10.2). Per-level
  extents + value ranges come from the `multiscales` block + catalog stats (doc 04 §5/§9).
  We do not restate a divergent scheme; §12 just lists what the viewer reads.

---

## 4. Slicing & cross-sections

Three primitives, all sampling the *same* property field:

| Slice type | Geometry | Default sampling |
|---|---|---|
| **Orthogonal planes** | X / Y / Z planes, draggable along their axis | client-side from the resident volume (M1) / resident bricks (M2+) |
| **Arbitrary cross-section** | a tilted plane or a polyline-swept vertical curtain | server `slice` endpoint (`POST /property-models/{id}/slice`, doc 04 §9) |
| **Fence diagram** | a set of connected vertical panels along a path | server `slice` per panel, draped as textured quads |

### 4.1 Sampling strategy — client-side vs server-side

- **Client-side (preferred when the volume is resident):** an orthogonal/arbitrary plane
  is a quad whose fragment shader **samples the same 3D texture** as the ray-marcher (the
  single resident `Data3DTexture` in M1; the brick page-table texture in M2+) and applies
  the same transfer function. Zero extra fetch, instant drag, perfectly registered with
  the volume. This is the default for orthogonal planes.
- **Server-side (`POST /property-models/{id}/slice`, doc 04 §9):** for **arbitrary**
  planes/fences, for volumes **not loaded client-side** (too big / not the active layer),
  or for **publication-quality** native-resolution sections, request a resampled 2D
  array from the backend and drape it as a textured quad. Backend samples the native Zarr
  at full res regardless of client LOD — sharper than the client path.
- **Decision:** **client-side for the common interactive case, server-side for arbitrary
  geometry and full-resolution.** A per-slice "HQ" button forces the server path.

> **Slice contract (doc 04 §9.3 is authoritative — do not restate a divergent shape):**
> the endpoint is **`POST /property-models/{id}/slice`** with a `SliceRequest`
> (`plane` ∈ x/y/z/arbitrary, `position` or `origin`+`normal`, `level`, `t`, `bounds`,
> `encoding`). **Slices default to raw float32 to the client** (`encoding:"f32"`, doc 04
> Resolved / DECISIONS.md) so the client applies the live transfer function — keeping
> slice and volume colours locked. The server-rendered `png` mode is an export/no-GPU
> fallback, not the default.

---

## 5. Isosurfaces, surfaces, wells, points, glyphs

### 5.1 Isosurfaces (marching cubes) — client vs server

| Path | When | How |
|---|---|---|
| **Client, WebGPU compute** *(progressive enhancement — see Appendix §13)* | iso value scrubbed interactively, volume resident, `navigator.gpu` present | marching cubes in a compute shader over the resident volume → mesh in GPU buffers, no round-trip. Best UX. |
| **Client, Web Worker** *(WebGL2 floor)* | WebGL2, modest volumes | MC in a worker (transferable typed arrays) off the resident/downsampled grid; throttle/debounce on scrub. |
| **Server (`POST /property-models/{id}/isosurface`, doc 04 §9)** | huge volumes, native-res surface, or no client GPU | backend runs MC (skimage/VTK) → returns/streams **glTF** (inline if small, else `{job_id}` → `GET /features/{id}/geometry`); client just loads the mesh. |

**Decision:** **interactive iso scrubbing on the client** (WebGPU compute where
available, worker fallback) over the resident LOD; a **"bake at native resolution"**
action calls the server for the publication mesh. Iso meshes get smooth normals and the
property's color (flat iso color or a second property mapped onto the surface).

### 5.2 Surfaces / horizons & fault meshes (glTF)

- Loaded as **glTF** (doc 02 / OVERVIEW §5) — already in Engineering coordinates, so they
  drop straight into the Z-up scene. Draco/meshopt compression for big horizons.
- Double-sided, with optional **per-vertex property draping** (e.g. temperature on a
  horizon) via a sampled transfer function. Faults rendered semi-transparent with edge
  highlighting; horizons can show their **uncertainty** as a color or a displaced
  envelope (doc 07 feeds the uncertainty field).
- Picking via three-mesh-bvh for fast hover/identify.

### 5.3 Well paths as tubes with logs color-mapped

- Well trajectory comes from the **deviation survey** (doc 01 §4: MD/incl/azimuth →
  Engineering XYZ). We build a `TubeGeometry` along the polyline.
- **Log curves** (LAS, resampled to MD) are mapped to **vertex colors** along the tube
  via a chosen log's transfer function — e.g. gamma-ray or resistivity painted down the
  well. A switcher picks which log colors the tube; multiple logs can render as **offset
  ribbons/spirals** beside the path for several curves at once.
- Markers for casing shoes, formation tops, perforations; hover shows MD/TVD/elevation
  (depths reported true, see §2.3). Wells are also the natural pick target that syncs the
  **log-track panel** (§10.3).

### 5.4 Microseismic point clouds with time animation

- Events render as a `THREE.Points` cloud (instanced quads/sprites). Per-point
  attributes: position (Engineering XYZ), **time**, magnitude, and any scalar (e.g.
  b-value, confidence).
- **Size** ∝ magnitude (attenuated by distance); **color** by time, depth, or magnitude
  via a transfer function.
- **Time animation** driven by the global time slider (§9.4): a `uTimeWindow` uniform
  fades/reveals events inside a moving window (cumulative or rolling). GPU-side
  filtering — no re-upload per frame; supports tens–hundreds of thousands of events.
- Scale path: **3D Tiles / Potree** if a catalog ever exceeds the single-buffer budget.

### 5.5 Vector glyphs / streamlines

- **Glyphs:** instanced arrows/ellipsoids for vector or tensor fields (EM/MT field
  vectors, MT phase-tensor ellipses, modeled fluid velocity), sized/colored by
  magnitude, sub-sampled to a glyph budget.
- **Streamlines:** integrated client-side (RK4 over a sampled vector volume) in a worker
  / WebGPU compute, rendered as tubes or lines; seed rakes placed in-scene.
- Lower priority (Phase 4+), but the layer slot and instancing pattern are reserved.

---

## 6. Terrain & basemap (binds doc 01 §6)

### 6.1 DEM surface

- Doc 01 §6 fetches **Copernicus GLO-30**, reprojects to the project CRS, and stores a
  surface grid already in **Engineering elevation**. The viewer loads that grid as a
  `PlaneGeometry`-style mesh displaced in Z — it lands exactly in the ENU scene with **no
  reprojection at render time**. Subsurface volumes/slices/wells **hang beneath it**
  naturally because they share the frame and Z is elevation.
- The terrain mesh respects vertical exaggeration (§2.3) and the clip box, so you can
  cut a box that slices terrain + subsurface together.
- **Local mode:** `surfaceModel = flat:0` or a synthetic surface — a flat/served grid,
  no tiles (doc 01).

### 6.2 Basemap tiles — Web-Mercator reconciled at render time only

> **Resolved (DECISIONS.md):** basemap = **DEM shaded-relief by default + optional online
> tiles when available** (offline-safe). The georeferenced viewer never *requires* an
> external tile provider; the shaded-relief DEM (§6.1) is the self-contained default and
> online XYZ tiles are an opt-in enhancement.

Per doc 01 §3/§6, **the model CRS is never Web Mercator**; tiles are a render-time-only
concern. When online tiles are enabled, we drape imagery without contaminating the
measurement frame:

```
For each terrain-mesh vertex (Engineering XYZ):
  engineering → CRS easting/northing   (SpatialFrame, doc 01 §2 transform)
  CRS → lat/lon                        (pyproj on backend / proj4 in client)
  lat/lon → Web-Mercator tile UV       (standard XYZ slippy-tile math)
```

- Computed as a **texture-coordinate / vertex attribute** on the terrain mesh (and a
  quadtree of terrain tiles for LOD), so Web-Mercator distortion lives *only* in the UV
  lookup — geometry stays in the true CRS. We fetch standard XYZ basemap tiles
  (satellite/topo) and sample them by those UVs.
- The backend can precompute the engineering→lat/lon mapping per terrain tile (cheap,
  static for a project) so the client doesn't run pyproj; client-side proj4 is the
  fallback. This isolates the only place a slippy-tile scheme touches the app.
- Survey-coverage footprints, ROI outline, and a north arrow draw on/above the surface.

---

## 7. Performance budget & LOD strategy

### 7.1 Target hardware

Local-first, single-user R&D on a **modern discrete-GPU desktop/laptop** (the realistic
operator profile). Budget assumes WebGL2-class GPU, ~2–8 GB VRAM available to the page,
60 fps interactive / acceptable down to 30 fps while streaming.

### 7.2 Budgets

| Resource | Client-side budget (target) | Notes |
|---|---|---|
| Active ray-marched volume (working set) | ≤ ~512³ effective, via bricks | brick pool caps VRAM regardless of full volume size |
| Brick pool VRAM | ~256–512 MB (one atlas 3D texture) | fixed; LRU eviction |
| Simultaneous ray-marched volumes | 1–2 full-quality (more via MIP/blend) | each march costs fill-rate |
| Draw calls | ≤ ~1–2k/frame | instance glyphs/points/wells; merge static meshes |
| Microseismic points | ≤ ~10⁵–10⁶ in one buffer | beyond → 3D Tiles |
| Triangles (meshes resident) | ≤ ~5–10 M | Draco/meshopt, BVH for picking |
| Ray-march steps | adaptive 128–512; drop while moving | quality-on-idle |

### 7.3 LOD & adaptive quality

- **Quality-on-interaction:** during camera move / slider scrub, reduce ray-march step
  count and render at a lower internal resolution (dynamic-resolution scaling), then
  **refine to full quality on idle**. Keeps interaction smooth on heavy volumes.
- **Octree LOD per brick** (§3.4): screen-space-error + distance + frustum + clip-box
  culling decide which bricks/levels load.
- **Foveated/region refinement:** prioritize bricks near the camera target / inside the
  clip box.

### 7.4 When to fall back to server-side rendering

The client is the default renderer; we escalate to the backend (doc 04) when any of:

| Trigger | Server action |
|---|---|
| WebGL2 unavailable / weak GPU | server renders volume/slice **images** (`POST .../slice` with `encoding:"png"`), client shows them (2D-ish pan/zoom) |
| Volume far exceeds brick-streaming budget at needed fidelity | server `POST .../slice` / `POST .../isosurface` (doc 04 §9) |
| Native-resolution slice / isosurface ("HQ"/"bake") | server returns full-res slice array (`encoding:"f32"`) or glTF |
| Headless export / report figure | server renders a fixed-view image |

This is the bottom of the §1.2 fallback ladder. It is **graceful degradation, not a
second app** — same layer state, same transfer functions, the backend just becomes the
rasterizer. **Resolved (DECISIONS.md):** the client-side ceiling is **~512³ effective
working set (~256–512 MB brick pool), 1–2 full-quality volumes**; beyond that we
**auto-escalate to server-side** slice/isosurface/image rendering.

### 7.5 Memory management

- Single fixed **brick pool** + **mesh/texture LRU caches** keyed by layer; unmounting a
  layer frees its GPU resources (dispose geometries/materials/textures explicitly — R3F
  doesn't always auto-dispose pooled atlases).
- Decode bricks/meshes in **Web Workers** (transferable buffers) to keep the main thread
  responsive; throttle concurrent fetches/uploads.
- Watch `WEBGL_lose_context` / context-loss and rebuild from state.

---

## 8. (reserved — folded into §1.2 & §7.4)

---

## 9. Layer manager & UX model

### 9.1 Datasets → layers

Every dataset (property model, feature set, well, point cloud, raster, terrain) becomes a
**Layer** in a layer-manager panel — the toggleable/blendable unit:

```ts
interface Layer {
  id: string;
  datasetId: string;
  kind: 'volume' | 'slice' | 'isosurface' | 'surface' | 'well'
      | 'points' | 'glyphs' | 'terrain' | 'raster';
  visible: boolean;
  opacity: number;          // 0..1
  order: number;            // compositing / draw order
  blend: 'over' | 'additive' | 'mip' | 'minip';
  transferFn?: TransferFn;  // colormap + opacity + domain + log (volumes/slices/iso/wells)
  property?: string;        // which property/log is mapped
  time?: TimeBinding;       // 4D layers
  clip: boolean;            // obey global clip box
  zExplode?: number;        // exploded-layers offset
  lodBias?: number;         // quality nudge
}
```

- **Grouping** by survey method / fusion product; drag-to-reorder controls compositing.
- A new ingested dataset auto-creates a layer with registry-seeded defaults (§3.2) — it
  shows up correctly placed and colored with zero config (the doc 01 promise: synthetic
  and real render through the same path).

### 9.2 Transfer-function editing

A dockable editor (per selected layer): colormap gallery, **opacity curve over the value
histogram**, domain min/max, log/linear toggle, invert, "isolate band". Live — edits push
a new LUT texture, no refetch. Histogram from doc 04 stats / client compute.

### 9.3 Clip box, exploded view, sections

Global clip-box gizmo (§2.4), exploded-layer slider, and a "create section from clip
face" action that spawns a slice layer.

### 9.4 Time slider (4D)

A global **time axis** built from the union of all time-bearing layers (microseismic,
InSAR deformation rasters, repeat surveys):

- Playhead + window (instant / cumulative / rolling-window), play/pause/scrub, speed.
- Drives uniforms (microseismic `uTimeWindow`, InSAR raster frame select, volume time
  index) — **no geometry rebuild per tick** where possible.
- Layers declare a `TimeBinding` (their own sample times); the slider reconciles
  heterogeneous cadences onto one timeline and snaps/interpolates per layer.

---

## 10. State management (Zustand) & panel sync

### 10.1 Store shape

```ts
interface ViewerStore {
  // scene frame (mirrors SpatialFrame from doc 01; viewer never recomputes CRS)
  frame: { mode; anchor; roi; depthRange; verticalExaggeration; };

  camera: { mode: 'perspective'|'ortho'; target; position; preset?; };
  clipBox: { enabled; min; max; rotationDeg; capMode; };
  time:    { t: number; window: {mode; widthSec}; playing; speed; bounds; };

  layers: Record<string, Layer>;        // §9.1
  layerOrder: string[];

  selection: { kind; id; detail? };     // picked well/event/voxel → drives panels
  hover:     { kind; id; worldXYZ } | null;

  // derived render bookkeeping (not persisted)
  brickPool: { residentBricks; vramUsed; pending };
  quality:   { interacting: boolean; stepScale; resScale };
}
```

- **Selectors** keep R3F components subscribed to minimal slices (avoid re-rendering the
  whole scene on a slider tick — time goes to a uniform, not React state churn).
- **Persistence:** layer set, transfer functions, camera presets, clip box and time
  state serialize to a **saved view** (project-scoped) so a session is reproducible.

### 10.2 Engineering-frame invariants in state

The store mirrors doc 01's `SpatialFrame` read-only; **all stored positions are
Engineering metres**. Vertical exaggeration and Web-Mercator UVs are render-time
transforms, never written back into data or selection coordinates.

### 10.3 Analysis-panel sync (cross-plots, log tracks)

The 3D view and the 2D analysis panels (Observable Plot / D3, OVERVIEW §5) share the same
store, so they stay linked:

| Interaction | Effect |
|---|---|
| Pick a **well** in 3D | log-track panel opens that well's curves (MD-indexed) |
| Hover a **log depth** in the track panel | a marker rides the well tube at that MD |
| Brush a region on a **cross-plot** (resistivity vs density) | matching voxels **highlight in 3D** (a selection mask layer) |
| Pick a **voxel / event** in 3D | cross-plot/inspector shows its multi-property values (doc 04 `sample`) |
| Move the **time slider** | both 3D animation and any time-series panel advance together |

Cross-plot point↔voxel data and multi-property samples at a location come from doc 04's
`sample` endpoint; the **fusion math** (what to cross-plot, clustering) is doc 07. The
viewer owns only the *linking and brushing*, not the statistics.

---

## 11. Phasing (aligns to OVERVIEW §9)

| Phase | Viewer deliverable |
|---|---|
| 1 (**M1, first proof**) | Z-up scene, camera/controls, **one ray-marched volume held in a single resident `Data3DTexture`** (§1.3) at a modest size, orthogonal slice planes (same texture), clip box, transfer function, terrain (flat/synthetic). **Includes the Zarr v3 + Blosc/zstd browser-decode SPIKE** (server-decode fallback if not ready). *End-to-end vertical slice with the least moving parts.* |
| 2 (**M2+**) | **Brick-pool / page-table / octree-LOD streaming** (§3.4); layer manager, multi-volume compositing/blending, transfer-function editor, registry-seeded defaults. |
| 3 | Cross-plot brushing ↔ 3D highlight; uncertainty/fused-volume layers (doc 07 feeds). |
| 4 | Horizons/faults (glTF), well tubes + log coloring, microseismic + InSAR + time slider, isosurfaces. |
| 5 | Geomodel surfaces, well-planning overlays (target picks, trajectories, intersections — doc 09). |
| — | **WebGPU compute path** (iso + slice resampling — Appendix §13) lit up opportunistically once stable; never gates the build. |

---

## 12. Contract from the parallel docs (now resolved — doc 04 §9 is authoritative)

> **Doc 04 is the authoritative API + storage contract** (its §1, §9). The viewer
> *references* these endpoints; it does **not** restate a divergent shape. The earlier
> drafts of this doc used `GET /tiles/...` and `GET /slice` — both **superseded** by the
> doc 04 endpoints below.

**From doc 04 (storage & serving) — what the viewer reads:**
- **Bricks (hot path):** `GET /property-models/{id}/bricks/{level}/{t}/{bz}/{by}/{bx}`
  (or read the store directly via `GET /property-models/{id}/zarr/{path}`). Brick = 64³
  Zarr chunk, address == chunk key `<property>/<level>/c/<bz>/<by>/<bx>` (doc 02 §10.2);
  encoding Blosc+zstd float32; **NaN** no-data; per-level extents/value-range from the
  `multiscales` block + catalog stats (doc 04 §5/§6/§9).
- **Slice:** `POST /property-models/{id}/slice` (`SliceRequest`) → **raw f32 by default**
  (`encoding:"f32"`), `png` for export/no-GPU (doc 04 §9.3).
- **Isosurface:** `POST /property-models/{id}/isosurface` (`IsoRequest`) → inline **glTF**
  if small, else `{job_id}` then `GET /features/{id}/geometry` (doc 04 §9.3).
- **Sample / stats:** `POST /property-models/{id}/sample`, `POST /projects/{pid}/sample`
  (multi-property panel sync); value histograms come from the catalog `stats_json`
  exposed on `GET /property-models/{id}` (doc 04 §9.2).
- (Fallback) server-side **image render** via the slice `png` encoding for the no-GPU ladder.

**From doc 02 (data model) / doc 01 §5:**
- Property-type registry fields the viewer reads to seed transfer functions: canonical
  unit, default colormap, default log/linear, default display range.
- glTF conventions for horizons/faults/solids (Engineering coords, Draco/meshopt, normals).
- Well deviation-survey + LAS log schema (MD-indexed) for tube building & coloring.
- Microseismic event schema (XYZ, time, magnitude, scalars).

**From doc 01 (locked) — consumed, not requested:** Engineering Frame, `SpatialFrame`,
floating origin, depth/elevation conventions, terrain (Copernicus GLO-30) and the
engineering→CRS→lat/lon transform for basemap UVs.

---

## Decisions locked in

1. **Three.js via react-three-fiber**, Zustand-driven declarative scene graph; analysis
   panels in Observable Plot / D3 (DOM, not WebGL).
2. **WebGL2 is the default/floor; WebGPU is detected progressive enhancement** behind a
   `RenderBackend` seam (DECISIONS.md). WebGPU compute (interactive marching-cubes
   isosurfaces, on-GPU slice resampling, histograms) is an **Appendix §13 later phase** —
   it must not dominate the first build. Fallback ladder ends at server-side rendering —
   never a blank screen.
3. **Scene world space *is* the Engineering Frame** (doc 01): Z-up root
   (`DEFAULT_UP=(0,0,1)`), positions are Engineering metres straight to the GPU, float32
   safe via the floating origin. No CRS coordinates ever reach the GPU.
4. **Vertical exaggeration and the clip box are render-only**, applied at the scene root,
   uniform across all layers; picking/measurements report true Engineering coordinates.
   Volume ray-marcher clips in-shader.
5. **GPU single-pass ray-marching** with per-property transfer functions (colormap +
   opacity, seeded from the property-type registry); compositing modes over / additive /
   MIP / minIP; multi-volume blend supported.
6. **Renderer staging (critique #17):** **M1 = one resident `Data3DTexture`** (single
   modest volume + orthogonal slice + clip box + transfer function, WebGL2 only) is the
   first proof. **M2+ = brick/LOD streaming** against doc 04's octree-compatible pyramid
   into a **fixed-VRAM brick-pool atlas + page table** with coarse-first refinement —
   bounds client memory regardless of full volume size; volumes are never blank, detail
   streams in. Brick address == Zarr chunk key (doc 02 §10.2 / doc 04 §6).
7. **Slices client-side from resident bricks** for the interactive case (same textures &
   transfer fn as the volume); **server `slice` for arbitrary geometry / native-res**.
8. **Isosurfaces interactive on the client** (WebGPU compute or worker over resident LOD);
   **server bake at native resolution** for publication meshes.
9. **Terrain renders the doc-01 Engineering-elevation DEM directly**; **Web-Mercator
   basemap tiles reconciled at render time only** as terrain-mesh UVs — distortion never
   enters the measurement frame; subsurface hangs beneath terrain by construction.
10. **Layers** are the toggle/blend unit; auto-created with registry defaults on ingest.
    A **global time slider** drives 4D via uniforms (no per-tick geometry rebuild). 3D
    view ↔ analysis panels share the Zustand store (pick/hover/brush linking).
11. **Adaptive quality:** reduce ray-march steps + dynamic resolution while interacting,
    refine on idle. Fixed brick pool + LRU caches; workers for decode.

---

## 13. Appendix — WebGPU compute path (progressive enhancement, later phase)

> **Floor vs enhancement (DECISIONS.md):** **WebGL2 is the shipping floor**; everything in
> this appendix is **progressive enhancement that must not dominate the first build**. It
> lights up only when `navigator.gpu` is present, behind the `RenderBackend` seam (§1.2),
> and always has a WebGL2 / server-side equivalent so it is never load-bearing.

When WebGPU is detected, the same scene gains a **compute** path that moves work off the
server / off fragment-shader hacks onto the client GPU:

- **Marching-cubes isosurfaces in a compute shader** over the resident volume → mesh in
  GPU buffers, no server round-trip (§5.1, best UX for iso scrubbing).
- **On-GPU slice resampling** from bricks (arbitrary planes resampled in compute instead
  of a `POST .../slice` round-trip when the volume is resident).
- **Brick → 3D-texture transforms** and **histogram / transfer-function previews** computed
  on-GPU (feeds §3.2 / §9.2 instead of the server stats call).
- **Multi-volume binding** via bind-groups (cleaner N-volume compositing, §3.3).
- **Streamline integration** (RK4 over a sampled vector volume, §5.5).

**Three.js path:** `WebGPURenderer` + TSL node materials (compile to WGSL), with the
hand-written GLSL ray-marcher kept as the guaranteed WebGL2 floor. None of the above is a
build gate; each falls back to the WebGL2 fragment path or the doc 04 server endpoints
(the §1.2 fallback ladder).

---

## Resolved (was: open questions)

These were the doc's open forks; **DECISIONS.md** has settled them. Recorded here so the
section is not mistaken for still-open work.

1. **Render backend** → **WebGL2 floor + WebGPU progressive enhancement** (§1.2, §1.3,
   Appendix §13). We may require a WebGL2-class GPU; WebGPU is never a hard gate.
2. **Client-side performance ceiling** → **~512³ effective working set (~256–512 MB brick
   pool), 1–2 full-quality volumes; auto-escalate to server-side beyond** (§7.4).
3. **Basemap / terrain dependency** → **DEM shaded-relief default + optional online tiles
   when available** (offline-safe); never requires an external tile provider (§6.1, §6.2).

> **Genuinely still open (non-blocking):** sparse-octree vs dense-pyramid brick skipping is
> deferred to doc 04 (its "Still open" note) — the M2+ streaming client (§3.4) reads the
> same addressing either way, so it is not a viewer-side blocker.
