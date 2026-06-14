# The 3D viewer

!!! abstract "What you'll learn / why it matters"
    This page explains how the browser turns the fused earth model into an interactive 3-D
    scene you can fly through, slice, and drill virtual wells into. The centrepiece is
    **GPU volume ray-marching** — the technique that lets you render a *translucent cloud* of
    a physical property (temperature, resistivity, favorability) rather than just its surface.
    If you know what a texture, a shader, a sampling rate, and a level-of-detail scheme are,
    you already have every CS concept you need; we map the geoscience onto those. By the end
    you'll understand the scene graph, the ray-marching fragment shader, [transfer functions](#3-transfer-functions),
    slicing, multi-volume compositing, brick/LOD streaming, terrain, and the 4-D time slider —
    and exactly which TypeScript/GLSL files implement each piece.

The viewer is a [Three.js](glossary.md) scene driven declaratively by
[react-three-fiber (R3F)](glossary.md) and a [Zustand](glossary.md) store. It is a pure
*consumer* of the [data model](data-model.md) and the [fused earth model](fusion.md): the
backend produces gridded volumes and meshes; the viewer's only job is to draw them
faithfully and let you interrogate them. It targets **WebGL2** as the guaranteed floor, with
**WebGPU** as an optional accelerator (more on that at the end).

---

## 1. The Engineering-Frame scene (Z is up)

Everything you see lives in one coordinate system: the **Engineering Frame**, defined on the
[coordinates & units](spatial-framework.md) page. It is a local, right-handed,
metres-based grid — **X = East, Y = North, Z = Up** — with the origin "floated" to sit near
the project so coordinates stay in the ±tens-of-kilometres range. That floating origin is
what makes `float32` safe on the GPU: every vertex, every texture-box corner, every well
station sits comfortably inside ~millimetre precision, so **no map-projection (CRS)
coordinate ever reaches the GPU**.

!!! note "Define: scene graph"
    A **scene graph** is a tree of nodes (cameras, lights, meshes) that a renderer walks each
    frame to draw the world — the spatial equivalent of a DOM tree. In R3F you write that tree
    as React components, and React's reconciler keeps the Three.js scene in sync with your
    state. Toggling a layer literally mounts/unmounts a subtree.

There is one wrinkle. Three.js is **Y-up** by default (Y points to the sky), but our frame is
**Z-up** (Z is elevation, because depth is the dimension that matters underground). Rather
than rotate every dataset, the viewer flips the *whole scene* once, before any object is
created:

```ts title="frontend/src/scene/Scene.tsx"
// One-time Z-up: ENU Z is up (doc 06 §2.1). Set before any Object3D is created.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
// ...
camera={{ up: [0, 0, 1], near: 0.1, far: 1e6, position: [3000, -3000, 2000] }}
```

After that, a position fed to the GPU is *already* Engineering XYZ in metres — no per-object
transform. The scene tree mirrors the layer types you'd expect:

```
SceneRoot (Z-up, Engineering metres)
├── CameraControls        target-centric orbit/truck/dolly (drei)
├── Lights                hemispheric + key (subsurface look-dev)
├── TerrainLayers         DEM mesh + draped basemap
├── RasterLayers          2-D rasters (e.g. InSAR deformation)
├── SliceLayer            orthogonal plane(s) sampling the same volume texture
├── VolumeLayers          N × ray-marched property volumes  ◀── the centrepiece
├── FeatureLayers         horizons / faults / solids (glTF)
├── WellLayers            well tubes + log colouring + markers
├── PointCloudLayers      microseismic (time-animated)
├── PickTargetLayer       drill-target picking gizmo
└── ClipBox               6 user planes → renderer clipping
```

### 1.1 Subsurface-aware camera

The camera is a [drei](glossary.md) `CameraControls` rig — **target-centric**, so you orbit a
point of interest (a well, an anomaly) rather than the origin. Because the region of interest
extends *downward* (to e.g. −8000 m), the default framing tilts the camera to look obliquely
*down into* the volume. When data first loads, `CameraFramer` computes the bounding-box
diagonal and places the eye up-and-back:

```ts title="frontend/src/scene/Scene.tsx — CameraFramer"
const eye = new THREE.Vector3(c[0] + diag * 0.9, c[1] - diag * 0.9, c[2] + diag * 0.7);
controls.setLookAt(eye.x, eye.y, eye.z, c[0], c[1], c[2], false);
```

A perspective camera is the default; an orthographic toggle exists for measurement and
cross-section work (no perspective foreshortening when you read a section). The viewer also
supports **vertical (Z) exaggeration** — a render-only scale on the scene root that stretches
depth so subtle layering is visible — and crucially divides it back out for any reported
depth or coordinate, so a "2× exaggerated" horizon still reports its true elevation.

---

## 2. Volume ray-marching from scratch (the centrepiece)

This is the heart of the viewer. A **property volume** is a 3-D array of one scalar field —
think `float32[nz][ny][nx]` — for example temperature at every cell of the
[fused grid](fusion.md). We want to render it not as a solid block but as a *translucent fog*
where you can see hot regions glowing through cooler ones. That is **volume rendering**, and
the GPU technique we use is **single-pass ray-marching**.

!!! note "Define: voxel, 3-D texture, fragment shader"
    A **voxel** is a 3-D pixel — one cell of the volume array. A **3-D texture**
    (`Data3DTexture` / GLSL `sampler3D`) is that array uploaded to GPU memory so a shader can
    sample it with hardware trilinear interpolation at *any* continuous `(x,y,z)` — exactly
    like a 2-D texture but with three coordinates. A **fragment shader** is a tiny program the
    GPU runs *once per output pixel*, in massive parallel, to decide that pixel's colour.

### 2.1 The mental model: one ray per pixel

To draw the volume we render a **box mesh** — the volume's axis-aligned bounding box (AABB)
in Engineering metres — and attach a custom fragment shader to it. For every screen pixel the
box covers, the shader:

1. **Builds a ray** from the camera through that pixel into the box.
2. **Finds where the ray enters and exits** the box (and the user clip box).
3. **Walks the ray in small steps** (it "marches"), sampling the 3-D texture at each step.
4. **Maps each sample value → colour + opacity** via a [transfer function](#3-transfer-functions).
5. **Composites** those coloured, semi-transparent samples front-to-back into one final pixel.

It is conceptually identical to **alpha-blending a stack of translucent slides**: each step
is a slide; nearer slides partly occlude farther ones. The "resolution" of the image in depth
is set by the step size — fewer steps is faster but blockier (a classic
sampling/aliasing trade-off).

### 2.2 The actual shader, annotated

Here is the real WebGL2 (GLSL ES 3.00) ray-marcher, lightly trimmed. Read it top-to-bottom;
the comments tie each block to the five steps above.

```glsl title="frontend/src/lib/shaders.ts — VOLUME_FRAG (trimmed)"
uniform sampler3D uVolume;       // the property volume, in GPU memory
uniform sampler2D uTransferFn;   // 256×1 RGBA lookup table: value → colour+opacity
uniform vec3  uBoxMin, uBoxMax;  // volume AABB (Engineering metres)
uniform vec3  uClipMin, uClipMax;// user clip box (Engineering metres)
uniform vec3  uCameraPos;        // camera position (Engineering metres)
uniform float uDomainMin, uDomainMax, uLog;   // value→[0,1] mapping (linear or log)
uniform int   uSteps;            // max ray-march steps (quality knob)
uniform int   uBlend;            // 0=over 1=additive 2=MIP 3=minIP

// Ray ∩ axis-aligned box (the "slab" method). Returns (tNear, tFar).
vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (bmin - ro) * inv, t1 = (bmax - ro) * inv;
  float tNear = max(max(min(t0,t1).x, min(t0,t1).y), min(t0,t1).z);
  float tFar  = min(min(max(t0,t1).x, max(t0,t1).y), max(t0,t1).z);
  return vec2(tNear, tFar);
}

void main() {
  vec3 ro = uCameraPos;                          // (1) ray origin
  vec3 rd = normalize(vWorldPos - uCameraPos);   //     ray direction toward this pixel

  // (2) entry/exit: march only the OVERLAP of the volume box and the clip box.
  vec2 tv = intersectBox(ro, rd, uBoxMin, uBoxMax);
  vec2 tc = intersectBox(ro, rd, uClipMin, uClipMax);
  float t0 = max(max(tv.x, tc.x), 0.0);
  float t1 = min(tv.y, tc.y);
  if (t1 <= t0) discard;                          // ray misses → transparent pixel

  vec3  span = uBoxMax - uBoxMin;
  float dt   = max(span.x, max(span.y, span.z)) / float(uSteps);  // step size
  vec4  acc  = vec4(0.0);                          // accumulated colour (premultiplied)

  // (3) march
  for (int i = 0; i < 4096; ++i) {
    if (i >= uSteps || t > t1) break;
    vec3 p   = ro + rd * t;
    vec3 uvw = (p - uBoxMin) / span;              // world point → [0,1]³ texcoord
    float raw = texture(uVolume, uvw).r;          // sample the scalar field
    t += dt;
    if (isnan(raw)) continue;                     // NaN = no-data → skip (stays transparent)

    // (4) value → colour+opacity via the transfer-function LUT
    float vn = applyScaling(raw);                 // raw → [0,1] over [domainMin,domainMax]
    vec4  c  = texture(uTransferFn, vec2(vn, 0.5));

    // (5) composite front-to-back, corrected for step size
    float a = 1.0 - pow(1.0 - clamp(c.a * uOpacityGain, 0.0, 1.0), dt / uRefStep);
    acc.rgb += (1.0 - acc.a) * a * c.rgb;
    acc.a   += (1.0 - acc.a) * a;
    if (acc.a > 0.98) break;                       // early ray termination (opaque enough)
  }
  if (acc.a <= 0.0) discard;
  fragColor = vec4(acc.rgb, clamp(acc.a, 0.0, 1.0));
}
```

A few details worth their weight:

- **NaN is the no-data sentinel.** The fused grid stores `NaN` where there is no coverage;
  the shader skips those samples so empty regions stay transparent instead of rendering as a
  spurious value. (This matches the on-disk `NaN` fill convention in the
  [data model](data-model.md).)
- **Opacity correction** (`pow(..., dt / uRefStep)`) keeps the apparent density of the fog
  constant even when the step size changes — essential once LOD changes the sampling rate
  (otherwise a coarser march would look more transparent).
- **Early ray termination** stops marching once the accumulated alpha is ~opaque — you can't
  see through stuff that's already solid, so why keep sampling. A cheap, large win.
- **`side: THREE.BackSide`** on the material (see `VolumeLayer.tsx`) renders the box's *back*
  faces, so the ray is generated correctly even when the camera is *inside* the volume.

### 2.3 Clipping happens in the shader

Hardware clip planes carve *geometry*, but a ray-marched volume has no geometry along the ray
— it's all samples. So the **clip box is intersected in the shader** (the `tc` term above):
the user drags an axis-aligned box (`frontend/src/scene/ClipBox.tsx`), the store stores it as
fractions of the scene bounding box, and `VolumeLayer.tsx` converts those to Engineering
metres each frame and feeds `uClipMin`/`uClipMax`. The same box also drives
`renderer.clippingPlanes` for the *mesh* layers (terrain, faults), so one gizmo cuts the
whole scene consistently.

---

## 3. Transfer functions

A raw scalar (say 472 K, or 240 Ω·m) means nothing to a GPU. A **transfer function** is the
mapping `value → (colour, opacity)` that makes the volume legible. Implementationally it is a
tiny **1-D lookup table (LUT)**: a 256×1 RGBA texture the shader samples with the normalised
value `vn ∈ [0,1]`.

!!! note "Define: transfer function"
    Think of it as a *colourmap plus an opacity curve*. The colourmap turns a number into a
    hue; the opacity curve decides how *visible* each value is. By making mid-range values
    transparent and only one band opaque, you "isolate" a feature — e.g. show only the
    conductive (low-resistivity) geothermal anomaly and let everything else fade out.

Two normalisation choices live alongside the LUT:

- **Domain** (`uDomainMin`/`uDomainMax`): the value range mapped across the LUT.
- **Scaling** (`uLog`): linear or **logarithmic**. Resistivity spans many orders of magnitude
  (1 → 10⁴ Ω·m), so it is shown log-scaled by default — like a log-axis plot.

```glsl title="frontend/src/lib/shaders.ts — applyScaling"
float applyScaling(float raw) {
  float lo = uDomainMin, hi = uDomainMax, v = raw;
  if (uLog > 0.5) { v = log(max(raw,1e-12)); lo = log(max(uDomainMin,1e-12)); hi = log(max(uDomainMax,1e-12)); }
  return clamp((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0);   // → [0,1] index into the LUT
}
```

Defaults are **seeded from the property-type registry** (see the
[data model](data-model.md) and [coordinates & units](spatial-framework.md)): each property
type carries a canonical unit, a default colourmap, a default log/linear flag, and a default
display range. So a freshly ingested resistivity volume arrives log-scaled with a sensible
Ω·m range and an appropriate colourmap **with zero configuration**. Editing is live: the LUT
texture is re-baked in place (`updateTransferFnTexture`), no refetch, the volume updates next
frame. The editor UI is `frontend/src/ui/TransferFnEditor.tsx`; the LUT machinery is
`frontend/src/lib/transferFn.ts` and the palettes are `frontend/src/lib/colormaps.ts`.

### 3.1 Confidence-modulated opacity (the "honest view")

A standout feature: the shader can scale each sample's opacity by a **co-registered
confidence (or σ) volume**, so low-confidence regions render *faint*. Because favorability and
its confidence share the [fused grid](fusion.md), they sample at the same `[0,1]³` texcoord:

```glsl title="frontend/src/lib/shaders.ts — confidenceWeight"
float confidenceWeight(vec3 uvw) {
  if (uConfidenceOn < 0.5) return 1.0;            // modulation is opt-in
  float c = texture(uConfidence, uvw).r;
  if (isnan(c)) return 1.0;                        // no coverage → don't hide the data
  float w = clamp((c - uConfMin) / max(uConfMax - uConfMin, 1e-12), 0.0, 1.0);
  if (uConfInvert > 0.5) w = 1.0 - w;             // σ-style: high uncertainty ⇒ low confidence
  return mix(uConfFloor, 1.0, w);                  // keep a floor so faint ≠ invisible
}
```

This is the visual expression of the project's [uncertainty](uncertainty.md) discipline: you
literally *see less* where the model knows less.

---

## 4. Orthogonal slices and the clip box

A **slice** is a 2-D cut through the volume. The key design point: an orthogonal slice samples
the *same* `Data3DTexture` through the *same* transfer function as the ray-marcher, so slice
and volume colours stay perfectly locked. The slice is just a quad whose fragment shader looks
up a per-vertex `[0,1]³` texcoord:

```glsl title="frontend/src/lib/shaders.ts — SLICE_FRAG (core)"
if (any(lessThan(uvw, uClipMin)) || any(greaterThan(uvw, uClipMax))) discard;  // obey clip box
float raw = texture(uVolume, uvw).r;
if (isnan(raw)) discard;                            // no-data
vec4 c = texture(uTransferFn, vec2(applyScaling(raw), 0.5));
fragColor = vec4(c.rgb, uSliceOpacity);
```

| Slice type | Geometry | Sampling |
|---|---|---|
| **Orthogonal plane** | X / Y / Z plane, draggable along its axis | client-side, the same resident texture (instant, zero fetch) |
| **Arbitrary cross-section** | tilted plane / polyline-swept curtain | server `slice` endpoint, draped as a textured quad |
| **Fence diagram** | connected vertical panels along a path | one server slice per panel |

Client-side is the default for the interactive case; the backend's full-resolution `slice`
endpoint is used for arbitrary geometry or a publication-quality "HQ" render. The orthogonal
slice lives in `frontend/src/scene/SliceLayer.tsx`; the clip box in
`frontend/src/scene/ClipBox.tsx` and `frontend/src/scene/clipPlanes.ts`.

---

## 5. Multi-volume compositing

You rarely look at one property in isolation — you want resistivity *and* temperature *and* a
fused favorability field in the same frame to see where they coincide. Each volume is its own
`VolumeLayer` (its own box mesh + texture + transfer function), and they co-render. The
**blend mode** chosen per layer sets *both* the in-shader accumulation *and* the GPU's
hardware blend equation:

| Mode | What it does | Typical use |
|---|---|---|
| **over** | standard front-to-back alpha (nearer occludes farther) | the default; layered translucency |
| **additive** | emission, no occlusion (colours sum) | overlay an anomaly as a glow |
| **MIP** (maximum-intensity projection) | keep the single brightest sample along each ray | "show me the hottest / most-conductive surface" |
| **minIP** | keep the single dimmest sample | the inverse — coldest / most-resistive |

```ts title="frontend/src/scene/VolumeLayer.tsx — applyGLBlend (excerpt)"
if (blend === "mip") {                  // hardware max-blend so N layer-meshes composite right
  mat.blending = THREE.CustomBlending;
  mat.blendEquation = THREE.MaxEquation;
  mat.blendSrc = THREE.OneFactor; mat.blendDst = THREE.OneFactor;
}
```

!!! tip "MIP is the standard geophysics view"
    A **maximum-intensity projection** flattens a volume to "the strongest thing behind each
    pixel" — like an X-ray. It is the canonical way to ask "where is the anomaly?" without
    occlusion getting in the way.

The simplest path (and the WebGL2 default) is **separate ray-marches, alpha-composited by
layer order**. A fully [fused/derived volume](fusion.md) (favorability, uncertainty, cluster
labels) is produced server-side and simply rendered as one more single layer.

---

## 6. Volumes bigger than GPU memory: brick / LOD streaming

WebGL2 only *guarantees* a 256³ 3-D texture (`MAX_3D_TEXTURE_SIZE`); real GPUs do
512³–1024³, but a full region of interest at fine resolution easily blows past both that limit
and available VRAM. The solution is the same trick game engines use for huge textures:
**virtual texturing**, here in 3-D.

!!! note "Define: brick, page table, atlas, LOD"
    A **brick** is a small fixed-size cube of the volume (64³ voxels) — analogous to a tile in
    2-D mapping. **LOD** (level of detail) is a multiresolution pyramid: level 0 is full
    detail, each higher level is a 2× coarser downsample (like mipmaps). An **atlas** is one
    big fixed-size texture holding many resident bricks side by side. A **page table** is the
    indirection: it maps "which brick do I want" → "which atlas slot holds it (if any)".

### 6.1 Two milestones

The viewer ships in two stages, by design:

| Milestone | Volume backing | What it proves |
|---|---|---|
| **M1 (first proof)** | one **resident `Data3DTexture`** (a coarse level or a ≤256³ crop) | the Z-up frame, the shader, the Zarr→GPU path, with the fewest moving parts |
| **M2+** | streamed bricks into a **fixed-VRAM atlas + page table** | unbounded volumes via coarse-first, hole-free LOD streaming |

`VolumeLayer.tsx` routes each layer between the two automatically: small volumes with a
resident buffer stay on the proven single-resident shader (`isStreamingLayer` → `false`);
volumes whose level-0 size exceeds the budget (or that have no resident buffer but do have a
pyramid) go to `StreamingVolumeLayer`.

### 6.2 How streaming works

```
View change ─▶ pick a LOD level per brick (screen-space error / distance / frustum / clip-box cull)
           ─▶ request the needed bricks (each brick == one Zarr chunk on the backend)
           ─▶ decode in a Web Worker → upload into the brick-pool ATLAS (fixed VRAM)
           ─▶ update the PAGE TABLE texture (brick key → atlas slot)
  ray-march ─▶ shader walks the page table per sample, samples the resident brick,
               FALLS BACK to a coarser resident level on a miss → no holes
```

Two invariants make this never-blank and never-unbounded:

- **The coarsest level is pinned** in the pool (never evicted), so the volume is *always*
  drawable; finer bricks refine in progressively (a clear "loading detail" feel, not pop-in
  holes). The streaming shader walks *down* from the finest selected level until it finds a
  resident brick.
- **The atlas is a fixed-size pool** with **LRU eviction** — VRAM is capped regardless of
  full-volume size, exactly matching a CPU page cache.

The pool/page-table math is deliberately kept **pure and headlessly unit-testable** in
`frontend/src/lib/brickPool.ts` (slot allocation, LRU, atlas coordinate math); the GPU
textures are created and uploaded by `frontend/src/scene/StreamingVolumeLayer.tsx`; the
streaming shader is `frontend/src/lib/brickShaders.ts`; LOD selection is
`frontend/src/lib/lod.ts`; brick addressing/decoding is `frontend/src/lib/bricks.ts`,
`brickDecode.ts`, and the decode worker `brick.worker.ts`.

!!! tip "It's a CPU cache, in your GPU"
    Page table + atlas + LRU is literally virtual memory paging applied to voxels. A "page
    fault" (brick not resident) doesn't crash — it falls back to a coarser level, just as a
    progressive JPEG shows a blurry image first. This is lossy compression *in space*.

---

## 7. Terrain, wells, points, and the 4-D time slider

The viewer renders more than volumes; each maps cleanly onto the [data model](data-model.md).

- **Terrain** (`frontend/src/scene/TerrainLayer.tsx`, `frontend/src/lib/terrain.ts`): a DEM
  (digital elevation model) surface mesh, already in Engineering elevation, so subsurface
  volumes/slices/wells hang beneath it by construction. Optional online basemap tiles are
  draped as terrain-mesh UVs, so map-projection distortion lives *only* in the texture lookup
  and never contaminates the measurement frame. DEM shaded-relief is the offline-safe default.

- **Features** (`frontend/src/scene/FeatureLayer.tsx`): horizons, faults, and solids loaded as
  **glTF** (a standard 3-D mesh format) already in Engineering coordinates. Faults render
  semi-transparent; a property (e.g. temperature) can be draped onto a horizon via the
  transfer function.

- **Wells** (`frontend/src/scene/WellLayer.tsx`): a well trajectory is a tube built along the
  [deviation survey](well-planning.md), with a chosen log (gamma-ray, resistivity, the
  [predicted log](well-planning.md)) painted along it as vertex colours. This is the bridge to
  the next page — see [Drilling & well planning](well-planning.md).

- **Microseismic** (`frontend/src/scene/PointCloudLayer.tsx`): earthquake-like events as a
  `THREE.Points` cloud, sized by magnitude and coloured by time/depth.

- **The time slider** (`frontend/src/ui/TimeSlider.tsx`, `frontend/src/lib/time.ts`): a global
  time axis unifying all time-bearing layers (microseismic, repeat InSAR rasters, repeat
  surveys). Crucially it drives **uniforms**, not geometry rebuilds — a `uTimeWindow` uniform
  fades events inside a moving window, and the `TimePlayer` loop in `Scene.tsx` advances the
  playhead each frame. So scrubbing tens of thousands of events stays smooth (no per-tick
  re-upload). This is "4-D": three space dimensions plus time.

!!! example "Everything is one Zustand store"
    The 3-D view and the 2-D analysis panels (cross-plots, log tracks) share
    `frontend/src/store.ts`. Pick a well in 3-D → its log-track panel opens. Brush a region on
    a resistivity-vs-density cross-plot → the matching voxels highlight in 3-D (a selection
    mask layer). Move the time slider → both the 3-D animation and any time-series panel
    advance together. The viewer owns the *linking and brushing*; the
    [fusion](fusion.md) layer owns the statistics.

---

## 8. WebGL2 floor, WebGPU enhancement

!!! note "Define: WebGL2 vs WebGPU"
    Both are browser GPU APIs. **WebGL2** is universal and the project's guaranteed floor;
    its fragment-shader ray-marcher covers everything above. **WebGPU** is newer and adds
    **compute shaders** — general-purpose GPU programs not tied to drawing pixels.

The renderer is architected behind a thin backend seam so the *same* scene can target either
API. WebGPU, when `navigator.gpu` is present, moves a few heavy jobs off the server (or off
fragment-shader hacks) and onto the client GPU: interactive **marching-cubes isosurfaces**,
**on-GPU slice resampling** from bricks, and histogram/transfer-function previews. It is
**never a hard gate** — it is progressive enhancement, and every WebGPU path has a WebGL2 or
server-side fallback. The fallback ladder is, in order:

`WebGPU compute → WebGL2 fragment → server-side render (slice/isosurface/image) → 2-D basemap + slices`

…so the user **never sees a blank screen**.

---

## Key takeaways

- The scene world space **is** the Engineering Frame: Z-up, metres, float32-safe via a
  floating origin — no CRS coordinates ever reach the GPU.
- **Volume ray-marching** is the centrepiece: one ray per pixel marches a `sampler3D`, maps
  each sample through a **transfer function** (a 1-D colour+opacity LUT), and composites
  front-to-back. `NaN` is the no-data skip; opacity is step-size corrected; rays terminate
  early when opaque.
- **Transfer functions** are seeded from the property-type registry (sensible colourmap, log
  scaling, range with zero config) and can modulate opacity by **confidence** for an honest view.
- **Slices** sample the same texture + transfer function as the volume, so colours stay locked.
- **Multi-volume compositing** supports over / additive / **MIP** / minIP.
- **Brick/LOD streaming** is virtual texturing in 3-D: a fixed-VRAM atlas + page table + LRU,
  coarsest level pinned so the volume is never blank — bounded memory for unbounded volumes.
- **WebGL2 is the floor; WebGPU is optional acceleration**, with a fallback ladder that never
  shows a blank screen.

## Where this lives in the code

| Concern | File(s) |
|---|---|
| Scene root, Z-up, camera, time loop | `frontend/src/scene/Scene.tsx` |
| Ray-marcher + slice GLSL | `frontend/src/lib/shaders.ts` |
| Volume layer (M1) + blend modes + routing | `frontend/src/scene/VolumeLayer.tsx` |
| Streaming shader / pool / page-table / LOD | `frontend/src/lib/brickShaders.ts`, `brickPool.ts`, `bricks.ts`, `brickDecode.ts`, `brick.worker.ts`, `lod.ts`; `frontend/src/scene/StreamingVolumeLayer.tsx` |
| Transfer functions + colourmaps | `frontend/src/lib/transferFn.ts`, `colormaps.ts`; `frontend/src/ui/TransferFnEditor.tsx` |
| Slice / clip box | `frontend/src/scene/SliceLayer.tsx`, `ClipBox.tsx`, `clipPlanes.ts` |
| Terrain / features / wells / points / rasters | `frontend/src/scene/TerrainLayer.tsx`, `FeatureLayer.tsx`, `WellLayer.tsx`, `PointCloudLayer.tsx`, `RasterLayer.tsx` |
| Time slider | `frontend/src/ui/TimeSlider.tsx`, `frontend/src/lib/time.ts` |
| Shared store (3-D ↔ panels) | `frontend/src/store.ts` |
