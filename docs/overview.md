# What this simulator is

> **What you'll learn / why it matters.** This page gives you the *whole system at a glance* before any
> geoscience. You'll see the six-layer architecture as a programmer would draw it, the "progressive fusion
> ladder" that defines what the product does at each stage of ambition, what "integration & visualization
> platform" actually means (and what it deliberately does **not** do yet), the tech stack and the reasons
> behind each choice, and a map of every other page in these docs. If you only read one architecture page,
> read this one — everything else hangs off it.

## The one-sentence version

It is a **browser-based 3-D earth model builder** that ingests every kind of subsurface survey (gravity,
electrical, electromagnetic, seismic, satellite radar, borehole logs, geology maps…), reconciles them into a
single coordinate system and data model, **fuses** them onto a shared 3-D grid, transforms the raw physics into
the quantities a geothermal engineer cares about, and renders the result as an interactive volume you can slice,
fly through, and plan a well into — always carrying **uncertainty**.

If the geoscience words above are fuzzy, read [the geothermal primer](geothermal-primer.md) first, then come
back. If you want to know *why fusing surveys is genuinely hard*, read [the core problem](core-problem.md).

## Think of it as an ETL + rendering pipeline for the Earth

Strip away the geophysics and the shape is one a backend engineer already knows:

```
 EXTRACT            TRANSFORM                           LOAD / SERVE        RENDER
 raw survey files → normalize → store → fuse → derive → tile/stream API  → GPU volume viewer
 (many formats)     (one frame   (Zarr)  (one   (rock                       (Three.js)
                     one schema)          grid) physics)
```

- **Extract** = parse a dozen messy scientific file formats (SEG-Y, LAS, EDI, `.stg`, GeoTIFF, QuakeML, GeoJSON).
- **Transform** = put everything in one coordinate frame, one set of units, one schema — three primitive types.
- **Load/serve** = chunked array storage (think tiled image pyramids, but 3-D) streamed over HTTP.
- **Render** = a WebGL/WebGPU volume renderer that ray-marches a 3-D texture on the GPU.

The hard, domain-specific part lives in the middle — the "fuse" and "derive" steps — because, as the
[core problem](core-problem.md) page explains, the inputs do not measure the same physical thing, are not on the
same grid, and disagree in ways that are *physically correct*. Reconciling them is the product.

## The six-layer architecture

The system is layered, bottom-up, so each layer depends only on the ones below it. This is OVERVIEW §4. A
programmer can read it like a dependency graph: the **Spatial Framework** is the kernel everything imports; the
**Client** is the top of the stack.

```
┌─────────────────────────────────────────────────────────────┐
│  CLIENT (browser) — React + TypeScript + Three.js            │  ← what the user sees
│  3D scene · layer manager · volume render · slices · sections │
│  well paths · microseismic · time slider · cross-plot panels  │
└───────────────▲──────────────────────────────┬───────────────┘
                │ REST / WebSocket (tiles, slices, queries, jobs)│
┌───────────────┴──────────────────────────────▼───────────────┐
│  API / SERVING — FastAPI                                      │  ← HTTP boundary
│  project CRUD · upload · chunk/tile streaming · sample/slice/ │
│  isosurface queries · job management                          │
├──────────────────────────────────────────────────────────────┤
│  PROCESSING / COMPUTE (Python geoscience)                     │  ← the domain logic
│  resample→fused grid · gridding/interp · rock-physics ·       │
│  derived-property engine · [pluggable] inversion · geomodel   │
├──────────────────────────────────────────────────────────────┤
│  DOMAIN / MODEL — unified earth model                         │  ← the schema
│  observations · property models · features · fused grid · time│
├──────────────────────────────────────────────────────────────┤
│  INGESTION — per-method format adapters (plugins)            │  ← the parsers
├──────────────────────────────────────────────────────────────┤
│  STORAGE — catalog DB + chunked array store + raw store      │  ← persistence
├──────────────────────────────────────────────────────────────┤
│  SPATIAL FRAMEWORK — CRS, datums, units registry, provenance │  ← the kernel
└──────────────────────────────────────────────────────────────┘

         SYNTHETIC DATA GENERATOR (feeds Ingestion)
   ground-truth earth → forward-model each method → datasets
```

Layer by layer, in CS terms:

| Layer | What it is | Programmer analogy | Page |
|---|---|---|---|
| **Spatial Framework** | The single coordinate system, vertical datum, and units registry every number is expressed in. The kernel: nothing above it invents its own coordinates. | The base types / units library every module imports. | [Coordinates, depth & units](spatial-framework.md) |
| **Storage** | A catalog database (metadata, bounding boxes, provenance) plus a chunked array store for bulk 3-D/4-D data, plus a verbatim copy of every raw file. | A metadata DB + an object store of tiled binary blobs. | [Codebase architecture](architecture.md) |
| **Ingestion** | One *adapter* per survey format that parses a raw file and emits records that fit the data model. Adapters are plugins — adding a method needs no core changes. | Pluggable deserializers behind a common interface. | [Ingestion](ingestion.md) |
| **Domain / Model** | The three primitive record types every method maps onto, plus the **fused earth model** grid and the **time** axis. | Your normalized schema / domain objects. | [The data model](data-model.md) |
| **Processing / Compute** | Resampling onto the shared grid, cross-plotting, **rock-physics** transforms, the derived-property engine, and (later) inversion and implicit geomodeling. | The business-logic / analytics service. | [Fusion](fusion.md), [Rock physics](rock-physics.md), [Inversion](inversion.md) |
| **API / Serving** | A FastAPI app exposing project CRUD, uploads, chunk/tile streaming, sample/slice/isosurface queries, and async job control. | Your REST + WebSocket gateway. | [Codebase architecture](architecture.md) |
| **Client** | A React + Three.js single-page app: a 3-D scene, a layer manager, volume rendering, slice planes, well paths, a time slider, and cross-plot panels. | A SPA with a GPU-accelerated viewport. | [The 3D viewer](visualization.md) |

Off to the side sits the **synthetic data generator** — a separate Python package that builds a *fully known fake
planet*, forward-models what each survey would measure over it, and writes those measurements out in the real
industry file formats. Crucially it feeds the **normal** ingestion path; it is just another data *source*, not a
back door. Because we keep the ground truth, every fusion result can be scored against a known answer. See
[the synthetic data generator](synthetic-data.md).

!!! note "Why bottom-up layering matters here"
    The whole reason the bottom layer is "Spatial Framework, not graphics" is that the deepest, most expensive bug
    in geoscience software is a coordinate mistake — a survey placed in the wrong spot, a depth confused with an
    elevation, feet mixed with metres. By forcing *every* number through one frame and one units registry at the
    bottom, those bugs are caught once, centrally, instead of in twelve different parsers. See
    [Coordinates, depth & units](spatial-framework.md) for the gory details.

## The three primitives (the heart of the model)

Everything that flows through the pipeline is one of exactly **three** record types. This is the schema the whole
system agrees on; the rest is plumbing around it.

| Primitive | Plain meaning | CS analogy | Example |
|---|---|---|---|
| **Observation** | What was measured, where — raw and immutable. | An append-only event log; the source of truth you never mutate. | 400 gravity readings at GPS points; an MT impedance tensor per station. |
| **Property Model** | A continuous 3-D field of *one* physical property. | A dense N-D array (a tensor) with units and a coordinate support. | A resistivity[^res] cube produced by an inversion; a temperature volume. |
| **Geological Feature** | A discrete shape a human (or algorithm) interpreted. | Vector geometry — points, lines, meshes — with attributes. | A fault[^fault] surface; a well[^well] path; a microseismic event cloud. |

[^res]: **Resistivity** — how strongly a rock opposes electrical current (units Ω·m). Its inverse is
    **conductivity** (S/m). Hot, salty, clay-rich rock conducts well (low resistivity); dry, cold, crystalline
    rock resists (high resistivity). It is the single most diagnostic property for geothermal work — see
    [the geothermal primer](geothermal-primer.md) and [electrical methods](survey-methods/electrical.md).
[^fault]: **Fault** — a fracture in the rock where the two sides have moved relative to each other. Faults can be
    the plumbing that lets hot water rise; see [the geothermal primer](geothermal-primer.md).
[^well]: **Well / borehole** — a hole drilled into the ground. The only place we measure rock *directly* rather
    than inferring it from surface physics. See [boreholes](survey-methods/boreholes.md).

Three cross-cutting concepts attach to all of them:

- **Engineering Frame** — one local right-handed metre grid (X = East, Y = North, Z = Up) that every coordinate
  lives in. Real-world georeferencing is an *optional* rigid transform on top. ([details](spatial-framework.md))
- **Provenance** — every artifact records where it came from: which raw files, which transforms, which versions.
  Lineage is mandatory; there is no record without provenance. ([details](data-model.md))
- **Uncertainty** — every property model carries an error / resolution estimate that *survives into fusion*, so a
  smooth, vague survey is never silently trusted as much as a sharp one. ([details](uncertainty.md))

The full schema (exact fields, on-disk layout, the **Fused Earth Model** grid that the primitives get resampled
onto) is [the data model](data-model.md).

## The progressive fusion ladder

The product is defined by a ladder of increasing ambition (OVERVIEW §6). Each rung delivers standalone value, and
each builds on the one below. Critically, **the MVP is rungs 1–3**; the higher rungs are later phases.

```
6. Drilling & well planning      pick a target, plan a trajectory, predict the log
5. Cooperative / joint inversion  solve for the earth that explains ALL surveys at once   ← later (plugin)
4. Geological modeling            build an implicit 3-D geology from all the data
3. Rock-physics transforms        turn resistivity/velocity → temperature/fluid/favorability   ┐
2. Cross-plotting & statistics    sample many volumes at shared points; correlate them          │ MVP
1. Visual co-registration & overlay  one frame; toggle layers, slice, clip, composite          ┘
```

1. **Visual co-registration & overlay** — get every dataset into one frame and *look* at them together: toggle
   layers, change transparency, drag orthogonal slice planes, clip with a box, composite multiple volumes. This
   alone is valuable — most teams cannot view all their surveys in one scene today. ([the 3D viewer](visualization.md))
2. **Cross-plotting & statistics** — because everything is resampled onto one grid, you can sample several
   property volumes *at the same points* and correlate them (e.g. resistivity vs density), cluster, histogram. In
   CS terms: a `JOIN` on cell index across volumes. ([fusion](fusion.md))
3. **Rock-physics transforms** — convert geophysical properties into the geothermal targets via physics
   relations (e.g. Archie's law: resistivity + temperature → fluid likelihood), producing **derived volumes**
   like a *geothermal favorability* field. ([rock physics & favorability](rock-physics.md))
4. **Geological modeling** — build an *implicit geomodel* (a continuous 3-D geology constrained by all the data).
5. **Cooperative / joint inversion** — instead of inverting each survey alone, solve for the single earth that
   best explains *all* surveys simultaneously, coupling them structurally. This is a hard inverse problem and is
   a **pluggable, later** module. ([inversion](inversion.md))
6. **Drilling target & well-path planning** — choose where to drill, design a trajectory that honours real
   drilling geometry, intersect it with the model, and predict temperature and risk along the path.
   ([well planning](well-planning.md))

**Uncertainty climbs every rung.** Confidence volumes are themselves renderable layers, so you can always see
*how much the model deserves to be trusted* at each point. ([uncertainty](uncertainty.md))

## "Integration & visualization platform" — what that means (and excludes)

This is the single most important scoping decision, so read it carefully.

!!! abstract "The MVP consumes already-inverted models; it does not (yet) do inversion"
    Geophysics has two big halves:

    - **Forward modeling** — *given an earth, predict what an instrument would measure.* (Easy-ish: simulate the
      physics.)
    - **Inversion** — *given the measurements, reconstruct the earth.* This is an **inverse problem**: ill-posed,
      non-unique, computationally heavy. It is the geophysical equivalent of asking "what code produced this
      output?" — many answers fit, and you need regularization/priors to pick one.

    The platform's MVP **consumes the *output* of inversion** — finished resistivity cubes, velocity models,
    density grids — and focuses on *fusing, transforming, and visualizing* them. It also stores the **raw
    observations** verbatim. What it does **not** do in the MVP is *run* the inversion to turn raw observations
    into property models. That capability is a **pluggable module added in a later phase**
    ([inversion](inversion.md)), behind a stable interface, so the day it arrives its output is *just another
    property model* and reuses all the existing storage, fusion, and rendering code.

This scoping is what makes a one-person, browser-first build tractable: the hardest research code (3-D inversion
solvers) is deferred, but designed-for from day one. Two design principles enforce it:

- **Ingestion emits a `PropertyModel` when a file is already inverted, an `Observation` when it is raw.** Raw-only
  surveys simply *wait* for the inversion plugin; they are still stored and visualizable as observations.
- **Inversion output is just another `PropertyModel`** → it flows into storage, fusion, and the viewer through the
  exact same path as an ingested model. No special case.

The other scoping decisions worth knowing:

- **Research / R&D platform.** Modularity is a first-class constraint: a new survey method is *add an adapter +
  a property type + (optionally) a transform/renderer*, with **no core changes**. ([plugins → architecture](architecture.md))
- **Local-first, single user** to start — but designed to grow to hosted/multi-user.
- **"Simulated data for now."** Because real multi-method datasets are hard to assemble, a
  [synthetic data generator](synthetic-data.md) ships with the build, so there is realistic, *ground-truthed*
  data on day one.

## The tech stack — and why each piece

| Concern | Choice | Why (the real reason) |
|---|---|---|
| **Backend language** | **Python** + FastAPI | The geoscience ecosystem *is* Python: `pyproj` (coordinates), `xarray`/Zarr (chunked arrays), `rasterio` (rasters), `verde` (gridding), `segyio` (seismic), `lasio` (well logs), `ObsPy` (seismology), and later `SimPEG`/`PyGIMLi` (inversion), `GemPy` (geomodeling). Re-implementing these would be a multi-year mistake. FastAPI gives async I/O and typed schemas on top. |
| **Frontend** | **React + TypeScript + Vite**, state via Zustand | Standard, fast, typed SPA stack. Zustand is a small global store for layer/visibility/time state. |
| **3-D rendering** | **Three.js** (via react-three-fiber) | The viewer's core need is **GPU volume rendering**: ray-marching a `Data3DTexture` in a custom shader, with per-property transfer functions. Three.js exposes raw WebGL/WebGPU and 3-D textures, which higher-level globe libraries do not. |
| **Geospatial context** | DEM[^dem] shaded-relief terrain + optional online basemap tiles | A subsurface model needs the *surface* it sits under; the terrain and basemap are draped so voxels and geography share one frame. |
| **Bulk 3-D/4-D storage** | **Zarr v3** (chunked, sharded, multiscale) | Zarr is "tiled image pyramids, generalized to N-D and streamable over HTTP range requests." A 64³ chunk maps directly to one GPU brick; a power-of-two pyramid gives level-of-detail. |
| **2-D rasters** | Cloud-Optimized GeoTIFF (COG) | The standard web-streamable raster format. |
| **Surfaces / vectors / point clouds** | glTF / VTK / GeoJSON / LAS | Right tool per geometry kind. |
| **Catalog DB** | PostgreSQL + PostGIS | Metadata, bounding boxes, provenance, with real spatial indexing. |
| **Async jobs** | RQ + Redis behind one job contract | Long compute (resampling, transforms, future inversions) runs as tracked background jobs; the client subscribes over WebSocket. |

[^dem]: **DEM (Digital Elevation Model)** — a raster grid of surface elevation, i.e. a heightmap of the terrain.
    The default source here is Copernicus GLO-30 (~30 m global). See [coordinates & depth](spatial-framework.md).

## A tour of the rest of the docs

These pages are ordered as a course; top-to-bottom you never hit an undefined term.

- **The Big Picture** — you are here. Also: [geothermal energy for programmers](geothermal-primer.md) (start
  there if "geothermal" is fuzzy) and [the core problem](core-problem.md) (why fusion is hard).
- **Foundations** — [coordinates, depth & units](spatial-framework.md) (the kernel) and
  [the data model](data-model.md) (the schema every method maps onto).
- **The Survey Methods** — one page per family of measurement, each with the physics, what it can and cannot see,
  the **real file format with an annotated example**, and the normalized output. Start with
  [how to read these pages](survey-methods/index.md), then:
  [potential fields](survey-methods/potential-fields.md),
  [electrical](survey-methods/electrical.md),
  [electromagnetic](survey-methods/electromagnetic.md),
  [seismic](survey-methods/seismic.md),
  [InSAR](survey-methods/insar.md),
  [boreholes](survey-methods/boreholes.md),
  [geology & geochemistry](survey-methods/geology-geochem.md).
- **Merging the Data** — [ingestion](ingestion.md), [fusion](fusion.md),
  [rock physics & favorability](rock-physics.md), [uncertainty](uncertainty.md).
- **Using the Model** — [the 3-D viewer](visualization.md), [drilling & well planning](well-planning.md),
  [forward modeling & inversion](inversion.md).
- **Reference** — [the synthetic data generator](synthetic-data.md),
  [codebase architecture](architecture.md), [glossary](glossary.md).

## Key takeaways

- It is an **ETL + GPU-rendering pipeline for the subsurface**: parse messy survey files → normalize to one
  frame/schema → store as chunked arrays → fuse onto a shared grid → derive geothermal targets → render and plan.
- The architecture is **six bottom-up layers** with the **Spatial Framework** as the kernel and the **three
  primitives** (Observation / Property Model / Feature) as the schema everything agrees on.
- The product is a **progressive fusion ladder**; the **MVP is rungs 1–3** (overlay → cross-plot → rock physics).
- It is an **integration & visualization platform**: it *consumes already-inverted models* and raw observations;
  running inversion is a **pluggable later phase**, designed-for but not in the MVP.
- The stack is **Python** (for the geoscience ecosystem) + **Three.js** (for GPU volume rendering) + **Zarr**
  (for streamable N-D arrays).
- A **synthetic, ground-truthed data generator** ships with the build so the whole pipeline can be exercised and
  *scored against a known answer* from day one.

## Where this lives in the code

- Backend package root: `backend/geosim/` — sub-packages mirror the layers: `catalog/` (storage metadata),
  `storage/` (Zarr/COG/glTF), `ingestion/` (adapters), `geomodel/` & `inversion/` (compute), `synthgen/`
  (synthetic generator), `api/` (FastAPI app, e.g. `api/app.py`), `plugins/` (the extension registry).
- Frontend root: `frontend/src/` — `scene/` (the Three.js layers: `VolumeLayer.tsx`, `SliceLayer.tsx`,
  `WellLayer.tsx`…), `lib/` (`api.ts`, `volume.ts`, `bricks.ts`, `favorability.ts`…), `store.ts` (Zustand state).
