# Geothermal Underground Simulator — Architecture Overview

> This is the top-level design document. Each section in §10 gets its own detailed
> design doc in this `design/` folder (e.g. `01-spatial-framework.md`). This file is
> the map; those are the territory.

## Context

The goal is a **browser-based 3D underground simulator** for geothermal drilling that fuses *every* subsurface survey method — gravity, magnetics, electrical resistivity (ERT), induced polarization (IP), electromagnetic (EM/TEM/AEM), magnetotellurics (MT), seismic (reflection/refraction/passive), InSAR, well logs, heat-flow, geochemistry — into a single georeferenced 3D earth model and renders it in the browser.

Scoping decisions:
- **Integration & visualization platform.** The system consumes *already-processed/inverted* property models (resistivity volumes, velocity models, density grids) plus raw observations, fuses them, and visualizes. Forward-modeling and geophysical inversion are **pluggable modules added in a later phase**, not the MVP.
- **Research / R&D platform.** Modularity and pluggable survey methods are first-class design constraints.
- **Stack.** React + TypeScript + Three.js frontend; Python + FastAPI backend (to access the geoscience ecosystem).
- **Local-first, single user** to start; designed so it can grow to hosted/multi-user later.
- "Simulated data for now," so a **synthetic data generator** is part of the build: define a ground-truth synthetic earth, forward-model each survey type, and emit realistic datasets the platform ingests.

---

## 1. The core problem to solve

Every survey method produces data that differs along several axes simultaneously. The architecture exists to reconcile these differences into one coherent, queryable, viewable model:

| Axis | Range across methods |
|---|---|
| **Dimensionality** | 0D point samples → 1D soundings/well logs → 2D profiles/grids → 3D volumes → 4D time-lapse (monitoring) |
| **Physical property measured** | density, magnetic susceptibility, resistivity/conductivity, chargeability, seismic velocity (Vp/Vs), surface deformation, temperature, fluid chemistry |
| **Coordinate frame** | lat/lon, UTM, local grid; elevation vs depth; differing vertical datums |
| **Resolution & depth of investigation** | shallow & sharp (ERT, seismic) vs deep & smooth (MT, gravity); each has a resolution kernel |
| **Sensitivity / non-uniqueness** | gravity & MT are smooth and non-unique; seismic gives sharp structure; each "sees" different rock physics |
| **Uncertainty** | every model carries error/resolution that must survive into fusion |

**Key insight:** methods don't measure the same thing, so fusion happens through (a) a **shared spatial frame**, (b) a **common resampling grid** for cross-comparison, and (c) **rock-physics relationships** that link geophysical properties to the geothermal targets you actually care about (temperature, permeability, fluid, lithology).

---

## 2. Conceptual data model (the heart of the system)

Three primitive types, all georeferenced into one **Project Spatial Frame**:

1. **Observations** — raw/measured survey data tied to acquisition geometry. Immutable record of *what was measured where* (gravity station readings, ERT apparent-resistivity pseudosections, MT impedance tensors, seismic traces, InSAR scenes, LAS log curves). Never destroyed.
2. **Property Models** — derived continuous fields of one physical property over a region (regular grid, octree, or unstructured mesh): resistivity, density, susceptibility, velocity, temperature, chargeability. Output of inversion/processing. Each carries **units**, **support geometry**, and **uncertainty**.
3. **Geological Features** — discrete geometric interpretation: horizons/surfaces, faults, geological unit solids, well paths, fracture networks, microseismic event clouds. Vector geometry.

**Project Spatial Frame** = chosen projected CRS (e.g. a UTM zone) + vertical datum + local origin + region-of-interest (ROI) bounding box + depth range. Everything is transformed into this frame on ingest via `pyproj`.

**The Fused Earth Model** = a canonical 3D model grid (regular voxels, or octree/unstructured mesh) covering the ROI, onto which any property model can be *resampled* to enable overlay, cross-plotting, and joint analysis — **without destroying** the native-resolution originals. The fused grid is the common ground for visualization compositing and derived-property math.

**Time (4D)** is a first-class dimension: monitoring datasets (InSAR, microseismic, repeat surveys) attach a time axis; the UI gets a time slider.

---

## 3. Survey method catalog → data formats → normalized output

Each method gets an **ingestion adapter** (plugin). This table is the contract for the ingestion layer:

| Method | Sensitive to | Native formats | Normalized primitive |
|---|---|---|---|
| Gravity / gradiometry | density | CSV/columns, `.grd`, netCDF, BGI | point obs + anomaly grid → density volume |
| Magnetics (ground/aero) | susceptibility | ASEG-GDF, CSV, `.grd` | grid → susceptibility volume |
| ERT | resistivity | AGI `.stg`, Res2DInv, UBC, ABEM | resistivity volume |
| Induced Polarization | chargeability | AGI, UBC | chargeability volume |
| EM (TEM / airborne EM) | conductivity | ASEG-GDF, USF, `.xyz` | conductivity-depth volume |
| Magnetotellurics (MT) | deep resistivity | EDI (impedance tensor), ModEM/UBC inverted | app-resistivity curves + 3D resistivity volume |
| Seismic reflection/refraction | impedance/structure/velocity | SEG-Y, velocity cubes, horizon exports | velocity volume + horizons/faults |
| Passive / microseismic | fracture activity | QuakeML, CSV catalogs | event point cloud (4D) |
| InSAR | surface deformation | GeoTIFF time-series, CSV | deformation raster time-series (4D) |
| Well logs | ground-truth props | LAS, DLIS | 1D curves along well path |
| Temperature / heat-flow | geothermal target | CSV, LAS | point/1D temperature field |
| Geology maps / structure | lithology/structure | Shapefile, GeoJSON, GeoPackage | surface features + unit solids |
| Geochemistry | fluid/gas indicators | CSV/LIMS exports | point samples |

Adapter interface (sketch): `parse(raw) → { observations[], propertyModels[], features[], crs, units, provenance }`.

---

## 4. System architecture (layers)

```
┌─────────────────────────────────────────────────────────────┐
│  CLIENT (browser) — React + TS + Three.js (react-three-fiber)│
│  3D scene · layer manager · volume render · slices · sections │
│  well paths · microseismic · time slider · cross-plot panels  │
└───────────────▲──────────────────────────────┬───────────────┘
                │ REST/WebSocket (tiles, slices, queries, jobs)  │
┌───────────────┴──────────────────────────────▼───────────────┐
│  API / SERVING — FastAPI                                      │
│  project CRUD · upload · chunk/tile streaming · sample/slice/ │
│  isosurface queries · job management                          │
├──────────────────────────────────────────────────────────────┤
│  PROCESSING / COMPUTE (Python geoscience)                     │
│  resample→fused grid · gridding/interp (verde/kriging) ·      │
│  rock-physics transforms · derived-property engine ·          │
│  [pluggable] inversion (SimPEG/PyGIMLi) · geomodel (GemPy)    │
├──────────────────────────────────────────────────────────────┤
│  DOMAIN / MODEL — unified earth model                         │
│  observations · property models · features · fused grid · time│
├──────────────────────────────────────────────────────────────┤
│  INGESTION — per-method format adapters (plugins)             │
├──────────────────────────────────────────────────────────────┤
│  STORAGE — catalog DB + chunked array store + raw store       │
├──────────────────────────────────────────────────────────────┤
│  SPATIAL FRAMEWORK — CRS, datums, units registry, provenance  │
└──────────────────────────────────────────────────────────────┘

         SYNTHETIC DATA GENERATOR (feeds Ingestion)
   ground-truth earth → forward-model each method → datasets
```

**Plugin/extensibility framework** spans ingestion, properties, transforms, and inversion engines — the R&D requirement. New survey method = register an adapter + property type + (optional) transform/renderer, no core changes.

---

## 5. Technology stack

**Frontend**
- React + TypeScript + Vite; state via Zustand.
- 3D engine: **Three.js via react-three-fiber**. Volume rendering through GPU ray-marching (`Data3DTexture` + custom shaders) with per-property transfer functions.
- Geospatial context (terrain, basemap, georeferencing): **deck.gl** layer or CesiumJS terrain; custom CRS/camera bridge so subsurface voxels and geographic terrain share one frame.
- Analysis panels (sounding curves, log tracks, cross-plots): Observable Plot / D3.

**Backend**
- **Python + FastAPI** — unlocks the geoscience ecosystem: `pyproj`, `xarray`, `rasterio`, `verde`, `discretize`, `segyio`, `lasio`, `ObsPy`, and later `SimPEG`/`PyGIMLi` (inversion), `GemPy` (implicit geomodel).
- Long jobs: background tasks first; Celery/RQ when needed.

**Storage (local-first)**
- **Catalog/metadata:** PostgreSQL + PostGIS (or SQLite + SpatiaLite for pure-local). Projects, datasets, CRS, units, bounding boxes, provenance, layer definitions.
- **Bulk arrays:**
  - 3D/4D volumes → **Zarr** (chunked, lazy, multiresolution pyramids, web-streamable) via xarray.
  - 2D grids/rasters → **Cloud-Optimized GeoTIFF (COG)**.
  - Surfaces/solids/meshes → glTF / VTK; lightweight vectors → GeoJSON.
  - Point clouds (microseismic) → LAS/LAZ or 3D Tiles / Potree.
- **Raw store:** original survey files kept verbatim with provenance links.

**Streaming to the viewer:** multiresolution Zarr bricks → 3D textures client-side (ray-marched); large volumes get octree LOD; option for server-side isosurface extraction (marching cubes) and 3D-Tiles streaming later.

---

## 6. Fusion / interpretation — progressive levels

The platform delivers value incrementally up this ladder:

1. **Visual co-registration & overlay** *(MVP)* — everything in one frame; toggle layers, transparency, orthogonal slice planes, clipping box, multi-volume compositing.
2. **Cross-plotting & statistics** — sample multiple property volumes at shared points; cross-plot (e.g. resistivity vs density), cluster, histogram.
3. **Rock-physics transforms** — convert/constrain properties toward geothermal targets (e.g. resistivity + temperature relations → fluid/temperature likelihood); produce **derived volumes** like a "geothermal favorability" field.
4. **Geological modeling** — build an implicit geomodel (GemPy) constrained by all data sources.
5. **Cooperative / joint inversion** *(pluggable, later)* — structural coupling (cross-gradient), shared-mesh inversion via SimPEG/PyGIMLi.
6. **Drilling target & well-path planning** — pick targets, plan trajectories, intersect with model, estimate temperature/risk along path, export.

Uncertainty propagates through every level; confidence volumes are renderable layers.

---

## 7. Visualization techniques (the 3D viewer)

- **Volume rendering** (ray-marched) with per-property transfer functions and blending.
- **Orthogonal slice planes** (X/Y/Z) and **arbitrary cross-sections / fence diagrams**.
- **Isosurfaces** (marching cubes) for thresholded properties.
- **Surfaces/horizons & fault meshes**.
- **Well paths** as tubes with log curves color-mapped along them.
- **Microseismic point clouds** animated over time.
- **Vector glyphs / streamlines** (EM/MT fields, modeled fluid flow).
- **Terrain surface** with draped basemap and survey-coverage footprints.
- **Time slider** for 4D datasets; exploded-layer and clipping-box modes.

---

## 8. Synthetic data generator (so there's data on day one)

A standalone module that:
1. Defines a **ground-truth synthetic earth** (layered geology + faults + a geothermal anomaly: hot, conductive, altered zone) as property volumes (density, susceptibility, resistivity, velocity, temperature).
2. **Forward-models** what each survey *would* measure over it (with realistic noise, resolution loss, and depth-of-investigation effects), emitting files in the native formats from §3.
3. Feeds those through the normal ingestion adapters — so the integration pipeline is exercised end-to-end and fusion can be validated against known ground truth.

---

## 9. Build roadmap (phases)

- **Phase 0 — Foundations:** repo scaffold (frontend + backend), Project Spatial Frame, CRS/units registry, catalog DB schema, storage layout (Zarr/COG/raw), provenance model.
- **Phase 1 — Synthetic earth + one method:** synthetic generator producing a resistivity volume; one ingestion adapter; Zarr storage; basic 3D viewer with one volume + slice planes. *Proves the vertical slice end-to-end.*
- **Phase 2 — Multi-method ingestion:** adapters for gravity, magnetics, MT, seismic, well logs; normalized primitives; layer manager; multi-volume overlay & compositing.
- **Phase 3 — Fusion L1–L3:** fused-grid resampling, cross-plotting, rock-physics transforms, derived "favorability" volume, uncertainty layers.
- **Phase 4 — Features & 4D:** horizons/faults, well paths with logs, microseismic + InSAR time-series, time slider.
- **Phase 5 — Geomodel + planning:** GemPy implicit model; drilling target & well-path planning/export.
- **Phase 6 — Pluggable inversion:** SimPEG/PyGIMLi forward+inverse modules behind the plugin interface; joint/cooperative inversion.

---

## 10. Detailed design docs (one per section)

Each becomes its own document in this folder before any code is written:

| # | Doc | Scope |
|---|---|---|
| 1 | `01-spatial-framework.md` | CRS/datum strategy, depth vs elevation, units registry, ROI definition |
| 2 | `02-data-model.md` | Exact schema for observations / property models / features / fused grid / provenance; on-disk Zarr/COG conventions |
| 3 | `03-ingestion-adapters.md` | Plugin interface, per-method parsers, normalization rules, the format table as a contract |
| 4 | `04-storage-and-serving.md` | Catalog DB schema, chunking/pyramid strategy, tile/slice/sample API design |
| 5 | `05-synthetic-data-generator.md` | Ground-truth earth spec and per-method forward models |
| 6 | `06-visualization-engine.md` | Three.js scene graph, volume ray-marching, slicing, terrain/CRS bridge, performance/LOD |
| 7 | `07-fusion-and-rock-physics.md` | Resampling, cross-plot, transform engine, uncertainty |
| 8 | `08-plugin-architecture.md` | How new methods, properties, transforms, and inversion engines register |
| 9 | `09-drilling-well-planning.md` | Targets, trajectories, intersection & risk |
| 10 | `10-inversion-integration.md` *(later)* | SimPEG/PyGIMLi boundaries and job orchestration |

---

## Verification approach

Because this is greenfield, "verification" at each phase = an **end-to-end vertical slice**:
- Phase 1 success = synthetic resistivity volume generated → ingested → stored as Zarr → streamed → ray-marched in the browser with a working slice plane, all in the correct geographic location.
- Each later phase adds a method/feature and is verified by visually confirming correct co-registration against the synthetic ground truth, plus unit tests on adapters (format round-trips) and transforms (known-input/known-output).
