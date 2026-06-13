# Design Decisions Log

Single source of truth for resolved design forks. Each row is a decision the user
confirmed during the §10 design-doc pass. **Bold "DEVIATION"** marks where the choice
differs from the doc's originally-drafted default (those docs were patched to match).

## Spatial framework (doc 01)
- Internal **Engineering Frame** (ENU, metres, Z-up, floating origin); georeferencing is an optional rigid transform.
- Arrays always stored in Engineering coords → local→georeferenced is free.
- Horizontal CRS: **auto UTM by default** (1–10 km common case); custom ROI-centred TM for basin/regional; polar stereographic at high latitude.
- Vertical canonical: **orthometric elevation, EGM2008 / MSL**; depth/MD/TVD derived.
- Units: **pint** registry, canonical SI-ish per property; °C default display.
- Terrain: **Copernicus GLO-30** DEM.
- Typical extent: **single-site 1–10 km**, larger basin/regional supported.

## Data model (doc 02)
- Fused-grid resolution: **auto from data (median native spacing), clamped to max cell count**, overridable.
- Derived-artifact versioning: **keep all versions** (content-addressed bulk; unchanged arrays shared).
- Categorical/lithology fields: **PropertyModel with a class-probability axis** (one channel per class).
- Zarr v3, sharded, `(t,z,y,x)` Z-up, ~64³ chunks, OME-Zarr multiscale; arrays in Engineering coords only.
- Observations immutable; gridding is a separate provenance-tracked PropertyModel.

## Ingestion (doc 03)
- Raw→volume gridding: **separate user-initiated, parameterized step** (raw stays raw).
- Pre-inverted files: **emit PropertyModel when already inverted, Observation when raw** (raw-only waits for inversion plugin).
- Partial-file failure: **fixed threshold >10% records dropped = fail**, overridable per upload.
- Adapters are pure format readers; CRS/datum/unit normalization delegated to doc 01.

## Storage & serving (doc 04)
- Catalog DB: **PostgreSQL + PostGIS from the start**. **[DEVIATION — drafted default was SQLite+SpatiaLite]**
- Slice delivery: **raw float32 to client** (shared GPU transfer function).
- Async jobs: **RQ + Redis from the start**, behind one stable job contract. **[DEVIATION — drafted default was FastAPI BackgroundTasks]**
- Volumes: Zarr v3, 64³ chunks = GPU bricks, Blosc+zstd, power-of-two pyramid; brick address = Zarr chunk path.
- API: REST + HTTP-range/Zarr-over-HTTP for data; WebSocket for job progress.

## Synthetic generator (doc 05)
- Forward-model fidelity: **rigorous physics for 2–3 key methods first (MT, gravity, seismic)**; other methods plausible/analytic, upgraded later. **[DEVIATION — drafted default was T0-plausible for all methods first]**
- Flagship scenario: **`great-basin-v1`** (Basin & Range / Nevada hydrothermal play).
- Rock-property values: **defensible textbook defaults now**, refine later.
- Principle: one geology → all properties (mutually consistent); emit native formats; retain ground truth as scoring oracle.

## Visualization (doc 06)
- Render backend: **WebGL2 floor + WebGPU progressive enhancement**.
- Client volume ceiling: **~512³ working set (~256–512 MB brick pool)**, escalate to server beyond.
- Basemap: **DEM shaded-relief default + optional online tiles** (offline-safe).
- Slices client-side from resident bricks (interactive) + server-side for arbitrary geometry.
- Viewer world space IS the Engineering Frame; CRS never touches the GPU.

## Fusion & rock-physics (doc 07)
- Starter rock-physics library: **full table** (resistivity→temp/fluid, velocity→porosity, alteration index, microseismic→fracture density, Waxman-Smits/dual-water, permeability proxies). **[DEVIATION — drafted default was the minimal resistivity→temp + velocity→porosity set]**
- Favorability combination: **weighted-linear + fuzzy-logic shipped; Bayesian deferred** until known-occurrence training data. **[DEVIATION — drafted default was weighted-linear only by default]**
- Uncertainty: **delta-method everywhere, Monte-Carlo opt-in** per nonlinear transform.
- Non-destructive resampling; footprint honesty (NaN beyond coverage/DOI); derived volumes are PropertyModels.

## Plugin architecture (doc 08)
- Discovery: **hybrid — decorators (first-party) + entry-points (third-party)**, one registry.
- Execution/trust: **in-process, trusted** (local single-user); sandbox isolated to one future hosted seam.
- Frontend: **backend declares renderers/colormaps via `/api/capabilities`; client has a fixed renderer catalog**; no third-party JS for now.
- Six extension points; Method Bundle packages adapter + property type(s) + transfer function + optional forward model/transform.

## Drilling & well planning (doc 09)
- Trajectory: **geometric (min-curvature + dogleg) + a crude drillability flag in core**; full mechanics as later plugin. **[DEVIATION — drafted default was pure geometric, no drillability flag]**
- Exports: **CSV survey + CSV log + WITSML-trajectory**. **[DEVIATION — drafted default was CSV only]**
- Induced-seismicity risk: **later, as a RiskPlugin** (core risk score stays simple).
- Planned well = deviation survey identical to ingested wells; predicted log via sample-along-line.

## Inversion (doc 10 — later phase)
- First engine: **gravity (plumbing proof) → then ERT (PyGIMLi)**; MT deferred to server tier.
- Compute boundary: **local default + optional remote/GPU worker** per-engine compute profile.
- Coupling: **build 5a single-method + 5b cooperative (job DAG); 5c joint (cross-gradient/PGI) stays roadmap**.
- Inversion output is just another PropertyModel → storage/fusion/viz fully reused.
