# Build Roadmap — Milestones

Concrete, dependency-ordered milestones derived from `OVERVIEW.md §9` and the
resolved `DECISIONS.md`. Each milestone is a **shippable vertical increment** with
an **exit criterion** you can actually observe. Effort tags: **S** ≈ days,
**M** ≈ 1–2 wks, **L** ≈ 3–5 wks, **XL** ≈ 6+ wks (solo-ish pace; rough).

## Guiding principle — lean spine first, heavy science layered on

Your decisions front-load real scientific weight: **rigorous physics for MT/gravity/
seismic**, the **full rock-physics table**, **cooperative inversion**, **Postgres+Redis**
infra. The risk is sinking months into rigorous solvers before anything renders.

**Recommendation:** build the **integration spine end-to-end on cheap (T0) synthetic
data first** (M0–M2), so storage → ingestion → serving → viewer → fusion is proven and
visible. Then deepen the science (rigorous forward models M3, full rock-physics M5) as
*upgrades to a working pipeline* rather than prerequisites. The rigorous-physics work
(M3) can run **in parallel** with fusion work (M4–M5) once the pipeline consumes data.

```
        ┌── M3 rigorous forward models (MT/grav/seismic) ──┐  (parallelizable)
M0 ─ M1 ─ M2 ─┤                                            ├─ M6 ─ M7 ─ M8 ─ M9
        └── M4 ─ M5 fusion + rock-physics ─────────────────┘
critical path: M0 → M1 → M2 → M4 → M5 → M6 → M7   (M3, M8 sit beside it; M9 later phase)
```

---

## M0 — Foundations & walking skeleton  **(L)**
*Docs: 01, 02, 04, 08* · *Phase 0*

Stand up the bones so everything later has a home.

- Monorepo scaffold: `backend/` (FastAPI, Python) + `frontend/` (React+TS+Vite) + `design/`.
- **Spatial framework** (doc 01): `SpatialFrame`, Engineering-Frame transforms (pyproj), units registry (pint), depth/MD/TVD helpers.
- **Catalog DB** (doc 04): **PostgreSQL + PostGIS**, SQLAlchemy models for the doc-02 schema (datasets, property_models, observations, features, provenance, raw_files, layers, views, jobs), Engineering-metre bbox index. Migrations (Alembic).
- **Storage layout** (doc 04 §3): project-as-directory for `arrays/ grids/ meshes/ vectors/ points/ raw/ cache/`; Zarr v3 writer/reader helpers; raw store (content-addressed).
- **Job system** (doc 04): **RQ + Redis** worker, `jobs` table, WS progress endpoint — one job contract.
- **Plugin registry** (doc 08): the `PluginRegistry` + 6 extension points + property-type registry, decorator + entry-point discovery. (Empty but wired.)

**Exit:** create a project via API → row in Postgres + a project directory on disk; enqueue a trivial RQ job → progress streams over WS to a stub frontend page. No science yet.

---

## M1 — The vertical slice  **(M)**
*Docs: 02, 04, 05(min), 06* · *Phase 1* · **the make-or-break milestone**

One property volume, all the way through, rendered correctly in the browser.

- Minimal synthetic source: a hand-built **resistivity volume** (a conductive blob in layers) written as a doc-02 `PropertyModel` Zarr group (+ `_sigma`, pyramid).
- One ingestion path → catalog row + Zarr on disk + provenance.
- Serving: `/property-models/{id}/zarr/...` brick streaming + group metadata + a `/slice` endpoint (raw f32).
- **Viewer** (doc 06): react-three-fiber scene in the Engineering Frame, **WebGL2 GPU ray-marching** of the volume via `Data3DTexture`, a transfer-function (colormap + opacity), one **orthogonal slice plane**, clip box, orbit camera. Brick/LOD streaming against the pyramid.

**Exit:** load the synthetic resistivity volume in a browser, ray-marched in the right place, scrub a slice plane through it, change the colormap live. This proves storage↔serving↔viewer.

---

## M2 — Synthetic earth + multi-method ingestion + layer manager  **(L)**
*Docs: 03, 05 (T0 tier), 06* · *Phase 2*

Make the platform multi-method and give it real-feeling data.

- **Synthetic generator `great-basin-v1`** (doc 05) at **T0 fidelity** (degrade-the-truth / analytic) for *all* methods: one ground-truth earth → density, susceptibility, resistivity, Vp/Vs, temperature (+ the extra fields the full rock-physics table needs: alteration, fracture density, porosity, salinity). Emits native files (CSV, GeoTIFF, EDI, SEG-Y, LAS, …) + retained ground-truth bundle.
- **Ingestion adapters** (doc 03) for the core methods: gravity, magnetics, MT, ERT, seismic, well logs, InSAR. Each emits doc-02 primitives; raw stays raw; pre-inverted → `PropertyModel`, raw → `Observation`.
- **Gridding** as an explicit user-initiated step (scattered obs → volume).
- **Layer manager** (doc 06): datasets → toggleable/blendable layers, per-layer transfer functions, multi-volume compositing, terrain surface (Copernicus DEM in georef mode / flat in local).

**Exit:** generate `great-basin-v1`, ingest 5–6 methods, and see them co-registered as blendable layers over terrain in one 3D scene — the "many surveys, one earth" demo.

---

## M3 — Rigorous forward models: MT, gravity, seismic  **(XL)**  *(parallelizable)*
*Doc: 05 (T1 tier)* · *your fidelity decision*

Upgrade the synthetic generator's three flagship methods from plausible to physically rigorous, so fusion/inversion can be validated against real physics.

- **Gravity** T1: proper Newtonian integration of the density volume (prism/FFT).
- **MT** T1: 1D/3D EM forward (e.g. via SimPEG/`empymod`) → realistic impedance/apparent-resistivity with skin-depth-correct DOI.
- **Seismic** T1: at least convolutional/acoustic synthetic + realistic velocity→reflectivity (toward an elastic FD later).
- Each behind the uniform `ForwardModel` contract; T0 stays as the fast fallback.

**Exit:** the MT/gravity/seismic synthetic responses are physically defensible (skin depth, DOI, resolution falloff visible) and round-trip through ingestion unchanged. *Can land any time after M2; doesn't block M4–M5.*

---

## M4 — Fusion L1–L2: fused grid, resample, cross-plot  **(L)**
*Docs: 02 (§11), 07, 04* · *Phase 3 (part 1)*

- **FusedEarthModel** (doc 02 §11): regular voxel grid, auto-resolution (median native spacing, clamped); native models **resampled in as referenced layers**, originals read-only.
- **Resampling engine** (doc 07): per-support interpolation, footprint honesty (NaN beyond coverage/DOI).
- **Cross-plot & stats** (doc 07): multi-volume `sample`, 2D/3D cross-plots, clustering → analysis panels (Observable Plot/D3) linked to the 3D view.

**Exit:** build a fused grid from the `great-basin-v1` layers; cross-plot resistivity vs density vs velocity at shared cells and see the geothermal anomaly separate as a cluster.

---

## M5 — Rock-physics (full table) + favorability + uncertainty  **(XL)**
*Doc: 07* · *Phase 3 (part 2)* · *your full-table decision*

- **Transform engine** (doc 07 + doc 08 registry): declarative + Python `apply()`, pint-checked, versioned.
- **Full rock-physics library:** resistivity→temperature/fluid (Archie + Arps), velocity→porosity, clay/alteration index, microseismic→fracture density, Waxman-Smits/dual-water, permeability proxies.
- **Geothermal favorability:** weighted-linear **and** fuzzy-logic combination (Bayesian deferred), user-configurable evidence weights + membership curves → a `[0,1]` derived volume.
- **Uncertainty:** delta-method propagation everywhere, Monte-Carlo opt-in per nonlinear transform → paired confidence volumes; validate against the retained ground truth.

**Exit:** produce a "geothermal favorability" volume over `great-basin-v1`, with a confidence volume, that highlights the known synthetic anomaly — and tune the weights live.

---

## M6 — Features & 4D  **(L)**
*Docs: 02 (§5), 06* · *Phase 4*

- Horizons/faults (glTF meshes), **well paths** with deviation surveys + log curves color-mapped along the tube, unit solids.
- **4D**: microseismic point clouds (time-animated), InSAR deformation time-series, the **time slider** unioning dataset epochs.

**Exit:** scrub a time slider and watch microseismic events accumulate + InSAR deformation evolve, with horizons/faults/wells in the same scene.

---

## M7 — Drilling target & well planning  **(L)**
*Doc: 09* · *Phase 5 (part 1)*

- Pick a target on a favorability/temperature isosurface; plan a trajectory (deviation survey, **min-curvature + dogleg**, **crude drillability flag**).
- **Predicted log** via `sample_along_line` across fused volumes (with uncertainty); geothermal outputs (BHT, reservoir intersection length, productive-fracture count); transparent weighted risk score.
- **Export:** CSV survey + CSV log + **WITSML-trajectory**.

**Exit:** plan a well to the synthetic hot zone, see the predicted temperature/lithology/risk log along its path, and export a WITSML trajectory.

---

## M8 — Implicit geomodel (GemPy)  **(L)**  *(optional/parallel)*
*Doc: 07/OVERVIEW §6 L4* · *Phase 5 (part 2)*

Build an implicit geological model (GemPy) constrained by horizons/faults/wells; expose it as unit-solid features + a lithology PropertyModel (class-probability axis). Sits beside M7; not on the critical path.

---

## M9 — Inversion (later phase)  **(XL+)**
*Doc: 10* · *Phase 6* · *explicitly non-blocking*

- **5a single-method:** gravity engine first (plumbing proof) → **ERT** (PyGIMLi) as first geothermally meaningful engine. Output is just a `PropertyModel` → reuses all storage/fusion/viz.
- **Compute:** local default + optional remote/GPU worker per engine (MT deferred to server tier).
- **5b cooperative:** sequential job-DAG coupling (one method's model constrains another). **5c joint** (cross-gradient/PGI) stays roadmap.
- **Validation:** invert the synthetic data, score against ground truth.

**Exit:** invert synthetic gravity then ERT, land the results as fused layers, and score recovery against the known `great-basin-v1` truth.

---

## Suggested first demo cut

**M0 → M1 → M2** gives the headline "every survey method, one 3D earth" product on
synthetic data — the most compelling thing to show early. **M4 → M5** turns it from a
viewer into an *interpretation* tool (favorability). Treat **M3** and **M8** as
parallel deepenings, and **M9** as the next major phase once the platform proves out.

## Cross-cutting tracks (run continuously, not milestones)

- **Testing:** adapter format round-trips; transform known-in/known-out; fusion scored vs synthetic ground truth (the generator's reason for existing).
- **Plugin conformance:** every new method/transform/renderer registers through doc 08 — no core edits.
- **Provenance/versioning:** every derived artifact carries lineage from day one (doc 02 §7).
