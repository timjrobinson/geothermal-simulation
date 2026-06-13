# 10 — Inversion Integration (Forward Modeling & Geophysical Inversion)

> Parent: `OVERVIEW.md` §6 (levels 4–5), §9 (Phase 6), §10 (row 10).
>
> **⚠️ LATER-PHASE / NON-BLOCKING.** The MVP platform *consumes* already-inverted
> property models. This document plans the later phase where the platform itself
> runs forward modeling and inversion as **pluggable modules**. Nothing here is on
> the MVP critical path; it exists so that the MVP's data model, storage, and
> plugin seams are shaped to accept inversion cleanly when the time comes. If a
> decision here would force MVP work, the MVP decision wins and this doc adapts.
>
> **Parallel docs referenced at contract level** (assumptions stated where the doc
> does not yet exist): plugin framework (08), Observation/PropertyModel schemas
> (02), job queue + storage (04), resample-to-fused-grid (07), synthetic generator
> (05), spatial frame (01, LOCKED), drilling/planning (09).

---

## 0. The one idea that makes this tractable

An inversion engine is **just another producer of a `PropertyModel`.** It consumes
`Observation`s (02) plus a model domain (a mesh over the Engineering Frame, 01),
runs forward + inverse physics, and emits a `PropertyModel` with uncertainty and
provenance — the *same* primitive the platform already stores (04), resamples onto
the fused grid (07), fuses (07), and visualizes (06).

```
Observations (02) ──┐
                    ├──▶  [ Inversion Engine plugin (08) ]  ──▶  PropertyModel (02)
Model domain  ──────┘         forward + inverse                  + uncertainty
(mesh over Engineering Frame, 01)                               + provenance
                                                                      │
                                                          (already-known path)
                                                          store (04) · resample (07)
                                                          · fuse (07) · visualize (06)
```

**This closure is the whole architectural bet.** Because inversion output re-enters
the system as an ordinary property model, the platform needs *no new downstream
machinery* to support inversion — only an upstream plugin type and a job to run it.
Everything past the engine boundary is reuse. Keep that boundary clean and this
phase stays additive rather than invasive.

---

## 1. Where this sits in the fusion ladder (OVERVIEW §6)

| Level | Capability | This doc |
|---|---|---|
| 1–3 | Co-registration, cross-plot, rock-physics transforms | MVP (docs 06, 07) — *consumes* inverted models |
| 4 | Geological modeling (GemPy) | Phase 5 (doc 09 area) |
| **5** | **Cooperative / joint inversion** | **this doc** |
| 6 | Drilling target & well planning | doc 09 |

We split level 5 into the engine layer it stands on:

- **5a — Single-method inversion in-platform** (forward + inverse for one method).
- **5b — Cooperative inversion** (one method's model constrains another's, sequentially).
- **5c — Joint inversion** (multiple methods on a shared mesh, coupled in one objective).

Each stage is independently shippable and each delivers value alone. **Do not build
5c before 5a is solid.**

---

## 2. The inversion engine plugin interface

Conforms to the plugin framework (08). **Assumed contract from 08** (flag if 08
diverges): plugins are registered Python entry points discovered at startup; each
declares a `kind`, a stable `id`, a JSON-schema'd parameter block, and a typed
`run()` entrypoint; the host owns lifecycle, sandboxing, and dependency isolation
(see §9 — inversion deps are heavy and may warrant a separate environment/process).

We add one new plugin **kind**: `inversion-engine`.

### 2.1 Capability declaration (static, at registration)

```jsonc
InversionEngineSpec {
  "id": "simpeg.gravity.l2",          // stable, namespaced
  "kind": "inversion-engine",
  "library": "SimPEG",                // SimPEG | PyGIMLi | custom | ...
  "version": "0.22.x",

  // What it can invert. Drives UI ("which datasets can feed this?")
  "methods": ["gravity"],             // method ids from OVERVIEW §3 catalog
  "outputProperty": "density",        // canonical property type (01 §5 registry)
  "dimensionality": "3D",             // 1D | 2D | 3D | 2.5D

  // What kind of model domain it needs (see §4)
  "meshTypes": ["TensorMesh", "TreeMesh"],   // discretize / PyGIMLi mesh kinds it accepts
  "supportsTimeLapse": false,                 // 4D / monitoring inversion

  // Coupling capability (see §6). single = standalone only.
  "coupling": "single",               // single | cooperative-reference | joint-member

  // Compute profile (see §9) — lets the scheduler place the job
  "compute": { "device": "cpu", "memoryHintGB": 8, "scalable": "coarse" },

  // JSON schema for the parameter block (regularization, bounds, etc. — §5)
  "paramsSchema": { /* JSON Schema */ }
}
```

### 2.2 Runtime entrypoint (per job)

```python
def run(ctx: InversionContext) -> InversionResult: ...
```

```jsonc
// INPUT
InversionContext {
  "observations":   [ObservationRef, ...],   // 02 — resolved to data + geometry + uncertainty
  "domain":         ModelDomain,             // §4 — mesh + active cells in Engineering Frame
  "startingModel":  PropertyModelRef | const | null,   // initial m0
  "referenceModel": PropertyModelRef | const | null,   // m_ref for regularization (§5, §6)
  "params":         { /* validated against paramsSchema */ },
  "frame":          SpatialFrame,            // 01 (LOCKED) — read-only
  "report":         ProgressReporter,        // §3 — progress(iter, phi_d, phi_m, ...)
  "checkpoint":     CheckpointIO,            // §3 — save/load resumable state
  "cancelled":      Callable[[], bool]       // cooperative cancellation
}

// OUTPUT — becomes an ordinary PropertyModel (02)
InversionResult {
  "model":       PropertyModel,        // 02 — on the inversion mesh, Engineering Frame
  "uncertainty": UncertaintyField,     // 02 — see §2.3
  "predicted":   [PredictedData, ...], // forward response of recovered model (for fit QC)
  "diagnostics": {                     // convergence record → provenance + UI
    "iterations": int, "phi_d": [...], "phi_m": [...],
    "chi_factor": float, "beta_schedule": [...], "wall_seconds": float
  },
  "provenance":  InversionProvenance   // §7 — full reproducibility record
}
```

### 2.3 Uncertainty is mandatory, not optional

Every method "sees" differently (OVERVIEW §1: gravity/MT smooth & non-unique; seismic
sharp). Fusion (07) and well planning (09) are only honest if that survives. The
engine must emit *something* in the `UncertaintyField` slot, in descending order of
rigour:

| Tier | Source | Cost |
|---|---|---|
| A | Posterior covariance diagonal / model resolution matrix (when linearizable) | moderate |
| B | Sensitivity-/depth-of-investigation–weighted confidence (e.g. DOI index, cumulative sensitivity) | cheap |
| C | Ensemble spread (multiple regularizations / starting models / bootstrap) | expensive |
| D | Flat per-property prior from the registry (01) if nothing better — **explicitly flagged low-confidence** | free |

The contract: never emit a bare model with no uncertainty. Tier B is the expected
default for the first methods; tier A/C are opt-in. **Need from doc 02:** the
`UncertaintyField` must support at least {per-cell stddev, per-cell confidence/DOI
mask}. Flag to 02.

---

## 3. Job orchestration (long, heavy jobs)

Inversions are minutes-to-hours, CPU- (sometimes GPU-) bound, and must survive
restarts. They run as **background jobs on the task queue from doc 04** (OVERVIEW §5:
"background tasks first; Celery/RQ when needed"). **Assumed job contract from 04**
(flag if it diverges): submit → `job_id`; status polling + WebSocket push; cancel;
artifacts written to storage and registered in the catalog.

```
POST /jobs/inversion
  { engineId, observationIds[], domainSpec, params, startingModelId?, referenceModelId? }
    → { jobId }                       # validated against engine.paramsSchema first
GET  /jobs/{jobId}                    → { state, progress, diagnostics, resultModelId? }
WS   /jobs/{jobId}/stream             → live { iter, phi_d, phi_m, beta } for plots
POST /jobs/{jobId}/cancel
```

| Concern | Approach |
|---|---|
| **Progress** | Engine calls `ctx.report(iter, phi_d, phi_m, beta, ...)` each Gauss–Newton / outer iteration. Streamed to a live convergence plot (Tikhonov curve, data misfit vs target χ). |
| **Checkpointing** | Engine periodically `ctx.checkpoint.save(state)` (current model, β, iteration). On worker restart, resume from last checkpoint instead of restarting the solve. Required for multi-hour jobs and pre-emptible/laptop-sleep reality. |
| **Cancellation** | Cooperative: engine checks `ctx.cancelled()` between iterations, saves a checkpoint, exits cleanly. The partial model is still a valid (flagged) `PropertyModel`. |
| **Parameterization** | See §5. All params validated against `paramsSchema` *before* the job is queued — fail fast, not three hours in. |
| **Concurrency** | One heavy inversion can saturate a workstation. Scheduler respects `compute.memoryHintGB` and a configurable max-concurrent-heavy-jobs (default **1** local). |
| **Reproducibility** | Every job records the full `InversionProvenance` (§7) so a result can be regenerated bit-for-bit (seeded) or re-run with one changed param. |

**Need from doc 04:** checkpoint blob storage keyed by `job_id`, and a job-artifact
link so the produced `PropertyModel` is discoverable from the job and vice-versa.

---

## 4. Meshing & the model domain

The inversion mesh is **not** the Fused Earth Model grid (02/07), and must not be
forced to be. Each method has its own optimal discretization; forcing all onto one
grid would either over-resolve cheap methods or under-resolve sharp ones. So:

> **The inversion mesh lives in the Engineering Frame (01) but is method-chosen.
> Its results are *resampled* onto the canonical Fused grid afterward (07) — the
> same way every other property model is.** Originals kept at native resolution
> (OVERVIEW §2: "without destroying the native-resolution originals").

### 4.1 ModelDomain object

```jsonc
ModelDomain {
  "frame": "engineering",            // ALWAYS Engineering Frame (01) — never CRS coords
  "meshType": "TreeMesh",            // TensorMesh | TreeMesh | SimplexMesh (PyGIMLi) | ...
  "coreRegion": { "xmin":-2000,"xmax":2000,"ymin":-2000,"ymax":2000,
                  "zmin":-3000,"zmax":500 },   // region of interest, Engineering m
  "baseCellSize": [25, 25, 25],      // core cell size (m); may be refined for TreeMesh
  "padding": {                       // expansion cells to absorb boundary effects (§4.3)
      "factor": 1.3, "nCells": 8, "directions": ["x","y","z-"] },
  "refinement": [ /* method-specific: refine near receivers / topography / targets */ ],
  "activeCells": "below-surface",    // topography-aware active set (01 surfaceModel)
  "topography": "frame.surfaceModel" // tie air/ground interface to the project surface (01)
}
```

### 4.2 Mesh type by method/library (see §8 landscape)

| Mesh | Library | Used by | Why |
|---|---|---|---|
| `TensorMesh` | discretize/SimPEG | gravity, mag, MT, simple DC | regular, simple, fast assembly |
| `TreeMesh` (octree) | discretize/SimPEG | EM/TDEM/FDEM, MT, focused gravity/mag | refine near sources/receivers, coarsen at depth → fewer cells |
| `SimplexMesh` / tetra | PyGIMLi (+ discretize SimplexMesh) | ERT/IP, traveltime, complex topography | conforms to topography & electrodes |

### 4.3 Core, padding, and the air

- **Core region** = the ROI we care about, at target resolution. Subset of the
  project ROI (01) — usually tighter than the full Fused grid extent.
- **Padding cells** expand outward (geometrically growing) so the solution boundary
  doesn't contaminate the core. Gravity/mag/MT/EM need substantial padding;
  potential/EM fields are non-local. These cells exist for physics, are **not**
  exported to the fused grid.
- **Topography / air**: cells above `surfaceModel` (01) are inactive (potential
  fields) or air-valued (EM). The active-cell set is topography-aware and tied to
  the project surface — *one source of truth for the ground*.

### 4.4 Handoff back to the Fused grid (links doc 07)

After convergence:
1. Take recovered values on **active core cells only** (drop padding/air).
2. Resample onto the canonical Fused Earth Model grid via the **resampling service
   in doc 07** (volume→volume; conservative/linear per property). *No new resampler
   here — reuse 07's.*
3. Carry the uncertainty field through the *same* resampling (so confidence stays
   co-registered with values).
4. Register both native-mesh model and fused-resampled model in the catalog (04);
   visualization (06) and well planning (09) consume the fused one.

**Need from doc 07:** resampler must accept arbitrary source meshes (TensorMesh,
TreeMesh, SimplexMesh), not only regular grids → from→to handled by a mesh
interpolation interface. Flag to 07.

**Need from doc 02:** `PropertyModel.support` must already represent non-regular
meshes (octree, tetra), per OVERVIEW §2 ("regular grid, octree, or unstructured
mesh"). Confirm with 02.

---

## 5. Parameterization (what the user/UI controls)

Geophysical inversion is non-unique; the result is shaped as much by *regularization
choices* as by the data. Those choices must be explicit, validated, and recorded
(reproducibility, §7). The standard Tikhonov knobs, exposed through `paramsSchema`:

| Param | Meaning | Typical default |
|---|---|---|
| `regularization` | smoothness/structure: `l2` (smooth) · `sparse/l1` (compact, blocky) · `cross-gradient` (§6) | `l2` |
| `alpha_s, alpha_x/y/z` | smallness vs smoothness weighting per axis | library defaults |
| `referenceModel` (m_ref) | model pulled toward in smallness term (a `PropertyModelRef` → §6 coupling) | half-space / prior |
| `startingModel` (m0) | initial guess | reference / best-guess |
| `bounds` | physical bounds (e.g. ρ ∈ [1, 10⁴] Ω·m) | property registry (01) |
| `beta / chi_factor` | trade-off; cooling schedule; target data misfit (χ=1 ≈ fit to noise) | β cooling, χ=1 |
| `noiseFloor / dataWeights` | uncertainty model on the data (from Observation errors, 02) | from 02 obs errors |
| `maxIterations`, `tol` | stopping | 20–40 / library |
| `seed` | RNG seed for any stochastic step | fixed for reproducibility |

**Defaults must be sane out of the box** — a user picks a dataset + an engine and
gets a defensible inversion with no tuning, then refines. The data uncertainty
(`noiseFloor`/`dataWeights`) **comes from the Observation error fields (02)** — flag
that 02 carries per-datum noise estimates.

---

## 6. Cooperative & joint inversion roadmap

This is where the platform's unique asset pays off. **Every method already lives in
one co-registered Engineering Frame (01) on one canonical grid (07).** Coupling
multiple methods normally requires painstaking re-registration and re-meshing across
toolchains; here it is a property of the platform. That is the strategic reason this
doc exists. Staged:

### Stage 5a — Single-method inversion *(build first)*
One method → one model. No coupling. Proves engine plugin, job orchestration,
mesh handoff, uncertainty, and validation against synthetic ground truth (§7/§ valid).
`coupling: "single"`. **Everything else depends on this working.**

### Stage 5b — Cooperative (sequential) inversion
Invert method A; use A's model as a **reference/constraint** for method B:
- **Reference-model coupling** — A's recovered model becomes B's `referenceModel`
  (m_ref) or starting model. Cheapest, robust, no new solver math.
- **Structure-guided weighting** — derive a structural weight from A (e.g. edges
  from a seismic/ERT model) that relaxes smoothness across A's interfaces in B.

Implemented as an **orchestration of single-method jobs** (a small DAG of §3 jobs)
plus a "model→reference/weight" adapter. No monolithic joint solver yet — leverages
5a engines directly. `coupling: "cooperative-reference"`.

### Stage 5c — Joint (simultaneous) inversion
Multiple methods on a **shared mesh**, coupled inside one objective function:
- **Cross-gradient** (structural): penalize ∇m_A × ∇m_B → structures align without
  assuming a petrophysical law. The general-purpose default.
- **Petrophysically coupled** (PGI): a learned/assumed property-property relationship
  (rock physics, links 07) ties the models — e.g. SimPEG's PGI framework couples
  via a Gaussian-mixture petrophysical model. Strongest when rock physics is known.

Requires a `joint-member` engine capability and a **joint driver** that owns the
shared mesh, assembles the combined objective, and steps all physics together. This
is the heaviest lift (single shared mesh, multiphysics, much larger problem) and is
explicitly the *last* milestone. SimPEG's joint/PGI machinery is the intended host.

| Stage | Coupling | Solver | Effort | Prereq |
|---|---|---|---|---|
| 5a | none | per-method | baseline | engine plugin + jobs |
| 5b | reference / structure | sequential (job DAG) | small over 5a | 5a |
| 5c | cross-gradient / PGI | shared-mesh joint | large | 5a, shared-mesh meshing, 07 rock-physics |

**Sequencing rule:** do not start 5c until ≥2 methods work standalone (5a) and
cooperative coupling (5b) is validated against synthetic ground truth.

---

## 7. Reproducibility & provenance

Inversion results are **interpretations**, not measurements. They must be fully
auditable and regenerable (OVERVIEW §2: property models carry provenance; 01 §7: all
transforms record provenance). The result's `PropertyModel.provenance` (02) is
extended with an `InversionProvenance`:

```jsonc
InversionProvenance {
  "engineId": "simpeg.gravity.l2", "engineVersion": "...", "library": "SimPEG@0.22.1",
  "observationIds": [...],           // exact inputs (02), immutable
  "domain": { /* full ModelDomain, §4 */ },
  "params": { /* resolved params incl. defaults applied */ },
  "startingModelId": ..., "referenceModelId": ...,
  "seed": 12345,
  "diagnostics": { /* φ_d, φ_m, β schedule, χ achieved, iterations, wall time */ },
  "softwareEnv": { "python": "...", "deps": {...} },   // pinned for bit-reproducibility
  "coupling": { "stage": "5a", "partners": [] },       // §6
  "createdAt": "...", "jobId": "..."
}
```

This makes "what produced this volume?" answerable from the catalog, and "re-run with
χ=1.5" a one-field change. **Need from doc 02:** provenance must be an open/extensible
block so domain-specific records (inversion, rock-physics, gridding) attach without
schema churn.

---

## 8. Library landscape (methods → libraries → needs)

The backend is Python + FastAPI specifically to reach this ecosystem (OVERVIEW §5).
Engines wrap these; the platform owns the boundary, not the math.

| Method (OVERVIEW §3) | Library | Mesh / data objects | Output property | Notes |
|---|---|---|---|---|
| Gravity / gradiometry | **SimPEG** | TensorMesh / TreeMesh; `gravity.Survey` | density | linear → fast; non-unique, smooth |
| Magnetics | **SimPEG** | TensorMesh / TreeMesh; `magnetics.Survey` | susceptibility | linear; remanence/MVI as extension |
| DC resistivity | **SimPEG** / **PyGIMLi (ERT)** | Tensor/Tree (SimPEG) · SimplexMesh (PyGIMLi) | resistivity | PyGIMLi excels at topography-conforming ERT |
| Induced Polarization | **SimPEG** / **PyGIMLi** | as DC | chargeability | usually after a DC inversion |
| FDEM / TDEM / AEM | **SimPEG** (EM) | **TreeMesh** (octree near tx/rx) | conductivity | heavy; octree essential |
| Magnetotellurics (MT) | **SimPEG** (NSEM) | TensorMesh / TreeMesh; impedance/tipper | resistivity (deep) | big 3D; padding-heavy; smooth/deep |
| Seismic refraction (traveltime) | **PyGIMLi** (Refraction) | SimplexMesh | velocity | ray-based tomography |
| Seismic reflection / FWI | **ObsPy** (I/O) + (Devito / specialist) | external/regular grids | velocity/impedance | FWI is out of scope near-term; ingest results |
| Passive / microseismic | **ObsPy** | event catalogs | event cloud (4D) | location/detection, not field inversion |
| Implicit geology | **GemPy** | implicit scalar field | lithology / surfaces | constrained "geometry inversion"; feeds 5c structure |

**Mesh vocabulary (discretize / PyGIMLi):** `TensorMesh` (regular rectilinear),
`TreeMesh` (octree, adaptive), `SimplexMesh`/tetrahedral (unstructured, conforms to
topography & electrodes). The `ModelDomain` (§4) abstracts these behind one object so
the platform UI doesn't bake in any one library.

**Boundary discipline:** SimPEG `Survey`/`Data` and PyGIMLi data containers are built
**inside the engine plugin** from platform `Observation`s — they never leak across the
plugin boundary. The platform speaks `Observation`/`PropertyModel`/`ModelDomain`
only. This keeps libraries swappable and versions isolated (§9).

---

## 9. Compute reality check (local-first)

OVERVIEW: local-first, single-user. Inversion stresses that hard. Honest tiering:

| Workload | Feasible on a workstation? | Notes |
|---|---|---|
| Gravity / magnetics 3D (linear) | **Yes** | minutes–tens of min; modest RAM. Good first target. |
| DC/ERT, IP (2D/3D) | **Yes** (moderate) | PyGIMLi efficient; 3D ERT grows. |
| Traveltime tomography | **Yes** | light. |
| MT 3D | **Marginal** | RAM- and time-heavy; coarse OK locally, production wants a server. |
| FDEM/TDEM 3D, AEM lines × many | **No / marginal** | octree + many transmitters → server/cluster, GPU helps. |
| Joint / PGI (5c) | **Marginal→No** | shared big mesh × multiphysics → server class. |
| Seismic FWI | **No** | cluster/GPU; out of near-term scope (ingest results instead). |

**Implications for deployment assumptions (revisits OVERVIEW §5 "local-first"):**

- The plugin `compute` profile (§2.1) lets the **same engine** run locally or on a
  remote worker. The job system (04) must allow a **remote/larger worker pool** for
  `device:"gpu"` / high `memoryHintGB` engines — i.e. local-first does **not** mean
  local-only for inversion. Flag to 04: optional remote worker tier.
- **Dependency isolation:** SimPEG/PyGIMLi/GemPy pull heavy, sometimes conflicting
  native deps (PETSc, pymatsolver, etc.). Engines should run in **isolated
  environments / subprocesses** (or containers), not in the FastAPI process. Flag to
  08 (plugin isolation) and 04 (worker images).
- **Default posture:** ship 5a for the *workstation-feasible* methods (gravity, mag,
  ERT, traveltime) first; treat MT/EM/joint as "bring a server." The platform stays
  useful locally and scales out per-engine when the user opts in.

---

## 10. Validation (against synthetic ground truth)

The synthetic generator (05) already defines a **known ground-truth earth** and
forward-models each method (OVERVIEW §8). That closes the loop for inversion:

```
05 ground truth ──forward(05)──▶ simulated Observations ──▶ [engine] ──▶ recovered PropertyModel
       │                                                                        │
       └────────────────────────── score (07 on shared fused grid) ◀───────────┘
```

| Check | Metric |
|---|---|
| Data fit | predicted vs observed misfit reaches target χ≈1 (no over/under-fit) |
| Model recovery | recovered vs ground-truth on fused grid: RMSE, structural similarity, anomaly localization & amplitude |
| Uncertainty honesty | ground truth falls within stated confidence at expected rate (calibration) |
| Resolution realism | recovered smoothness/DOI matches the method's physics (gravity smear vs ERT sharpness) |

This is the per-phase "vertical slice" verification (OVERVIEW §"Verification"): an
engine is *done* when it recovers the synthetic anomaly within its physical
resolution limits and its uncertainty is calibrated. Reuses 05 (data) and 07
(resample + cross-compare) — **no bespoke validation harness beyond a scoring step.**
**Need from doc 05:** ground-truth volumes must be retrievable for scoring, on (or
resamplable to) the fused grid.

---

## Decisions locked in (for the later phase)

1. **Closure principle.** An inversion engine is a plugin (kind `inversion-engine`,
   08) that consumes `Observation`s (02) + a `ModelDomain` and emits a
   `PropertyModel` + uncertainty + provenance (02). No new downstream machinery —
   storage (04), resample (07), fusion (07), visualization (06) are all reused.
2. **Uncertainty is mandatory.** Every result carries an `UncertaintyField`; tier B
   (sensitivity/DOI-weighted) is the default floor, never a bare model.
3. **Mesh independence.** Inversion runs on a method-chosen mesh (Tensor/Tree/Simplex)
   in the **Engineering Frame (01)**; results are **resampled onto the Fused grid via
   doc 07** afterward. Native-resolution originals are preserved. The fused grid is
   never the forced inversion mesh.
4. **Jobs on the doc-04 queue** with progress streaming, checkpointing, cooperative
   cancellation, and param validation *before* enqueue. Default max 1 concurrent
   heavy job locally.
5. **Full reproducibility.** Every result records an `InversionProvenance` (inputs,
   params incl. applied defaults, seed, env, diagnostics) → regenerable / re-runnable.
6. **Staged coupling.** 5a single-method → 5b cooperative (job DAG, reference/structure
   coupling, reuses 5a engines) → 5c joint (cross-gradient / PGI, shared mesh).
   Strict ordering; 5c last.
7. **Library boundary discipline.** SimPEG/PyGIMLi/GemPy objects are constructed
   *inside* engine plugins; the platform speaks only `Observation`/`PropertyModel`/
   `ModelDomain`. Engines run in isolated environments/subprocesses (heavy/conflicting
   deps).
8. **Local-first ≠ local-only for inversion.** A `compute` profile per engine lets
   jobs target a remote/GPU worker tier; workstation-feasible methods (gravity, mag,
   ERT, traveltime) ship first, MT/EM/joint are "bring a server."
9. **Validation via the synthetic generator (05).** Invert simulated data, score
   recovered vs ground truth on the fused grid (07); calibrated uncertainty is part
   of "done."
10. **Non-blocking.** None of this gates the MVP. It only shapes MVP seams in 02
    (extensible provenance, non-regular mesh support, per-datum obs error,
    `UncertaintyField`), 04 (checkpoint store, remote worker option), 07 (mesh→grid
    resampling), 08 (plugin isolation) so inversion drops in additively later.

### Contract needs flagged to parallel docs (summary)

| Doc | Need |
|---|---|
| 02 | extensible `provenance` block; `support` covers octree/tetra meshes; per-datum noise on Observations; `UncertaintyField` (stddev + DOI/confidence mask) |
| 04 | checkpoint blob store keyed by job; job↔artifact links; optional remote/GPU worker tier |
| 05 | ground-truth volumes retrievable & resamplable to the fused grid for scoring |
| 07 | resampler accepts arbitrary source meshes (Tensor/Tree/Simplex), carries uncertainty through |
| 08 | new plugin kind `inversion-engine`; isolated-environment/subprocess execution for heavy deps |

---

## Open questions for you

1. **Which method's inversion do we build first (5a)?** Gravity is the cleanest
   proof (linear, fast, workstation-friendly, fewest moving parts) but lowest
   geothermal information; ERT/MT carry the resistivity signal that actually maps a
   geothermal reservoir (hot, conductive, altered) but are heavier/harder.
   *Recommended default:* **gravity first** as the engine-plumbing proof, then **ERT**
   (PyGIMLi, topography-conforming, still local-feasible) as the first
   geothermally-meaningful one — defer MT until a server tier exists.

2. **Where is the local-vs-server compute boundary, and do we commit to a remote
   worker tier in this phase?** Options: (a) **strictly local** — only ship engines
   that run on a workstation (gravity/mag/ERT/traveltime), punt MT/EM/joint entirely;
   (b) **local default + optional remote worker** — engines declare a compute profile,
   heavy ones target an opt-in remote/GPU worker (changes 04's assumptions); (c)
   **server-first** for inversion from day one. *Recommended default:* **(b)** — keeps
   the local-first promise for the common case while not architecturally excluding the
   heavy methods.

3. **How far do we commit to joint inversion (5c) now vs. leaving it as a roadmap
   stub?** Full joint/PGI is the largest lift and stresses meshing, compute, and the
   data model the most. Options: (a) **5a only** for this phase, 5b/5c documented but
   unbuilt; (b) **5a + 5b** (cooperative via sequential job DAG — high value, modest
   cost, no joint solver); (c) **commit to 5c** with a shared-mesh joint driver.
   *Recommended default:* **(b)** — cooperative inversion delivers most of the
   multi-method payoff (the platform's unique co-registration asset) without the
   shared-mesh joint-solver expense; keep 5c as a validated-prerequisite roadmap item.
