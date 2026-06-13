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

Conforms to the plugin framework (08). **Contract from 08:** plugins are registered
via 08's dual discovery (decorators for first-party / entry points for third-party,
one registry); each declares a `kind`, a stable `id`/`key`, a JSON-schema'd parameter
block, and a typed entrypoint; the host owns lifecycle and load-time validation.
**Process placement is doc 08's `executionMode` (08 §2.1), not a scheme invented here:**
inversion engines declare `worker_process` / `container` / `remote_worker` because
their deps are heavy and conflicting (see §9). This **AGREES** with doc 08's
in-process-trusted default — that default covers *lightweight* contributions; heavy
inversion engines opt into a separate process/container/remote worker on doc 08's
*process-isolation axis* (an engineering placement choice), which is orthogonal to
the *trust axis* (engines are still trusted code, single-user). No separate isolation
or sandboxing scheme is defined in this doc.

We add one new plugin **kind**: `inversion-engine`.

### 2.1 Capability declaration (static, at registration)

```jsonc
InversionEngineSpec {
  "id": "simpeg.gravity.l2",          // stable, namespaced
  "kind": "inversion-engine",
  "library": "SimPEG",                // SimPEG | PyGIMLi | custom | ...
  "version": "0.22.x",

  // What it can invert. Drives UI ("which datasets can feed this?")
  "methods": ["gravity"],             // canonical MethodKey(s) (doc 02 §2)
  "submethods": ["dc_resistivity"],   // optional: canonical submethod(s) under method (doc 02 §2); omit if N/A
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
  "observations":   [ObservationRef, ...],   // 02 — data + geometry + per-obs σ (02 §3 sigma cols / methodData.errorModel)
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
restarts. They run as **background jobs on doc 04's job system** — **RQ + Redis
workers** (doc 04 §9.4, the locked default), against doc 04's `jobs` table and its
**single job contract** (table + endpoints + WebSocket). **Doc 04 is the authoritative
job/API contract; this doc invents no new endpoints.** Inversion is just another
job `kind` on that contract (alongside `ingest`/`fuse`/`pyramid`/`transform`), and
its produced `PropertyModel` is registered in the catalog (04) like any other.

Submission uses doc 04's **resource-style** endpoints (a sub-resource action that
returns a `job_id`), mirroring `POST .../datasets:ingest → {job_id}` — *not* a
bespoke `POST /jobs/inversion`. Concretely (doc 04 §9.2 owns the exact shapes):

```
POST /property-models:invert            # resource action → enqueue on doc 04's queue
  { engineId, observationIds[], domainSpec, params, startingModelId?, referenceModelId? }
    → { job_id }                        # params validated against engine.paramsSchema first
GET  /jobs/{job_id}                     → Job { status, progress, message, result_json }  (doc 04 §9.2)
WS   /jobs/{job_id}/progress            → doc 04's progress channel; engine pushes { iter, phi_d, phi_m, beta }
POST /jobs/{job_id}:cancel              → 202   (doc 04 §9.2)
```

Live convergence telemetry (`iter, phi_d, phi_m, beta`) rides doc 04's existing job
progress WebSocket — a richer `message`/progress payload, not a separate stream.

| Concern | Approach |
|---|---|
| **Progress** | Engine calls `ctx.report(iter, phi_d, phi_m, beta, ...)` each Gauss–Newton / outer iteration. Streamed to a live convergence plot (Tikhonov curve, data misfit vs target χ). |
| **Checkpointing** | Engine periodically `ctx.checkpoint.save(state)` (current model, β, iteration). On worker restart, resume from last checkpoint instead of restarting the solve. Required for multi-hour jobs and pre-emptible/laptop-sleep reality. |
| **Cancellation** | Cooperative: engine checks `ctx.cancelled()` between iterations, saves a checkpoint, exits cleanly. The partial model is still a valid (flagged) `PropertyModel`. |
| **Parameterization** | See §5. All params validated against `paramsSchema` *before* the job is queued — fail fast, not three hours in. |
| **Concurrency** | One heavy inversion can saturate a workstation. Scheduler respects `compute.memoryHintGB` and a configurable max-concurrent-heavy-jobs (default **1** local). |
| **Reproducibility** | Every job records the full `InversionProvenance` (§7) so a result can be regenerated bit-for-bit (seeded) or re-run with one changed param. |

**Need from doc 04** (additive to its locked contract): checkpoint blob storage keyed
by `job_id`, and a job↔artifact link (`jobs.result_json` → the produced
`PropertyModel` id) so the result is discoverable from the job and vice-versa.

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
| `TreeMesh` (octree) | discretize/SimPEG | `em` (tdem/fdem/aem), `mt`, focused `gravity`/`magnetics` | refine near sources/receivers, coarsen at depth → fewer cells |
| `SimplexMesh` / tetra | PyGIMLi (+ discretize SimplexMesh) | `ert`/`ip` (dc_resistivity/ip_*), `seismic` refraction/tomography, complex topography | conforms to topography & electrodes |

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
| `noiseFloor / dataWeights` | data weights = 1/σ², σ from the **per-observation `sigma` columns / `methodData.errorModel` (doc 02 §3)** | from doc 02 §3 sigmas; default noise floor when absent |
| `maxIterations`, `tol` | stopping | 20–40 / library |
| `seed` | RNG seed for any stochastic step | fixed for reproducibility |

**Defaults must be sane out of the box** — a user picks a dataset + an engine and
gets a defensible inversion with no tuning, then refines. The data uncertainty
(`noiseFloor`/`dataWeights`) **comes from doc 02 §3's per-observation error
convention (now defined), not a parallel uncertainty schema invented here:** each
measured value column carries a paired `sigma` column (`role:"sigma"`,
`errorFor:"<value>"`); tensors/traces carry `methodData.errorModel`. The engine
builds data weights = 1/σ² directly from these. When a source has no errors, doc 02
§3's **per-property default noise floor** (from the property registry, recorded in
provenance) is consumed — the engine never treats data as error-free.

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

Method names below are the **canonical `method` + `submethod` keys (doc 02 §2)** —
never ad-hoc groupings like "DC resistivity"/"FDEM/TDEM"/"AEM".

| `method` / `submethod` (doc 02 §2) | Library | Mesh / data objects | Output property | Notes |
|---|---|---|---|---|
| `gravity` | **SimPEG** | TensorMesh / TreeMesh; `gravity.Survey` | density | linear → fast; non-unique, smooth |
| `magnetics` | **SimPEG** | TensorMesh / TreeMesh; `magnetics.Survey` | susceptibility | linear; remanence/MVI as extension |
| `ert` / `dc_resistivity` | **SimPEG** / **PyGIMLi (ERT)** | Tensor/Tree (SimPEG) · SimplexMesh (PyGIMLi) | resistivity | PyGIMLi excels at topography-conforming ERT |
| `ip` / `ip_time`,`ip_freq` | **SimPEG** / **PyGIMLi** | as `ert` | chargeability | usually after a `dc_resistivity` inversion |
| `em` / `tdem`,`fdem`,`aem` | **SimPEG** (EM) | **TreeMesh** (octree near tx/rx) | conductivity | heavy; octree essential |
| `mt` | **SimPEG** (NSEM) | TensorMesh / TreeMesh; impedance/tipper | resistivity (deep) | big 3D; padding-heavy; smooth/deep |
| `seismic` / `refraction`,`tomography` | **PyGIMLi** (Refraction) | SimplexMesh | velocity | ray-based traveltime tomography |
| `seismic` / `reflection` (+ FWI) | **ObsPy** (I/O) + (Devito / specialist) | external/regular grids | velocity/impedance | FWI is out of scope near-term; ingest results |
| `microseismic` | **ObsPy** | event catalogs | event cloud (4D) | location/detection, not field inversion |
| `geology` (implicit) | **GemPy** | implicit scalar field | lithology / surfaces | constrained "geometry inversion"; feeds 5c structure |

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
| `gravity` / `magnetics` 3D (linear) | **Yes** | minutes–tens of min; modest RAM. Good first target. |
| `ert` (dc_resistivity), `ip` (2D/3D) | **Yes** (moderate) | PyGIMLi efficient; 3D ERT grows. |
| `seismic` traveltime tomography | **Yes** | light. |
| `mt` 3D | **Marginal** | RAM- and time-heavy; coarse OK locally, production wants a server. |
| `em` (fdem/tdem 3D, aem lines × many) | **No / marginal** | octree + many transmitters → server/cluster, GPU helps. |
| Joint / PGI (5c) | **Marginal→No** | shared big mesh × multiphysics → server class. |
| Seismic FWI | **No** | cluster/GPU; out of near-term scope (ingest results instead). |

**Implications for deployment assumptions (revisits OVERVIEW §5 "local-first"):**

- The plugin `compute` profile (§2.1) lets the **same engine** run locally or on a
  remote worker. The job system (04) must allow a **remote/larger worker pool** for
  `device:"gpu"` / high `memoryHintGB` engines — i.e. local-first does **not** mean
  local-only for inversion. Flag to 04: optional remote worker tier.
- **Dependency isolation via doc 08's `executionMode`:** SimPEG/PyGIMLi/GemPy pull
  heavy, sometimes conflicting native deps (PETSc, pymatsolver, etc.). Engines
  therefore declare a non-`in_process` `executionMode` (08 §2.1) —
  `worker_process` (RQ/Redis worker, doc 04), `container` (conflicting native deps),
  or `remote_worker` (remote/GPU). This is doc 08's per-contribution
  **process-isolation axis** (engineering placement for deps/CPU/GPU), *not* a
  sandbox and *not* a security decision — engines remain trusted code, AGREEING with
  08's in-process-trusted default for the lightweight common case. No new isolation
  scheme here; doc 04 supplies the worker/container images.
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
4. **Jobs on doc 04's RQ+Redis queue and single job contract** (jobs table +
   resource-style endpoints + WS progress) — inversion is just another job `kind`,
   submitted via a `:invert` resource action returning a `job_id`, never a bespoke
   inversion endpoint. Progress streaming, checkpointing, cooperative cancellation,
   and param validation *before* enqueue. Default max 1 concurrent heavy job locally.
5. **Full reproducibility.** Every result records an `InversionProvenance` (inputs,
   params incl. applied defaults, seed, env, diagnostics) → regenerable / re-runnable.
6. **Staged coupling.** 5a single-method → 5b cooperative (job DAG, reference/structure
   coupling, reuses 5a engines) → 5c joint (cross-gradient / PGI, shared mesh).
   Strict ordering; 5c last.
7. **Library boundary discipline.** SimPEG/PyGIMLi/GemPy objects are constructed
   *inside* engine plugins; the platform speaks only `Observation`/`PropertyModel`/
   `ModelDomain`. Heavy/conflicting-dep engines declare a non-`in_process`
   `executionMode` (doc 08 §2.1: `worker_process`/`container`/`remote_worker`) — doc
   08's process-isolation axis, not a bespoke sandbox.
8. **Local-first ≠ local-only for inversion.** A `compute` profile per engine lets
   jobs target a remote/GPU worker tier; workstation-feasible methods (gravity, mag,
   ERT, traveltime) ship first, MT/EM/joint are "bring a server."
9. **Validation via the synthetic generator (05).** Invert simulated data, score
   recovered vs ground truth on the fused grid (07); calibrated uncertainty is part
   of "done."
10. **Non-blocking.** None of this gates the MVP. It only shapes MVP seams in 02
    (extensible provenance, non-regular mesh support, per-observation obs error,
    `UncertaintyField`), 04 (checkpoint store, remote worker option), 07 (mesh→grid
    resampling), 08 (`executionMode` already covers heavy-engine placement) so
    inversion drops in additively later.

### Contract needs flagged to parallel docs (summary)

| Doc | Need |
|---|---|
| 02 | extensible `provenance` block; `support` covers octree/tetra meshes; per-observation `sigma` columns / `methodData.errorModel` + default noise floor (02 §3, now defined — inversion consumes this, no parallel schema); `UncertaintyField` (stddev + DOI/confidence mask) |
| 04 | (additive to its locked RQ+Redis job contract) checkpoint blob store keyed by `job_id`; job↔artifact link (`result_json`); optional remote/GPU worker tier |
| 05 | ground-truth volumes retrievable & resamplable to the fused grid for scoring |
| 07 | resampler accepts arbitrary source meshes (Tensor/Tree/Simplex), carries uncertainty through |
| 08 | new plugin kind `inversion-engine`; heavy engines use doc 08's `executionMode` (`worker_process`/`container`/`remote_worker`, 08 §2.1) — no new isolation scheme |

---

## Decisions resolved (were open questions)

These three forks are **settled** (see `DECISIONS.md` → Inversion). They are no
longer open; recorded here only as rationale. **All of this remains LATER-PHASE and
non-blocking** — these choices shape *when* and *how* the later inversion phase
proceeds, not the MVP.

1. **First engine = `gravity`, then `ert`.** Gravity is the cleanest plumbing proof
   (linear, fast, workstation-friendly, fewest moving parts); `ert` (PyGIMLi,
   topography-conforming, still local-feasible) follows as the first
   geothermally-meaningful method. **`mt` is deferred to a server tier.** Rejected:
   leading with a heavy resistivity method before the engine plumbing is proven.

2. **Compute boundary = local default + optional remote/GPU worker, per engine.**
   Each engine declares a `compute` profile (§2.1) and a doc 08 `executionMode`;
   workstation-feasible methods run locally, heavy ones opt into an
   optional remote/GPU worker (doc 04's optional worker tier). Rejected: strictly
   local (would permanently exclude MT/EM/joint) and server-first (breaks the
   local-first promise for the common case).

3. **Coupling = build 5a single + 5b cooperative; 5c joint stays roadmap.** Build
   5a (single-method) then 5b (cooperative, sequential **job DAG** on doc 04's queue,
   reference/structure coupling, reuses 5a engines). 5b delivers most of the
   multi-method payoff (the platform's unique co-registration asset) without a
   shared-mesh joint solver. **5c joint (cross-gradient / PGI) stays a
   validated-prerequisite roadmap item**, not built in this phase.
