# Forward modeling & inversion

!!! abstract "What you'll learn / why it matters"
    This is the **deepest concept** in the platform, and an optional later phase. Everywhere
    else, the platform *consumes* already-inverted models (a resistivity cube, a density cube)
    as if they fell from the sky. Here you learn where they actually come from. You'll meet the
    **forward problem** (model → predicted data) and the **inverse problem** (data → model),
    understand *why inversion is the hard direction* (it's ill-posed and non-unique — an
    under-determined optimization), see how **regularization** (Tikhonov, smoothness, the
    trade-off knob β) makes it tractable, and grasp the one architectural idea that makes the
    whole thing fit: **an inversion engine is just another `PropertyModel` producer.** We'll
    cover meshes (Tensor/Tree/Simplex), the real engines ([SimPEG](glossary.md) for gravity,
    [PyGIMLi](glossary.md) for ERT), cooperative vs joint inversion, and validating against
    synthetic ground truth. Every concept is framed in terms a programmer already knows:
    optimization, loss functions, overfitting, and lossy reconstruction.

!!! warning "Later-phase, non-blocking"
    None of this is on the critical path. The MVP renders and plans on top of property models
    produced *elsewhere*. This phase teaches the platform to produce them itself, as pluggable
    modules. It exists mainly so the data model, storage, and plugin seams are shaped to accept
    inversion cleanly later.

---

## 1. Forward vs inverse: the two directions

You cannot see through rock, so geophysics measures the Earth *indirectly*. Tie a sensor to
some physics — gravity, electrical resistance, seismic travel time — and record what the
surface readings are. Two problems connect the **model** (what the rock actually is, e.g.
density at every cell) and the **data** (what you measured at the surface).

!!! note "Define: the forward problem"
    The **forward problem** is `model → data`: *given* a known earth, simulate what each sensor
    would read. This is "just physics" — a deterministic function. If you know the density
    everywhere, Newtonian gravity tells you exactly the surface gravity anomaly. We write it
    $d = F(m)$, where $m$ is the model vector (one value per mesh cell) and $d$ is the predicted
    data vector (one value per measurement). For a programmer: `forward` is a pure function from
    a big array to a smaller array.

!!! note "Define: the inverse problem"
    The **inverse problem** is `data → model`: given the surface readings, *recover* the earth
    that produced them. This is the direction we actually want (we have data; we want the
    subsurface) and it is fundamentally harder. We are trying to invert $F$ — but $F$ usually
    has no clean inverse.

The synthetic generator runs the *forward* direction to manufacture realistic fake surveys
from a known earth (see [synthetic data](synthetic-data.md)); the inversion engines run the
*inverse* direction to reconstruct an earth from surveys.

---

## 2. Why inversion is hard: ill-posed and non-unique

Think of inversion as an **optimization problem**: find the model $m$ whose predicted data
$F(m)$ best matches the observed data $d_{obs}$. The obvious loss is **data misfit**:

$$
\phi_d(m) = \bigl\lVert W_d\,(F(m) - d_{obs}) \bigr\rVert^2
$$

where $W_d$ weights each datum by $1/\sigma$ (its measurement uncertainty), so noisy data
counts less. Minimise $\phi_d$ and you've "fit the data." Easy, right?

No — for three reasons every CS person will recognise:

1. **Under-determined.** There are *far* more unknowns (mesh cells, often $10^5$–$10^6$) than
   measurements (often $10^2$–$10^4$). It's a linear system with vastly more columns than rows:
   infinitely many models fit the data **exactly**. This is **non-uniqueness**.

2. **Ill-posed / unstable.** Tiny changes in the noisy data can produce wildly different models.
   The naive least-squares solution amplifies noise into geologic-looking garbage (high-frequency
   "checkerboard" artifacts).

3. **Depth blindness.** Some physics has *no intrinsic depth resolution*. Gravity, for instance:
   a small shallow dense blob and a large deep one can produce the *same* surface anomaly. The
   data simply doesn't constrain depth — so the optimizer, left alone, smears everything to the
   surface.

!!! tip "The CS analogy: overfitting & lossy reconstruction"
    Fitting $\phi_d$ alone is like training a model with more parameters than data points and no
    regularisation — it memorises the noise and generalises to nonsense. Inversion is also a
    **lossy reconstruction**: the forward operator throws away information (it's a low-pass,
    smoothing operator), so you can never recover the original exactly — only a plausible
    earth consistent with what survived. Different "plausible" choices = non-uniqueness.

---

## 3. Regularization: choosing *which* model among the infinitely many

Since the data alone can't pick one model, we add a second term encoding what we *expect* an
earth to look like — a prior. This is **regularization**, and the classic form is **Tikhonov**.
We minimise a combined objective:

$$
\phi(m) = \phi_d(m) + \beta\,\phi_m(m)
$$

- $\phi_d(m)$ — **data misfit** (fit the measurements; defined above).
- $\phi_m(m)$ — **model regularization** (penalise "unreasonable" models).
- $\beta$ — the **trade-off knob** (the single most important inversion parameter).

The model term is itself a blend of **smallness** (stay close to a reference model $m_{ref}$ —
don't invent structure you can't justify) and **smoothness** (penalise sharp jumps between
neighbouring cells — geology is mostly smooth):

$$
\phi_m(m) = \alpha_s \lVert m - m_{ref}\rVert^2
          + \alpha_x \lVert \partial_x m\rVert^2
          + \alpha_y \lVert \partial_y m\rVert^2
          + \alpha_z \lVert \partial_z m\rVert^2
$$

Symbols: $\alpha_s$ weights smallness; $\alpha_{x,y,z}$ weight smoothness along each axis;
$\partial$ is a finite-difference derivative (the discrete gradient between adjacent cells).

### 3.1 β is the dial between underfitting and overfitting

$\beta$ trades the two terms:

- **β large** → regularization dominates → a very smooth, featureless model that *ignores* the
  data (underfitting).
- **β small** → data misfit dominates → a noisy, overfit model chasing measurement noise.

!!! note "Define: the discrepancy principle / χ target"
    You don't fit the data *exactly* — you fit it to *its noise level*. With data weighted by
    $1/\sigma$, the expected misfit of a correct model is about the number of data points
    ($\chi \approx 1$ per datum). So the inversion **cools** β from large to small until
    $\phi_d$ reaches that target — stop fitting when you've explained the signal but not the
    noise. This is the textbook way to pick β automatically.

In SimPEG terms this is a directive stack: `BetaEstimate_ByEig` picks a starting β,
`BetaSchedule` cools it each iteration, and `TargetMisfit` stops at the χ target. The platform
streams the live `(iteration, φ_d, φ_m, β)` telemetry so a UI can plot the **Tikhonov
trade-off curve** (the classic "L-curve") as it converges.

```python title="backend/geosim/inversion/engines/gravity_simpeg.py — the directive stack"
directive_list = [
    directives.UpdateSensitivityWeights(every_iteration=False),  # depth weighting (gravity has none)
    directives.BetaEstimate_ByEig(beta0_ratio=...),              # pick starting β
    directives.BetaSchedule(coolingFactor=..., coolingRate=1),    # cool β each iteration
    directives.TargetMisfit(chifact=...),                         # stop at χ ≈ target
    progress_directive,                                           # stream (iter, φ_d, φ_m, β)
]
```

### 3.2 Depth weighting — the cure for depth blindness

For gravity, an extra trick: **sensitivity (depth) weighting** counteracts the optimizer's urge
to pile everything at the surface, by boosting the weight of deeper cells so they can hold
structure. The platform also reuses that same sensitivity information as the model's
*uncertainty* (next section).

---

## 4. The closure: an inversion engine is just another `PropertyModel` producer

This is the whole architectural bet. An inversion engine **consumes** [Observations](data-model.md)
plus a model domain (a mesh over the Engineering Frame) and **emits** a [PropertyModel](data-model.md)
with uncertainty and provenance — the *same* primitive the platform already
[stores](data-model.md), [resamples and fuses](fusion.md), and [visualizes](visualization.md).

```
Observations ──┐
               ├──▶  [ Inversion engine (plugin) ]  ──▶  PropertyModel + uncertainty + provenance
model domain ──┘        forward + inverse                       │
(mesh over Engineering Frame)                          (already-known downstream path)
                                                       store · resample · fuse · visualize
```

Because the output re-enters the system as an *ordinary* property model, the platform needs **no
new downstream machinery** to support inversion — only an upstream plugin type and a job to run
it. Everything past the engine boundary is reuse. This is exactly the
[plugin architecture](architecture.md) bet: a clean boundary keeps a heavy new capability
*additive* rather than invasive.

### 4.1 The engine contract

An engine is a small object: a declarative `spec` (capabilities) + a `run(ctx) → result`
method. Crucially, **no SimPEG/PyGIMLi type ever crosses the boundary** — the platform speaks
only NumPy + its own `Observation` / `PropertyModel` / `ModelDomain`. The solver containers are
built *inside* `run`. This keeps libraries swappable and version-isolated.

```python title="backend/geosim/inversion/engine.py — the spec + result (excerpt)"
@dataclass(frozen=True)
class InversionEngineSpec:
    id: str                       # "simpeg.gravity"
    kind: str                     # "gravity"
    library: str                  # "simpeg" | "pygimli" | "mock"
    methods: Sequence[str]        # canonical MethodKeys it can invert (e.g. ["gravity"])
    output_property: str          # PropertyType it recovers (e.g. "density")
    mesh_types: Sequence[str] = ("tensor",)
    coupling: str = "standalone"  # standalone | joint | petrophysical
    compute: str = "in_process"   # executionMode hint (heavy engines → worker_process)
    params_schema: dict = ...     # JSON-Schema for params; validated BEFORE the job runs

@dataclass
class InversionResult:
    values: np.ndarray   # (z, y, x) recovered CORE model, canonical units
    sigma:  np.ndarray   # (z, y, x) 1σ uncertainty — MANDATORY (see §6)
    iterations: int; final_phi_d: float | None; final_phi_m: float | None
```

The engine returns the recovered model on the *core* cells as a Z-up `(z, y, x)` array — the
same axis order and frame as everything else in [storage](data-model.md).

---

## 5. Meshes & the model domain

The inversion mesh is **not** the [fused grid](fusion.md), and must not be forced to be. Each
method has its own optimal discretization; forcing all onto one grid would either over-resolve
cheap methods or under-resolve sharp ones. So the inversion runs on a **method-chosen mesh** in
the Engineering Frame, and its result is *resampled onto the fused grid afterward* — exactly the
way every other property model is.

!!! note "Define: the three mesh types"
    - **TensorMesh** — a regular rectilinear grid (uniform boxes). Simple, fast to assemble.
      Used by gravity, magnetics, MT, simple DC. (This is the one the framework builds itself.)
    - **TreeMesh (octree)** — adaptive: cells subdivide where you need detail (near sensors)
      and coarsen where you don't (at depth), so you spend cells where they matter — like a
      spatial quadtree/octree. Used by EM and focused potential-field inversions.
    - **SimplexMesh (tetrahedral)** — unstructured triangles/tetrahedra that *conform to
      topography and electrodes*. Used by ERT/IP and seismic tomography over rough terrain.

A `ModelDomain` wraps the mesh plus two pieces of bookkeeping:

- **Core region** — the volume you care about, at target resolution. The recovered model is
  only trusted here, and only the core resamples onto the fused grid.
- **Padding** — geometrically expanding cells outward so the solver's boundary sits *far* from
  the target (gravity/EM fields are non-local; a near boundary would contaminate the core).
  Padding cells exist for physics and **never** leave the engine.
- **Active cells** — air above the surface model is inactive; the active set is topography-aware,
  tying the air/ground interface to the project surface.

```python title="backend/geosim/inversion/domain.py — extract_core (drop padding/air)"
def extract_core(self, cell_values):
    nx, ny, nz = self.mesh.shape_cells          # discretize order (x, y, z)
    cube = cell_values.reshape((nz, ny, nx))    # → Z-up (z, y, x) to match storage
    sz, sy, sx = self.core_slices()             # slice OFF the n_pad padding cells
    return cube[sz, sy, sx]
```

After convergence the recovered core is resampled onto the canonical fused grid via the
[fusion](fusion.md) resampler (carrying its uncertainty through the *same* resampling so
confidence stays co-registered), and both the native-mesh model and the fused model are
registered in the catalog. Native-resolution originals are preserved.

---

## 6. Uncertainty is mandatory, not optional

Every method "sees" differently — gravity and MT are smooth and non-unique; seismic is sharp —
and [fusion](fusion.md) and [well planning](well-planning.md) are only honest if that survives.
So an engine **must** emit an uncertainty field; a bare model with no uncertainty is *invalid*
and the result constructor rejects it:

```python title="backend/geosim/inversion/engine.py — sigma is mandatory"
if self.sigma is None:
    raise ValueError("InversionResult.sigma is MANDATORY — an inversion with no "
                     "uncertainty is invalid (doc 10 §2.3)")
```

There is a tiered menu, in descending rigour: (A) posterior covariance / model-resolution
matrix; (B) sensitivity-/depth-of-investigation–weighted confidence; (C) ensemble spread; (D) a
flat registry prior, explicitly flagged low-confidence. **Tier B is the expected default.** The
gravity engine, for example, derives σ from SimPEG's per-cell sensitivity weights: a *low*
weight means a poorly-constrained (deep / edge) cell, which becomes a *large* σ — so the
uncertainty literally reflects where the physics can and can't resolve.

---

## 7. The real engines

The backend is Python + FastAPI specifically to reach this ecosystem. Engines *wrap* mature
libraries; the platform owns the boundary, not the math.

### 7.1 SimPEG gravity — the plumbing-proof engine

A workstation-feasible, **linear** gravity inversion is the first engine because it wires the
whole phase together with the fewest moving parts (gravity's forward operator is linear, so it's
fast and well-understood). Inside `run` it builds a SimPEG `gravity.Survey` from the observation
stations + Bouguer anomaly, a `Simulation3DIntegral` over the active TensorMesh cells (the model
unknown is density anomaly $\Delta\rho$), runs the L2 Tikhonov inversion with the directive
stack from [§3.1](#31-is-the-dial-between-underfitting-and-overfitting), then adds a background
density to recover an absolute density model on the core. See
`backend/geosim/inversion/engines/gravity_simpeg.py`.

### 7.2 PyGIMLi ERT — the first geothermally-meaningful engine

[ERT (electrical resistivity tomography)](survey-methods/electrical.md) is next, because
resistivity is a direct clue to hot conductive fluid. The engine consumes an ERT observation
(a dipole-dipole **apparent-resistivity pseudosection** plus electrode geometry), runs a 2-D
PyGIMLi inversion on a topography-conforming **SimplexMesh**, and recovers a **resistivity**
model with PyGIMLi's model **coverage** (cumulative sensitivity) as the tier-B uncertainty. ERT
is intrinsically a *line* survey — the inversion lives in the vertical section under the
electrode line — so the 2-D result is swept across the thin `y` extent of the core, and the
coverage-derived σ inflates off-section, properly distrusting cells the line can't see. See
`backend/geosim/inversion/engines/ert_pygimli.py`.

| `method` | Library | Mesh | Output property | Local feasibility |
|---|---|---|---|---|
| `gravity` | SimPEG | Tensor/Tree | density | yes (minutes) |
| `magnetics` | SimPEG | Tensor/Tree | susceptibility | yes |
| `ert` / `ip` | PyGIMLi / SimPEG | Simplex | resistivity / chargeability | yes (moderate) |
| `seismic` (refraction) | PyGIMLi | Simplex | velocity | yes (light) |
| `mt` | SimPEG (NSEM) | Tensor/Tree | resistivity (deep) | marginal — "bring a server" |
| `em` (tdem/fdem/aem) | SimPEG | Tree (octree) | conductivity | no / marginal — server/GPU |

The compute reality is honest: workstation-feasible methods (gravity, magnetics, ERT,
traveltime) ship first; MT/EM/joint are "bring a server." Each engine declares a `compute`
profile and an `executionMode` so the same engine can run locally or on a remote/GPU worker —
**local-first does not mean local-only** for inversion.

---

## 8. Cooperative vs joint inversion

The platform's unique asset is that **every method already lives in one co-registered
Engineering Frame**. Coupling methods normally requires painful re-registration across
toolchains; here it's a property of the platform. Coupling ships in strict stages:

- **5a — single-method** *(build first)*. One method → one model, no coupling. Proves the engine
  plugin, jobs, mesh handoff, uncertainty, and validation. Everything depends on this working.

- **5b — cooperative (sequential)**. Invert method A; feed A's recovered model into method B as a
  **reference/starting model** or a **structure-guided weight** (high where A has sharp gradients
  → likely a geological boundary → relax B's smoothness across it). Implemented as a tiny **DAG
  of ordinary 5a jobs** plus a "model → reference/weight" adapter — *no* new solver math. Each
  stage is a real, separately-persisted PropertyModel, so the whole DAG reuses storage/fusion/
  serving unchanged.

```python title="backend/geosim/inversion/cooperative.py — the structure weight (excerpt)"
def structure_weight(self):   # high where the partner model has sharp gradients
    grad = np.gradient(self.values)
    mag  = np.sqrt(np.sum([g**2 for g in grad], axis=0))
    return (mag / np.max(mag)).astype(np.float32)   # normalised [0,1] cell weighting
```

- **5c — joint (simultaneous)** *(roadmap, last)*. Multiple methods on one **shared mesh**,
  coupled inside one objective: **cross-gradient** (penalise $\nabla m_A \times \nabla m_B$ so
  structures align without assuming a rock-physics law) or **petrophysically coupled (PGI)** (a
  rock-physics relationship ties the models). This is the heaviest lift and is explicitly *not*
  built in this phase.

!!! warning "Sequencing rule"
    Do **not** start 5c until ≥2 methods work standalone (5a) and cooperative coupling (5b) is
    validated against synthetic ground truth. 5b already delivers most of the multi-method payoff
    without a shared-mesh joint solver.

---

## 9. Validating against synthetic ground truth

Because inversion is non-unique, you can't trust an engine until you've checked it against a
*known* answer. The [synthetic generator](synthetic-data.md) provides exactly that: a known
ground-truth earth that it forward-models into simulated observations. Invert those, then score
the recovered model against the truth on the shared fused grid:

```
synthetic truth ──forward──▶ simulated observations ──▶ [engine] ──▶ recovered model
       │                                                                   │
       └──────────────────── score on the fused grid ◀────────────────────┘
```

| Check | Metric |
|---|---|
| **Data fit** | predicted-vs-observed misfit reaches χ ≈ 1 (neither over- nor under-fit) |
| **Model recovery** | recovered-vs-truth RMSE, structural similarity, anomaly localisation & amplitude |
| **Uncertainty honesty** | truth falls inside the stated confidence at the expected rate (calibration) |
| **Resolution realism** | recovered smoothness/DOI matches the method's physics (gravity smear vs ERT sharpness) |

An engine is "done" when it recovers the synthetic anomaly within its physical resolution limits
*and* its uncertainty is calibrated.

---

## 10. Jobs, reproducibility & the harness

Inversions are minutes-to-hours and must survive restarts, so they run as background jobs on the
platform's job queue (just another job `kind`, alongside ingest/fuse/pyramid). The **harness**
(`backend/geosim/inversion/harness.py`) is the conductor: **validate** params against the spec's
`paramsSchema` *before* enqueueing (fail fast, not three hours in) → **run** the engine →
supply a **default tier-B uncertainty** if the engine has none → **persist** the recovered core
as a PropertyModel → **resample** onto the fused grid. Every result records an
`InversionProvenance` (engine id/version, exact params incl. applied defaults, mesh fingerprint,
observation inputs, seed, convergence diagnostics) so any model answers "what produced this?"
and "re-run with χ = 1.5" is a one-field change. Inversion results are *interpretations*, not
measurements — they must be fully auditable.

---

## Key takeaways

- **Forward** = `model → data` (just physics, a pure function); **inverse** = `data → model`
  (the hard direction we actually want).
- Inversion is **ill-posed and non-unique**: more unknowns than data, unstable to noise, often
  depth-blind — a textbook under-determined, overfitting-prone optimization.
- **Regularization** (Tikhonov: smallness + smoothness) picks one model among the infinitely
  many; **β** is the trade-off knob, cooled until the misfit hits the χ ≈ 1 target.
- **Closure**: an inversion engine is *just another `PropertyModel` producer* — consumes
  Observations + a mesh, emits a model + mandatory uncertainty + provenance; everything
  downstream is reuse.
- The inversion **mesh is method-chosen** (Tensor/Tree/Simplex) and resampled onto the fused
  grid afterward; **uncertainty is mandatory** (tier B by default).
- **SimPEG** (gravity, the plumbing proof) and **PyGIMLi** (ERT, the first meaningful method)
  are the first engines; heavy methods go to a server.
- Coupling stages strictly: **5a single → 5b cooperative (a DAG of single jobs) → 5c joint
  (roadmap)**; validated against **synthetic ground truth**.

## Where this lives in the code

| Concern | File(s) |
|---|---|
| Engine contract (spec / context / result / provenance) | `backend/geosim/inversion/engine.py` |
| Model domain & meshing | `backend/geosim/inversion/domain.py` |
| Run harness (validate → run → persist → resample) | `backend/geosim/inversion/harness.py` |
| SimPEG gravity engine | `backend/geosim/inversion/engines/gravity_simpeg.py` |
| PyGIMLi ERT engine | `backend/geosim/inversion/engines/ert_pygimli.py` |
| Cooperative (5b) orchestration | `backend/geosim/inversion/cooperative.py` |
| Mock engine (for tests / plumbing) | `backend/geosim/inversion/mock.py` |
| API surface | `backend/geosim/api/inversion.py` |
