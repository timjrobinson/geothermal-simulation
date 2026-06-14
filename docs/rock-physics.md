# Rock physics & favorability

> **What you'll learn / why it matters.** [Fusion](fusion.md) puts every survey method on
> one shared 3-D grid, so each cell becomes a feature vector like
> `[resistivity, density, P-velocity, …]`. But a driller doesn't care about *resistivity* —
> they care about **heat**, **fluid**, and **permeability**. **Rock physics** is the bridge:
> a library of equations that turn *what geophysics measures* into *what geothermal cares
> about*. This page teaches that bridge from first principles — the actual equations,
> symbol-by-symbol — then shows how the platform combines those derived fields into one
> `[0,1]` **"drill here" favorability index**, and why the default combination rule is
> deliberately *not* a weighted average. If you know what a pure function, a type-checked
> interface, and lossy compression are, you already have the mental scaffolding; we'll attach
> the geoscience to it.

!!! abstract "Where this sits in the pipeline"
    [Ingestion](ingestion.md) → normalized [primitives](data-model.md) → [fusion](fusion.md)
    co-registers them onto a grid → **rock-physics transforms** (this page) derive new fields →
    **favorability** (this page) combines them → the [3-D viewer](visualization.md) renders it,
    always carrying [uncertainty](uncertainty.md). Rock physics is **stage ③** of the fusion
    engine; favorability is the headline product of stage ③.

---

## 1. What "rock physics" actually is

A geophysical survey never measures temperature or permeability directly. It measures a
**physical property** of the rock+fluid system — how it conducts electricity, how fast sound
travels through it, how dense it is — because *those* are the things that bend a field a
sensor can read at the surface. (If "physical property" is fuzzy, see the
[survey methods overview](survey-methods/index.md): each method is good evidence for *one*
property and blind to the others.)

**Rock physics** is the body of empirical and theoretical relationships that link these
properties to each other and to the quantities we actually want:

| What geophysics measures | What rock physics infers | What geothermal wants it for |
|---|---|---|
| electrical resistivity $\rho$ | pore-fluid conductivity, **temperature**, water saturation | hot brine = conductive |
| seismic P-velocity $V_p$ | **porosity**, fracturing | open space + cracks for fluid |
| bulk density $\rho_b$ | **porosity**, lithology | pore space |
| low resistivity + structure | **alteration** (clay cap) | fossil/active hydrothermal signature |
| microseismic event cloud | **fracture density** → **permeability** | a *path* for the fluid to flow |

!!! note "A CS analogy: rock physics is a decoder"
    Think of the subsurface as having compressed the "truth" (temperature, fluid,
    permeability) into a lossy, noisy encoding (resistivity, velocity, density) via the laws
    of physics. Rock physics is the **decoder**. Like any lossy decoder it is *approximate*
    and *assumption-laden*: the same resistivity can decode to different temperatures
    depending on porosity and salinity, exactly as the same JPEG bytes decode differently
    under different quantization tables. The platform's honesty rules (below, and on the
    [uncertainty page](uncertainty.md)) exist because this decode is never exact.

These relationships are **site-calibratable approximations, not universal truths**. Every
parameter (porosity, the cementation exponent, salinity, matrix density…) is first-class and
user-tunable, and where well/core/geochem data exist they are the
[calibration](#7-calibration-turning-proxies-into-measurements-honestly) anchor. Until calibrated, **every output is a likelihood /
proxy field**, never a measured value — a rule the engine enforces mechanically (see
[§4](#4-the-transform-engine) and [the uncertainty page](uncertainty.md)).

---

## 2. The key transforms, symbol by symbol

Each transform below is a real, registered class in
`backend/geosim/fusion/rockphys/`. We give the equation, define **every symbol**, and note
the assumptions the code declares (these `assumptions` strings are surfaced in the UI next to
the output layer — honesty is part of the data, not a footnote).

### 2.1 Archie's law — resistivity ↔ porosity & saturation

Archie's law is the workhorse of formation evaluation. It says the bulk electrical
conductivity of a clean (clay-free) rock comes entirely from the conductive **brine** in its
pore space — the rock matrix is an insulator. In conductivity form:

$$
\sigma_t \;=\; \frac{\sigma_w \, \phi^{m} \, S_w^{n}}{a}
$$

| Symbol | Meaning | Units |
|---|---|---|
| $\sigma_t$ | bulk (true) conductivity of the rock = $1/\rho_t$ | S/m |
| $\sigma_w$ | conductivity of the pore **brine** (the salty water) = $1/\rho_w$ | S/m |
| $\phi$ | **porosity** — fraction of rock volume that is pore space (0–1) | — |
| $S_w$ | **water saturation** — fraction of *pore* space filled with water (0–1) | — |
| $m$ | **cementation exponent** — how tortuous/connected the pores are (≈1.3–2.5) | — |
| $n$ | **saturation exponent** (≈1.5–2.5) | — |
| $a$ | **tortuosity factor** (≈0.5–2.5; often taken as 1) | — |

The two ratios $F = a/\phi^m$ (the *formation factor*) and $I = S_w^{-n}$ (the *resistivity
index*) are the classic packaging. Geothermally, **low resistivity ⇒ either high porosity,
high saturation, or hot/salty fluid** — Archie is how we disentangle which.

Solving for water saturation (the `archie_saturation` transform,
`rockphys/fluid.py`, `target="fluid"`):

$$
S_w \;=\; \left( \frac{a\,\rho_w}{\phi^{m}\,\rho_t} \right)^{1/n}
$$

```python
# backend/geosim/fusion/rockphys/fluid.py — ArchieSaturation.apply (clay-free)
rho_t = np.maximum(resistivity, 1e-6)          # ρ_t : measured true resistivity (Ω·m)
phi   = np.clip(porosity, 1e-4, 1.0)           # φ   : porosity (another derived field or a param)
sw    = (a_tortuosity * rho_w_ohm_m
         / (phi**m_cementation * rho_t)) ** (1.0 / n_saturation)   # Sw ∈ [0,1] (harness clamps)
```

!!! warning "Archie is nonlinear in $\phi$"
    Because $\phi$ enters as $\phi^m$, a small porosity error blows up in $S_w$. This is why
    the docstring recommends the **Monte-Carlo** uncertainty mode for this transform rather
    than the linearized delta-method default — see [uncertainty propagation](uncertainty.md#4-propagating-uncertainty-through-the-pipeline).

### 2.2 Arps relation — resistivity → temperature *likelihood*

Brine doesn't just conduct better when it's saltier; it conducts better when it's **hotter**
— ions move faster. The **Arps relation** captures this: fluid conductivity rises roughly
**2 % per °C** at fixed salinity. That gives us a thermometer made of electricity. The
`resistivity_to_temperature.arps` transform (`rockphys/temperature.py`,
`target="temperature"`) is the canonical example in the design doc, and it works in two
steps.

**Step 1 — Archie, to back out the pore-fluid conductivity** (assuming $a=1$, full
saturation):

$$
\sigma_w \;=\; \frac{\sigma_{\text{bulk}}}{\phi^{m}}, \qquad \sigma_{\text{bulk}} = \frac{1}{\rho}
$$

**Step 2 — invert the Arps temperature line** for absolute temperature in **kelvin** (the
canonical unit everywhere — see [coordinates, depth & units](spatial-framework.md)):

$$
\sigma_w(T) \;=\; \sigma_w(T_{\text{ref}})\,\bigl(1 + \alpha\,(T - T_{\text{ref}})\bigr)
\quad\Longrightarrow\quad
T \;=\; T_{\text{ref}} + \frac{\sigma_w/\sigma_w(T_{\text{ref}}) - 1}{\alpha}
$$

| Symbol | Meaning | Units / default |
|---|---|---|
| $\rho$ | input bulk resistivity (from MT/ERT inversion) | Ω·m |
| $\phi$ | porosity (param) | default `0.10` |
| $m$ | cementation exponent (param) | default `2.0` |
| $\sigma_w(T_{\text{ref}})$ | brine conductivity at the reference temperature | from `brine_conductivity(salinity, T_ref)` |
| $\alpha$ | **Arps slope** ≈ 0.02 per K (the "2 %/°C") | param `arps_slope_per_K`, default `0.02` |
| $T_{\text{ref}}$ | reference temperature | param, default `298.15` K (25 °C) |

```python
# backend/geosim/fusion/rockphys/temperature.py — ResistivityToTemperature.apply
sigma_bulk = 1.0 / np.maximum(resistivity, 1e-6)
sigma_w    = sigma_bulk / porosity**m_cementation              # Step 1: Archie, a=1
sigma_w_ref = brine_conductivity(fluid_salinity_ppm, T_ref_K)  # σ_w(T_ref)
temperature_K = T_ref_K + (sigma_w / sigma_w_ref - 1.0) / arps_slope_per_K   # Step 2: Arps
```

!!! danger "This output is a *likelihood*, not a measurement"
    `calibration_status = "uncalibrated"` on this class. The execution
    [harness](#4-the-transform-engine) therefore **retitles** the output layer
    `"temperature likelihood"` and stamps its uncertainty `tier = "proxy"`. It cannot present
    as a measured temperature volume until a [calibration run](#7-calibration-turning-proxies-into-measurements-honestly) promotes it —
    and even then only *near the wells*. This is the scientific-honesty gate; the
    [uncertainty page](uncertainty.md) explains why it's non-negotiable.

### 2.3 Velocity → porosity (Wyllie / Raymer-Hunt-Gardner)

Sound travels fast through solid rock and slow through fluid-filled pores, so **velocity is a
porosity proxy**. The `velocity_to_porosity` transform (`rockphys/porosity.py`) offers two
classic models.

**Wyllie time-average** — total travel time is the volume-weighted sum of time through matrix
and time through fluid:

$$
\frac{1}{V_p} \;=\; \frac{1-\phi}{V_{\text{matrix}}} + \frac{\phi}{V_{\text{fluid}}}
\quad\Longrightarrow\quad
\phi \;=\; \frac{1/V_p - 1/V_{\text{matrix}}}{1/V_{\text{fluid}} - 1/V_{\text{matrix}}}
$$

**Raymer-Hunt-Gardner (RHG)** — empirically more accurate at low porosity, solved via its
quadratic:

$$
V_p \;=\; (1-\phi)^2\,V_{\text{matrix}} + \phi\,V_{\text{fluid}}
$$

| Symbol | Meaning | Units / default |
|---|---|---|
| $V_p$ | measured P-wave velocity | m/s |
| $V_{\text{matrix}}$ | velocity in the solid mineral frame | param, default `5500` m/s |
| $V_{\text{fluid}}$ | velocity in the pore fluid (brine ≈ 1500 m/s) | param, default `1500` m/s |
| $\phi$ | output porosity | fraction, clamped to `[0, 0.5]` |

A **density → porosity** alternative (`density_to_porosity`) uses simple mass balance:
$\phi = (\rho_{\text{matrix}} - \rho_b)/(\rho_{\text{matrix}} - \rho_{\text{fluid}})$, with
$\rho_{\text{matrix}}$ defaulting to 2650 kg/m³ (quartz) and $\rho_{\text{fluid}}$ to
1000 kg/m³ (brine).

### 2.4 Waxman-Smits / dual-water — the clay correction

Plain Archie assumes the matrix is an insulator. But **clay** conducts on its own (its
charged mineral surfaces hold mobile ions), so in shaly/altered rock Archie **over-reads**
how much water is present — it mistakes clay conduction for brine conduction. The
**Waxman-Smits** model (`rockphys/fluid.py`, `WaxmanSmitsSaturation`) adds a clay term in
*parallel* with the brine path:

$$
\frac{1}{\rho_t} \;=\; \frac{\phi^{m}\,S_w^{n}}{a\,\rho_w} \;+\; B\,Q_v\,S_w^{\,n-1}
$$

| Symbol | Meaning |
|---|---|
| $B$ | equivalent counter-ion conductance (≈4.6 S/m per meq/mL at 25 °C) |
| $Q_v$ | cation-exchange capacity per unit pore volume, taken $\propto$ clay volume ($Q_v \approx Q_{v,\max}\,V_{\text{clay}}$) |

There is no closed form for $S_w$, so the code solves it by a handful of fixed-point
iterations. `DualWaterSaturation` subclasses it with the same solver but a different spec
(bound vs free pore water) so the UI lists both options. **Why this matters geothermally:**
hydrothermal systems are *full* of conductive alteration clay, so the clay correction is
often the difference between "there's fluid here" and "that's just the clay cap."

### 2.5 Alteration index — the clay cap

Hydrothermal alteration produces conductive smectite clay that forms a low-resistivity
**cap** over many geothermal reservoirs. `alteration_index` (`rockphys/alteration.py`,
`target="alteration"`) is a heuristic: a smooth **low-resistivity membership**

$$
L \;=\; \sigma\!\left(\frac{\log_{10}\rho_{\text{thresh}} - \log_{10}\rho}{w}\right)
$$

(where $\sigma$ is the logistic function, $\rho_{\text{thresh}}$ the clay-cap threshold, $w$
a log-width) optionally fused with a clay-volume structure proxy via a weighted geometric
mean. A **data-driven** sibling, `gmm_alteration_posterior`, fits a 2-component Gaussian
mixture to $\log_{10}\rho$ and returns the posterior probability of the low-resistivity class
— "let the data find the threshold." This is *clustering-as-a-transform* (see
[fusion §clustering](fusion.md)).

### 2.6 Microseismic → fracture density → permeability

Permeability — the existence of connected cracks for fluid to flow through — is the hardest
of the three geothermal ingredients to see remotely. Two proxies:

- **`microseismic_density`** (`rockphys/fracture.py`): tiny earthquakes ("microseismic
  events") cluster where rock is actively fracturing. The transform takes an event cloud
  *binned onto the grid* as per-cell counts (`events_to_count_volume` does the binning), then
  applies a 3-D **Gaussian kernel density estimate (KDE)** and peak-normalizes to a `[0,1]`
  fracture-density index. The bandwidth (kernel σ in cells) is the tunable smoothing knob —
  exactly the bandwidth/smoothing tradeoff a CS person knows from KDE plots.
- **`vp_vs_fracture_proxy`**: open or fluid-filled fractures lower the $V_p/V_s$ ratio, so a
  low-$V_p/V_s$ membership flags fractured zones.

Then **`fracture_to_permeability`** (`rockphys/permeability.py`) maps the fracture index to
intrinsic permeability $k$ (in m²) by **log-linear interpolation** between a tight-matrix
floor and a well-fractured ceiling, damped by alteration (clay gouge *seals* fractures):

$$
\log_{10} k \;=\; \log_{10} k_{\min} + d_{\text{frac}}\,\bigl(\log_{10} k_{\max} - \log_{10} k_{\min}\bigr),
\qquad
k \;\leftarrow\; k \cdot \bigl(1 - s\cdot \text{alteration}\bigr)
$$

| Symbol | Meaning | Default |
|---|---|---|
| $d_{\text{frac}}$ | fracture-density index (0–1) | input |
| $k_{\min}$ | tight-matrix floor (≈0.01 mD) | `1e-17` m² |
| $k_{\max}$ | well-fractured ceiling (≈1 darcy) | `1e-12` m² |
| $s$ | alteration-sealing factor | `0.5` |

Its first declared assumption is blunt: *"HEURISTIC relative-perm index, NOT a
flow/percolation simulation (low confidence)."* The platform never pretends a proxy is a
reservoir simulator.

---

## 3. The starter transform library at a glance

This is the **full** library shipped (a deliberate scope decision recorded in the design
doc), one registered transform per row, all in `backend/geosim/fusion/rockphys/`:

| `target` | Transform `id` | Inputs → output | Relationship |
|---|---|---|---|
| temperature | `rp.resistivity_to_temperature.arps` | ρ → temperature *likelihood* (K) | Archie + Arps |
| fluid | `rp.archie_saturation` | ρ, φ → $S_w$ | Archie |
| fluid | `rp.waxman_smits`, `rp.dual_water` | ρ, φ, clay → $S_w$ | clay-corrected |
| porosity | `rp.velocity_to_porosity` | $V_p$ → φ | Wyllie / RHG |
| porosity | `rp.density_to_porosity` | $\rho_b$ → φ | mass balance |
| alteration | `rp.alteration_index` | ρ (+clay) → index | low-ρ clay cap |
| alteration | `rp.gmm_alteration_posterior` | ρ → class prob | data-driven GMM |
| fracture | `rp.microseismic_density` | events → density | KDE |
| fracture | `rp.vp_vs_fracture_proxy` | $V_p, V_s$ → index | low-$V_p/V_s$ |
| permeability | `rp.fracture_to_permeability` | fracture (+alteration) → $k$ | heuristic, low-confidence |

Every one is `calibration_status = "uncalibrated"` out of the box.

---

## 4. The transform engine

The design principle is a clean separation a programmer will recognize: **a declarative spec
+ a pure function**.

- The **spec** (class attributes: `id`, `version`, `title`, `target`, `inputs`, `output`,
  `params`, `assumptions`, `calibration_status`) makes a transform *discoverable,
  parameterizable, versionable, and UI-drivable*. It's the typed interface.
- The **pure function** `apply(self, ctx, **inputs, **params)` does the physics. It receives
  each input as a NumPy array (already unit-converted) and each param as a scalar, and
  returns the output array. **No I/O, no units, no masking, no storage** — those are the
  harness's job. Pure in, pure out, trivially testable. (See `Transform` in
  `backend/geosim/fusion/transform.py`.)

```python
# The contract every transform implements (abbreviated from transform.py)
class ResistivityToTemperature(Transform):
    id      = "rp.resistivity_to_temperature.arps"
    version = "1.0.0"                       # semver — bump on any math/param change (§versioning)
    target  = "temperature"
    inputs  = [InputSpec("resistivity", unit="ohm*m", required=True)]
    output  = OutputSpec("temperature", unit="kelvin", valid_range=(273, 673),
                         proxy_when_uncalibrated=True)
    params  = [Param("porosity", float, default=0.10, range=(0.01, 0.5), sigma=0.03), ...]
    assumptions = ["single liquid brine phase (no boiling / steam)", ...]
    calibration_status = "uncalibrated"
    def apply(self, ctx, resistivity, *, porosity, ...): ...   # pure math
```

### 4.1 The execution harness — what `run_transform` guarantees

`run_transform()` is the common pipeline that lets every `apply()` stay pure math. It runs
eight steps (numbered in the code, `transform.py`):

1. **Resolve inputs** to the fused grid — auto-[resample](fusion.md) a named native model if
   it isn't on the grid yet.
2. **Unit-check & convert** each input to the transform's declared unit via `pint`. A
   wrong-*dimension* input is a hard error — you cannot accidentally feed velocity where
   resistivity is expected.
3. **Build the valid mask** = logical AND of every *required* input's coverage. A cell
   missing any required input becomes **NaN (nodata)**, never silently zero-filled. (This is
   the footprint-honesty rule from [fusion](fusion.md); it's why a derived field has *holes*
   exactly where its inputs don't overlap.)
4. **Vectorized `apply`** over only the valid cells.
5. **Clamp** to the output's `valid_range`; the out-of-range fraction is recorded (a high
   fraction usually means bad parameters).
6. **Propagate σ** — see [§4.3](#43-uncertainty-rides-through-every-transform).
7. **Stamp calibration honesty** — if `uncalibrated`, retitle to `"<target> likelihood"` and
   set `tier = "proxy"`. The output tier is the **minimum tier over the inputs**, then
   *capped* by the transform's own calibration status (an uncalibrated transform can never
   emit better than `proxy`).
8. **Write** the output + paired σ as a derived `PropertyModel` carrying a full provenance
   recipe.

### 4.2 Output is a first-class `PropertyModel`

A transform's output is **indistinguishable downstream from an ingested property model** —
same [data-model](data-model.md) schema, same Zarr storage, same renderer. The only
difference is a **provenance block** that is a complete, reproducible recipe:

```jsonc
// the derivation block written into provenance (transform.py:_write_derived)
"derivation": {
  "kind": "transform",
  "transformId": "rp.resistivity_to_temperature.arps",
  "transformVersion": "1.0.0",
  "fusedGridId": "fem_…",
  "inputs": [ {"property": "resistivity", "propertyModelId": "…", "version": "…"} ],
  "params": { "porosity": 0.10, "m_cementation": 2.0, "arps_slope_per_K": 0.02, ... },
  "calibrationStatus": "uncalibrated",   // ⇒ output is a likelihood/proxy field
  "tier": "proxy",
  "assumptions": [ "single liquid brine phase (no boiling / steam)", ... ],
  "independence": "assumed_independent"  // see the uncertainty page
}
```

Because the output *is* a `PropertyModel`, you can **chain** transforms: velocity → porosity,
then (porosity + resistivity) → saturation. And because the derivation block is a complete
recipe, any derived volume is **fully reproducible** from its inputs + transform version +
params. Re-running with different params yields a **new versioned instance**, never an in-place
mutation.

### 4.3 Uncertainty rides through every transform

By default the harness propagates 1σ by the **delta method** (first-order error propagation),
computing the partial derivatives numerically so transform authors never hand-derive a
Jacobian:

$$
\sigma_y^2 \;\approx\; \sum_i \left(\frac{\partial f}{\partial x_i}\right)^2 \sigma_{x_i}^2
\;+\; \sum_\theta \left(\frac{\partial f}{\partial \theta}\right)^2 \sigma_\theta^2
$$

The second sum is **parameter uncertainty** — declare a `sigma` on a `Param` (e.g. porosity
$\sigma = 0.03$) and it contributes, often *dominating* once a field is calibrated. For
strongly nonlinear transforms (Archie!) you can opt into **Monte-Carlo** mode instead. The
full treatment — including why a *low* σ does **not** mean a cell is *well resolved* — lives
on the dedicated [uncertainty page](uncertainty.md).

---

## 5. Geothermal favorability — the "drill here" index

Everything so far produces *evidence layers*. **Favorability** is the special transform that
fuses them into one targetable index in `[0,1]` — the thing a driller actually points at
(`backend/geosim/fusion/favorability.py`).

### 5.1 The headline geothermal idea

> A producible geothermal play needs **heat AND fluid AND permeability** to occur in the
> **same cell**. Heat with no fluid is dry hot rock. Fluid with no permeability is trapped.
> Each survey method is good evidence for *one* of the three and blind to the others.
> Favorability is where they line up.

### 5.2 Why the default is fuzzy-conjunction, not a weighted average (a CS design lesson)

This is the most instructive design decision in the whole engine, and it's pure computer
science. The naïve approach is a **weighted linear sum**:

$$
F \;=\; \frac{\sum_i w_i\, e_i}{\sum_i w_i}, \qquad e_i \in [0,1]
$$

It's simple and transparent — and **catastrophically wrong as a default**, because it is
**compensatory**: a soaring temperature score can numerically *offset* absent permeability
and paint a **dry-hot cell as "favorable."** That's the worst possible failure mode — it
sends a drill rig to a place with no fluid path.

The fix is to encode "AND" the way the physics means it: **fuzzy-conjunction**
(non-compensatory). The shipped default combines the *required* evidence with a fuzzy-AND so
that **any absent required layer pulls the cell toward 0**:

$$
F_{\text{AND}} \;=\; \min_i(e_i) \quad\text{(Zadeh AND)} \qquad\text{or}\qquad F_{\text{AND}} \;=\; \prod_i e_i \quad\text{(product AND)}
$$

Supporting evidence can only *boost* via a soft fuzzy-OR ($\max$), never rescue a missing
required conjunct:

$$
F \;=\; F_{\text{AND}} + (1 - F_{\text{AND}})\cdot F_{\text{OR}}, \qquad
F_{\text{OR}} = \max_{j \in \text{supporting}} e_j
$$

!!! tip "The analogy: AND vs a weighted sum is a `min` vs a mean"
    A weighted average is *forgiving* — one big term dominates. A `min` (or product) is
    *unforgiving* — the weakest link decides. Geothermal favorability is a weakest-link
    problem: you cannot drill heat you cannot reach. So the default operator must be the
    unforgiving one. Weighted-linear still ships, but as an explicit **exploratory** mode —
    and even then, cells missing a `required` layer are **flagged and excluded from
    top-targets**, never silently averaged away.

```python
# backend/geosim/fusion/favorability.py — _fuzzy_and (NaN-aware)
s = np.where(np.isfinite(stack), stack, 0.0)   # a missing required conjunct → 0 (pulls toward 0)
return np.prod(s, axis=0) if op == "product" else np.min(s, axis=0)
```

The three combination methods:

| Method | Per-cell formula | Character | When |
|---|---|---|---|
| **fuzzy** *(default)* | fuzzy-AND over required, fuzzy-OR over supporting | non-compensatory; weakest-link | physically faithful — the default |
| **weighted** *(exploratory)* | $\sum w_i e_i / \sum w_i$ | compensatory; transparent | exploration, with a missing-required guard |
| **bayesian** | posterior odds (weights-of-evidence) | calibratable to known plays | **deferred** — raises `NotImplementedError` until training data exists |

### 5.3 Membership curves — turning raw fields into `[0,1]`

Each evidence layer is mapped to a fuzzy membership in `[0,1]` by a per-layer **transfer
function** (`TransferFn`), user-editable in the UI exactly like a viewer
[transfer function](visualization.md):

| Type | Shape | Use |
|---|---|---|
| `ramp` | linear from `lo` (→0) to `hi` (→1); `hi<lo` descends | "favorable T ramps 150→250 °C" |
| `sigmoid` | logistic at `center`, steepness `k` | soft threshold |
| `gaussian-band` | peak at `center`, half-width `width` | favorable *around* a value |

```python
# membership() — NaN (no coverage) stays NaN; the curve never invents evidence (favorability.py)
if tf.type == "ramp":
    m = (xf - lo) / (hi - lo)        # hi < lo ⇒ descending ramp automatically
    out[finite] = np.clip(m, 0.0, 1.0)
```

A `FavorabilitySpec` (the wire format) declares the evidence list, each with its `source`
model, `target` property, `transferFn`, `weight`, and `role` (`required` | `supporting`),
plus a `missingPolicy` (`nodata` | `neutral` | `drop`) governing how a cell lacking one layer
is treated. Because everything is user-set, favorability is explicitly a **research
instrument**, not a fixed score.

### 5.4 The two honesty diagnostics

A favorability score alone is dangerous — a high number over a cell where only **one of three**
required layers actually has data is a *warning*, not a target. So favorability ships **two
companion diagnostic volumes** alongside the score and its confidence volume (all renderable
`[0,1]` fields):

- **Evidence overlap** — per cell, *what fraction of the required evidence layers actually
  cover it* (respecting each layer's footprint/DOI). `overlap_frac = covered_required /
  total_required`. A favorability of 0.9 at overlap 0.33 means "looks great, but we only
  measured one of the three things."
- **Assumption burden** — per cell, *what fraction of the contributing evidence rides on
  uncalibrated/proxy transforms*. It surfaces where a hotspot is essentially "the rock-physics
  guessed," so [calibration](#7-calibration-turning-proxies-into-measurements-honestly) can be prioritized exactly there. (Native
  ingested sources don't add burden; only proxy-tier derived ones do.)

```python
# favorability.py — overlap and confidence
overlap_frac = overlap_count / float(len(req_idx))   # fraction of required layers present
confidence   = np.where(any_coverage,
                        np.clip(overlap_frac * (1.0 - burden), 0.0, 1.0),
                        np.nan)                        # low overlap OR high burden ⇒ low confidence
```

!!! example "Reading a favorability result"
    Cell A: favorability 0.85, overlap 1.0, burden 0.0 → strong target, all three ingredients
    measured, all calibrated. **Drill candidate.**
    Cell B: favorability 0.85, overlap 0.33, burden 0.66 → the score is a mirage: two required
    layers are absent and the present one is an uncalibrated proxy. **Go calibrate, don't drill.**

---

## 6. Cross-plots & clustering as transforms

Once N properties share the grid, every cell is a feature vector, which unlocks ordinary
multivariate analysis — covered in depth on the [fusion page](fusion.md): 2-D/3-D
cross-plots (e.g. $\log\rho$ vs density to spot an anomaly cluster), histograms, correlation
matrices, and **clustering** (k-means / Gaussian Mixture) that turns the property stack into
interpretable rock classes. GMM posteriors double as a soft alteration likelihood that
favorability can consume (that's `gmm_alteration_posterior`, §2.5). The platform clusters;
the geologist names the clusters.

---

## 7. Calibration — turning proxies into measurements, honestly

A proxy field becomes a *measurement* only when it is anchored to ground truth, and **only
where that ground truth reaches**. Calibration is the *centre* of the rock-physics workflow,
not an afterthought (`backend/geosim/fusion/calibration.py`). The loop has four stages:

```
① INGEST ground truth   well logs / core / geochem sampled ALONG the well path
                        (MD → Engineering XYZ via minimum-curvature; NOT voxelized)
        ▼
② FIT site params       least-squares fit of the transform's params to the
                        (measured ↔ predicted) pairs at the probes →
                        a PARAMETER DISTRIBUTION (mean + σ), not a point estimate
        ▼
③ RE-RUN               push the calibrated params (and their σ) through apply()
                        over the full grid; param σ now feeds σ-propagation (§4.3)
        ▼
④ PROMOTE (spatially)   calibrationStatus → well_calibrated and tier proxy → quantitative
                        ONLY within `resolving_distance` of a probe; far cells STAY proxy
```

Three points a programmer should internalize:

- **Parameter distributions, not point fits.** The fit returns a per-param 1σ from the
  least-squares covariance ($\mathrm{Cov} \approx s^2 (J^\top J)^{-1}$). An unidentifiable
  parameter gets $\sigma = \infty$ — reported as un-pinned-down rather than spuriously tight.
  That σ then flows into propagation and is *often the dominant uncertainty term* after
  calibration.
- **Promotion is spatially honest.** One well calibrates its *neighbourhood*, not the basin.
  `promote_spatial` marks only cells within `resolving_distance` of a probe as
  `quantitative`; everything beyond stays `proxy` / "likelihood." The assumption-burden
  diagnostic (§5.4) shows exactly where that frontier is.
- **Synthetic vs real.** The [synthetic earth](synthetic-data.md) ships full *truth fields*,
  so `score_against_truth` can grade a calibration against an oracle — but its result is
  **flagged `synthetic_only`** so it can never masquerade as a real-data quality metric. Real
  projects have only the sparse probes; the calibrated transform is the best estimate, never a
  checked-against-truth value.

```python
# calibration.py — the spatial-honesty gate (a single well does not calibrate the basin)
promoted = dist_to_nearest_probe <= resolving_distance   # near probes → quantitative
# everything else keeps tier="proxy" and its "likelihood" labelling
```

---

## Key takeaways

- **Rock physics is a lossy, assumption-laden decoder** from measured physical properties
  (resistivity, velocity, density) to geothermal targets (temperature, fluid, porosity,
  permeability). Know the equation *and* its assumptions.
- The **headline equations**: Archie ($\sigma_t = \sigma_w \phi^m S_w^n / a$) links
  resistivity to porosity/saturation; **Arps** turns it into a thermometer; **Wyllie/RHG**
  turn velocity into porosity; **Waxman-Smits** corrects for clay; **microseismic KDE →
  fracture density → permeability** is an explicit low-confidence proxy.
- A transform is a **declarative spec + a pure `apply()`**; the **harness** handles units,
  nodata masking, clamping, σ-propagation, calibration stamping, and storage, so the physics
  stays pure and testable. Outputs are first-class, versioned, chainable `PropertyModel`s.
- **Favorability** combines evidence into a `[0,1]` index. The default is
  **fuzzy-conjunction** (heat ∧ fluid ∧ permeability), *not* a weighted average, because a
  weighted average is **compensatory** and would call a dry-hot cell "favorable" — a
  weakest-link problem demands a `min`/product, not a mean.
- Nothing presents as more certain than the data supports: uncalibrated outputs are
  **"likelihood" / proxy** fields, favorability carries **evidence-overlap** and
  **assumption-burden** diagnostics, and **calibration** promotes a field to a measurement
  **only where a well actually reaches**.

## Where this lives in the code

| Concern | Path |
|---|---|
| Transform base class + execution harness + σ propagation | `backend/geosim/fusion/transform.py` |
| Temperature (Archie+Arps) | `backend/geosim/fusion/rockphys/temperature.py` |
| Fluid / saturation (Archie, Waxman-Smits, dual-water) | `backend/geosim/fusion/rockphys/fluid.py` |
| Porosity (Wyllie/RHG, density) | `backend/geosim/fusion/rockphys/porosity.py` |
| Alteration (index + GMM posterior) | `backend/geosim/fusion/rockphys/alteration.py` |
| Fracture density (microseismic KDE, Vp/Vs) | `backend/geosim/fusion/rockphys/fracture.py` |
| Permeability proxy | `backend/geosim/fusion/rockphys/permeability.py` |
| Favorability + honesty diagnostics | `backend/geosim/fusion/favorability.py` |
| Calibration loop | `backend/geosim/fusion/calibration.py` |
| Cross-plot / clustering | `backend/geosim/fusion/analysis.py` |

See also: [the data model](data-model.md), [fusion](fusion.md),
[uncertainty & scientific honesty](uncertainty.md), [the synthetic data generator](synthetic-data.md),
and the [glossary](glossary.md) for any term defined here.
