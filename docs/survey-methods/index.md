# The Survey Methods — How to read these pages

> **What you'll learn / why it matters.** This is the entrance to the heart of the
> documentation: one page per *family* of geophysical survey. Before you dive into any
> single method, this page gives you the **shared template** every method page follows, a
> **map** of which method senses which physical property (and why a geothermal explorer
> cares), and the four pieces of vocabulary — **forward problem**, **inverse problem**,
> **depth of investigation**, **resolution kernel**, and **non-uniqueness** — that recur on
> *every* page. Read this once and the rest of the section will feel like a familiar API.

## Why "many surveys, one earth" is the whole game

You cannot see through kilometres of rock. So a geothermal explorer measures the Earth
*indirectly* — from the surface, the air, a satellite, or down a borehole — using physics.
Each method is a different **sensor** that responds to a different **physical property** of
the rock (its density, its electrical resistance, the speed of sound through it, …). No
single sensor tells you where to drill. The platform's job, described in
[the core problem](../core-problem.md) and built out in [fusion](../fusion.md), is to line
all these sensors up in one 3-D frame so you can see where their stories agree.

These method pages are where you learn *what each sensor actually measures*, *what it can
and cannot see*, and *what its raw files look like* before the platform normalizes them
into the three primitives of [the data model](../data-model.md).

!!! abstract "A CS analogy for the whole section"
    Think of each survey method as a **lossy, band-limited sensor** sampling a hidden 3-D
    signal (the true earth). Each sensor has its own **sampling rate** (station spacing),
    its own **bandwidth** (it only resolves features above some size), its own **dynamic
    range** (what it's sensitive to at all), and its own **noise floor**. Reconstructing
    the original signal from these degraded measurements is an **inverse problem**, and —
    like deconvolving a blurred image — it has *many* possible answers. Holding several
    independent sensors over the same scene is how you pin down the one answer that's real.

## The five terms you need before reading any method page

Every method page uses these. Learn them here; we link back to this section throughout, and
they all appear in the [glossary](../glossary.md).

### Forward problem vs inverse problem

This is the single most important idea in geophysics, and it maps cleanly onto computer
science.

- **Forward problem** — *given the earth, predict the measurement.* "If the rock at this
  spot has density $\rho$, what gravity reading would my instrument show?" The forward
  problem is a well-defined function: one earth in, one (noisy) dataset out. It is a
  **simulator**. In this codebase, the [synthetic data generator](../synthetic-data.md)
  *is* a collection of forward models — see `backend/geosim/synthgen/forward/` — that take a
  known truth earth and produce realistic survey files.

- **Inverse problem** — *given the measurement, recover the earth.* "I measured these
  gravity values at the surface; what density distribution underground produced them?" This
  is the function run **backwards**, and running it backwards is hard, slow, and
  ambiguous. It is the central subject of [forward modeling & inversion](../inversion.md).

$$
\underbrace{m}_{\text{earth model}} \;\xrightarrow{\;\;\text{forward } F(m)\;\;}\; \underbrace{d}_{\text{data}}
\qquad\qquad
\underbrace{d}_{\text{data}} \;\xrightarrow{\;\;\text{inverse } F^{-1}(d)\;\;}\; \underbrace{m}_{\text{earth model}}
$$

Here $m$ is the model (e.g. a 3-D density volume), $d$ is the data (e.g. gravity readings),
and $F$ is the physics that turns one into the other. The forward operator $F$ is unique and
computable. The inverse operator $F^{-1}$ generally **does not exist as a clean function** —
which leads straight to the next term.

!!! tip "The compiler analogy"
    The forward problem is like **running a program** to get output. The inverse problem is
    like staring at the output and trying to reconstruct the source code. Many different
    programs print `42`. Many different earths produce the same gravity map.

### Non-uniqueness

**Non-uniqueness** means *different earths can produce identical data.* The inverse problem
has many solutions, and the data alone cannot choose between them. A small dense body close
to the surface and a large dense body deep down can pull on your gravimeter exactly the
same way. Potential-field methods (gravity, magnetics) are the textbook offenders, but every
method is non-unique to some degree.

The platform deals with non-uniqueness in three ways, all of which you'll meet later:

1. **Carry uncertainty everywhere** so a number is never trusted more than it deserves
   ([uncertainty & scientific honesty](../uncertainty.md)).
2. **Fuse independent methods** so one method's ambiguity is constrained by another's
   strength ([fusion](../fusion.md)).
3. **Regularize the inversion** — inject prior assumptions (smoothness, known geology) to
   pick *a* plausible answer ([inversion](../inversion.md)).

!!! warning "This is why you can't just trust one pretty resistivity cube"
    An inverted volume *looks* definitive — it's a crisp 3-D image with colors. But it is
    one of infinitely many models consistent with the data, chosen by the inversion's
    assumptions. Treat every inverted property model as a hypothesis, not a photograph.

### Depth of investigation (DOI)

**Depth of investigation** is *how deep a method can actually "see"* with usable signal.
Below the DOI the measurement contains essentially no information about the rock — the
sensor's sensitivity has decayed to noise, and anything an inversion shows there is coming
from its regularization assumptions, not the data.

In signal terms: DOI is where the **signal-to-noise ratio crosses 1**. The platform takes
DOI seriously — when 1-D soundings are
[stitched into a 3-D volume during ingestion](../ingestion.md), everything below each
sounding's DOI is *masked and flagged low-confidence* rather than silently extrapolated
(see `backend/geosim/ingestion/` and design doc 03 §4).

DOI varies enormously across methods and is one of the main reasons you need more than one:

| Method | Rough DOI | Why |
|---|---|---|
| ERT (electrical) | ~15–20 % of array length | current spreads only so far from finite electrodes |
| Seismic reflection | km, but layer-dependent | energy attenuates and scatters with depth |
| Gravity / magnetics | "everything, blurrily" | the field sums *all* mass/magnetization, with no depth resolution |
| Magnetotellurics (MT) | tens of m to tens of km | controlled by frequency (the **skin effect**) |

### Resolution kernel

A real sensor never reports the value at a single point. It reports a **weighted average**
of the true property over a *region* — and the shape of that weighting is the **resolution
kernel** (also called the *sensitivity kernel* or *footprint*).

!!! abstract "The convolution / point-spread-function analogy"
    The resolution kernel is exactly a **point spread function**. The measurement is the
    true earth **convolved** with the kernel: $d(\mathbf{x}) = \int K(\mathbf{x},\mathbf{x}')\,m(\mathbf{x}')\,d\mathbf{x}'$.
    A small, sharp kernel means high resolution (you see fine detail); a broad kernel means
    a blurry, smoothed image. Inversion is, in part, an attempt to *deconvolve* the kernel —
    and just like image deconvolution, it amplifies noise and is unstable, which is another
    face of non-uniqueness.

You can see a literal resolution kernel in the code: the ERT forward in
`backend/geosim/synthgen/forward/electrical.py` computes each apparent-resistivity reading
as a **Gaussian-sensitivity-weighted average** of the true resistivity column beneath the
electrode array — a depth-decaying "banana"-shaped kernel that fades out below the DOI. That
function *is* a resolution kernel, written in NumPy.

Two related qualities every method page reports:

- **Lateral resolution** — the smallest *horizontal* feature you can distinguish (set
  mostly by station/line spacing — a sampling-rate question).
- **Vertical resolution** — the smallest *vertical* feature you can distinguish (set by the
  physics; usually degrades with depth as the kernel broadens).

## The shared template every method page follows

So you always know where to look, each page in this section is laid out the same way:

1. **The physics** — what natural law the method exploits, with the real equations
   (symbols all defined).
2. **What property it senses** — the single physical property (density, resistivity, …)
   the method actually responds to, and how that connects to rock and to heat/fluid.
3. **What it CAN and CAN'T see** — the honest strengths and blind spots. (Every method is
   blind to something; this is why fusion exists.)
4. **Depth of investigation & resolution** — how deep, how sharp, how the kernel behaves.
5. **Native file format, annotated** — a few real lines of the industry format (CSV, `.stg`,
   GeoTIFF header, EDI, LAS, SEG-Y…), commented field-by-field, so you recognize a file
   when you see one.
6. **The normalized primitive it becomes** — what
   [ingestion](../ingestion.md) turns the file into: an **`Observation`**, a
   **`PropertyModel`**, or a **`GeologicalFeature`** (see [the data model](../data-model.md)),
   and which `geometryKind` / `support.kind` tag it carries.
7. **Where it's strong for geothermal** — which of the three targets (heat, fluid,
   permeability) the method is good evidence for.
8. **Key takeaways** and **Where this lives in the code.**

### Refresher: the three normalized primitives

Every method, no matter how exotic its file format, collapses into one of three data
structures defined in [the data model](../data-model.md). Keep these in mind on every page:

| Primitive | What it is | Carries |
|---|---|---|
| **`Observation`** | the immutable raw measurement, tied to where/when it was taken | a `geometryKind` ∈ `points`, `soundings`, `profile2d`, `traces`, `raster2d`, `wellcurve`, `tensor` |
| **`PropertyModel`** | a continuous field of *one* physical property | a `support.kind` ∈ `volume`, `grid2d`, `section`, `mesh` |
| **`GeologicalFeature`** | a discrete interpreted shape (a fault, a well path) | vector geometry |

A method that arrives **already inverted** (a finished resistivity cube) becomes a
`PropertyModel` directly. A method that arrives **raw** (gravity station readings) becomes
an `Observation`, and only later — through an explicit, provenance-tracked step — produces a
`PropertyModel`. **Raw stays raw**; this is a core design rule of
[ingestion](../ingestion.md).

## The method → property → geothermal-relevance map

This is the table to bookmark. It maps every survey family to the **physical property** it
senses and to its **geothermal relevance** — specifically which of the three things a
geothermal target needs in the same place: **heat** (hot rock/fluid), **fluid** (water to
carry the heat), and **permeability** (cracks for the fluid to flow). Each row links to its
detailed page.

| Method | Senses (physical property) | What that reveals | Geothermal relevance |
|---|---|---|---|
| [Gravity](potential-fields.md) | bulk **density** | basin shape, faults, intrusions, voids | structure; dense intrusions = possible heat source |
| [Magnetics](potential-fields.md) | magnetic **susceptibility** | magnetic rocks, and **demagnetized** alteration zones | **magnetic low** over hydrothermal alteration — a heat/fluid fingerprint |
| [ERT (DC resistivity)](electrical.md) | electrical **resistivity** | the conductive **clay cap**, faults | shallow, sharp map of the clay cap that seals a reservoir (fluid + heat indicator) |
| [IP (induced polarization)](electrical.md) | **chargeability** | disseminated clay & sulphide minerals | distinguishes conductive clay from conductive brine; alteration mapping |
| [TEM / AEM](electromagnetic.md) | electrical **conductivity** | conductive layers, clay cap, fast coverage | airborne reconnaissance of the conductive cap |
| [Magnetotellurics (MT)](electromagnetic.md) | deep electrical **resistivity** | conductor down to many km | the *deep* conductive clay cap above a reservoir — flagship geothermal method |
| [Seismic reflection/refraction](seismic.md) | acoustic **velocity / impedance** | layer boundaries, faults, structure | sharp structural framework; locating permeable faults |
| [Microseismic (passive)](seismic.md) | tiny **earthquakes** (fracture activity) | where rock is cracking/moving now | direct evidence of **permeability** and active fractures (4-D) |
| [InSAR](insar.md) | surface **deformation** (mm) | ground swelling/sinking over time | pressure/volume change in a reservoir; monitoring (4-D) |
| [Well logs](boreholes.md) | **ground-truth** rock properties | resistivity, density, porosity *at the borehole* | calibration "ground truth" tying surface surveys to reality |
| [Temperature / heat-flow](boreholes.md) | **temperature** vs depth | the geothermal gradient directly | the most direct measurement of **heat** |
| [Geology & geochemistry](geology-geochem.md) | **lithology**, fluid/gas chemistry | rock units, faults, surface fluid chemistry | structural context; fluid chemistry hints at reservoir temperature |

!!! note "Read the pattern, not just the rows"
    Notice that **no single row covers all three of heat + fluid + permeability.** Gravity
    and magnetics are deep but blurry and structural. ERT is shallow but sharp. MT is deep
    and the geothermal workhorse but smooth. Seismic gives structure but not temperature.
    Wells give truth but only along a thin line. The genius of the platform is that each
    method's blind spot is another method's strength — fusion exploits exactly this
    complementarity.

## A note on units and frames (so the file examples make sense)

Every method page shows native files in the units and coordinate convention the instrument
produces. Two things happen on ingest, both covered in depth in
[coordinates, depth & units](../spatial-framework.md):

- **Coordinates** get reprojected into the project's **Engineering Frame** (a local
  X-East / Y-North / Z-Up metre grid). In this codebase's synthetic data, files are already
  written in that local frame with no CRS, so what you see in the examples *is* metres east
  / north / up.
- **Units** get canonicalized (mGal, nT, ohm·m, kelvin, …) by a units registry. The native
  unit is recorded in provenance so nothing is silently mis-scaled.

## Key takeaways

- Each survey method is a **lossy, band-limited sensor** for one physical property; no
  single method is sufficient, which is why the platform fuses many.
- **Forward problem** = earth → data (a simulator, what the synthetic generator does).
  **Inverse problem** = data → earth (hard, slow, ambiguous).
- **Non-uniqueness**: different earths produce identical data, so every inverted model is a
  hypothesis, not a photograph.
- **Depth of investigation (DOI)** is where signal-to-noise crosses 1 — below it, the data
  is silent and the platform masks it.
- The **resolution kernel** is a point-spread function: measurements are the true earth
  convolved with each method's footprint.
- Every method normalizes into one of three primitives — **`Observation`**,
  **`PropertyModel`**, **`GeologicalFeature`** — and geothermal success means finding
  **heat + fluid + permeability** coinciding, which only fusion across methods can show.

## Where this lives in the code

- Forward models (the "earth → data" simulators) per method:
  `backend/geosim/synthgen/forward/` (e.g. `potential_field.py`, `electrical.py`).
- Ingestion adapters that turn native files into primitives:
  `backend/geosim/ingestion/adapters/` (one module per method).
- The primitive definitions every page refers to: `backend/geosim/ingestion/base.py`
  (`RawObservation`, `RawPropertyModel`, `RawFeature`) and the data model in design doc
  `design/02-data-model.md`.
- Design references: `design/OVERVIEW.md` §3 (the method→format→primitive table) and
  `design/03-ingestion-adapters.md` (the adapter contract).
