# The core problem — many surveys, one earth

> **What you'll learn / why it matters.** This page explains *why fusion is genuinely hard* — not a plumbing
> detail but the reason the whole system exists. You'll learn the axes along which survey methods differ
> (dimensionality, physical property, coordinate frame, resolution & depth-of-investigation, sensitivity /
> non-uniqueness, uncertainty), the key insight that **different methods do not measure the same thing**, and the
> three machinery pieces that make fusion possible at all: a **shared frame**, a **common grid**, and **rock
> physics**. We introduce — at a conceptual level — the three primitives, the fused earth model, the Engineering
> Frame, provenance, and uncertainty (the [Foundations](spatial-framework.md) pages have the details). It ends
> with a worked story of one anomaly seen six different ways.

## The naïve expectation vs reality

A programmer new to this reasonably expects: "I have a dozen datasets of the underground; I'll just overlay them
and read off the answer." If they all measured the *same quantity* on the *same grid* in the *same coordinates*
with the *same reliability*, that would be true — fusion would be a trivial `merge`.

They do **none** of those things. Two surveys over the *identical* patch of ground can:

- have **different dimensionality** — one is a handful of points, another a full 3-D cube;
- measure **different physical properties** — density vs resistivity vs sound speed — that are only *indirectly*
  related to what you care about;
- arrive in **different coordinate systems and depth conventions** — lat/lon vs UTM, elevation vs depth, feet vs
  metres;
- have **wildly different resolution and depth reach** — one is sharp but shallow, the other deep but blurry;
- be **non-unique and disagree** — and that disagreement can be *physically correct*, not a bug;
- carry **different, often unstated, uncertainty**.

Fusion is the work of reconciling all six axes at once. The rest of this page takes them one at a time.

## The six axes of difference

This is OVERVIEW §1 — the table the entire architecture is built to reconcile.

| Axis | Range across methods | CS analogy |
|---|---|---|
| **Dimensionality** | 0-D point samples → 1-D soundings/well logs → 2-D profiles/grids → 3-D volumes → 4-D time-lapse | scalars → arrays → matrices → tensors → tensor *series* |
| **Physical property measured** | density, magnetic susceptibility, resistivity/conductivity, chargeability, seismic velocity (Vp/Vs), surface deformation, temperature, fluid chemistry | different *schemas* — you can't just concatenate columns |
| **Coordinate frame** | lat/lon, UTM, local grid; elevation vs depth; differing vertical datums | the same address written in incompatible formats |
| **Resolution & depth of investigation** | shallow & sharp (ERT, seismic) vs deep & smooth (MT, gravity); each has a resolution kernel | different sampling rates *and* different low-pass filters |
| **Sensitivity / non-uniqueness** | gravity & MT are smooth and non-unique; seismic gives sharp structure; each "sees" different rock physics | a lossy, many-to-one encoding you must invert |
| **Uncertainty** | every model carries error/resolution that must survive into fusion | every value needs an error bar attached, not stripped |

Let's unpack each.

### 1. Dimensionality — 0-D to 4-D

Methods live at different points on a dimensionality ladder, exactly like data structures:

- **0-D** — a single point sample. *Example:* one temperature reading at the bottom of a well; one geochemistry
  sample from a spring.
- **1-D** — a profile along one axis. *Example:* a **well log** is a curve of a property vs depth along the
  borehole; an **MT sounding** is resistivity vs depth at one station.
- **2-D** — a grid or a vertical section. *Example:* a gravity **anomaly grid** (a 2-D image draped over the
  surface); a seismic **line** (a 2-D vertical slice).
- **3-D** — a full volume. *Example:* an inverted resistivity **cube**.
- **4-D** — a 3-D field that changes over **time**. *Example:* **InSAR** ground-deformation time series, or a
  **microseismic** event cloud accumulating during a stimulation.

Fusing a 0-D temperature point with a 3-D resistivity cube is not a `merge` of like records; it is an
*interpolation/constraint* problem. Time (4-D) is treated as a first-class axis throughout (the viewer has a time
slider), so a static cube and a time-lapse stack coexist in one model. The
[data model](data-model.md) gives every dataset an optional time axis.

### 2. Physical property — nobody measures "where to drill"

This is the deepest point on the page, so slow down here. **No instrument measures heat, fluid, or
permeability directly** (except a thermometer in a well, at that one point). Each method measures *some physical
property of the rock* that is only *indirectly* and *non-uniquely* related to what you actually care about.

| Method family | Measures (physical property) | Connected to the target via… |
|---|---|---|
| Gravity / gradiometry | **density** | dense bodies, basin shape, faults |
| Magnetics | **magnetic susceptibility** | alteration destroys magnetism → magnetic *low* over a system |
| Electrical (ERT/IP) | **resistivity / chargeability** | conductive = hot/salty/clay-rich fluid; chargeable = clay/sulphide |
| Electromagnetic (TEM/MT) | **conductivity** (the inverse of resistivity), to *depth* | deep conductors = deep fluid/clay |
| Seismic | **velocity / impedance** (sound speed, contrasts) | structure (layers, faults); porosity/fracture soften velocity |
| Microseismic | **fracture activity** (tiny earthquakes) | active, permeable fractures |
| InSAR | **surface deformation** | fluid pressure changes pushing the ground |
| Boreholes | density, resistivity, velocity, **temperature**… *directly* | ground truth — but only along one line |

In CS terms, each method is a **lossy projection** of the true 3-D earth onto one property, through a physics
that throws information away. You never get "the earth"; you get a shadow of it cast by one kind of light. The
job of [rock physics](rock-physics.md) is to relate those shadows back toward the target — and the job of
*fusion* is to combine many shadows so the true shape is constrained.

### 3. Coordinate frame — the address-format mess

Geoscience data is a swamp of coordinate conventions, and a single mistake silently puts a survey in the wrong
place or flips a depth into the sky. The variations:

- **Horizontal:** geographic **lat/lon** vs projected grids (**UTM** zones, national grids), each a different
  CRS[^crs].
- **Vertical:** **elevation** (positive *up* from a sea-level datum) vs **depth** (positive *down* from the
  surface) — and depth itself splits into **MD** (measured depth along a curved borehole) vs **TVD** (true
  vertical depth). Mix these up and a well's bottom lands in the wrong layer.
- **Vertical datums:** "sea level" is not one thing — ellipsoidal height vs geoid/orthometric height differ by
  tens of metres.
- **Units:** metres vs feet; Ω·m vs S/m; °C vs K vs °F.

[^crs]: **CRS (Coordinate Reference System)** — the precise definition of how a number-pair maps to a place on
    Earth (e.g. `EPSG:32612` is UTM zone 12 North). The [spatial framework page](spatial-framework.md) covers
    these in depth.

The system's answer (detailed in [coordinates, depth & units](spatial-framework.md)) is to convert *everything*,
once, on ingest, into a single internal **Engineering Frame** and a single canonical unit per property — so no
code above the bottom layer ever touches the mess again.

### 4. Resolution & depth of investigation — sampling rate *and* low-pass filter

Two distinct ideas, both crucial.

- **Resolution** is how fine a feature a method can distinguish — its effective *sampling rate* and *blur*. A
  shallow electrical survey can resolve metre-scale layers; a gravity survey blurs everything into broad smooth
  highs and lows.
- **Depth of investigation (DOI)** is *how deep* a method can sense at all. Beyond its DOI, a method is simply
  **blind** — it returns nothing meaningful, and the model must mark that region as "no data," not zero.

A signal-processing analogy makes this precise: each method applies its own **low-pass filter** to the true
earth, and the cutoff frequency *gets lower with depth*. Deep features are smeared out; sharp deep edges are
unrecoverable. Worse, different methods have *different* filters:

- **ERT / seismic** — shallow and sharp (high spatial frequencies preserved near surface), but resolution and
  reach fall off quickly with depth.
- **MT / gravity** — deep but smooth (only low spatial frequencies survive); they "see" kilometres down but
  cannot draw a crisp edge.

!!! warning "Smoothness is not certainty"
    A gravity or MT model looks like a smooth blob *because the method is low-pass*, not because the earth is
    actually smooth there. Treating a smooth model as a precise one is a classic mistake. This is exactly why
    **uncertainty must travel with every model** (axis 6) and why fusion must honour each method's resolution
    kernel rather than naïvely averaging.

The system records the **DOI/coverage footprint** and marks beyond-reach regions as `NaN` (not zero) so the
viewer can be honest about where a method *knows nothing*. See [uncertainty](uncertainty.md).

### 5. Sensitivity & non-uniqueness — the inverse problem

Turning surface measurements into an earth model is an **inverse problem**, and it is **non-unique**: *many
different earths produce the same data.* For a programmer: it's like being handed a program's output and asked
for the source — infinitely many programs match, and you need extra assumptions (priors, regularization) to pick
one. Gravity and MT are notoriously non-unique (their smoothing throws away the information that would pin a
unique answer); seismic is sharper but still under-constrained.

This is not pessimism — it is *the* reason to fuse. Each independent method *constrains* a different aspect of the
true earth. A model that must simultaneously fit gravity **and** MT **and** seismic **and** a well is far more
tightly pinned than any one alone. Adding methods removes ambiguity. Fusion is, formally, the act of intersecting
the solution sets of many non-unique problems.

### 6. Uncertainty — error bars that must survive

Every property model carries error and resolution. The cardinal rule is that this uncertainty must **survive
into the fused model**, not get silently dropped when datasets are combined. If a vague, smooth survey and a
sharp, well-constrained one disagree, the fused result must *weight them by how much they deserve to be trusted*,
and the viewer must be able to show *confidence* as its own renderable layer. A number without its error bar is a
lie of false precision. The [uncertainty page](uncertainty.md) covers how it is propagated (delta-method by
default, Monte-Carlo where needed).

## The key insight, stated plainly

!!! abstract "Why fusion can't be a simple merge"
    > **Different methods do not measure the same thing.** They measure different *physical properties*, at
    > different *dimensionalities*, in different *coordinate frames*, at different *resolutions and depths*, with
    > different *non-uniqueness*, carrying different *uncertainty*. You cannot stack them like layers of a PNG.

Fusion is therefore only possible through three pieces of machinery:

1. **A shared spatial frame** — put every dataset in one coordinate system, one vertical datum, one set of units,
   so "this point" means the same thing for all of them. *(the [Engineering Frame](spatial-framework.md))*
2. **A common resampling grid** — resample every property model onto one canonical 3-D voxel grid so they can be
   compared, cross-plotted, and combined **cell by cell** — *without destroying* the native-resolution originals.
   *(the [Fused Earth Model](data-model.md))*
3. **Rock-physics relationships** — convert the disparate geophysical properties (resistivity, velocity,
   density…) toward the geothermal targets (temperature, fluid, permeability) so the methods can finally be
   talking about the *same thing*. *(the [rock-physics engine](rock-physics.md))*

The next sections introduce the data-model concepts those pieces rely on, at a conceptual level only.

## The concepts that make fusion possible (a first look)

These are introduced here so the worked story below makes sense; the full treatment is in
[the data model](data-model.md), [coordinates & depth](spatial-framework.md), and [uncertainty](uncertainty.md).

### The three primitives

Everything normalizes into exactly three record types:

- **Observation** — *what was measured, where.* Raw, immutable, tied to acquisition geometry (the gravity station
  readings, the MT impedance tensors, the seismic traces). Never destroyed. Think *append-only event log*.
- **Property Model** — a continuous 3-D field of *one* physical property (a resistivity cube, a velocity model),
  carrying its **units**, its **support geometry**, and its **uncertainty**. Think *a dense tensor with metadata*.
- **Geological Feature** — a discrete interpreted shape (a fault surface, a well path, a microseismic cloud).
  Think *vector geometry with attributes*.

### The fused earth model

The **Fused Earth Model** is the canonical 3-D voxel grid covering the region of interest, onto which any property
model can be *resampled*. It is the **common ground** for overlay, cross-plotting, and derived-property math —
the shared coordinate space where a `JOIN across volumes by cell index` finally becomes possible. Resampling is
**non-destructive**: the native originals are kept at full resolution; the fused grid is a *view* for comparison,
not a replacement. A project may hold several (a coarse overview plus a zoomed target grid).

### The Engineering Frame

The **Engineering Frame** is the single internal coordinate system: a local right-handed metre grid, **X = East,
Y = North, Z = Up**, with Z as elevation. Every coordinate in the system lives here. Connecting it to a real place
on Earth ("georeferencing") is just an *optional* rigid transform applied at the edges (ingest, terrain, export).
This is what lets a synthetic cube and a real survey "fuse through exactly the same code path." See
[coordinates, depth & units](spatial-framework.md).

### Provenance and uncertainty

- **Provenance** — every artifact records its full lineage: which raw files, which transforms, which parameter
  values, which versions produced it. Mandatory; there is no record without provenance. This is what makes a
  fused model *auditable* — you can always answer "where did this number come from?"
- **Uncertainty** — every property model carries an error/resolution estimate that propagates through fusion and
  every transform, surfacing as confidence layers in the viewer.

## A worked story: one anomaly, seen six ways

Here is the payoff, made concrete with the flagship `great-basin-v1` earth (see
[the geothermal primer](geothermal-primer.md)). There is **one** real feature underground: a fault-controlled
hot-water upflow with a conductive clay cap and a fractured, saline reservoir. Now watch six methods "see" the
*same* feature completely differently:

| Method | What it reports | What it's really sensing | What it's *blind* to |
|---|---|---|---|
| **Gravity** | a smooth, broad low/high tied to the basin and fault | density contrast; basin geometry | the temperature; the fluid; sharp edges |
| **Magnetics** | a magnetic **low** over the system | alteration that *destroyed* magnetite | depth of the source; whether it's hot *now* |
| **MT** (electromagnetic) | a deep, smooth **conductor** | conductivity to several km depth (fluid + clay) | the *sharp* top of the clay cap; structure |
| **ERT** (electrical) | a sharp, shallow **conductor** | the clay cap, crisply, near surface | anything below its DOI — the deep reservoir |
| **Seismic** | reflectors outlining layers and the **fault** | impedance contrasts = structure | the temperature and fluid almost entirely |
| **Borehole temperature** | one hot number at one well | heat, *directly* | everything more than metres from the hole |

Three things to notice — they are the whole reason this platform exists:

1. **They disagree, and the disagreement is correct.** MT shows a smooth deep conductor; ERT shows a sharp
   shallow one; they "conflict" at depth only because ERT is *blind* below its DOI while MT is smooth there.
   Resolving this requires honouring each method's resolution kernel — *not* averaging them. A naïve merge would
   produce nonsense.
2. **No single method finds the target.** Gravity/seismic give *structure* (where permeability might be); MT/ERT
   give *fluid/alteration*; the well gives *heat* at one point; magnetics confirms *alteration*. Only when you
   overlay them in one frame does the volume where **heat ∧ fluid ∧ permeability** all coincide become visible.
3. **You need the three machinery pieces to even line them up.** The shared **frame** puts the gravity grid, the
   MT cube, the seismic line, and the well in the same place; the common **grid** lets you sample them all at the
   same cell to cross-plot resistivity vs density; **rock physics** turns "low resistivity + hot well" into a
   *fluid/temperature likelihood* you can AND with a *permeability* proxy to get a favorability volume.

That favorability volume — the one place all the independent evidence agrees — is your drilling target. Building
it correctly, with uncertainty intact and provenance attached, is the core problem this whole system solves.

## Key takeaways

- Fusion is **not a merge.** Survey methods differ on **six axes** simultaneously: dimensionality, physical
  property, coordinate frame, resolution & depth-of-investigation, sensitivity/non-uniqueness, and uncertainty.
- **No method measures the target directly** — each is a *lossy projection* of the true earth onto one physical
  property, through physics that throws information away.
- Methods apply different **low-pass filters** with depth (shallow-sharp vs deep-smooth); smoothness is a
  property of the *method*, not proof the earth is smooth.
- Reconstructing the earth is a **non-unique inverse problem**; fusing independent methods is how you *constrain*
  it — each method removes some of the others' ambiguity.
- Fusion needs three machinery pieces: a **shared frame**, a **common resampling grid** (the Fused Earth Model),
  and **rock physics** — plus **provenance** and **uncertainty** carried throughout.
- The payoff is the volume where independent evidence for **heat ∧ fluid ∧ permeability** overlaps — the drilling
  target.

## Where this lives in the code

- The shared-frame machinery: `backend/geosim/catalog/spatial.py` and the `SpatialFrame` handling (doc 01).
- Resampling onto the common grid and cross-plotting: `backend/geosim/geomodel/builder.py`,
  `backend/geosim/api/fusion.py`, and on the client `frontend/src/lib/fusion.ts` /
  `frontend/src/lib/crossplot.ts`.
- The three primitives' catalog records: `backend/geosim/catalog/models.py` (`Observation`, `PropertyModel`,
  `Feature`, plus `FusedModel`/`FusedLayer` and `Provenance`).
- The "many surveys, one anomaly" ground truth comes from `backend/geosim/synthgen/` (the `great-basin-v1`
  scenario).
