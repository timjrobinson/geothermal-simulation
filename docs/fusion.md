# Fusion — the fused earth model

> **What you'll learn / why it matters.** [Ingestion](ingestion.md) gave every survey
> method one coordinate frame and one set of units — but each method still lives on its own
> grid, at its own resolution, covering its own patch of ground. You still cannot answer the
> only question that matters: *at this exact spot underground, what does gravity say AND what
> does resistivity say AND what does seismic say?* **Fusion** is the answer. It resamples
> every model onto **one shared 3-D voxel grid** so that a single cell holds a value from
> every method at once — turning a pile of incompatible datasets into a co-registered stack
> you can cross-plot, cluster, and reason about cell-by-cell. This is the heart of the whole
> platform. If you've ever joined several time-series onto a common timestamp axis, or
> resampled audio to a common sample rate before mixing, you already know the shape of this
> problem; fusion is that, in 3-D, with strict rules about *never inventing data*.

A quick orientation. Three things flow into fusion and three things flow out:

```
PropertyModels (native grids/meshes, Engineering coords, doc 02)
        │  ① RESAMPLE onto one canonical FusedEarthModel voxel grid (non-destructive)
        ▼
  Fused property stack  ──②──►  CROSS-PLOT / STATS / CLUSTER  ──►  analysis panels
        │
        │  ③ ROCK-PHYSICS TRANSFORMS (resistivity→temperature, …) — see rock-physics.md
        ▼
  Derived volumes (temperature, porosity, favorability, …)
        │  ④ UNCERTAINTY propagated end-to-end → confidence volumes
```

This page covers ① the fused grid and resampling, and ② cross-plotting / statistics /
clustering. The rock-physics transforms (③) and favorability are their own subject — see
[rock physics & favorability](rock-physics.md) — and the honesty machinery (④) is detailed
in [uncertainty & scientific honesty](uncertainty.md). Everything here operates purely in
the [Engineering Frame](spatial-framework.md) (metres, Z-up); no coordinate math happens at
this layer — the data arrives already reconciled.

---

## 1. The Fused Earth Model — one grid to compare them all

A **Property Model** (see the [data model](data-model.md)) is a continuous 3-D field of one
physical property — a resistivity cube, a velocity cube, a density field. Each arrives on its
*own* grid: the seismic cube might be 10 m cells, the magnetotelluric (MT) resistivity model
500 m cells, the gravity-derived density field something else again. You cannot subtract two
arrays that don't share a shape and an origin.

The **`FusedEarthModel`** solves this. It is a single **regular voxel grid** — think of it as
a dense 3-D array, a `float32[nz][ny][nx]`, evenly spaced in Engineering metres — that acts
as a common **container**. Crucially, the fused grid carries **no property of its own**: it is
an empty coordinate scaffold into which each native model is *resampled* as a referenced
**layer**. (In CS terms: the fused grid is the index; the layers are columns joined onto that
index.)

!!! note "A voxel is a 3-D pixel"
    A **voxel** (volume element) is the 3-D analogue of a pixel: a little box of space that
    holds one value. The fused grid is a regular lattice of voxels, addressed by integer
    indices `(z, y, x)`, with a known `origin` and `spacing` in metres so each index maps to
    a real location. Axis order throughout is **`[z, y, x]`, Z-up** — depth-first, matching
    the storage layout.

A project may hold **several** fused models — e.g. a coarse overview of the whole
region plus a fine one zoomed on a target zone. They are not forced to be singular.

### 1.1 Choosing the resolution (and why not "as fine as possible")

What spacing should the fused grid use? The naive answer — "the finest of any input, so we
lose no detail" — is exactly **wrong**, and understanding why is central to honest fusion.

!!! warning "Resampling a smooth model onto a fine grid fabricates resolution it never had"
    If you upsample the 500 m MT model onto a 10 m grid, you get a 10 m array of numbers —
    but those numbers are pure interpolation between far-apart real samples. The grid *looks*
    detailed; the data is not. That is fabricated resolution: a lie the eye believes. (CS
    analogy: upscaling a 64×64 JPEG to 4K does not recover detail; it invents smooth
    nonsense between the real pixels.)

So the fused grid is deliberately a **comparison / compositing grid, not a super-resolution
grid**. The default heuristic (`backend/geosim/fusion/grid.py`, `auto_resolution`):

| Rule | Value |
|---|---|
| Default spacing $dx=dy=dz$ | **median** of native cell sizes across the loaded models |
| Clamp | to $[\,\text{extent}/512,\ \text{extent}/64\,]$ |
| Hard cap | $n_x \cdot n_y \cdot n_z \le 256^3 \approx 16.7\text{M}$ cells; coarsen until it fits |
| Anisotropy ($dz<dx$) | allowed when data is mostly layered/depth-resolved; off by default |

The **median** native spacing is fine enough not to throw away the sharpest method, coarse
enough to keep the whole volume streamable, and honest about the fact that the smooth methods
were never that detailed. The cap keeps a single level-0 brick a sane size; finer target
grids are an explicit opt-in. Uncertainty (below and in [uncertainty](uncertainty.md)) is
what keeps the remaining interpolation honest.

```python
# auto_resolution(): median native spacing, clamped, then coarsened to honour the cell cap
base    = float(np.median(spacings)) if spacings else max_extent / 256.0
spacing = float(np.clip(base, max_extent/512, max_extent/64))
while spacing > 0.0 and n_cells(spacing) > cell_cap:   # 256³ default
    spacing *= 2.0
```

---

## 2. Resampling native models onto the fused grid

**Resampling** is the act of reading a native model and writing its values onto the fused
voxel centres. The goal is a *shared support* — every property sampled at the same cells, so
cell `(z,y,x)` of the resistivity layer and cell `(z,y,x)` of the density layer describe the
**same cubic metre of rock**.

### 2.1 The non-destructive principle

!!! abstract "Originals are never touched"
    The native Property Model is opened **read-only** and is *never modified*. Resampling
    produces a **new** array — a `FusedLayer` written into the fused model's storage —
    referencing the source by `sourcePropertyModelId@sourceVersion`. Resampling is a pure
    function of `(native model + fused grid + method)`, so it is cacheable, re-runnable, and
    versioned. If a layer's cache key already exists, the existing layer is returned instead
    of recomputing.

The cache key is `(propertyModelId, version, fusedGridId, method, params)`. This is the
materialized-view pattern again: the fused layer is a cached derivation, invalidated when its
source version changes, and it carries provenance back to that source.

### 2.2 Picking the interpolation method

The native model's **support geometry** drives the method. The codebase resolves
`method="auto"` from the support kind and the relative resolution
(`backend/geosim/fusion/resample.py`, `resolve_method` / `_do_resample`):

| Native support | Default method | What it does |
|---|---|---|
| regular grid, **coarser** than fused | **trilinear** | smooth 3-D linear interpolation; no sharpening |
| regular grid, **finer** than fused | **block-mean** then trilinear | volume-weighted averaging first → anti-aliasing |
| unstructured **mesh** | **barycentric** (linear) | linear interpolation inside the containing cell |
| scattered **points** | **spline** gridding then trilinear | grid the scatter, then sample (a gridding step) |
| **categorical** (e.g. a lithology class id) | **nearest** only | never average class labels — `2.5` is not a rock type |

**Trilinear** interpolation is the 3-D version of linear interpolation: a query point's value
is the distance-weighted blend of the 8 surrounding voxel corners. **Block-mean** is the
downsampling counterpart — when the native model is *finer* than the fused grid, naive
point-sampling would alias (miss detail between samples, like undersampling a high-frequency
signal), so each fused cell is set to the volume-weighted average of the native cells it
covers. This is exactly **anti-aliasing**, and it preserves the field's mean.

### 2.3 Log space for orders-of-magnitude properties

Some properties span many orders of magnitude. **Resistivity** (how strongly rock resists
electrical current) ranges from ~1 Ω·m (hot, conductive brine) to ~10,000 Ω·m (dry crystalline
rock) — four decades. Linearly interpolating between 1 and 10,000 gives 5,000 at the midpoint,
which is geologically meaningless; the *geometric* midpoint (~100) is what's physical. So
resistivity, conductivity, and permeability interpolate in **$\log_{10}$ space** by default;
density, velocity, and temperature interpolate linearly. This flag lives on the **property
type registry** (see [coordinates, depth & units](spatial-framework.md)), not hard-coded in
the resampler:

```python
def _to_interp_space(arr, space):       # interpolate in log10, then map back
    if space == "log10":
        out = np.full_like(arr, np.nan, dtype=float)
        pos = np.isfinite(arr) & (arr > 0.0)
        out[pos] = np.log10(arr[pos])
        return out
    return arr.astype(float)
```

### 2.4 Footprint honesty — the most important rule on this page

A method only measured *some* of the ground, down to *some* depth. Outside that region it
knows **nothing** — and "nothing" must read as **nothing**, never as a number.

!!! warning "NaN outside coverage — never zero, never edge-bleed, never extrapolate"
    Every Property Model has a **footprint** (its coverage region — bbox / convex hull of
    valid cells) and often a **depth-of-investigation (DOI)** floor below which it is blind.
    Resampling fills fused cells **only inside** the footprint; everywhere else the value is
    **`NaN`** (not-a-number). No extrapolation past the native edge. No zero-fill. No
    nearest-edge bleed.

Why does a CS person care so much about this? Because the alternative is **fabricating data**,
and downstream everything trusts these arrays. Two specific traps:

- **Zero-fill is a silent lie.** If "no measurement" were stored as `0.0`, then a
  cross-plot, a correlation, or a cluster would treat that 0 as a *real low value*. A
  density of 0 g/cm³ is physically absurd, but the math doesn't know that — it would happily
  pull a cluster centroid toward it. `NaN` is the explicit "absent" sentinel that every
  downstream operation knows to skip.
- **Extrapolation is confident nonsense.** Bleeding the nearest edge value outward says "the
  anomaly continues past where we looked," which is precisely the unsupported claim you must
  not make.

In code, the interpolators are masked to the native bounds (`fill_value=np.nan`,
`bounds_error=False`), and `respect_footprint=True` (the default) blanks anything outside the
coverage mask:

```python
value, sigma, mask = _do_resample(reader, pm, grid, method, space)
if respect_footprint:
    value = np.where(mask, value, np.nan)   # outside footprint → NaN
    sigma = np.where(mask, sigma, np.nan)
coverage = mask.astype(np.float32)          # a per-layer boolean coverage volume
```

Every resampled layer therefore ships **three** co-registered arrays: the value, its
propagated $\sigma$ (1-sigma uncertainty), and a **coverage mask**. The mask lets the viewer
shade "no data here" *distinctly* from "low value here" — a distinction that is invisible if
you zero-fill.

### 2.5 The fused stack: co-registered, each with its own holes

The result of resampling N models is **N co-registered volumes on one grid, each with its own
NaN pattern**. They overlap where the surveys overlap and disagree-by-absence elsewhere. Any
later operation (cross-plot, transform, cluster) acts **only where the inputs it needs are all
present** — the holes propagate honestly through everything.

---

## 3. Cross-plotting, statistics & clustering

Once N properties share the grid, **every fused cell becomes a feature vector**
`[resistivity, density, Vp, …]` (with NaN where a method is absent). This is the unlock: a 3-D
geological problem becomes a tabular multivariate-statistics problem, one row per cell. From
here it is ordinary data science.

### 3.1 Co-located sampling

`sample_fused` (`backend/geosim/fusion/analysis.py`) stacks the chosen layers and extracts a
feature matrix. By default it uses **listwise deletion** — keep a cell only if *all* selected
layers are non-NaN there (`mode="all"`), which is what joint analysis needs; an `"any"` mode
keeps any-present cells for single-property histograms. An optional bounding box restricts the
working set to "just the anomaly."

```python
stack  = np.stack([read(layer) for layer in layers], axis=-1)  # (z, y, x, p)
finite = np.isfinite(stack)
keep   = finite.all(axis=-1) if mode == "all" else finite.any(axis=-1)   # honest joins skip NaN
feats  = stack.reshape(-1, p)[np.flatnonzero(keep.reshape(-1))]          # (n_cells, p)
```

A special, important case is **volume × well log**. Well logs are *not* resampled to voxels
(they are sparse, precious ground-truth probes); instead the fused volumes are sampled
**along the well path** with `sample_path`, giving co-located `(measured_log, model_value)`
pairs at each point down the borehole. That paired view is the foundation of **calibration**
(turning a model into something anchored to measured truth — see [rock physics](rock-physics.md)
and [well planning](well-planning.md)).

### 3.2 Cross-plots, histograms, correlation, and linked brushing

With the feature matrix in hand:

| Panel | What it shows | Returns |
|---|---|---|
| **2-D cross-plot** | property A vs B (e.g. $\log\rho$ vs density), points coloured by depth/class | point set, or a 2-D density grid for big N |
| **Histogram / KDE** | one property's distribution over the volume or a selection | bin edges + counts (+ optional KDE) |
| **Correlation matrix** | how every loaded property co-varies with every other | a matrix → heatmap |
| **Profile along well** | the calibration scatter, per depth | paired arrays |

The 2-D cross-plot has a nice scalability touch: below `BIG_N_SCATTER` (50,000) points it
returns a raw scatter; above it, returning hundreds of thousands of points would choke the
browser, so it returns a 2-D **density grid** (a `histogram2d`, the hexbin equivalent)
instead.

The killer feature is **linked brushing**: you lasso a cluster of interesting points *in the
cross-plot*, and the platform turns that selection into a **cell-index mask** that lights up
the corresponding cells *in 3-D* (and vice versa). Because every sampled cell remembers its
flattened grid index, the round-trip is exact:

```python
def selection_to_mask(sample, selected_local_indices):
    flat = np.zeros(np.prod(sample.grid_shape), dtype=bool)
    flat[sample.cell_index[selected_local_indices]] = True   # plot selection → 3-D cells
    return flat.reshape(sample.grid_shape)                    # itself storable as a volume
```

This is what makes the anomaly *findable*: you spot an outlier population in the abstract
property space and immediately see *where in the Earth* it sits.

### 3.3 Clustering → rock classes

Cross-plots are for eyes; **clustering** is for turning the property stack into labelled
**rock classes** automatically. The pipeline (`cluster_fused`):

```
1. assemble the feature matrix at all-present cells (§3.1)
2. log-transform log10-flagged properties (registry), then standardize (z-score each column)
3. fit k-means (hard classes) OR a Gaussian Mixture Model (soft probabilities)
4. predict a label (k-means) or posteriors (GMM) for every valid cell
5. write back: a categorical "lithology_class" PropertyModel,
               + for GMM, one probability volume per class
6. return centroids, sizes, and cross-plot ellipses
```

Two algorithms, two characters:

- **k-means** partitions cells into $k$ hard groups by minimizing within-cluster variance —
  fast, great for "how many rock types are there?" exploration.
- **Gaussian Mixture Model (GMM)** models the data as a blend of $k$ Gaussian blobs and gives
  each cell a *soft* per-class **probability**, which is better when populations overlap. Those
  posteriors double as a soft alteration/lithology *likelihood* that favorability can consume.

The standardization in step 2 matters: clustering uses distances, so a property measured in
huge numbers (resistivity in the thousands) would otherwise dominate one in small numbers
(porosity, 0–1). Z-scoring puts every property on a comparable scale; log-transforming the
order-of-magnitude properties first puts them in the space they're physically compared in.

```python
def _standardize(feats, properties):
    logged = _log_transform_features(feats, properties)  # log10 the flagged columns
    scaler = StandardScaler()                             # z-score every column
    return scaler.fit_transform(logged), scaler
```

!!! tip "The platform clusters; the geologist names"
    Clustering finds statistically distinct groups — it does **not** know that cluster 2 is
    "propylitic alteration" or "fresh basement." Those interpretation labels are
    user-assigned to cluster ids. The output **class volume** is an ordinary derived
    Property Model, rendered with a categorical colourmap, fully toggleable in the
    [3-D viewer](visualization.md).

### How clustering reveals the anomaly

This is the payoff of the whole pipeline. A geothermal target is rock that is, *all at once*,
hot, fluid-bearing, and altered — which shows up as a distinctive **combination** of low
resistivity, low density, and an anomalous velocity. No single property singles it out, but in
the multivariate feature space those cells form their **own cluster**, well-separated from the
boring host rock. You cross-plot $\log\rho$ vs density, see a tight low-low population, GMM it
into its own class, light it up in 3-D via linked brushing — and there is your anomaly,
discovered by statistics over co-located cells that fusion made comparable in the first place.

---

## 4. Where this leaves you (and what comes next)

After fusion you have: one or more fused voxel grids; each native property resampled onto them
as a non-destructive, footprint-honest layer carrying value + $\sigma$ + coverage; the ability
to cross-plot, correlate, and cluster every cell as a feature vector; and a class volume
naming the rock types. What you do *not* yet have is the leap from geophysics to
geothermal-engineering quantities (temperature, porosity, permeability, favorability) — that is
the **rock-physics transform engine**, covered in [rock physics & favorability](rock-physics.md).
And the $\sigma$ / coverage / resolution machinery that keeps every step honest is detailed in
[uncertainty & scientific honesty](uncertainty.md).

---

## Key takeaways

- **One shared voxel grid.** The `FusedEarthModel` is an empty regular-voxel container; native
  Property Models are resampled *into* it as referenced layers so every cell is co-located and
  comparable.
- **Resolution is chosen, not maximized.** Default spacing is the **median** native cell size,
  clamped and capped — fine enough to keep the sharpest method, coarse enough to stay honest
  and streamable. Upsampling everything to the finest grid would fabricate resolution.
- **Non-destructive.** Originals are read-only; each resampled layer is a new, cached,
  versioned, provenance-linked array.
- **Footprint honesty is non-negotiable.** Outside a method's coverage / DOI the value is
  **NaN** — never zero, never edge-bleed, never extrapolated. Each layer ships a coverage mask
  so "no data" is visibly distinct from "low value."
- **Method by support, log-space where it matters.** Trilinear (coarse→fine), block-mean
  (fine→coarse, anti-aliased), barycentric (mesh), spline (scattered), nearest (categorical);
  resistivity/conductivity/permeability interpolate in $\log_{10}$.
- **Every cell is a feature vector.** Co-located sampling + cross-plots + correlation +
  k-means/GMM clustering turn the 3-D problem into multivariate statistics; **linked brushing**
  ties the abstract property space back to 3-D cells, which is how the anomaly is found.

## Where this lives in the code

| Concern | Module |
|---|---|
| Fused-grid container + auto-resolution heuristic | `backend/geosim/fusion/grid.py` |
| Resampling, log-space, footprint masking, σ inflation | `backend/geosim/fusion/resample.py` |
| Co-located sampling, cross-plot, histogram, correlation, clustering, linked brushing | `backend/geosim/fusion/analysis.py` |
| Rock-physics transforms + favorability (next page) | `backend/geosim/fusion/transform.py`, `rockphys/`, `favorability.py` |
| The fused-model catalog rows (container + layers) | `backend/geosim/catalog/` (`FusedModel`, `FusedLayer`) |
