# 07 — Fusion & Rock-Physics Engine

> Parent: `OVERVIEW.md` §6 (levels 1–3) + §10 row 7. This doc defines the **fusion
> engine**: resampling native property models onto the canonical Fused Earth Model
> grid, cross-plotting/statistics, the **rock-physics transform engine** that converts
> geophysical properties toward geothermal targets, **derived volumes** (e.g.
> "geothermal favorability"), and **uncertainty propagation** through all of it.
>
> **Out of scope (here):** joint/cooperative *inversion* — structural coupling,
> cross-gradient, shared-mesh inverse problems — that is doc 10. This doc only
> *combines and transforms already-derived property models*; it never solves an
> inverse problem.
>
> **Binds to:** spatial/units conventions from doc 01 (LOCKED). **References (does not
> redefine):** `PropertyModel` / `FusedGrid` / uncertainty schemas from doc 02;
> storage & serving of derived volumes from doc 04; the transform-plugin registration
> mechanism from doc 08. Where those schemas are needed, names from OVERVIEW §2/§6 are
> used as the contract and assumptions are flagged explicitly.

> ### ⚠️ Revision — user decisions applied (see `DECISIONS.md`)
> - **Starter rock-physics library = the FULL table** (not the minimal
>   resistivity→temp + velocity→porosity set): resistivity→temperature/fluid
>   (Archie + Arps), velocity→porosity, clay/alteration index, microseismic→fracture
>   density, **Waxman-Smits / dual-water**, and **permeability proxies**. This expands
>   Phase-3 scope and requires the synthetic earth (doc 05) to emit the supporting
>   property/state fields these transforms consume.
> - **Favorability combination = ship weighted-linear AND fuzzy-logic; defer Bayesian**
>   until known-occurrence training data exists. (Drafted default was weighted-linear
>   only by default.)
> - **Uncertainty = delta-method everywhere, Monte-Carlo opt-in** per nonlinear
>   transform — confirmed.

---

## 0. Where this sits in the stack

This is the **PROCESSING / COMPUTE** layer of OVERVIEW §4 ("resample→fused grid · rock-physics transforms · derived-property engine"). It consumes `PropertyModel`s (doc 02) from storage (doc 04), produces new `PropertyModel`s (derived volumes) that are **stored and served identically to ingested ones**, and returns statistics/cross-plot payloads to the viewer's analysis panels (OVERVIEW §5, Observable Plot / D3).

Everything operates in the **Engineering Frame** (doc 01 §1): ENU, metres, Z-up, floating origin. No CRS math happens here — coordinates are already Engineering by the time a `PropertyModel` reaches storage. All quantities carry canonical units (doc 01 §5).

```
PropertyModels (native grids/meshes, Engineering coords, doc 02)
        │  ① RESAMPLE onto canonical FusedGrid (non-destructive)
        ▼
  Fused property stack  ──②──►  CROSS-PLOT / STATS / CLUSTER  ──►  analysis panels
        │                                                          (D3/Observable Plot)
        │  ③ ROCK-PHYSICS TRANSFORMS (registry: declarative + Python fn)
        ▼
  Derived volumes (temperature-likelihood, porosity, alteration, favorability …)
        │  stored & served as PropertyModels (doc 04) → renderable layers (doc 06)
        ▼
  ④ UNCERTAINTY propagated end-to-end → confidence volumes + low-sensitivity flags
```

The four numbered stages map to OVERVIEW §6 fusion levels **1→2→3 + uncertainty**.

---

## 1. The Fused Earth Model grid (consumer of doc 02)

The canonical fused grid is **defined in doc 02** (`FusedGrid`); this doc is its primary *consumer*. We restate only the contract we depend on:

```jsonc
// Assumed shape from OVERVIEW §2 / doc 02 — names are the contract, not a redefinition.
FusedGrid {
  "id": "fused-default",
  "kind": "regular",                 // MVP: regular voxel grid. (octree/unstructured = later)
  "origin":  [x0, y0, z0],           // Engineering metres (within SpatialFrame.roi/depthRange)
  "spacing": [dx, dy, dz],           // metres; default isotropic (see §1.1)
  "shape":   [nx, ny, nz],
  "crs": "engineering"               // always; doc 01 guarantees this
}
```

**Reconciled with doc 02 (§11 `FusedEarthModel`) — now confirmed, not assumed:**
- **A1 ✓ confirmed.** The fused grid **is a regular voxel grid** (doc 02 §11 default `gridType:"regular_voxel"`). Octree/unstructured fusion is a later extension delivered via the LOD pyramid, not an irregular topology; the resampling API (§2.4) doesn't assume regularity but only the regular path ships first.
- **A2 ✓ confirmed (naming aligned).** The canonical object is doc 02's **`FusedEarthModel`** (id prefix `fem_`), bounded by `SpatialFrame.roi` × `depthRange`. A project **may hold several** (coarse overview + zoomed target-zone), each its own `fusedModel` Dataset — *not* forced-singular. The `FusedGrid` sketch above is illustrative; the authoritative shape is `FusedEarthModel.support` (a doc 02 `VolumeSupport`), so use doc 02's field names and **axis order `shape:[nz,ny,nx]`, origin/spacing `[…z,y,x]`** (z-leading, Z-up) — not the `[nx,ny,nz]` ordering sketched here.
- **A3 ✓ confirmed.** A derived/fused volume **is a `PropertyModel`** (doc 02 §4) on the fused-grid `VolumeSupport`, carrying provenance to its inputs + transform (§4.3). Each native property enters as a `FusedLayer` referencing `sourcePropertyModelId@sourceVersion` (doc 02 §11) — originals stay read-only.

### 1.1 Choosing fused-grid resolution

The fused grid is a **comparison and compositing** grid, not a super-resolution grid. Default heuristic (overridable per project):

| Rule | Value |
|---|---|
| Default spacing | `dx=dy=dz = ` median of native cell sizes across loaded property models, clamped to `[roi_extent/512, roi_extent/64]` |
| Rationale | fine enough to not throw away the sharpest method, coarse enough to keep a single volume streamable (doc 04/06 LOD handles the rest) |
| Hard cap | `nx·ny·nz ≤ 256³` for the default level-0 brick (≈ 16.7 M cells); finer target grids are opt-in |
| Anisotropy | allowed (`dz < dx`) when most data is layered/depth-resolved; off by default |

We deliberately **do not** resample everything to the finest native grid — that fabricates resolution the smooth methods (gravity, MT) never had, and uncertainty (§5) is what keeps that honest.

---

## 2. Resampling native models onto the fused grid

**Goal (OVERVIEW §6.1/§2):** put every property on a *shared support* so cells are co-located and comparable — **without destroying native originals**. Native models stay in storage untouched; resampling produces a **new** array bound to a `FusedGrid`.

### 2.1 Non-destructive principle

| | |
|---|---|
| Native `PropertyModel` | immutable input; never modified or overwritten |
| Resampled output | a **new** derived `PropertyModel` (`derivation: "resample"`), or a cached fused-stack layer |
| Re-derivable | resampling is a pure function of (native model + FusedGrid + method); cacheable, re-runnable, versioned (§4.4) |
| Storage | written as a Zarr array via doc 04, same as any property |

Resampled layers are **cached** keyed by `(propertyModelId, propertyModelVersion, fusedGridId, method, params)`. Cache invalidates if the native model version changes.

### 2.2 Interpolation method by support type

The native support geometry (declared by doc 02 on the `PropertyModel`) drives the method. Compute is `xarray` + `scipy`/`verde`; resampling never invents data outside the native footprint (§2.3).

| Native support | Default resampling onto fused voxels | Library | Notes |
|---|---|---|---|
| **Regular grid → regular grid** | trilinear interpolation | `scipy.ndimage.map_coordinates` / `xarray.interp` | fast path; order-1 default, nearest for categorical |
| **Regular grid (coarser) → finer fused** | trilinear (smooth, *no* sharpening) | `xarray.interp` | uncertainty inflates (§5.3) — we never fake detail |
| **Regular grid (finer) → coarser fused** | block average (area/volume-weighted) | `xarray.coarsen` then align | anti-aliasing; preserves the mean |
| **Unstructured mesh / octree** | barycentric (linear) interpolation per fused-cell centroid | `scipy` `LinearNDInterpolator` / `discretize` cell-lookup | tetrahedral/cell containment; nearest fallback at boundary |
| **Scattered points (gravity stations, geochem)** | grid via continuous-curvature / spline, *then* trilinear | `verde` (`Spline`/`KNeighbors`) | this is a *gridding* step; gridding uncertainty tracked (§5) |
| **1D well log along path** | not volume-resampled; sampled *as* a 1D track for cross-plot (§3.1) | — | logs are ground-truth probes, not volumes; see §3.1 |
| **Categorical (lithology id)** | nearest-neighbour only | `xarray.interp(method="nearest")` | never average class labels |

**Default per-property method** comes from the **property type registry** (doc 01 §5 / doc 08): each property declares whether it interpolates **linearly in native space or in log space**. Resistivity, conductivity, and permeability span orders of magnitude → interpolate in **log10** by default; density, velocity, temperature → linear. This flag lives with the property type, not hard-coded here.

### 2.3 Footprint, coverage, and nodata

This is the crux of honest fusion: **a method that didn't sample a region must read as nodata there, not zero, not extrapolated.**

- Each `PropertyModel` carries (or we derive) a **coverage mask / footprint** — the region of valid support (doc 02 should store this; if absent we compute the convex-or-alpha hull of native cells + a depth-of-investigation cap).
- Resampling fills fused cells **only inside the footprint**; outside → **`NaN` (nodata)**.
- **No extrapolation beyond footprint, ever.** Interpolators are masked to the hull; values past the native edge are nodata, not nearest-edge bleed.
- **Depth of investigation (DOI):** smooth methods (MT, gravity) have a DOI floor below which the model is unconstrained. If doc 02/03 attaches a DOI surface, cells below it become nodata (and feed the low-sensitivity flag, §5.4).
- The fused stack is therefore a set of **co-registered volumes each with its own nodata pattern**; cross-plots and transforms operate only where the *required* inputs are all present (§3, §4.5).

A per-layer **coverage volume** (boolean/float fraction) is emitted alongside each resampled property so the viewer can shade "no data here" distinctly from "low value here" (doc 06 transfer functions).

### 2.4 Resampling API (backend)

```python
# compute layer — pure-ish, cached. Returns a derived PropertyModel handle (doc 02/04).
resample_to_fused(
    property_model_id: str,
    fused_grid_id: str = "fused-default",
    method: Literal["auto","trilinear","block_mean","barycentric","spline","nearest"] = "auto",
    interp_space: Literal["auto","linear","log10"] = "auto",   # auto → property registry
    respect_footprint: bool = True,        # False is a dev-only escape hatch
    cache: bool = True,
) -> ResampledLayerRef       # {modelId, fusedGridId, coverageMaskRef, sigmaRef}
```

`method="auto"` and `interp_space="auto"` resolve via the table above + the property registry. The returned ref also points at the **propagated σ volume** (§5.3) and **coverage mask** (§2.3).

---

## 3. Cross-plotting, statistics & clustering (OVERVIEW §6.2)

Once N properties share the fused grid, every fused cell is a **feature vector** `[resistivity, density, Vp, …]` (with nodata where a method is absent). This unlocks multivariate analysis.

### 3.1 Co-located sampling

- **Volume × volume:** sample any subset of fused layers at the same cells. Only cells where **all selected layers are non-nodata** enter the joint sample (listwise deletion by default; "any-present" mode optional for histograms).
- **Volume × well log:** well logs are *not* resampled to voxels (§2.2). Instead the fused volumes are **sampled along the well path** (trilinear at MD→Engineering-XYZ points via doc 01 `md_to_tvd` + frame transform), producing co-located `(log_value, volume_value)` pairs — the key **calibration** view (e.g. measured temperature log vs resistivity-derived temperature).
- **Region of interest:** sampling can be restricted to a clipping box / polygon / depth slab / inside-an-isosurface selection coming from the viewer (doc 06), so users cross-plot "just the anomaly."

### 3.2 Cross-plots, histograms, density

| Panel | What | Compute | Returns to viewer |
|---|---|---|---|
| **2D cross-plot** | property A vs B, e.g. log(ρ) vs density | `numpy`; optional 2D hist / hexbin for big N; per-point color by depth, class, or a 3rd property | downsampled point set or 2D density grid (JSON) → D3/Observable Plot |
| **3D cross-plot** | A vs B vs C | `numpy` | point set → Three.js mini-scene or rotatable plot |
| **Histogram / KDE** | one property's distribution (whole vol or selection) | `numpy.histogram` / `scipy` KDE | bin edges + counts |
| **Correlation / covariance** | cross-correlation matrix across loaded properties | `numpy.corrcoef` | matrix → heatmap panel |
| **Profile/scatter along well** | calibration scatter + per-depth | as §3.1 | paired arrays |

**Linked brushing (key R&D feature):** a selection (lasso/threshold) made in a cross-plot returns a **cell-index mask**, which the viewer can light up in 3D and vice-versa. The mask is computed backend-side and is itself storable as a (categorical) derived volume.

### 3.3 Clustering / classification → lithology / alteration classes

Multivariate clustering turns the property stack into interpretable **classes** (OVERVIEW §6.2: "cluster").

| Algorithm | Use | Library |
|---|---|---|
| **k-means** | fast, hard classes; "how many rock types" exploration | `scikit-learn` |
| **Gaussian Mixture (GMM)** | soft/probabilistic membership → per-class probability volumes; better for overlapping populations | `scikit-learn` |
| **Agglomerative / DBSCAN** | optional; structure discovery, outliers | `scikit-learn` |

Pipeline (backend, `scikit-learn`):

```
1. assemble feature matrix from selected fused layers at valid (all-present) cells
2. per-feature scaling: standardize; log-transform flagged props first (registry)
3. fit (k or n_components user-set; optional BIC/silhouette sweep returned as a hint)
4. predict labels (k-means) or posteriors (GMM) for every valid cell
5. write back:
     - categorical "class" volume  (PropertyModel, kind=categorical)
     - per-class probability volumes (GMM)  → feed uncertainty (§5)
6. return cluster centroids + sizes + cross-plot ellipses to the panel
```

The class volume is a derived `PropertyModel` (stored/served via doc 04) and renders with a categorical colourmap (doc 06). **Interpretation labels** ("propylitic alteration", "fresh basement") are user-assigned to cluster ids — the platform clusters; the geologist names. GMM posteriors double as a soft alteration/lithology likelihood that the favorability index (§4.6) can consume.

### 3.4 Compute placement (cross-plot/stats)

| Operation | Mode | Why |
|---|---|---|
| Sample / cross-plot / histogram on a selection (≲ a few M cells) | **synchronous** REST | sub-second on numpy; interactive |
| Whole-volume clustering, full-grid GMM, big sweeps | **job-based** (background task → WebSocket progress) | seconds–minutes; matches doc 04 job API |

Threshold for sync vs job is **cell count of the working set** (default 5 M cells), not operation type.

---

## 4. Rock-physics transform engine (OVERVIEW §6.3 — the core of this doc)

A **transform** maps one or more input property volumes → an **output derived volume**, applying a rock-physics relationship that moves from *what geophysics measures* toward *what geothermal cares about*: **temperature, fluid/permeability, alteration, fracture density**.

### 4.1 What a transform is

A transform = **declarative spec** (metadata, I/O contract, parameters, units) **+ a pure Python function** (the math). The spec makes it discoverable, parameterizable, versionable, and UI-drivable; the function does the physics. This split mirrors the plugin pattern of doc 08 (which owns the *registration* mechanism — see §4.7).

```python
@register_transform   # registry hook — mechanism defined in doc 08
class ResistivityToTemperature(Transform):
    id      = "rp.resistivity_to_temperature.arps"
    version = "1.2.0"
    title   = "Resistivity → Temperature (Arps fluid-conductivity)"
    target  = "temperature"            # geothermal target taxonomy (§4.2)

    inputs  = [InputSpec("resistivity", unit="ohm.m", required=True)]
    output  = OutputSpec("temperature", unit="degC",
                         valid_range=(0, 400), colormap="thermal")

    params  = [
        Param("porosity",      float, default=0.10, range=(0.01, 0.5)),
        Param("m_cementation", float, default=2.0,  range=(1.3, 2.5)),
        Param("fluid_salinity_ppm", float, default=5000, range=(100, 250000)),
        Param("T_ref_degC",    float, default=25.0),
    ]

    def apply(self, ctx, resistivity, *, porosity, m_cementation,
              fluid_salinity_ppm, T_ref_degC):
        # 1) Archie: bulk ρ + porosity → pore-fluid conductivity σ_w
        sigma_bulk = 1.0 / resistivity
        sigma_w = sigma_bulk / (porosity ** m_cementation)      # a=1
        # 2) fluid conductivity ↑ ~2%/°C (Arps) at fixed salinity → invert for T
        sigma_w_ref = brine_conductivity(fluid_salinity_ppm, T_ref_degC)
        temperature = T_ref_degC + (sigma_w / sigma_w_ref - 1.0) / 0.02
        return ctx.as_output(temperature)     # carries units, masks nodata, σ (§5)
```

### 4.2 Target taxonomy & the starter transform library

Transforms declare a **geothermal `target`** so the UI can group them and favorability (§4.6) can pull "all evidence for target X." Starter library (each is a registered transform; physics references in comments):

| Target | Transform(s) | Inputs → output | Relationship |
|---|---|---|---|
| **Temperature** | `resistivity_to_temperature` | ρ (+φ, salinity) → T-likelihood / °C | Archie + Arps fluid-conductivity vs T |
| **Fluid / saturation** | `archie_saturation` | ρ, φ → water saturation Sw | Archie's law (`Sw = ((a·ρ_w)/(φ^m·ρ_t))^{1/n}`) |
| **Fluid / clay-conduction** | `dual_water` / `waxman_smits` (opt) | ρ, φ, clay → Sw with surface conduction | corrects Archie in clay/altered rock |
| **Porosity** | `velocity_to_porosity` | Vp → φ | Wyllie time-average / Raymer-Hunt-Gardner |
| **Porosity (alt)** | `density_to_porosity` | ρ_b → φ | `φ = (ρ_matrix − ρ_b)/(ρ_matrix − ρ_fluid)` |
| **Alteration** | `alteration_index` | low-ρ ∧ structure proxies | clay/conductive alteration cap (smectite ⇒ low ρ) — common geothermal indicator |
| **Alteration (data-driven)** | GMM class posteriors (§3.3) | property stack → class prob | clustering-as-transform wrapper |
| **Fracture density** | `microseismic_density` | event cloud → smoothed density vol | KDE of microseismic events → permeability proxy |
| **Fracture density (struct.)** | `vp_vs_fracture_proxy` (opt) | Vp/Vs, attenuation → fracture index | Vp/Vs anomalies / low-velocity zones |
| **Permeability** | `fracture_to_permeability` (proxy) | fracture density (+ alteration) → relative perm index | heuristic; flagged low-confidence |

> **Physics honesty note.** These relations are **site-calibratable approximations**, not universal truths. Every transform's params (porosity, cementation exponent, salinity, matrix density, fluid velocity) are **first-class and user-tunable**, and where well logs exist they are the **calibration anchor** (§3.1). The platform's R&D value is making it trivial to swap relationships and re-parameterize, then *see* the effect in 3D against ground truth.

### 4.3 Output = a `PropertyModel`, stored & served identically (doc 04)

A transform output is **indistinguishable downstream from an ingested property model** — same `PropertyModel` schema (doc 02), same Zarr storage + tiling/streaming (doc 04), same renderer path (doc 06). The only difference is **provenance**:

```jsonc
// derived PropertyModel provenance block (doc 02 owns full schema; this is the contract)
"derivation": {
  "kind": "transform",                         // vs "ingest" | "resample" | "cluster" | "favorability"
  "transformId": "rp.resistivity_to_temperature.arps",
  "transformVersion": "1.2.0",
  "fusedGridId": "fused-default",
  "inputs": [ {"propertyModelId": "...", "version": "..."} ],
  "params": { "porosity": 0.10, "m_cementation": 2.0, "fluid_salinity_ppm": 5000, "T_ref_degC": 25.0 },
  "createdAt": "...", "createdBy": "tim@…",
  "sigmaRef": "<zarr path to confidence volume>"   // §5
}
```

This makes derived volumes **layers like any other** (toggle, slice, isosurface, cross-plot, even feed *another* transform → chaining). Re-running a transform with different params yields a **new versioned derived model**, never an in-place mutation (§4.4).

### 4.4 Versioning

| Thing versioned | How |
|---|---|
| **Transform code/spec** | semver `version` on the class; bump on math/param change. Old versions resolvable so old derived volumes stay reproducible. |
| **Derived volume instance** | immutable; identified by `(transformId, transformVersion, inputs+versions, params, fusedGridId)`. New params → new instance. |
| **Reproducibility** | the provenance block (§4.3) is a complete recipe — any derived volume can be regenerated from inputs + transform version + params. |

### 4.5 Execution semantics (nodata, units, broadcasting)

Every transform runs through a common harness so individual `apply()` functions stay pure math:

1. **Resolve inputs** to the fused grid (auto-resample via §2 if a named input isn't yet on `fusedGridId`).
2. **Unit-check & convert** each input to the transform's declared unit (`pint`, doc 01 §5). Wrong-dimension input → hard error.
3. **Build the valid mask** = AND of input coverage masks (§2.3). Cells missing any *required* input → **nodata** in output (no silent zero-fill).
4. **Vectorized apply** over valid cells (`numpy`/`xarray`, chunk-wise so it streams; large jobs are Dask-able later).
5. **Clamp/validate** to the output's `valid_range`; out-of-range flagged (often a sign of bad params) and recorded.
6. **Propagate σ** (§5.2) into the paired confidence volume.
7. **Write** output + σ + mask as a derived `PropertyModel` (doc 04).

### 4.6 The "geothermal favorability" derived volume (OVERVIEW §6.3)

Favorability is a **special transform** that combines *multiple evidence layers* into one **targetable index in `[0,1]`** — the thing a driller actually points at. It is the headline fusion product.

**Evidence layers** are any derived/native volumes tagged as favorable indicators, e.g.:
- high **temperature** (or T-likelihood),
- **fluid/saturation** present,
- **permeability / fracture density** elevated (need a *path* for fluid),
- **alteration** present (fossil/active hydrothermal signature),
- (optionally) structural proximity to a mapped fault (from features, doc 02).

The classic geothermal play needs **heat + fluid + permeability** co-located; favorability encodes that conjunction.

**Three pluggable combination methods** (user-selectable — this is the R&D knob):

| Method | Formula (per cell) | Character | When |
|---|---|---|---|
| **Weighted linear** *(default)* | `F = Σ wᵢ·eᵢ / Σ wᵢ`, each `eᵢ∈[0,1]` normalized | simple, transparent, fast; compensatory (a strong layer offsets a weak one) | first pass, explainable |
| **Fuzzy-logic** | fuzzy-AND (`min` / product) for *required* conjunctions, fuzzy-OR (`max`) for alternatives, via a small expression tree | encodes "heat **AND** fluid **AND** perm" non-compensatorily | when conjunction matters (recommended for the real geothermal model) |
| **Bayesian (weights-of-evidence)** | combine evidence as posterior odds; `logit(P) = logit(prior) + Σ Wᵢ⁺/⁻` | probabilistic, calibratable to known occurrences, well-founded uncertainty | when training points / known plays exist |

Spec sketch:

```jsonc
FavorabilitySpec {
  "method": "weighted" | "fuzzy" | "bayesian",
  "evidence": [
    { "source": "<derivedModelId|nativeModelId>",
      "target":  "temperature",
      "transferFn": { "type": "ramp", "lo": 150, "hi": 250 },  // map raw→[0,1] favorability
      "weight": 0.4,                       // weighted/bayesian
      "role":   "required" | "supporting"  // fuzzy AND/OR
    },
    { "source": "...", "target": "permeability", "transferFn": {...}, "weight": 0.3, "role": "required" },
    { "source": "...", "target": "fluid",        "transferFn": {...}, "weight": 0.3, "role": "supporting" }
  ],
  "missingPolicy": "nodata" | "neutral(0.5)" | "drop"   // how to treat cells missing an evidence layer
}
```

- **Each evidence layer is normalized to `[0,1]`** via a per-layer **transfer/fuzzy-membership function** (ramp, sigmoid, gaussian-band) — e.g. "favorable temperature ramps 150→250 °C." These curves are **user-editable in the UI** (same control as the viewer's transfer functions, doc 06).
- **Weights / membership shapes / method are all user-configurable** — favorability is explicitly a *research instrument*, not a fixed score. The UI exposes sliders; re-running writes a new versioned favorability volume.
- **`missingPolicy`** decides whether a cell lacking one evidence layer is nodata (strict), neutral, or just dropped from the weighting — this interacts hard with footprints (§2.3) and is surfaced to the user, because favorability is only trustworthy where its evidence actually overlaps.
- Output is a `PropertyModel` (`derivation.kind = "favorability"`, colormap diverging/heat), with a paired **confidence volume** (§5) and an **evidence-overlap mask** (how many evidence layers were present per cell — itself a renderable diagnostic).

### 4.7 Registry mechanism (overlaps doc 08)

The **how-do-plugins-register** mechanism is owned by **doc 08** (plugin architecture). This doc defines only the **transform contract** that registers *into* it:

> A transform plugin exposes: `id`, `version`, `title`, `target`, `inputs[]` (name/unit/required), `output` (unit/range/colormap), `params[]` (name/type/default/range), and an `apply(ctx, **inputs, **params)` pure function. Registration, discovery, and lifecycle = doc 08. The fusion engine consumes the registry to (a) populate the UI's transform palette, (b) validate I/O, (c) run the harness (§4.5).

**Assumption flagged for doc 08:** the registry supports both **built-in** transforms (shipped, the §4.2 library) and **user/plugin** transforms (the R&D path), keyed by `id`, with versioned resolution.

---

## 5. Uncertainty propagation (OVERVIEW §6 closing line)

> "Uncertainty propagates through every level; confidence volumes are renderable layers."

This is non-optional and is what keeps fusion honest given the wildly different sensitivity/non-uniqueness of methods (OVERVIEW §1).

### 5.1 Source of uncertainty (from doc 02)

Each native `PropertyModel` **carries an uncertainty representation** — doc 02 owns the exact schema. We assume one of (in priority order):

| Form | Symbol | Source |
|---|---|---|
| per-cell σ volume | `σ(x)` | inversion posterior / model covariance |
| per-cell confidence/resolution scalar `[0,1]` | `c(x)` | resolution kernel, DOI |
| global σ or relative error | `σ` | fallback when nothing per-cell exists |
| coverage/DOI mask | — | binary support (§2.3) |

**A4 ✓ confirmed against doc 02 §6.** A `PropertyModel` exposes `uncertainty` as a co-registered **per-cell 1σ array** (`<property>_sigma`, canonical unit) and optionally a `ResolutionSpec` (DOI surface + smoothing kernel) — the "noisy vs blurry" complement. `uncertainty:null` means **unknown, not zero**, so we attach a **default conservative relative σ per property** (from the property registry) to keep propagation running. Fused layers also carry a `validMask` (doc 02 §11 `FusedLayer.validMask`) for coverage. (The `confidence`/`variance` forms in the table are alternate `UncertaintySpec.representation` values doc 02 §6 permits.)

### 5.2 Propagation rules

Uncertainty rides through the same pipeline as the values:

**Through resampling (§2):**
- σ is resampled by the **same interpolator** as the value.
- **Interpolation inflates σ:** upsampling a coarse model onto a finer grid adds an interpolation-variance term (grows with distance from native nodes), so faked detail reads as low-confidence. Gridding scattered points adds the gridder's prediction variance (`verde` gives this).
- Outside footprint/DOI → nodata (not high σ — *no* data).

**Through transforms (§4):** first-order error propagation (delta method) by default:

```
For output  y = f(x₁, … xₙ; θ):
    σ_y² ≈ Σᵢ (∂f/∂xᵢ)² · σ_xᵢ²     (+ optional param covariance Σ_θ term)
```

- `∂f/∂xᵢ` evaluated **numerically** (finite difference) by the harness so transform authors don't hand-derive Jacobians — they just supply `apply()`.
- **Optional Monte-Carlo mode** for strongly nonlinear transforms (Archie is nonlinear in φ): sample inputs ~ their distributions, push K samples through `apply()`, take output mean/σ/quantiles. Job-based (§5.5). This is the more correct path; delta method is the fast default.
- **Parameter uncertainty** (e.g. uncertain porosity in Archie) is includable: declare a σ on a `Param` → contributes to `σ_y` (and is often the *dominant* term — surfaced to the user).

### 5.3 Confidence volume output

Every resampled layer and every derived volume ships a **paired confidence volume**:
- stored as `sigmaRef` (a `PropertyModel`-like array, doc 04), in the value's units (σ) or normalized confidence `[0,1]`.
- renderable directly (doc 06) **or** bound to the value layer as an **opacity/desaturation modulator** — low-confidence regions render faint/greyed. This is the default "honest view": you literally see less where the model knows less.

### 5.4 Non-uniqueness & low-sensitivity flagging

Smooth, non-unique methods (gravity, MT) and below-DOI regions must be **flagged**, not silently trusted:

| Flag | Trigger | Effect |
|---|---|---|
| **Below-DOI / unconstrained** | cell below a model's DOI surface, or outside footprint | nodata for that input; favorability evidence-overlap drops |
| **Low-sensitivity** | confidence below a threshold, or resolution-kernel width ≫ fused spacing | mask layer + faint render; excluded from cluster fit by default |
| **High-disagreement** | where two methods *should* correlate (via a transform) but don't, residual flagged | a diagnostic "tension" volume — interesting, not necessarily wrong |
| **Extrapolated-detail** | upsampling-dominated cells (interp variance ≫ native σ) | high-σ, faint render |

These flags are themselves derivable mask/scalar volumes the user can toggle. They are advisory; nothing is deleted.

### 5.5 Compute placement (uncertainty)

| Mode | When |
|---|---|
| Delta-method σ (analytic-ish) | inline with the transform/resample (cheap; sync or same job) |
| Monte-Carlo σ | **job-based** always (K× the work); progress over WebSocket |

---

## 6. Compute & API surface summary

All fusion compute is **Python backend** (OVERVIEW §4/§5): `xarray` (labelled arrays + lazy/Dask), `numpy`/`scipy` (interp, stats), `verde` (gridding scattered data), `scikit-learn` (clustering/GMM), `pint` (units). No fusion math in the browser — the viewer only requests and renders.

**Sync vs job** (consistent rule across the engine):

| Sync (REST, ≲ sub-second) | Job-based (background task + WebSocket progress) |
|---|---|
| sample/cross-plot/histogram on a selection ≤ 5 M cells | whole-volume resample, full-grid clustering/GMM |
| correlation matrix on a selection | every transform over a full fused grid |
| reading an already-computed derived layer (→ doc 04) | favorability over full grid; Monte-Carlo uncertainty |

Job orchestration (queue, progress, result handle) is **doc 04's** job API; this doc only declares which operations are jobs.

**Backend API sketch** (mounts under FastAPI, OVERVIEW §4):

```
POST  /fused/{gridId}/resample            { propertyModelId, method, interp_space }  → ResampledLayerRef | job
POST  /fused/{gridId}/sample              { layers[], selection }                    → feature matrix (sync)
POST  /fused/{gridId}/crossplot           { x, y, [z], color, selection, density }   → plot payload (sync)
POST  /fused/{gridId}/cluster             { layers[], algo, k|components, scale }     → job → class+prob volumes
GET   /transforms                         → registry (palette: id, params, target, io)   [doc 08-backed]
POST  /fused/{gridId}/transform           { transformId, version, inputs, params }   → derived PropertyModel | job
POST  /fused/{gridId}/favorability        { FavorabilitySpec }                       → favorability volume | job
GET   /derived/{modelId}                  → PropertyModel handle (+sigmaRef, +maskRef)   [served by doc 04]
```

Derived/resampled/confidence/mask outputs are **all `PropertyModel`s served by doc 04** and **rendered by doc 06** — fusion adds no new serving or rendering path, only new *producers*.

---

## 7. Worked end-to-end example (validates against synthetic ground truth)

Ties to OVERVIEW §8 (synthetic earth has a "hot, conductive, altered zone") and the Phase-3 verification (OVERVIEW §9, §"Verification").

1. Ingest synthetic **resistivity** (MT-like, smooth, deep) + **velocity** (seismic, sharp) + **density** (gravity, smooth) models, each with σ/DOI (doc 05/03/02).
2. **Resample** all three onto `fused-default` (§2): resistivity & density block/trilinear, each masked to footprint/DOI → co-located stack + coverage + σ.
3. **Cross-plot** log(ρ) vs density, color by depth → spot the low-ρ/low-density anomaly cluster (§3.2).
4. **GMM cluster** the stack → a class volume; the anomalous class ≈ the altered zone; label it (§3.3).
5. **Transform** resistivity→temperature (Archie+Arps) and velocity→porosity (§4.2); **microseismic→fracture density** for permeability proxy.
6. **Favorability** (fuzzy-AND of high-T **AND** high-perm **AND** fluid) → a single index volume (§4.6).
7. **Confidence** volume modulates opacity; below-DOI deep region renders faint (§5.3–5.4).
8. **Validate:** the favorability hot-spot should coincide with the synthetic ground-truth geothermal anomaly — the end-to-end fusion correctness check.

---

## Decisions locked in

1. **Non-destructive resampling.** Native property models are immutable; fusion produces *new* derived `PropertyModel`s bound to a `FusedGrid`. Resampled layers are cached and re-derivable, never overwritten.
2. **Footprint-honest fusion.** No extrapolation beyond a method's coverage/DOI — outside support is **nodata (NaN)**, never zero or edge-bleed. Each layer ships a coverage mask; transforms/cross-plots only act where required inputs are present.
3. **Resampling method is support- and property-driven.** Trilinear (regular), block-mean (downsampling), barycentric (mesh), spline gridding (scattered); interpolate in **log space for orders-of-magnitude properties** (resistivity/conductivity/permeability) per the property registry. Categorical = nearest only.
4. **Transform = declarative spec + pure Python `apply()`**, with typed inputs/outputs (unit-checked via `pint`), tunable params, and **semver versioning**; the registration mechanism is doc 08's, the contract is here.
5. **Derived volumes are first-class `PropertyModel`s** — stored, served (doc 04), and rendered (doc 06) identically to ingested ones; they differ only by a provenance recipe and can be chained.
6. **Favorability is a configurable multi-evidence index in `[0,1]`** with three swappable combination methods (**weighted-linear default**, fuzzy-logic, Bayesian/weights-of-evidence), per-evidence user-editable membership curves and weights — an explicit R&D instrument.
7. **Uncertainty propagates end-to-end.** σ rides through resampling (with interpolation-variance inflation) and transforms (delta method default, optional Monte-Carlo); every derived/resampled layer ships a **paired confidence volume** that can modulate render opacity. Below-DOI / low-sensitivity / extrapolated-detail regions are **flagged**, never silently trusted.
8. **All fusion compute is Python backend** (`xarray`/`numpy`/`scipy`/`verde`/`scikit-learn`/`pint`); **sync for selections ≤ 5 M cells, job-based for whole-grid** work (job API = doc 04). The browser only renders.

## Open questions for you

1. **Which rock-physics relationships to prioritize for the starter library?** The §4.2 table is broad; building/calibrating all of them well is real work. *Why it matters:* it sets Phase-3 scope and which synthetic-earth properties (doc 05) must be forward-modeled to validate them.
   - **(a)** Lead with **resistivity→temperature/fluid (Archie + Arps)** only — the canonical geothermal link — plus velocity→porosity. *(recommended default)*
   - **(b)** Add alteration index + microseismic→fracture density up front (fuller "heat+fluid+perm" triad for favorability).
   - **(c)** Full table including Waxman-Smits/dual-water and permeability proxies.

2. **Default favorability combination method.** *Why it matters:* it shapes the headline product's behaviour and the UI controls. Weighted-linear is *compensatory* (a great temperature can mask absent permeability); fuzzy-AND is not — and geothermal genuinely needs the conjunction of heat **and** fluid **and** permeability.
   - **(a)** **Weighted-linear** as the shipped default (transparent, simplest UI), with fuzzy/Bayesian available. *(recommended default — easiest to reason about first)*
   - **(b)** **Fuzzy-AND** default (most physically faithful to the geothermal play conjunction).
   - **(c)** Ship weighted + fuzzy together, no Bayesian until known-occurrence training data exists.

3. **Uncertainty rigor for the MVP.** *Why it matters:* Monte-Carlo through transforms is the *correct* answer for nonlinear rock physics (Archie), but it is K× the compute and always job-based; the delta method is cheap and inline but approximate near nonlinearity.
   - **(a)** **Delta-method everywhere for MVP**, expose Monte-Carlo as an opt-in job for specific transforms. *(recommended default)*
   - **(b)** Monte-Carlo as default for flagged-nonlinear transforms, delta method elsewhere.
   - **(c)** Delta-method only for Phase 3; defer Monte-Carlo entirely to a later phase.
