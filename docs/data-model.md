# The data model

!!! abstract "What you'll learn / why it matters"
    A gravity survey, a borehole log, a seismic cube, and a fault interpretation are wildly
    different things. To fuse them you first need a *common type system* — a small set of data
    structures every method maps onto, carrying units, uncertainty, and a record of where every
    number came from. This page defines that type system: the **three primitives**
    (Observation, PropertyModel, GeologicalFeature), the **FusedEarthModel** container that
    grids them together, **provenance** (a DAG of how each artifact was derived), **versioning**
    and content-addressing, the **on-disk formats** (Zarr, COG, glTF, LAZ) and *why* each was
    chosen, and how **uncertainty** is represented. If [the spatial framework](spatial-framework.md)
    defined the coordinate system and units, this page defines the schemas those coordinates and
    units live inside.

For a programmer: this is the schema layer. Every ingested file and every derived artifact becomes
one of four typed records under a thin envelope, and the heavy numeric payload lands on disk in a
format chosen for *how it will be queried* — random-access sliceable cubes for volumes, streamable
meshes for surfaces, compressed clouds for points. The design goal throughout is **lossless
fidelity plus full traceability**: raw measurements are never overwritten, and every value traces
back to the byte it came from.

---

## 1. The envelope: `Dataset` and its four payloads

Everything ingested or derived is a **`Dataset`** — a thin catalog record (the envelope) with a
typed body called the `payload`. There are exactly four payload kinds:

```
Project ── owns ─→ SpatialFrame (the coordinate frame, see spatial-framework.md)
   │
   └─ contains many ─→ Dataset (one ingested source OR one derived artifact)
                          │  kind ∈ {observation, propertyModel, feature, fusedModel}
                          ├─ ObservationSet    →  what was measured, where (raw, immutable)
                          ├─ PropertyModel     →  a continuous 3-D field of ONE property
                          ├─ GeologicalFeature →  a discrete interpreted shape
                          └─ FusedEarthModel   →  the shared grid everything resamples onto
   Every Dataset ─ carries ─→ Provenance (lineage DAG; REQUIRED — no exceptions)
```

One uploaded file can yield several Datasets sharing a provenance root — e.g. a seismic
interpretation becomes one velocity `propertyModel` *plus* several `feature` horizons. The
`Dataset` row (see `geosim/catalog/models.py`) carries identity, the survey `method`, the
Engineering-metre bounding box (`extent`, used to build the catalog's spatial index), an optional
`TimeAxis` (present ⇒ the dataset is 4-D), the binding to the project `SpatialFrame`, the original
CRS it arrived in, a **mandatory** `provenanceId`, and version info.

!!! note "Term: ULID identity"
    Every record's `id` is a **ULID** — a sortable, time-ordered unique ID — with a **kind prefix**:
    `ds_`, `obs_`, `pm_`, `feat_`, `fem_`, `well_`, `prov_`. The prefix tells you the kind at a
    glance; the time-ordering means IDs sort chronologically (handy for an index).

The survey `method` is drawn from a **fixed registry** so nothing invents variants — every doc and
plugin uses these exact keys, with subtypes living in an optional `submethod` field:

```python
MethodKey = "gravity"|"magnetics"|"ert"|"ip"|"em"|"mt"|"seismic"|"microseismic"
          | "insar"|"welllog"|"heatflow"|"geology"|"geochem"
          | "derived"|"fused"|"synthetic"
# submethod examples: seismic → "reflection"|"refraction"; em → "tdem"|"fdem"|"aem"
```

### 1.1 The one universal rule: no bare numbers

A `Quantity` is the atom of measurement — a value plus its **canonical unit** plus its
**property type** (the registry key from [the units page](spatial-framework.md#6-the-units-registry)):

```python
Quantity { "value": number|null, "unit": string, "propertyType": PropertyTypeKey }
```

!!! warning "No number that means something physical is ever stored without a unit"
    Either it is a `Quantity` (a scalar) or it lives in a bulk array whose **per-array metadata**
    declares the unit and property type. A bare float with an implied unit is forbidden anywhere in
    the system — it is the data-model equivalent of an untyped void pointer, and it is exactly how
    real geoscience datasets get silently corrupted.

---

## 2. Primitive #1 — `ObservationSet` (immutable measured data)

An **Observation** is *what was measured, where* — raw survey data tied to its acquisition
geometry. It is **immutable**: once written it is never edited. A corrected re-import is a *new*
dataset whose provenance links back to the old one. This is the auditable measured-record guarantee.

!!! tip "Why observations are NOT gridded on ingest"
    It is tempting to interpolate every survey straight onto a nice regular grid. The platform
    refuses, because **gridding is interpretation** — it bakes in resampling choices and destroys
    the raw record. Observations stay faithful to the instrument; any continuous field is a
    *separate* PropertyModel (§3) with its own provenance. (CS analogy: keep the raw event log;
    derive aggregates separately and reproducibly.)

Acquisition geometry differs enormously across methods — a gravity survey is scattered points, a
seismic survey is a cube of traces, an InSAR scene is a raster. So `ObservationSet` is a **tagged
union on `geometryKind`**:

| `geometryKind` | Example methods | Records backing | Geometry held in |
|---|---|---|---|
| `points` | gravity stations, geochem, heat-flow | inline / Parquet | per-record XYZ columns |
| `soundings` | [TEM/AEM](survey-methods/electromagnetic.md), MT site curves | Parquet | XYZ + a depth/frequency column |
| `profile2d` | [ERT/IP](survey-methods/electrical.md) pseudosection | Parquet | a polyline + along-line + pseudo-depth |
| `traces` | [seismic SEG-Y](survey-methods/seismic.md) | Zarr (trace cube) | a polyline; samples in the array |
| `raster2d` | [InSAR scene](survey-methods/insar.md), anomaly grid | COG | an affine transform |
| `wellcurve` | [LAS/DLIS logs](survey-methods/boreholes.md) | Parquet (MD-indexed) | XYZ derived from the well path |
| `tensor` | [MT impedance EDI](survey-methods/electromagnetic.md) | Parquet/JSON | per-site XYZ; components in `methodData` |

Small record tables are stored inline; large ones spill to **Parquet** (a columnar table format —
column compression, predicate push-down, exactly what you want for millions of stations). The
schema declares every column: name, dtype, unit, property type, and a `role`
(`value` / `sigma` / `coord` / `index` / `meta`).

!!! note "Every measurement should carry its error"
    The schema strongly encourages a paired `sigma` column for every `value` column (`role:"sigma"`,
    `errorFor:"<value>"`), in the value's unit — the per-datum 1σ that [inversion](inversion.md)
    needs. Tensors and traces carry an error model in `methodData.errorModel`. When a source has no
    errors, ingestion applies a **default noise floor** from the property registry and records it in
    provenance — it **never** silently treats data as error-free.

### 2.1 Annotated example: a gravity `.csv` → an `ObservationSet`

Gravity is the simplest case: scattered stations, one value each. Here is a few lines of a typical
native gravity export, annotated:

```csv
# station,    lon,        lat,       elev_m,  gravity_mGal,  err_mGal
GRV001,    -118.1043,  38.6512,    1623.4,    -18.42,        0.05
GRV002,    -118.1031,  38.6519,    1625.1,    -18.37,        0.05
GRV003,    -118.1018,  38.6527,    1627.8,    -19.10,        0.06
#  ^ id     ^ WGS84 lon/lat (EPSG:4326)   ^ orthometric    ^ the      ^ per-datum
#                                            elevation       Bouguer    1σ error
#                                                            anomaly
```

On ingest (see [the ingestion page](ingestion.md)) this becomes:

- coordinates reprojected from `EPSG:4326` into the project CRS, then into **Engineering metres**
  (the [spatial transform](spatial-framework.md#2-the-engineering-frame-one-internal-coordinate-system)),
  with the original CRS recorded in provenance;
- `gravity_mGal` mapped to property `gravity_anomaly`, canonical unit `mGal` (already canonical);
- `err_mGal` becoming the paired `sigma` column;
- an `ObservationSet` with `geometryKind:"points"`, `primaryProperty:"gravity_anomaly"`, the table
  stored inline (small) with `xyzColumns:["x","y","z"]`.

The raw `.csv` is *also* copied verbatim into the raw store (see §6) — the normalized record never
replaces the original bytes.

---

## 3. Primitive #2 — `PropertyModel` (a continuous field of one property)

A **PropertyModel** is a derived **continuous field of exactly one physical property** over a
region, carrying **units + uncertainty + a support geometry**.

!!! note "Term: support"
    The **support** is *how the field is discretized* — the geometric scaffold the values hang on.
    Think of it as the shape/stride metadata of an array, generalized to four discretization styles.

`PropertyModel` is a tagged union on `support.kind`:

| Support kind | What it is | Native to |
|---|---|---|
| **`volume`** | a 3-D (or 4-D) regular voxel grid — `origin`, `spacing`, `shape` | the default; most inversions, the [fused grid](fusion.md) |
| **`grid2d`** | a *horizontal* 2-D field at a level (depth slice, anomaly map) | gravity/magnetic grids, depth slices |
| **`section`** | a vertical *curtain*: a 2-D field on (distance-along-line × depth) embedded in 3-D along a polyline | [ERT](survey-methods/electrical.md), 2-D seismic, AEM stitches |
| **`mesh`** | an unstructured mesh (tetrahedra, octree, voronoi) | [inversion](inversion.md)-native meshes |

!!! warning "One property per model"
    Each `PropertyModel` carries **exactly one** property. A seismic inversion yielding both Vp and
    Vs is **two** PropertyModels sharing provenance and support geometry. This keeps colormap,
    unit, scaling, and uncertainty unambiguous per field, and makes the
    [property-type registry](spatial-framework.md#6-the-units-registry) the single source of display
    behaviour. Display hints (`colormap`, `scaling`, `displayRange`) are seeded from that registry
    and stored in the Zarr attrs (§5).

The `section` support deserves a note: many real methods (resistivity profiles, 2-D seismic lines)
are *natively* a vertical slice along a survey line, not a horizontal grid and not a full cube. The
platform models that directly as a curtain embedded along a polyline; the [viewer](visualization.md)
draws it as a draped sheet, and [fusion](fusion.md) resamples it into the volume via its embedding.

### 3.1 What an MT inversion becomes

A [magnetotelluric (MT)](survey-methods/electromagnetic.md) inversion produces a resistivity cube.
As a PropertyModel:

```jsonc
PropertyModel {
  "property": "resistivity",            // ONE property
  "canonicalUnit": "ohm.m",
  "display": { "colormap":"turbo", "scaling":"log", "displayRange":[1,10000] },
  "support": { "kind":"volume", "origin":[x0,y0,z0], "spacing":[dx,dy,dz],
               "shape":[nz,ny,nx], "cellRef":"center" },
  "uncertainty": { "representation":"stddev", "tier":"quantitative", ... },  // §7
  "values": { "store":"zarr", "uri":"arrays/pm_….zarr", "path":"/resistivity" }
}
```

The bulk values land in a Zarr group (§5). Resistivity spans orders of magnitude, so its registry
entry sets `scaling:"log"` and `interp_space:"log10"` — fusion interpolates it in log space.

---

## 4. Primitive #3 — `GeologicalFeature` (discrete interpreted geometry)

A **GeologicalFeature** is vector geometry that *someone interpreted* — a shape, not a field. It is
a tagged union on `featureKind`:

| `featureKind` | Geometry store | What it is |
|---|---|---|
| `surface` | glTF / VTK triangle mesh | an interpreted horizon (a geological layer boundary), may be draped |
| `fault` | glTF / VTK mesh | a surface **plus** slip semantics (type, dip, strike, throw) |
| `unitSolid` | glTF / VTK solid | a geological unit as a watertight solid (for volume ops) |
| `wellPath` | GeoJSON line + survey table | a borehole trajectory + its deviation survey |
| `pointCloud` | LAZ / 3D-Tiles | a [microseismic](survey-methods/seismic.md) event cloud (4-D) |
| `fractureNetwork` | Parquet / VTK | a discrete fracture set, for permeability work |
| `polyline` | GeoJSON | generic lines (coverage footprints, picks) |

The `wellPath` feature is the one to study, because it binds back to
[the vertical/deviation machinery](spatial-framework.md#5-deviation-surveys-minimum-curvature):

```jsonc
WellPathDetail {
  "kind": "wellPath",
  "wellId": "well_…",                 // stable identity; logs attach via this id
  "reference": { "kind":"KB", "elevation": 1622.0, "head":[x,y,z] },  // doc 01 §4
  "deviationSurvey": { "ref": <table MD,inc,azi>, "method":"minimum_curvature" },
  "trajectory": <polyline MD→Engineering XYZ>,   // cached from the survey
  "totalDepthMd": 2400.0
}
```

!!! note "Where do well *logs* live? (a subtle split)"
    The borehole *trajectory* is this `wellPath` **feature** (re-editable interpretation). The LAS
    curve *values* (gamma, resistivity, temperature down the hole) are a separate **ObservationSet**
    with `geometryKind:"wellcurve"` (immutable measured data). They are joined by `wellId`, and the
    viewer colour-maps the curve along the trajectory tube. This cleanly separates the *measured
    curve* (never edited) from the *interpreted path* (editable). See
    [boreholes](survey-methods/boreholes.md).

---

## 5. On-disk formats — and *why* each

The catalog (a relational DB, `geosim/catalog/models.py`) holds only metadata, bounding boxes,
shapes, units, stats, and pointers. **Bulk numbers live on disk** in a format chosen for *how they
will be queried*. The project tree (`geosim/storage/layout.py`) is:

```
<storage_root>/<project_id>/
  arrays/   <datasetId>.zarr     # 3-D/4-D volumes (PropertyModels, FusedModels)
  grids/    <datasetId>.tif      # 2-D rasters as COG
  meshes/   <datasetId>.{glb,vtu}# surfaces / solids
  vectors/  <datasetId>.geojson  # lines / picks
  points/   <datasetId>.laz      # point clouds
  raw/      <sha256>/<name>      # verbatim source files (provenance roots)
  cache/    slices/ isosurfaces/ tiles/   # derivable; safe to delete
```

### 5.1 Zarr v3 for volumes — the format a CS person will appreciate

A 3-D property cube can be hundreds of millions of cells. You can't load it whole in a browser, and
you need to read *arbitrary slices and sub-bricks* over HTTP. **Zarr** is the answer: a chunked,
compressed, n-dimensional array format where each chunk is an independently addressable object. It
is conceptually NumPy arrays sharded into a directory tree of compressed blocks, plus JSON metadata.

The real writer (`geosim/storage/property_model.py`) encodes these conventions:

- **Chunking into "bricks."** The array is split into **isotropic cubic chunks, default 64³**
  (~1 MiB at float32). Cubic (never full-z-column) chunks mean an XY slice, an XZ slice, a YZ slice,
  *and* a ray-march brick all read roughly the same small number of chunks. This is the spatial
  equivalent of choosing a cache-line-friendly memory layout.

    ```python
    DEFAULT_CHUNK = 64   # cubic chunk edge (geosim/storage/property_model.py)
    ```

- **Pyramids (multiresolution / LOD).** Like mipmaps for a texture, the writer builds a pyramid:
  level `0` is full resolution, each coarser level halves every axis (`build_value_pyramid` in
  `geosim/storage/pyramid.py`). The viewer streams a coarse level first, then refines — so a
  whole-volume overview is a ~64³ thumbnail, not a gigabyte download. The pyramid is described by an
  **OME-Zarr `multiscales`** block so standard tooling reads it.

- **NaN-fill, not zero.** Masked / outside-coverage / outside-[DOI](uncertainty.md) cells are the
  explicit `fill_value = NaN`, **never 0**. Zero is a *valid measurement*; conflating "no data" with
  "zero" is a classic silent bug. Pyramid downsampling is NaN-aware (an all-NaN block stays NaN).

- **Axis order is fixed: `(z, y, x)`** for 3-D, `(t, z, y, x)` for 4-D — *t and z lead, x varies
  fastest*. **Z increases with elevation** (Z-up); ingestion *flips* depth-indexed sources so the
  convention always holds. This binds directly to the
  [Engineering Frame](spatial-framework.md#2-the-engineering-frame-one-internal-coordinate-system).

- **Compression: Blosc(zstd, shuffle), lossless.** Good ratios, fast decode.

Inside one `.zarr` group, **each property is a multiscale subgroup**, and its 1σ uncertainty is a
*parallel* subgroup with the same shape (§7):

```
<datasetId>.zarr/                 # Zarr v3 group, sharded
  zarr.json                       # group meta: multiscales spec, frame ref, property list
  resistivity/                    # value: multiscale subgroup
    0/  1/  2/ …                   #   level 0 = full res; each coarser level halves z,y,x
  resistivity_sigma/              # co-registered 1σ, SAME shape/levels (variance-correct downsample)
    0/  1/  2/ …
  resistivity_doi/                # optional depth-of-investigation surface (2-D)
```

Naming is fixed: value = `<propertyType>`, uncertainty = `<propertyType>_sigma`,
DOI = `<propertyType>_doi`. Each level array carries attrs that make it self-describing — the
*required minimum* a browser reader can rely on is `origin` + `spacing` in Engineering metres,
`(z,y,x)` order:

```jsonc
// per-array .zattrs (geosim/storage/property_model.py)
{
  "propertyType": "resistivity",
  "canonicalUnit": "ohm.m",
  "scaling": "log",            "colormap": "turbo",   "displayRange": [1, 10000],
  "origin":  [z0, y0, x0],     // Engineering m, z,y,x order (REQUIRED for regular grids)
  "spacing": [dz, dy, dx],     // Engineering m, z,y,x order
  "cellRef": "center",
  "_ARRAY_DIMENSIONS": ["z","y","x"],   // xarray/CF interop
  "categories": null           // present only for categorical (lithology) arrays
}
```

!!! note "Categorical (lithology) fields are special"
    A `lithology_class` field is not a continuous number, so it does **not** get a `_sigma`. It uses
    either **hard labels** (an integer array + a `categories` table `[{id,name,color}]`, with an
    entropy/confidence companion) or **class probabilities** (a `(class,z,y,x)` float array summing
    to 1 across the class axis). Downsampling uses *mode* for labels and *mean-then-renormalize* for
    probabilities.

### 5.2 The other three formats

- **COG (Cloud-Optimized GeoTIFF)** for 2-D grids/rasters (InSAR scenes, anomaly maps). A GeoTIFF
  with internal tiling + overviews so a browser can `Range`-request just the tile/zoom it needs —
  the 2-D analogue of Zarr chunking+pyramids. Coordinates are **Engineering metres** (the geotransform
  is the affine), and a metadata tag flags it non-geographic so it is never mistaken for a true
  georeferenced raster.
- **glTF (`.glb`)** for surfaces/faults/solids — the *viewer-native* mesh format the browser's
  GLTFLoader streams directly. The real writer (`geosim/storage/gltf.py`) emits one triangle mesh
  primitive (POSITION + optional per-vertex COLOR_0 + index buffer) in Engineering metres, Z-up. A
  VTK `.vtu` sidecar is added when richer cell/node attributes or true solids are needed.
- **LAZ** (compressed LAS) or **3D Tiles** for point clouds (microseismic), with `t` and `magnitude`
  as extra per-point attributes.

This format-per-primitive mapping is the contract the [storage/serving layer](architecture.md)
implements; design source is `design/02-data-model.md` §10 and `design/04-storage-serving.md`.

---

## 6. Provenance — a DAG of how every number was made

Every Dataset has exactly one **Provenance**, and **no dataset exists without it**
(`provenance_id` is `NOT NULL` in the catalog). Provenance is a **DAG of derivation steps rooted at
the original source files**, so any value traces back to the byte it came from.

```
raw file (verbatim, sha256)
        │  Step: parse
        ▼
ObservationSet ──Step: interpolate──▶ PropertyModel ──Step: resample_to_fused──▶ FusedEarthModel
        ▲                                    ▲                                          ▲
        └── Transform: reproject, unit_convert (every CRS/unit change logged with params)
```

There are **two deliberately distinct categories**, because "reversible" must not be a blanket
claim:

- **`Transform` = coordinate/unit changes** (reproject, unit-convert, anchor, depth↔elevation,
  vertical-datum). These record their exact parameters and a *scoped* reversibility:
  `exact` (affine unit conversions, anchoring, most reprojections), `with_pinned_deps`
  (vertical-datum changes — invertible *only* with the recorded geoid model + library version), or
  not used. The original CRS/unit are mirrored on the dataset and the raw source for fast filtering.
- **`Step` = derivations** (gridding, interpolation, downsampling, clipping,
  [inversion](inversion.md), synthesis). These are **repeatable, not reversible** — each pins
  `{module, gitSha}` + exact `params` so the result is *reproducible*, but you cannot recover the
  inputs from the output. The UI must never call these "reversible."

!!! tip "What this buys you"
    - **Raw files are never mutated** — copied verbatim into `raw/<sha256>/` (see
      `geosim/storage/raw_store.py`); the SHA-256 is the content address and the dedupe key.
    - Because lineage is a **DAG**, a fused voxel or a rock-physics-derived volume points back
      through its `inputs[]` to every contributing dataset and ultimately every source file. The UI
      can literally render "where did this number come from."
    - **Editing a feature** (dragging a horizon) appends an `edit` Step and a new version — the
      prior interpretation stays auditable.

In the catalog these are the `provenance`, `provenance_inputs`, and `raw_files` tables; see
[uncertainty & honesty](uncertainty.md) for why this traceability is non-negotiable for a tool
people drill wells from.

---

## 7. Uncertainty & resolution — and the tiers

Every number a survey produces is wrong by *some* amount. The platform carries that explicitly. The
decision: uncertainty is a **co-registered per-cell standard-deviation array** (1σ in the property's
canonical unit), sharing the PropertyModel's support — the `<property>_sigma` Zarr sibling from §5.

```jsonc
UncertaintySpec {
  "representation": "stddev" | "confidence" | "variance" | "categorical_prob",
  "values": <BulkRef, SAME shape as the values — co-registered>,
  "unit": string,                       // canonical unit (stddev) or "fraction" (0..1)
  "tier": "quantitative" | "proxy" | "qualitative" | "unknown",
  "independence": "assumed_independent" | "correlated_unmodeled"
}
```

!!! warning "`uncertainty = null` means UNKNOWN, not zero"
    A missing uncertainty field is treated as **un-weightable**, never as perfectly certain.
    [Fusion](fusion.md) weights cells by $1/\sigma^2$, so silently treating null as zero σ would
    give junk data infinite confidence — the opposite of honest.

The **tier** captures *how trustworthy the σ numbers themselves are* — a level of meta-honesty
specific to this domain:

| Tier | Meaning |
|---|---|
| `quantitative` | derived from a real posterior / propagated, calibrated errors |
| `proxy` | magnitude is indicative only (rule-of-thumb σ, uncalibrated transform) |
| `qualitative` | ordinal confidence only — viewers must render low/med/high, **not** a number |
| `unknown` | no basis — treat as un-weightable, not as zero |

Separately, a **`ResolutionSpec`** captures *blurriness* — the complement of σ's *noisiness*:

!!! note "Term: DOI (depth of investigation) & resolution kernel"
    Smooth methods like [gravity](survey-methods/potential-fields.md) and
    [MT](survey-methods/electromagnetic.md) are *low-resolution but low-variance* — they don't see
    fine detail, and below a certain depth they see nothing at all. The **DOI** is that surface
    below which the model is unconstrained. A **resolution kernel** is a characteristic smoothing
    length per axis. Keeping both σ (noisy) and resolution (blurry) lets fusion tell a *noisy* model
    from a merely *blurry* one. Full per-cell covariance is explicitly **out** of scope — it is
    $O(N^2)$ and infeasible for million-cell volumes; a marginal 1σ is what inversion engines emit.

Crucially, when the pyramid downsamples a `_sigma` array it does so **variance-correct**:
averaging $N$ independent cells reduces the mean's variance by $1/N$, so
$\sigma_\text{coarse} = \sqrt{\overline{\sigma_\text{fine}^2}/N_\text{valid}}$ (see
`downsample_sigma` in `geosim/storage/pyramid.py`) — confidence survives level-of-detail, instead of
being silently smeared away. More on all of this in [uncertainty & scientific honesty](uncertainty.md).

---

## 8. Versioning & content-addressing

```jsonc
VersionInfo { "rootId": Id, "seq": 1, "parent": Id|null, "immutable": bool, "reason": string|null }
```

- **IDs are ULIDs**; `rootId` is the stable identity across all versions, `seq` is `1,2,3…`. Viewer
  layers reference `rootId` and resolve to a pinned `seq` or "latest."
- **Observations are immutable** (`immutable:true`). A corrected re-import is a *new* dataset linked
  via provenance — never an edit.
- **Derived artifacts version on change.** Re-fusing, re-running a rock-physics transform, editing a
  horizon, or re-anchoring creates `seq+1` with a fresh provenance Step. **Old versions are
  retained cheaply** because bulk arrays are **content-addressed by `sha256`** — unchanged bulk is
  *shared, not copied*. (This is exactly how Git stores blobs: identical content = one object.)
- **Re-anchoring local→georeferenced does NOT re-version bulk arrays** — arrays are already in
  Engineering coordinates and untouched; only the
  [`SpatialFrame`](spatial-framework.md#24-georefstatus-quality-is-not-the-same-as-mode) metadata
  changes.
- **Garbage collection**: a bulk artifact is deletable once no live version references its `sha256`.

---

## 9. `FusedEarthModel` + `FusedLayer` — the shared grid

To overlay, cross-plot, and do cell-by-cell math across methods, every PropertyModel is resampled
onto **one shared grid** — the **FusedEarthModel** — *without ever modifying the originals*.

```jsonc
FusedEarthModel {
  "gridType": "regular_voxel",          // default — see below
  "support": VolumeSupport,             // covers the ROI; origin/spacing/shape in Engineering m
  "layers": FusedLayer[],               // each native property resampled IN as a layer
  "values": <Zarr group: one array per layer (+ sigma) sharing the grid>
}
FusedLayer {
  "property": "resistivity",
  "sourcePropertyModelId": "pm_…",      // the native original — NEVER overwritten
  "sourceVersion": "ver_…",             // pinned version resampled from (provenance)
  "resampleOp": { "method":"trilinear", "params": {…} },
  "sigmaArray": "…", "validMask": "…"   // resampled 1σ + a coverage mask
}
```

The default grid is a **regular voxel grid** because it is trivial to ray-march on the GPU, trivial
to cross-plot (cell-aligned sampling), and trivial for derived-property math (element-wise). Octree
behaviour is delivered via the LOD pyramid (§5.1) rather than an irregular topology; native
unstructured `mesh` PropertyModels are *kept* and *resampled onto* the voxel grid for comparison.

!!! tip "Non-destruction, concretely"
    Resampling opens `sourcePropertyModelId@sourceVersion` **read-only** and writes only into the
    fused Zarr group. Deleting or re-fusing the fused model never touches a native original. Re-fusing
    at a different resolution produces a new FusedEarthModel version (§8); the originals are immutable
    inputs in its lineage. The catalog tables are `fused_models` and `fused_layers`. The resampling
    engine itself lives in [fusion](fusion.md).

---

## 10. End-to-end: a SEG-Y seismic interpretation

To tie it together — ingesting a [seismic](survey-methods/seismic.md) interpretation over a real
UTM-zone-12 site:

1. **Source** `survey.sgy` copied verbatim → `raw/<sha>/survey.sgy`; provenance records its CRS
   (`EPSG:32612`), original units, and sha256.
2. The adapter parses it into: one **ObservationSet** (`geometryKind:"traces"`), one **PropertyModel**
   (`property:"velocity_p"`, a Zarr volume in Engineering coordinates via logged reproject + anchor
   Transforms), and two **GeologicalFeature** horizons (glTF surfaces).
3. Each gets a **Dataset** envelope sharing one provenance root.
4. The velocity model lands as `<ds>.zarr/velocity_p` (+ `_sigma`, + pyramid), attrs carrying
   `canonicalUnit:"m/s"`, `colormap`, `scaling:"linear"`.
5. The user builds a **FusedEarthModel**; the velocity model is resampled in as a `FusedLayer`
   (trilinear) — the native Zarr is untouched, lineage records the resample.
6. The [viewer](visualization.md) reads pyramid LOD bricks and ray-marches; cross-plot samples the
   fused grid; the [well planner](well-planning.md) intersects a trajectory with the velocity layer.

---

## Key takeaways

- Everything is a **`Dataset`** wrapping one of **four payloads**: `Observation` (immutable measured
  data), `PropertyModel` (one continuous property field), `GeologicalFeature` (interpreted geometry),
  `FusedEarthModel` (the shared grid).
- **No bare numbers** — every physical value is a `Quantity` or a bulk array with declared unit +
  property type.
- Observations keep their **native acquisition geometry** (tagged on `geometryKind`) and are never
  gridded in place — gridding is a separate, provenance-tracked PropertyModel.
- On-disk formats are chosen for *how data is queried*: **Zarr** (chunked/pyramided/NaN-filled
  `(z,y,x)` cubes) for volumes, **COG** for 2-D, **glTF** for meshes, **LAZ** for points; raw files
  kept verbatim and content-addressed.
- **Provenance is a DAG** rooted at verbatim sources; `Transform` (scoped-reversible coordinate/unit
  changes) is kept distinct from `Step` (repeatable-not-reversible derivations).
- **Uncertainty = co-registered per-cell 1σ** (variance-correct under LOD) plus a resolution/DOI
  complement, with **tiers** (quantitative→unknown); `null` means *unknown, not zero*.
- **Observations are immutable; derived artifacts version on change**, with content-addressed bulk
  shared like Git blobs. The **FusedEarthModel never overwrites originals**.

## Where this lives in the code

| Concern | Module |
|---|---|
| Catalog tables for all four record kinds + provenance + fused models | `backend/geosim/catalog/models.py` |
| ULID identity / kind prefixes | `backend/geosim/catalog/ids.py` |
| Zarr v3 PropertyModel writer/reader (layout, attrs, chunking) | `backend/geosim/storage/property_model.py` |
| Pyramid downsampling (mean for values, variance-correct for sigma) | `backend/geosim/storage/pyramid.py` |
| glTF mesh writer for surfaces/faults | `backend/geosim/storage/gltf.py` |
| Content-addressed raw store | `backend/geosim/storage/raw_store.py` |
| Project bulk-store directory layout | `backend/geosim/storage/layout.py` |
| Property-type registry (unit/colormap/scaling per property) | `backend/geosim/spatial/property_types.py` |

Design source of truth: `design/02-data-model.md` (with `design/04-storage-serving.md` for the
physical store). Continue to the [survey methods](survey-methods/index.md) to see real files map onto
these primitives, or to [fusion](fusion.md) to see the FusedEarthModel get built.
