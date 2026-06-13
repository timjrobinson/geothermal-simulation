# 02 — Data Model & On-Disk Conventions

> Parent: `OVERVIEW.md` §2. Binds to `01-spatial-framework.md` for all coordinates,
> datums, and units. This doc defines the **exact schemas** for the three primitives —
> Observations, Property Models, Geological Features — plus the **Fused Earth Model**
> grid and the **Provenance** model, and the on-disk **Zarr/COG/vector** layout.
> Ingestion (03), storage (04), viewer (06), fusion (07), and drilling (09) all read
> these schemas. Where a schema field is a coordinate or a quantity, it is in the
> **Engineering Frame** (ENU, metres, Z-up) and canonical units from doc 01 §5 unless
> a field name explicitly says otherwise.

## 0. Scope & layering

- This doc owns the **logical schema** (the shape of every record) and the **physical on-disk conventions** (how bulk arrays land on disk).
- It does **not** own the catalog **DB tables/indexes**, the tile/slice/sample **API**, or chunk-cache internals — those are doc 04. Per OVERVIEW §2/§5 the shared contract is: *catalog DB holds metadata + provenance; bulk arrays are Zarr (3D/4D) / COG (2D) / glTF-VTK-GeoJSON (vector); raw files kept verbatim.* **Assumption (flag for doc 04):** the catalog stores each record below as a JSONB document keyed by its `id`, with the spatial extent indexed in PostGIS/SpatiaLite. If doc 04 chooses a normalized relational layout instead, the *field semantics here are still the contract*.
- It does **not** own per-method parsing — doc 03 produces records that conform to these schemas.

### The four catalog record kinds + bulk backings

```
Project ── owns ─→ SpatialFrame (doc 01)
   │
   └─ contains many ─→ Dataset (catalog record; one ingested source OR one derived artifact)
                          │  payload kind ∈ {observation, propertyModel, feature, fusedModel}
                          ├─ ObservationSet   ──→ (small) inline / Parquet / COG / Zarr
                          ├─ PropertyModel     ──→ Zarr group (volume) | COG (2D) | VTK (mesh)
                          ├─ GeologicalFeature ──→ glTF/VTK (surfaces/solids) | GeoJSON | LAS/3D-Tiles (points)
                          └─ FusedEarthModel    ──→ Zarr group (one per fused grid)
   Every record ─ carries ─→ Provenance (lineage, sources, transforms)
```

`Dataset` is the **catalog envelope**; the four `payload` kinds are the typed bodies. One uploaded file can yield several Datasets (e.g. a SEG-Y interpretation → one velocity `propertyModel` + several `feature` horizons), all sharing a provenance root.

---

## 1. Common building blocks

Reused across every schema. Defined once.

```jsonc
// ---- Identity ----
Id        = string   // ULID (sortable, time-ordered). e.g. "ds_01J9Z3...". Prefix names the kind.
// prefixes: prj_ ds_ obs_ pm_ feat_ fem_ prov_ well_ run_ ver_

// ---- Quantity: a value (or array ref) that carries a unit ----
Quantity {
  "value":  number | null,        // scalar; null if carried in a bulk array
  "unit":   string,               // canonical unit string from doc 01 §5 registry (pint-parseable)
  "propertyType": PropertyTypeKey // FK into the property-type registry (doc 01 §5 / doc 08)
}

// ---- PropertyTypeKey: the registry handle (doc 01 §5) ----
PropertyTypeKey = "resistivity" | "conductivity" | "density" | "susceptibility"
                | "velocity_p" | "velocity_s" | "temperature"   // temperature canonical = K (doc 01 §5)
                | "chargeability_time_ms" | "chargeability_mv_v" | "phase_mrad"  // IP split (doc 01 §5, critique #22)
                | "gravity_anomaly" | "magnetic_field" | "deformation"
                | "favorability" | "lithology_class" | "<plugin-registered>"
// This list is the CANONICAL property-type registry (doc 01 §5 owns unit/colormap/scaling
// per key; doc 08 lets plugins register new keys). No doc invents property keys outside it.

// ---- AABB: axis-aligned bounding box, Engineering metres ----
Aabb { "xmin":num,"xmax":num, "ymin":num,"ymax":num, "zmin":num,"zmax":num }

// ---- TimeAxis: 4D support (see §8). Absent ⇒ static (3D). ----
TimeAxis {
  "kind":   "instant" | "interval" | "series",
  "epochs": string[],              // ISO-8601 UTC timestamps, one per time index
  "ref":    "acquisition" | "model"// what the epoch dates: when measured vs model valid-time
}

// ---- Ref to bulk data on disk (resolved by doc 04 storage layer) ----
BulkRef {
  "store":  "zarr" | "cog" | "gltf" | "vtk" | "geojson" | "las" | "parquet" | "inline",
  "uri":    string,                // relative path under project store root (doc 04 owns root)
  "path":   string | null,         // sub-path inside a Zarr group, e.g. "/resistivity"
  "sha256": string | null          // content hash of the bulk artifact (provenance/integrity)
}
```

**Rule:** any number that represents a physical measurement is either a `Quantity` (scalar) or lives in a bulk array whose **per-array metadata** declares the unit + propertyType. **No bare numbers with implied units**, anywhere.

---

## 2. `Dataset` — the catalog record

The envelope every ingested or derived thing gets. Thin; the heavy typed body is in `payload`.

```jsonc
Dataset {
  "id":        Id,                 // "ds_..."
  "projectId": Id,                 // FK → Project
  "name":      string,             // human label, e.g. "MT inversion — survey A 2024"
  "kind":      "observation" | "propertyModel" | "feature" | "fusedModel",
  "method":    MethodKey,          // canonical survey method (registry below)
  "submethod": string | null,      // canonical subtype, e.g. "reflection" under "seismic" (critique #7)
  "extent":    Aabb,               // Engineering metres; spatial footprint for catalog spatial index
  "time":      TimeAxis | null,    // 4D datasets only

  "spatialFrameId": Id,            // the project SpatialFrame this is expressed in (doc 01)
  "originCrs":      string | null, // CRS the source arrived in, BEFORE reprojection (provenance mirror)

  "provenanceId":   Id,            // → Provenance (§7). REQUIRED for every dataset.
  "version":        VersionInfo,   // §9
  "tags":           string[],
  "createdAt":      string,        // ISO-8601 UTC
  "createdBy":      string,        // user/email or "system:synthetic" | "system:fusion"

  "payload":        ObservationSet | PropertyModel | GeologicalFeature | FusedEarthModel
}

// CANONICAL method registry (resolves critique #7). Every doc — ingestion (03),
// plugins (08), inversion (10) — uses THESE keys, never variants like "seismic_reflection".
// Subtypes live in the optional `submethod` field, not in new top-level keys.
MethodKey = "gravity"|"magnetics"|"ert"|"ip"|"em"|"mt"|"seismic"|"microseismic"
          | "insar"|"welllog"|"heatflow"|"geology"|"geochem"
          | "derived"|"fused"|"synthetic"

// submethod: optional disambiguator under a MethodKey. Canonical values:
//   seismic → "reflection" | "refraction" | "ambient_noise" | "tomography"
//   em      → "tdem" | "fdem" | "aem"
//   ert/ip  → "dc_resistivity" | "ip_time" | "ip_freq"
// A plugin registers (method, submethod) pairs; routing/filtering keys on both.
```

| Field | Required | Notes |
|---|---|---|
| `id`,`projectId`,`name`,`kind`,`method` | ✓ | `kind` selects which `payload` schema is valid |
| `extent` | ✓ | Engineering AABB; doc 04 builds the spatial index from this |
| `spatialFrameId` | ✓ | always present — even local-mode data binds to a frame (doc 01) |
| `time` | optional | present ⇒ dataset is 4D; absent ⇒ static |
| `provenanceId` | ✓ | **no dataset exists without provenance** (§7) |
| `originCrs` | optional | null for synthetic/local-mode; mirror of provenance for quick filtering |

---

## 3. `ObservationSet` — immutable measured data

Raw/measured survey data tied to **acquisition geometry**. **Immutable** once written (§9). Stores *what was measured where*, not an interpolated field. Geometry is heterogeneous across methods, so the schema is a tagged union on `geometryKind`.

```jsonc
ObservationSet {
  "geometryKind": "points" | "soundings" | "profile2d" | "traces" | "raster2d"
                | "wellcurve" | "tensor",   // tags the acquisition geometry family
  "primaryProperty": PropertyTypeKey | null, // main measured quantity (null for raw traces/tensors)

  // ---- columnar table of stations/samples (small sets inline; large → Parquet BulkRef) ----
  "records": {
    "backing":  BulkRef,           // store="inline" (rows[]) | "parquet" | "zarr"
    "schema":   ColumnSpec[],      // declares every column: name, dtype, unit, propertyType
    "rowCount": integer
  },

  // ---- acquisition geometry: where each record sits, Engineering metres ----
  "geometry": {
    // points/soundings/wellcurve: per-record XYZ comes from named columns in `records`
    "xyzColumns": ["x","y","z"] | null,     // which columns are Engineering coords
    // profile2d / traces: line geometry
    "lineGeometry": BulkRef | null,         // polyline vertices (Engineering) for sections/lines
    // raster2d (e.g. InSAR scene, gravity anomaly grid): a 2D COG, not a table
    "raster": { "ref": BulkRef, "transform": Affine2D, "shape":[ny,nx] } | null
  },

  // ---- method-specific structured blob (kept faithful; doc 03 fills it) ----
  "methodData": object,            // e.g. MT impedance tensor cmpts, ERT electrode config, SEG-Y headers
  "acquisition": {
    "instrument": string | null,
    "surveyDate": string | null,   // ISO-8601
    "operator":   string | null,
    "notes":      string | null
  }
}

ColumnSpec { "name":string, "dtype":"f4"|"f8"|"i4"|"i8"|"bool"|"str",
             "unit":string|null, "propertyType":PropertyTypeKey|null, "nullValue":number|null,
             "role":"value"|"sigma"|"coord"|"index"|"meta",  // 'sigma' = per-datum 1σ for the value col it pairs
             "errorFor":string|null }                        // name of the value column this sigma describes

// ---- Per-observation error convention (REQUIRED for inversion, doc 10; critique #19) ----
// Every measured `value` column SHOULD have a paired sigma column (role:"sigma", errorFor:"<value>")
// in the value's unit. Tensors/traces carry an error model in `methodData.errorModel`
// (e.g. {kind:"relative", floor: <abs>, pct: <%>} or a covariance BulkRef). When a source
// has no errors, ingestion applies a per-property DEFAULT NOISE FLOOR (from the property
// registry) and records it in provenance — never silently treats data as error-free.
Affine2D = [ a, b, c, d, e, f ]   // GDAL-style: x = a + col·b + row·c ; y = d + col·e + row·f (Engineering m)
```

**Geometry-kind → backing cheat-sheet** (ingestion contract, doc 03):

| `geometryKind` | Example methods | `records` backing | Geometry held in |
|---|---|---|---|
| `points` | gravity stations, geochem, heat-flow | inline / parquet | `xyzColumns` |
| `soundings` | TEM/AEM, MT site curves | parquet | `xyzColumns` + depth/freq column |
| `profile2d` | ERT/IP pseudosection | parquet | `lineGeometry` + along-line + pseudo-depth cols |
| `traces` | seismic SEG-Y | zarr (trace cube) | `lineGeometry`; samples in records/zarr |
| `raster2d` | InSAR scene, anomaly grid | COG via `geometry.raster` | `raster.transform` |
| `wellcurve` | LAS/DLIS logs | parquet (MD-indexed) | `xyzColumns` derived from well path (§5) |
| `tensor` | MT impedance EDI | parquet/json | `xyzColumns` per site; cmpts in `methodData` |

> **Why observations keep their native geometry instead of being gridded immediately:** gridding is *interpretation*; doing it on ingest would destroy the raw record and bake in resampling choices. Observations stay faithful; any field is a separate `PropertyModel` (§4) with its own provenance.

---

## 4. `PropertyModel` — continuous field of one physical property

A derived **continuous field** of one property over a region, with **units + uncertainty + support geometry**. Backing is one of three support kinds; the schema is a tagged union on `support.kind`.

```jsonc
PropertyModel {
  "property":  PropertyTypeKey,    // the ONE physical property this field carries
  "canonicalUnit": string,         // from registry; the unit the bulk array is stored in
  "valueRange": [min, max] | null, // observed data range (for default transfer-fn / colormap autoscale)

  // ---- registry-driven display hints (resolved from doc 01 §5; copied here for self-containment) ----
  "display": {
    "colormap":  string,           // e.g. "viridis","turbo","RdBu" — registry default, user-overridable
    "scaling":   "linear" | "log", // log for resistivity/conductivity by default
    "displayRange": [min, max] | null,
    "opacityCurve": [[v,a],...] | null  // transfer-fn alpha control points (viewer, doc 06)
  },

  // ---- support geometry: HOW the field is discretized ----
  "support": VolumeSupport | Grid2DSupport | SectionSupport | MeshSupport,

  // ---- uncertainty (§6). Optional but strongly encouraged. ----
  "uncertainty": UncertaintySpec | null,

  // ---- resolution / sensitivity kernel (§6) ----
  "resolution": ResolutionSpec | null,

  // ---- the bulk values ----
  "values": BulkRef                // Zarr array path (volume) | COG (2D) | VTK cell/point data (mesh)
}

// ---- 3D/4D regular voxel volume: the default & most common ----
VolumeSupport {
  "kind":   "volume",
  "origin": [x0,y0,z0],            // Engineering metres, cell-(0,0,0) corner (or center; see cellRef)
  "spacing":[dx,dy,dz],            // metres per cell along each axis
  "shape":  [nz,ny,nx],            // array shape; see §10 for dim order/coord convention
  "cellRef":"corner" | "center",   // what origin refers to; default "center"
  "rotationDeg": 0.0               // in-plane rotation about Z, vs Engineering axes (usually 0)
}

// ---- 2D grid: a HORIZONTAL field at a level (depth slice, anomaly, surface-draped) ----
Grid2DSupport {
  "kind":"grid2d", "origin":[x0,y0], "spacing":[dx,dy], "shape":[ny,nx],
  "zLevel": number | "surface" | "draped", "transform": Affine2D
}

// ---- VERTICAL SECTION / CURTAIN: the native output of ERT, 2D seismic, AEM stitches ----
// (resolves critique #6) — a 2D field on a (distance-along-line × depth/elevation) grid
// that is EMBEDDED in 3D along an arbitrary polyline. Neither a horizontal grid2d nor a
// full volume. The viewer renders it as a draped curtain; fusion resamples it onto the
// fused volume via its embedding.
SectionSupport {
  "kind":"section",
  "polyline": BulkRef,             // path vertices in Engineering XY(Z) — the curtain's trace
  "alongAxis": { "n": nx, "spacing": number|null, "cumulativeDistRef": BulkRef|null }, // samples along line
  "vertAxis":  { "n": nz, "kind":"elevation"|"depth_below_surface",
                 "origin": number, "spacing": number },   // vertical samples (Z-up if elevation)
  "values": "[nz, nAlong]",        // array shape stored in the Zarr array (vert-major)
  "drape":  "vertical" | "follow_topography"
}

// ---- unstructured mesh (inversion native meshes: SimPEG TreeMesh, PyGIMLi tets) ----
MeshSupport {
  "kind":"mesh",
  "meshType":"tetra"|"hexa"|"octree"|"voronoi",
  "meshRef": BulkRef,              // VTK/.msh: node coords (Engineering m) + cell connectivity
  "valueLocation":"cell" | "node"
}
```

| Field | Required | Notes |
|---|---|---|
| `property`,`canonicalUnit`,`support`,`values` | ✓ | one property per model; multi-property = multiple PropertyModels |
| `display` | ✓ (defaultable) | seeded from registry; persists user overrides |
| `uncertainty` | optional | absent ⇒ unknown, *not* zero — viewer/fusion must treat as unweighted |
| `resolution` | optional | absent ⇒ resolution unknown; fusion uses geometry-only weighting |

> **One property per model.** A SEG-Y inversion that yields Vp *and* Vs is two PropertyModels sharing provenance + support geometry (support may be referenced by `BulkRef.path` into the same Zarr group — see §10). This keeps colormaps, units, and uncertainty unambiguous per field and makes the property-type registry the single source of display behavior.

---

## 5. `GeologicalFeature` — discrete geometric interpretation

Vector geometry: surfaces, faults, unit solids, well paths, point clouds, fracture networks. Tagged union on `featureKind`.

```jsonc
GeologicalFeature {
  "featureKind": "surface" | "fault" | "unitSolid" | "wellPath"
               | "pointCloud" | "fractureNetwork" | "polyline",
  "geometry":    BulkRef,          // glTF/VTK (mesh/solid) | GeoJSON (lines) | LAS/3D-Tiles (points)
  "attributes":  AttributeSpec[],  // per-vertex / per-cell / per-point scalar fields
  "style":       { "color":string, "opacity":number, "renderHint":string } | null,
  "detail":      SurfaceDetail | FaultDetail | UnitSolidDetail | WellPathDetail
                | PointCloudDetail | FractureNetworkDetail
}

AttributeSpec { "name":string, "dtype":string, "unit":string|null,
                "propertyType":PropertyTypeKey|null, "location":"vertex"|"cell"|"point" }

// ---- horizon / interpreted surface ----
SurfaceDetail { "kind":"surface", "geologicName":string|null,
                "isClosed":false, "draped":bool }

// ---- fault: a surface + slip semantics ----
FaultDetail   { "kind":"fault", "faultType":"normal"|"reverse"|"strikeslip"|"unknown",
                "dipDeg":number|null, "strikeDeg":number|null, "throwM":number|null }

// ---- geological unit as a watertight solid ----
UnitSolidDetail { "kind":"unitSolid", "lithology":string|null,
                  "stratOrder":integer|null, "watertight":bool }

// ---- WELL PATH: borehole trajectory + deviation survey (doc 01 §4 MD/TVD) ----
WellPathDetail {
  "kind":"wellPath",
  "wellId": Id,                    // "well_..." stable identity across logs/runs
  "wellName": string,
  "reference": {                   // doc 01 §4: each well stores its reference elevation
    "kind":"KB"|"GL"|"MSL"|"ellipsoid",
    "elevation": number,           // Engineering elevation (m) of the reference point
    "head": [x,y,z]                // Engineering XYZ of the wellhead
  },
  "deviationSurvey": {             // canonical source for MD↔TVD↔XYZ (doc 01 §4)
    "ref": BulkRef,                // table: MD, inclinationDeg, azimuthDeg
    "method":"minimum_curvature"|"tangential"|"balanced_tangential"
  },
  "trajectory": BulkRef,           // resolved polyline: MD → Engineering XYZ (cached from survey)
  "totalDepthMd": number
}

// ---- microseismic / event cloud (4D) ----
PointCloudDetail {
  "kind":"pointCloud",
  "eventCount": integer,
  "perEventColumns": ["x","y","z","t","magnitude","..."], // t drives the TimeAxis on the Dataset
  "magnitudeType": string | null
}

// ---- fracture network: discrete fracture set (planes/discs) ----
FractureNetworkDetail {
  "kind":"fractureNetwork",
  "representation":"planes"|"discs"|"polylines",
  "perFractureColumns":["x","y","z","dipDeg","azimuthDeg","radiusM","aperture","..."]
}
```

| `featureKind` | Geometry store | Key detail | Notes |
|---|---|---|---|
| `surface` | glTF/VTK triangle mesh | `SurfaceDetail` | horizons; may be draped |
| `fault` | glTF/VTK mesh | `FaultDetail` | surface + slip semantics |
| `unitSolid` | glTF/VTK / GemPy solid | `UnitSolidDetail` | watertight for volume ops |
| `wellPath` | GeoJSON line + survey table | `WellPathDetail` | **deviation survey is canonical** (doc 01 §4); logs attach via `wellId` |
| `pointCloud` | LAS/LAZ or 3D-Tiles | `PointCloudDetail` | microseismic; `t` column ⇒ 4D |
| `fractureNetwork` | parquet/VTK | `FractureNetworkDetail` | DFN for permeability work |
| `polyline` | GeoJSON | — | generic lines (coverage footprints, picks) |

> **Well logs live where?** The LAS curve *values* are an `ObservationSet` (`geometryKind:"wellcurve"`, MD-indexed); the borehole *trajectory* is this `wellPath` feature. They are joined by `wellId`. The viewer (doc 06) color-maps the curve along the trajectory tube. This keeps the immutable measured curve separate from the (re-editable) interpreted path.

---

## 6. Uncertainty & resolution

**Decision:** uncertainty is represented as a **co-registered per-cell standard-deviation array** (1σ in the property's canonical unit), with an **optional resolution kernel** describing spatial smearing. Both are separate bulk arrays sharing the PropertyModel's support geometry.

```jsonc
UncertaintySpec {
  "representation": "stddev" | "confidence" | "variance" | "categorical_prob",
  "values": BulkRef,               // SAME support/shape as the PropertyModel values (co-registered)
  "unit":   string,                // canonical unit (stddev) or "fraction"(0..1) for confidence
  "perCategory": false,            // true ⇒ extra axis of class probabilities (lithology fields)

  // ---- uncertainty TIER: how trustworthy the numbers themselves are (resolves critique #12) ----
  "tier": "quantitative" | "proxy" | "qualitative" | "unknown",
  // quantitative : derived from a real posterior / propagated calibrated errors
  // proxy        : magnitude is indicative only (rule-of-thumb σ, uncalibrated transform)
  // qualitative  : ordinal confidence only — viewers must render it as low/med/high, not a number
  // unknown      : no basis — treat as un-weightable, NOT as zero
  "independence": "assumed_independent" | "correlated_unmodeled"
  // delta-method fusion (doc 07) assumes the former; flag the latter so confidence isn't overstated
}

ResolutionSpec {
  "kind": "kernel" | "doi" | "raymask",
  // depth-of-investigation surface: below it the model is unconstrained (gravity/MT go smooth/deep)
  "doiRef":   BulkRef | null,      // 2D grid of DOI elevation (m) over XY — null if N/A
  // spatial resolution kernel: characteristic smoothing length per axis (may vary in space)
  "kernel":   { "lengthRef": BulkRef | null, "lengthScalar":[lx,ly,lz] | null, "unit":"m" }
}
```

**Why this and not alternatives:**

| Option | Verdict |
|---|---|
| **Per-cell 1σ array (chosen)** | ✓ Simplest renderable thing; co-registered so the viewer can show a confidence volume / fade low-confidence cells; fusion (doc 07) weights by 1/σ². Universal across methods. |
| Full posterior covariance | ✗ O(N²) — infeasible for million-cell volumes; only inversion engines (doc 10) hold this, and they can emit a marginal 1σ. |
| Resolution matrix only | △ Captured separately as `ResolutionSpec` (DOI surface + kernel length) — the *complement* to 1σ, since smooth methods are "low-resolution but low-variance." Keeping both lets fusion distinguish *noisy* from *blurry*. |
| Nothing | ✗ Violates OVERVIEW §1 ("uncertainty must survive into fusion"). |

> `uncertainty=null` means **unknown, not zero**. Fusion and the viewer must treat a null-uncertainty field as un-weightable rather than perfectly certain. The synthetic generator (doc 05) emits realistic 1σ + DOI so the path is exercised from day one.

---

## 7. Provenance — lineage, sources, transforms (reversible/auditable)

Every Dataset has exactly one `Provenance` (its `provenanceId`). Provenance is a **DAG of derivation steps** rooted at original source files, recording every CRS/unit transform (linking doc 01) so any value is traceable back to the byte it came from and any conversion is reversible.

```jsonc
Provenance {
  "id": Id,                        // "prov_..."
  "datasetId": Id,                 // the dataset this describes

  // ---- roots: original files, kept VERBATIM in the raw store (OVERVIEW §5) ----
  "sources": SourceFile[],

  // ---- the derivation chain that produced this dataset's bulk artifact ----
  "lineage": Step[],

  // ---- explicit record of EVERY spatial/unit transform applied (doc 01) ----
  "transforms": Transform[],

  "agent": { "tool":string, "version":string, "adapter":string|null }, // who/what produced it
  "createdAt": string
}

SourceFile {
  "uri": string,                   // path in raw store (verbatim copy)
  "sha256": string,                // integrity + dedupe
  "format": string,                // "SEG-Y","EDI","LAS","GeoTIFF",...
  "originalCrs": string | null,    // CRS as found in the source (pre-reprojection)
  "originalUnit": string | null,   // unit as found (pre-canonicalization)
  "bytes": integer
}

Step {
  "id": Id,                        // "run_..." per processing run (versioning, §9)
  "op": string,                    // "parse"|"reproject"|"unit_convert"|"resample_to_fused"
                                   //  |"rock_physics"|"interpolate"|"edit"|"invert"|"synthesize"
  "inputs":  Id[],                 // upstream dataset/source ids → makes lineage a DAG
  "params":  object,               // exact parameters (interp method, kernel, transform fn, ...)
  "code":    { "module":string, "gitSha":string|null }, // reproducibility
  "at":      string                // ISO-8601 UTC
}

// ---- coordinate/unit transform record. Reversibility is SCOPED, not blanket (critique #8) ----
Transform {
  "type": "crs_reproject" | "vertical_datum" | "unit_convert"
        | "engineering_anchor" | "depth_elevation",
  "from": string,                  // e.g. "EPSG:4326" | "ft" | "TVDSS"
  "to":   string,                  // e.g. "EPSG:32612" | "m"  | "elevation"
  "params": object,                // anchor, rotationDeg, geoid model+version, pint factor
  "reversible": "exact" | "with_pinned_deps" | "no",
  // exact            : affine unit_convert, engineering_anchor, depth_elevation, most crs_reproject
  // with_pinned_deps : vertical_datum — invertible ONLY with the recorded geoid model + lib version
  // no               : (not used by Transform; non-invertible work is a `Step` derivation, below)
  "deps": { "library": string, "version": string, "grid": string|null } | null  // for with_pinned_deps
}
```

**Auditability guarantees:**

- **Sources are never mutated** — raw files copied verbatim into the raw store with a `sha256`. (OVERVIEW §5.)
- **Two distinct provenance categories (resolves critique #8):**
  - **`Transform` = coordinate/unit changes** — reversibility scoped per record (`exact` / `with_pinned_deps` / never blanket). `originCrs`/`originalUnit` mirror the roots for fast filtering.
  - **`Step` = derivations** (gridding, interpolation, downsampling, clipping, inversion, synthesis) — these are **repeatable, not reversible**. Each `Step.code` pins `{module, gitSha}` and `params` so the result is *reproducible*, but the inputs cannot be recovered from the output. The UI/labels must never call these "reversible."
- **Lineage is a DAG**, so a fused model or a rock-physics derived volume points back through its `inputs[]` to every contributing dataset and ultimately every source file. The UI can render "where did this voxel come from."
- **Editing a feature** (e.g. dragging a horizon) appends an `edit` Step + new version (§9) rather than overwriting — the prior interpretation stays auditable.

---

## 8. Time / 4D representation

4D is a Dataset-level concern via the optional `TimeAxis` (§1). A dataset is 4D iff `Dataset.time != null`.

| Data | How time attaches |
|---|---|
| **InSAR time-series** | `PropertyModel` (`deformation`) with `TimeAxis{kind:"series"}`; Zarr gains a leading `t` dim → shape `[nt,nz?,ny,nx]` (often `[nt,ny,nx]` raster). |
| **Microseismic** | `pointCloud` feature; each event has a `t` column; `TimeAxis{kind:"series", ref:"acquisition"}`. Filtering is by event time, not array slicing. |
| **Repeat/time-lapse surveys** (4D seismic, repeat gravity) | each vintage is its own static Dataset; a `timeLapseGroupId` tag links vintages, and a derived "difference" PropertyModel carries the pair in its lineage. |
| **Model valid-time** (e.g. a forecasted temperature field) | `TimeAxis.ref:"model"` distinguishes "when measured" from "when the model is valid for." |

**Convention:** time is **always the leading array axis** when present (`t` first), epochs are **explicit ISO-8601 UTC** in `TimeAxis.epochs` (never implicit/regular — surveys are irregular in time). The viewer's time slider (OVERVIEW §7) reads `epochs`. **We do not** force all 4D data onto a shared global clock; each dataset keeps its own epochs and the slider unions them.

---

## 9. Identity, immutability & versioning

```jsonc
VersionInfo {
  "id": Id,                        // "ver_..." this specific version
  "rootId": Id,                    // the original dataset id — stable across all versions
  "seq": integer,                  // 1,2,3...
  "parent": Id | null,             // previous version (null for v1)
  "immutable": bool,               // observations: true. derived/features: false.
  "reason": string | null          // why this version exists ("re-anchored","horizon edited","reinverted")
}
```

**Rules:**

1. **IDs are ULIDs**, kind-prefixed (`obs_`, `pm_`, `feat_`, `fem_`, `well_`). `rootId` gives stable identity across versions; the viewer/layers reference `rootId` and resolve to a pinned `seq` or "latest."
2. **Observations are immutable.** Once an `ObservationSet` is written it is never edited. A re-import (e.g. corrected source) is a *new* observation dataset with provenance linking to the prior. This is the auditable measured-record guarantee (OVERVIEW §2).
3. **Derived artifacts version on change.** Re-running fusion, re-doing a rock-physics transform, editing a horizon, or re-anchoring the project creates a **new version** (`seq+1`, `parent` set) with a fresh provenance `Step`. Old versions are retained (cheap — bulk arrays are content-addressed by `sha256`; unchanged bulk is shared, not copied).
4. **Re-anchoring local→georeferenced (doc 01 §2) does NOT re-version bulk arrays** — arrays are in Engineering coords and untouched; only the `SpatialFrame` metadata changes. It *may* bump dataset versions only if an `extent`/CRS-mirror field changes, but no array reprocessing occurs.
5. **Garbage collection** (doc 04 owns mechanism): a bulk artifact is deletable when no live version references its `sha256`.

---

## 10. On-disk conventions

### 10.1 Store layout (illustrative — **doc 04 §3 is authoritative** for directory names)

> The sketch below shows the *grouping by bulk-type*; the actual on-disk directory
> names and root path are owned by **doc 04 §3** (e.g. `arrays/ grids/ meshes/
> vectors/ points/ raw/ cache/`). Files are named by the dataset's ULID id (which
> already encodes kind, doc 02 §1). What doc 02 owns is the **Zarr-group-internal
> layout** (§10.2) and the bulk-format-per-primitive mapping — not the folder names.

```
<project>/                                   # (doc-04 directory names differ — see doc 04 §3)
  raw/        <sha256>.<ext>                 # verbatim source files (provenance roots)
  zarr/       <datasetId>.zarr/              # property models, fused models, 4D volumes
  cog/        <datasetId>.tif                # 2D grids/rasters (InSAR scenes, anomaly grids)
  vector/     <datasetId>.{glb,vtu,geojson}  # surfaces, solids, well paths, fractures
  points/     <datasetId>.{laz,3dtiles}      # microseismic / large point clouds
  tables/     <datasetId>.parquet            # observation record tables
```

### 10.2 Zarr group layout — THE authoritative spec (resolves critique #5)

This is the **single source of truth** for on-disk Zarr structure. Doc 04 (storage),
doc 06 (viewer), and any writer **link here and do not restate a different layout.**
The earlier doc-04 `pm_<id>.zarr/0/1/2/...` sketch is **superseded by this.**

A PropertyModel is **one Zarr v3 (sharded) group**. **Each property is a multiscale
subgroup** whose members are the pyramid levels (`0` = full resolution). Co-supported
properties (e.g. Vp+Vs) and the sibling sigma/doi arrays are parallel subgroups sharing
the grid. This merges property-naming (was doc 02) with level-numbering (was doc 04):

```
<datasetId>.zarr/                     # Zarr v3 GROUP, sharded
  zarr.json                           # group meta: multiscales spec, frame ref, propertyType list
  resistivity/                        # multiscale subgroup for one property
    0/  c/0/0/0 ...                   #   level 0 = full resolution (chunks under c/)
    1/  2/  3/ ...                     #   coarser levels: each halves z,y,x (mean/anti-aliased)
  resistivity_sigma/                  # co-registered 1σ, SAME shape/levels (§6); VARIANCE-correct downsample
    0/ 1/ 2/ ...
  resistivity_doi/                    # optional DOI surface (2D, [y,x]); single level ok
  resistivity_classes/                # OPTIONAL categorical companion (see below)
```

**Naming is fixed:** value array = `<propertyType>`; uncertainty = `<propertyType>_sigma`;
DOI = `<propertyType>_doi`; categorical probabilities = `<propertyType>_classes`. The
**brick/chunk key is `<property>/<level>/c/<bz>/<by>/<bx>`** — doc 04's brick addressing
maps 1:1 onto this path.

**Dimension & coordinate convention (binds doc 01):**

| Item | Convention |
|---|---|
| **Axis order** | `(z, y, x)` for 3D; `(t, z, y, x)` for 4D — **t and z lead**, x fastest-varying |
| **Coordinates (REQUIRED minimum)** | regular grids carry `origin`+`spacing` in attrs (Engineering metres) — **this is the mandatory minimum a browser reader can rely on.** Explicit CF coordinate arrays `x,y,z(,t)` are **optional** and only required for *irregular* spacing. (No more "either/or" ambiguity.) |
| **Z direction** | increasing index = **increasing elevation** (Z-up). Ingestion flips depth-indexed sources. |
| **Datum/units** | never baked into coords beyond Engineering metres — CRS/datum live in the `SpatialFrame` (doc 01 §2) |
| **Fill** | explicit `fill_value` = **NaN** for floats; masked/outside-DOI/outside-coverage cells are NaN, never 0 |

**Categorical / lithology arrays (resolves critique #20).** A `lithology_class`
PropertyModel uses one of two fixed encodings, declared in attrs:
- **hard labels:** `(z,y,x)` (or `(t,z,y,x)`) **integer** array + a `categories` attr table `[{id,name,color}]`; `fill_value` = a reserved "no-data" id. Uncertainty = an **entropy/confidence** scalar field, *not* a `_sigma`.
- **class probabilities:** `(class,z,y,x)` (or `(t,class,z,y,x)`) **float** array summing to 1 across the leading `class` axis, with the `categories` table giving the class order. This is the `perCategory:true` form from §6.
A continuous-property `_sigma` is **not** valid for categorical fields.

**Per-array attrs (`.zattrs` / array `attributes`):**

```jsonc
{
  "propertyType":  "resistivity",
  "canonicalUnit": "ohm.m",
  "scaling":       "log",
  "colormap":      "turbo",
  "displayRange":  [1, 10000],
  "origin":        [z0,y0,x0],     // Engineering m, z,y,x order (REQUIRED for regular grids)
  "spacing":       [dz,dy,dx],     // Engineering m, z,y,x order
  "cellRef":       "center",
  "_ARRAY_DIMENSIONS": ["z","y","x"],  // xarray/CF interop
  "categories":    null            // present only for categorical arrays (see above)
}
```

### 10.3 Chunking & multiresolution

- **Chunking:** **isotropic cubic chunks, default `64³`** (~1 MiB at f4) so arbitrary slice planes (XY/XZ/YZ) and ray-march bricks all read efficiently. Never full-z-column chunks. (Doc 04 may tune the exact size but not the cubic shape.)
- **Sharding:** Zarr v3 sharding packs many chunks per shard file to avoid tiny-object blowup over HTTP.
- **Multiresolution:** levels live as numbered members **`<property>/0,1,2,…`** (`0` = full res; each level halves z,y,x), described by an **OME-Zarr `multiscales`** block in the property subgroup so standard tooling reads it. Value arrays downsample by **mean/anti-alias**; **`_sigma` arrays downsample variance-correct** (so confidence survives LOD); categorical arrays downsample by **mode** (labels) or **mean-then-renormalize** (probabilities).
- **Compression:** Blosc(zstd, shuffle), lossless. **⚠️ Browser decode is an early-validation spike (critique #5/#17):** confirm JS Zarr v3 + Blosc/zstd decode maturity in M0/M1 before committing the frontend; if not ready, the fallback is server-side decode-to-raw on the brick endpoint (doc 04) with the *same* chunk addressing — so this risk does not change the layout, only who decodes.

### 10.4 2D & vector conventions

- **COG** (2D grids/rasters): standard Cloud-Optimized GeoTIFF with internal tiling + overviews. **Coordinates are Engineering metres** (the GeoTIFF geotransform is the `Affine2D`); the real CRS is *not* written into the COG (it lives in the SpatialFrame) — a `metadata` tag records `engineering_frame=true` so it's never mistaken for a georeferenced raster.
- **Vector**: surfaces/solids as **glTF** (`.glb`, viewer-native) with a **VTK** (`.vtu`) sidecar when cell/node attributes or solids need richer typing. Lightweight lines/picks as **GeoJSON** with Engineering-metre coordinates (again flagged non-geographic).
- **Points**: microseismic as **LAZ** (compressed) or **3D Tiles** for large clouds (OVERVIEW §5), with `t`,`magnitude` as extra attributes.

---

## 11. `FusedEarthModel` — the canonical resampling grid

The common ground onto which any PropertyModel resamples for overlay, cross-plot, and derived-property math — **without destroying native originals** (OVERVIEW §2).

```jsonc
FusedEarthModel {
  "gridType": "regular_voxel",     // DEFAULT (see decision below)
  "support":  VolumeSupport,       // §4 — covers the ROI; origin/spacing/shape in Engineering m
  "time":     TimeAxis | null,     // present if any resampled layer is 4D

  // ---- each native property resampled in as a LAYER (originals untouched) ----
  "layers": FusedLayer[],

  "values":   BulkRef              // Zarr group: one array per layer (+ sigma) sharing the grid
}

FusedLayer {
  "layerId": Id,
  "property": PropertyTypeKey,
  "sourcePropertyModelId": Id,     // the native-resolution original (NEVER overwritten)
  "sourceVersion": Id,             // pinned version resampled from (provenance)
  "resampleOp": {
    "method":"trilinear"|"nearest"|"conservative"|"kriging"|"idw",
    "params": object               // doc 07 owns the resampling engine; this records what was done
  },
  "sigmaArray": string | null,     // path to resampled 1σ in the fused Zarr group
  "validMask":  string | null      // path to a coverage mask: which cells this layer actually informs
}
```

### Default grid: **regular voxel grid** — recommended, justified

| Option | Trade-off | Verdict |
|---|---|---|
| **Regular voxel (chosen default)** | Trivial to ray-march (`Data3DTexture`, OVERVIEW §5/§7), trivial cross-plot (cell-aligned sampling), trivial derived-property math (elementwise), simplest Zarr layout + pyramids. Cost: memory if uniformly fine. | ✓ **Default.** Matches the GPU volume renderer and the L1–L3 fusion ladder directly. |
| Octree / multiresolution | Saves memory when detail is localized; native to SimPEG TreeMesh. | △ Supported as **LOD pyramid of the regular grid** (§10.3) — we get octree-like streaming without an irregular topology to cross-plot against. True octree fusion deferred. |
| Unstructured mesh | Honors inversion-native meshes exactly. | ✗ Not the fused grid. Native unstructured PropertyModels are *kept* as `MeshSupport` and **resampled onto** the regular fused grid for comparison. |

**Resolution choice:** the fused grid spacing defaults to a **user-set target** (suggested: the median native spacing across loaded property models, clamped so total cells stay tractable — see open question). Native models are **resampled in** as `layers`; the source PropertyModel is referenced by id+version and never modified. Re-fusing or changing grid resolution produces a **new FusedEarthModel version** (§9); originals are immutable inputs in its lineage.

> **Non-destruction guarantee, concretely:** resampling reads `sourcePropertyModelId@sourceVersion` and writes into the fused Zarr group only. The native model's Zarr is opened read-only. Deleting/refusing the fused model never touches a native original.

---

## 12. End-to-end example (binds it together)

> Ingest a SEG-Y seismic interpretation over a real UTM-zone-12 site:

1. **Source** `survey.sgy` copied verbatim → `raw/<sha>.sgy`; `Provenance.sources[0]` records `EPSG:32612`, original units, sha256.
2. Adapter (doc 03) parses → emits: one `ObservationSet` (`geometryKind:"traces"`), one `PropertyModel` (`property:"velocity_p"`, Zarr volume, Engineering coords via `Transform[crs_reproject + engineering_anchor]`), and two `GeologicalFeature` horizons (glTF surfaces).
3. Each gets a `Dataset` envelope sharing one provenance root; transforms logged + reversible.
4. PropertyModel lands as `<ds>.zarr/velocity_p` (+ `_sigma`, `_pyramid/`), attrs carry `canonicalUnit:"m/s"`, `colormap`, `scaling:"linear"`.
5. User builds a `FusedEarthModel`; the velocity model is resampled in as a `FusedLayer` (trilinear) — the native Zarr is untouched, lineage records the resample.
6. Viewer (doc 06) reads pyramid LOD bricks → ray-marches; cross-plot (doc 07) samples the fused grid; well planner (doc 09) intersects the trajectory with the velocity layer and reports value-along-path.

---

## Decisions locked in

1. **Four catalog record kinds** under one `Dataset` envelope: `observation`, `propertyModel`, `feature`, `fusedModel`. Every record has an `id` (ULID, kind-prefixed), a `SpatialFrame` binding, and a **mandatory** `Provenance`.
2. **Observations keep native acquisition geometry** (tagged union on `geometryKind`) and are **immutable**; gridding is always a separate, provenance-tracked `PropertyModel`.
3. **One physical property per `PropertyModel`**; support is one of `volume` / `grid2d` / `mesh`; display behavior (colormap/scaling/range) is registry-driven (doc 01 §5) and stored in per-array Zarr attrs.
4. **Uncertainty = co-registered per-cell 1σ array**, plus an optional **resolution kernel / DOI surface** (the "blurry vs noisy" complement); `null` means *unknown, not zero*; full covariance is explicitly out.
5. **Bulk arrays are Engineering-coordinate only** (doc 01 §2): CRS/datum never baked into Zarr/COG/vector files; a flag marks them non-geographic. Re-anchoring never reprocesses arrays.
6. **Zarr v3, sharded, `(t,z,y,x)` axis order, Z-up, NaN fill**, ~`64³` isotropic chunks, **OME-Zarr-style multiscale pyramid** (variance-correct for sigma arrays). 2D → COG; surfaces/solids → glTF(+VTK); lines → GeoJSON; points → LAZ/3D-Tiles.
7. **Fused grid default = regular voxel**, with native models **resampled in as referenced layers** and originals kept read-only/immutable; octree behavior delivered via the LOD pyramid, not an irregular topology.
8. **Provenance is a DAG** rooted at verbatim raw files; **every CRS/unit/datum transform is logged with invertible params** (doc 01 §7). Observations immutable; derived artifacts version-on-change with content-addressed bulk sharing.
9. **4D**: optional `TimeAxis` at the Dataset level, **leading time axis**, **explicit ISO-8601 UTC epochs** (irregular allowed); time-lapse vintages are separate datasets linked by a group tag.

### Cross-doc assumptions flagged
- **Doc 04 (storage/serving):** owns the store-root path, the catalog DB physical schema (assumed JSONB-doc-per-record + PostGIS extent index), chunk-cache, tile/slice/sample API, and GC mechanism. Field semantics here are the contract.
- **Doc 03 (ingestion):** adapters emit records conforming to these schemas; the `geometryKind`/`MethodKey` tables are the shared contract.
- **Doc 07 (fusion):** owns the resample engine internals; `FusedLayer.resampleOp` only records *what* was done.
- **Doc 06 (viewer) / 08 (plugins):** read `display` hints + property-type registry; plugins register new `PropertyTypeKey`s.

### Open questions for you
1. **Fused-grid default resolution policy** — auto (median native spacing, cell-count-clamped) vs always user-set vs fixed default (e.g. 25 m)? Drives memory + first-fusion UX.
2. **Versioning depth** — keep *all* historical versions of derived artifacts (full audit, more storage) vs keep-latest + provenance-only (lighter, lossy)?
3. **Categorical/lithology fields** — represent as a `PropertyModel` with a class-probability axis (`perCategory`) or as a distinct primitive? Affects fusion math and the registry.
