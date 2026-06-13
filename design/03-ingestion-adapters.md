# 03 — Ingestion Adapters & Normalization

> Parent: `OVERVIEW.md` §3 (the method→format→primitive table) and §10 row 3.
> Depends on: `01-spatial-framework.md` (CRS/datum transforms, units registry) and
> `02-data-model.md` (exact primitive schemas — referenced here, **not redefined**).
> This doc defines how *any* survey file becomes normalized primitives in the
> Engineering Frame. Every byte that enters the model passes through an adapter.

## Goals & requirements

- **One plugin contract** every survey method implements; new method = new adapter, zero core changes (the R&D requirement, OVERVIEW §4).
- **Parse is dumb, normalize is shared.** An adapter only knows its file format. All CRS reprojection, datum conversion, and unit canonicalization is delegated to doc 01's pipeline — adapters never call `pyproj` or hardcode units themselves.
- **Raw is sacred.** Original files are stored verbatim; every emitted primitive links back to the exact bytes it came from (provenance, OVERVIEW §2 "never destroyed").
- **Partial success over hard failure.** A malformed row or missing optional field degrades gracefully with a structured warning; only missing *mandatory* spatial/identity metadata blocks ingest.
- **Idempotent.** Re-ingesting the same file produces the same primitives; re-ingest is content-addressed, not duplicate-creating.

---

## 1. The contract: what an adapter is

An adapter is a Python class implementing the `IngestionAdapter` protocol. It does exactly two things: **declare what it can handle**, and **parse raw bytes into a `ParseResult`**. It does *not* touch storage, the catalog, or coordinate transforms — the pipeline (§7) does that.

```python
# backend/ingestion/base.py
from typing import Protocol, runtime_checkable, Sequence
from dataclasses import dataclass, field

@dataclass
class SourceRef:
    """Where a quantity's coordinates/units live BEFORE normalization.
    Consumed by doc 01's frame.to_engineering() + units.to_canonical()."""
    crs: str | None            # EPSG code, WKT2, or None if unknown/local
    vertical_datum: str | None # EPSG, "ellipsoidal", "local", or None
    horizontal_unit: str       # e.g. "m", "deg", "ft"
    z_convention: str          # "elevation_up" | "depth_below_surface" | "depth_below_datum" | "MD"

@dataclass
class ParseResult:
    observations:   list["RawObservation"]   = field(default_factory=list)
    property_models:list["RawPropertyModel"] = field(default_factory=list)
    features:       list["RawFeature"]       = field(default_factory=list)
    source: SourceRef                          = None   # default frame for all coords below
    units:  dict[str, str]                     = field(default_factory=dict)  # property -> source unit
    provenance: "Provenance"                   = None
    warnings: list["IngestWarning"]            = field(default_factory=list)

@runtime_checkable
class IngestionAdapter(Protocol):
    method: str                      # "gravity" | "ert" | "mt" | ... (OVERVIEW §3)
    name: str                        # unique id, e.g. "segy-reflection-v1"
    extensions: Sequence[str]        # [".sgy", ".segy"]
    media_types: Sequence[str]       # optional MIME hints

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence in [0,1] that this adapter handles the file.
        Cheap — read magic bytes / header only. Used by format detection (§7)."""

    def parse(self, source: "RawSource") -> ParseResult:
        """Full parse. May be slow → runs as a job (§7). Coords/units stay in the
        file's NATIVE crs/datum/units; declare them via ParseResult.source so the
        pipeline can normalize. Adapter does NOT reproject or convert units itself."""
```

`RawObservation` / `RawPropertyModel` / `RawFeature` are the **pre-normalization** twins of the doc 02 primitives (ObservationSet / PropertyModel / GeologicalFeature, OVERVIEW §2). They carry native-frame coordinates and native units. The pipeline transforms them in place into the canonical doc 02 primitives, where geometry is classified by the **frozen vocabulary** (doc 02 §3–4): observations by `geometryKind` ∈ `points | soundings | profile2d | traces | raster2d | wellcurve | tensor`, and property models by `support.kind` ∈ `volume | grid2d | section | mesh`. The exact field set of the canonical primitives (geometry encoding, uncertainty representation, time-axis encoding) is owned by doc 02 — this doc assumes those names/shapes and aligns to them in §10.

### Registration

A decorator registers adapters into a global registry at import; entry-points allow third-party plugins (OVERVIEW §4) without editing core.

```python
# backend/ingestion/registry.py
_REGISTRY: dict[str, IngestionAdapter] = {}

def register(adapter_cls):                 # @register on the class
    inst = adapter_cls()
    assert IngestionAdapter.__instancecheck__... # protocol conformance check
    _REGISTRY[inst.name] = inst
    return adapter_cls

def adapters_for(method=None, ext=None): ...        # lookup helpers
def detect(sample, filename) -> IngestionAdapter:   # highest sniff() score wins

# Plugins outside core register via setuptools entry-point group:
#   [project.entry-points."geosim.plugins"]
#   my_adapter = "mypkg.adapters:MyAdapter"
```

The same registry pattern is shared with the property-type registry (doc 01 §5) and the plugin architecture (doc 08), under the unified `geosim.plugins` entry-point group (doc 08).

---

## 2. Per-method adapters — the enforced contract

This is OVERVIEW §3's table made executable. Each row is one (or more) registered adapter(s). Method/submethod use the **canonical `MethodKey` registry** (doc 02 §2): subtypes (seismic reflection/refraction, EM tdem/fdem/aem, ERT/IP dc_resistivity/ip_time/ip_freq) go in the `submethod` field, never as new top-level method keys. "Emits → support" is the normalized primitive and the geometry it attaches to — observations classify by `geometryKind`, property models by `support.kind` (doc 02 §3–4). Parsing libraries are from the OVERVIEW §5 stack.

| Method (`method` / `submethod`) | Native formats | Parse library | Emits → geometry |
|---|---|---|---|
| **gravity** | CSV/columns, `.grd` (Surfer/GMT), netCDF, BGI | `pandas`, `xarray`, `rasterio` (`.grd`) | `Observation`(`points`: g, gravity_anomaly) → **stations**; if pre-gridded: `PropertyModel`(gravity_anomaly, support.kind=`grid2d`) → optionally inverted density → `volume` |
| **magnetics** | ASEG-GDF, CSV, `.grd`, netCDF | `pandas` + `aseg_gdf2`, `xarray` | `Observation`(TMI/magnetic_field, `points`) → **point set**; `PropertyModel`(`grid2d`); inverted susceptibility → `volume` |
| **ert** (submethod=`dc_resistivity`) | AGI `.stg`, Res2DInv `.dat`, UBC, ABEM `.amp` | custom text parsers; `pygimli` readers where available | `Observation`(apparent ρ + electrode geometry, `profile2d`) → **pseudosection**; inverted ρ → `PropertyModel` support.kind=`section` (native curtain, doc 02 §4) or `volume` |
| **ip** (submethod=`ip_time` / `ip_freq`) | AGI, UBC (paired with ERT) | same as ERT | `Observation`(`profile2d`) carrying chargeability_time_ms (`ip_time`) / phase_mrad or chargeability_mv_v (`ip_freq`) → **pseudosection**; `PropertyModel` → `section` / `volume` |
| **em** (submethod=`tdem` / `fdem` / `aem`) | ASEG-GDF, USF, `.xyz`, netCDF | `pandas` + `aseg_gdf2`; `xarray` | `Observation`(decay curves per sounding, `soundings`) → **soundings**; layered/CDI inversion → `PropertyModel`(conductivity) → **stitched `volume`** (§4) |
| **mt** | EDI (impedance tensor), ModEM/UBC inverted, `.j` | `mtpy` (EDI), custom ModEM/UBC readers | `Observation`(Z(f), tipper, app-ρ/phase curves, `tensor`) → **sites**; inverted → `PropertyModel`(resistivity) → `volume` |
| **seismic** (submethod=`reflection`) | SEG-Y, velocity cubes (SEG-Y/netCDF), horizon ASCII | `segyio`, `xarray`; horizons via `pandas` | `Observation`(`traces`, optional) → **survey geometry**; `PropertyModel`(velocity_p/amplitude) → `volume`; `GeologicalFeature`(horizons, faults) → **surfaces/sticks** |
| **seismic** (submethod=`refraction` / `tomography`) | SEG-Y (first breaks), Rayfract/`.tomo` grids | `segyio`, custom tomo readers | `PropertyModel`(velocity_p) → `section` (2D line) or `volume` |
| **microseismic** | QuakeML, CSV catalogs, NonLinLoc `.hyp` | `obspy` (QuakeML/`.hyp`), `pandas` (CSV) | `Observation`/`GeologicalFeature`(event cloud) → **4D point cloud** (x,y,z,t,mag) |
| **insar** | GeoTIFF time-series, `.unw`, CSV (PS points) | `rasterio` (raster), `pandas` (PS) | `PropertyModel`(deformation, support.kind=`grid2d`) → **raster time-series (4D)**; PS → `Observation`(`points`, 4D) |
| **welllog** | LAS 1.2/2.0/3.0, DLIS | `lasio` (LAS), `dlisio` (DLIS) | `Observation`(curves vs MD, `geometryKind:"wellcurve"`) **+** a `wellPath` `GeologicalFeature` joined by `wellId`; needs deviation survey (§5) for MD→XYZ |
| **heatflow** (temperature logs) | CSV, LAS (continuous T log) | `pandas`, `lasio` | `Observation`(temperature, canonical **kelvin**) → `points` (point T) or `wellcurve` (T-vs-depth log, joined to a `wellPath`) |
| **geology** (maps) | Shapefile, GeoJSON, GeoPackage, KML | `geopandas`, `fiona` | `GeologicalFeature`(contacts, faults, unit polygons) → **surface features / unit solids** (2.5D draped, §5) |
| **geochem** | CSV / LIMS exports, XLSX | `pandas`, `openpyxl` | `Observation`(sample assays, `points`) → **point set** (often at surface or at a well MD) |

**Notes that are contract, not commentary:**
- Methods that arrive *already inverted* (resistivity/velocity/density volumes/sections) emit a `PropertyModel` directly — the platform is integration-first (OVERVIEW §1); forward/inverse modeling is later (doc 10). Raw-only files emit `Observation`s and a later inversion plugin produces the `PropertyModel`.
- A single file may emit multiple primitive kinds (SEG-Y volume + horizon export; ERT raw `profile2d` + inverted `section`). Adapters return all of them in one `ParseResult`.
- Every emitted primitive declares its `property_type` (doc 01 §5 registry key) so units and colormaps resolve automatically. An unknown property type is a hard error surfaced at ingest (it must be registered first — doc 08).
- **Per-observation errors (doc 02 §3 convention).** Each measured value column carries a paired sigma column (`role:"sigma"`, `errorFor:"<value>"`, in the value's unit); tensors/traces declare an `methodData.errorModel` instead. When a source carries no errors, ingestion applies the per-property **default noise floor** from the registry and records that substitution in provenance — measured data is never silently treated as error-free.

---

## 3. Normalization rules (delegated, not reinvented)

The pipeline (§7) runs these **after** `parse()`, identically for every method. Adapters never do them.

**a. CRS + vertical reprojection → Engineering Frame.** For each coordinate array, call doc 01 §7:
`frame.to_engineering(points_xyz, src_crs=source.crs, src_vertical=source.vertical_datum)`.
The `SourceRef.z_convention` selects the vertical handling: `elevation_up` is canonical; `depth_below_surface` → `depth_to_elevation` using `surfaceModel`; `depth_below_datum` → negate; `MD` → resolve via the well deviation survey (`md_to_tvd` → elevation, doc 01 §4). If `source.crs is None` and the project is **local mode**, coordinates are assumed already-Engineering (identity). If `None` and the project is **georeferenced**, it's a validation error (§6).

**b. Units → canonical.** For each property array, `units.to_canonical(values, unit=ParseResult.units[prop], property_type=...)` (doc 01 §5, `pint`). Source unit retained in provenance. Missing source unit → §6 policy.

**c. Gridding scattered → continuous.** Point/line observations that must become a volume/grid go through the gridding step (kept separate from parsing — it's a modeling choice, not a format fact). Defaults:

| Input shape | Default method | Library | Notes |
|---|---|---|---|
| 2D scattered points (gravity/mag stations) | bias-corrected gridding (Green's-function / spline) | `verde` | anti-aliased, handles gaps; output COG/2D grid |
| 2D → needs geostatistics / uncertainty | ordinary kriging | `pykrige` / `gstatsim` | when a variogram + uncertainty surface is wanted |
| sparse / quick preview | IDW | `verde` / `scipy` | fallback, no uncertainty |
| 1D soundings (TEM/MT/CDI) → 3D | per-sounding 1D model, then 3D interpolation between sites | `verde` 3D / `scipy` | "stitched" conductivity-depth volume (§4) |

Gridding is **never** applied silently to make a `PropertyModel` out of raw obs unless the adapter/user requests it; raw stays raw (`Observation`). Gridding parameters (method, spacing, variogram, search radius) are recorded in provenance so the grid is reproducible.

**d. 1D/2D → 3D placement.** Soundings, profiles, and pseudosections are intrinsically lower-dimensional but live in 3D:
- **1D sounding** → a vertical column at its (x,y); the conductivity-depth model becomes voxels along Z at that column. Many columns → §4 stitching.
- **2D profile/section** (ERT, seismic 2D) → a vertical "curtain" following the survey line's polyline in plan, extruded down. The raw measurements stay an `Observation` (`geometryKind:"profile2d"`); an inverted curtain is a `PropertyModel` with the native **`SectionSupport`** (`support.kind:"section"`, doc 02 §4) — a 2D field embedded in 3D along the polyline, not forced into the voxel grid until the user resamples to the fused grid (OVERVIEW §2, fusion is non-destructive).
- **Well log** → curves sampled against MD become a `wellcurve` `Observation` (immutable measured record), while the borehole trajectory (MD→XYZ via deviation survey) is a separate `wellPath` `GeologicalFeature`; the two are joined by `wellId` (doc 02 §3, §5). There is no `well_path` *support* kind.

---

## 4. Stitching 1D soundings into a volume (EM/TEM/MT special case)

This recurs enough to standardize. Each AEM/TEM/MT sounding yields a 1D resistivity-vs-depth (or conductivity-depth) function at one (x,y). To form a `PropertyModel` 3D grid:

1. Resample each sounding onto the canonical Engineering Z axis (elevation, m).
2. Interpolate laterally between soundings (default `verde` spline in 2D per depth slice; kriging optional for uncertainty).
3. Mask below each sounding's depth-of-investigation (DOI) — beyond DOI the value is flagged low-confidence, surfaced as an uncertainty layer (OVERVIEW §6, never silently extrapolated).

The native soundings are **kept as `Observation`s**; the stitched volume is a derived `PropertyModel` whose provenance references all contributing soundings. Same pattern applies to scattered 1D temperature logs → 3D temperature field.

---

## 5. Geometry helpers adapters rely on

- **Deviation survey** (doc 01 §4): well-log and temperature adapters must locate a borehole's `(MD, inclination, azimuth)` table. If the LAS/DLIS lacks one, ingest emits a *vertical-well assumption* warning and treats MD=TVD below the wellhead until a deviation survey is supplied. Wellhead (x,y,elev) is mandatory for placement (§6).
- **Geology maps are 2.5D**: polygons/lines are planar in CRS; they're draped onto `surfaceModel` (doc 01 §6) to get Z, unless the file carries explicit Z (e.g. modeled horizon). Unit *solids* require a geomodel (GemPy, doc 05/07) — the map adapter emits surface features only and flags that solids are downstream.
- **Microseismic / InSAR carry time**: `t` is parsed into the primitive's time axis (OVERVIEW §2 4D). Time-axis encoding is fixed by doc 02 §1/§8 — a Dataset-level `TimeAxis` with a leading `t` array axis and **explicit ISO-8601 UTC epochs** (not project-epoch offsets).

---

## 6. Validation & error handling

Every ingest produces a structured **`IngestReport`** (stored, shown in UI): counts of primitives, list of `IngestWarning` (code, severity, locus), and a terminal status `{ok | ok_with_warnings | failed}`.

**Mandatory vs optional metadata:**

| Field | Mandatory? | If missing |
|---|---|---|
| Identity (file readable, method detectable) | **yes** | `failed` — cannot ingest |
| Property type registered | **yes** | `failed` — register property first (doc 08) |
| Horizontal coords | **yes** | `failed` |
| Source CRS | conditional | **georef project:** `failed` unless user supplies CRS at upload. **local project:** assume Engineering, warn |
| Vertical datum | no | default to project `verticalDatum`, warn |
| Units (per property) | no, but risky | use property-type canonical-unit *assumption*, emit **high-severity** warning (silent wrong-unit is the worst failure mode) |
| Deviation survey (wells) | no | vertical-well assumption, warn (§5) |
| Time axis (4D methods) | yes for 4D | `failed` for that primitive only; others proceed |
| Uncertainty | no | absent → null uncertainty; flagged so fusion knows |

**Partial-file policy:** parsing is per-record where possible. A bad CSV row / corrupt SEG-Y trace is skipped, counted, and reported; the rest ingests (`ok_with_warnings`). A corrupt *header* (can't establish geometry/units for the whole file) is `failed`. Threshold: if **>10%** of records drop, escalate to `failed` (fixed default, overridable per upload — DECISIONS, doc 03).

**CRS/units missing** never silently guesses a real-world position. Worst case it lands in local/Engineering space with a loud warning, and can be georeferenced later (doc 01 §2 promote path).

---

## 7. Ingestion pipeline stages

```
 upload ─▶ store-raw ─▶ detect ─▶ parse ─▶ normalize ─▶ write ─▶ register
 (bytes)   (verbatim    (sniff)   (job)    (CRS+units   (Zarr/   (catalog
            + hash)                         +grid §3)    COG/raw)  + provenance)
```

1. **Upload** — file lands via FastAPI chunked upload (OVERVIEW §4). User may attach CRS/datum/method hints here (used if detection/headers are ambiguous).
2. **Store-raw** — write verbatim to the raw store (OVERVIEW §5), compute a content hash (sha256). The hash is the idempotency key (§8).
3. **Detect** — `registry.detect(sample, filename)` runs `sniff()` over candidate adapters; highest score wins; ties / low scores surface a "choose adapter" prompt. User hint overrides.
4. **Parse** — `adapter.parse(raw)` → `ParseResult`. **This is the long-running step.** Runs as a background job (FastAPI `BackgroundTasks` for small files; Celery/RQ when needed — OVERVIEW §5). Job status streams to the client over WebSocket. Large SEG-Y / DLIS / raster time-series are the heavy cases.
5. **Normalize** — §3 rules: reproject, unit-convert, optional gridding, 1D/2D→3D placement. Also a job step (gridding can be expensive).
6. **Write** — persist canonical primitives to bulk stores per doc 02/04 conventions (3D→Zarr, 2D→COG, vectors→GeoJSON/glTF, point clouds→LAS/3D-Tiles). Adapter does not choose storage; the writer does, keyed by support geometry.
7. **Register** — insert catalog rows (dataset, primitives, bounding box, CRS, units, support, time range) + provenance edges + the `IngestReport`. The dataset becomes visible/layerable in the UI only after this commits (atomic — partial primitives never appear).

**Where jobs fit:** steps 4–6 are one job chain per uploaded file; the catalog holds a `Job` row (`queued→running→done|failed`) the UI polls/streams. Re-runnable on failure from the last good step.

---

## 8. Idempotency, re-ingest, provenance

- **Idempotency key = sha256(raw bytes) + adapter name + adapter version + normalization params**. Same key already in catalog → skip parse, return the existing dataset (no duplicates). This makes the synthetic generator's repeated runs (OVERVIEW §8) cheap and exercising the pipeline safe.
- **Re-ingest with new params** (e.g. different gridding, a now-supplied CRS) → new key → new derived primitives, **same raw file** (deduped by content hash). Per doc 02 §9: observations are immutable (a corrected re-import is a *new* dataset linked to the prior); derived artifacts version-on-change (`seq+1`, parent set) with content-addressed bulk sharing — all versions retained.
- **Provenance linkage** (OVERVIEW §2, doc 01 §7): every primitive stores a `Provenance` edge to (a) the raw file hash + path, (b) adapter name+version, (c) source CRS/datum/units before normalization, (d) normalization params (gridding method, variogram, etc.). This makes every transform auditable and lets a re-projection later replay from raw. Doc 02 §7 owns the `Provenance` schema (`SourceFile` / `Step` / reversible `Transform`); this doc only populates it.

---

## 9. Adapter authoring checklist (for new methods — the R&D path)

A new survey method is added by one file:
1. Subclass/implement `IngestionAdapter`; set `method`, `name`, `extensions`, `media_types`.
2. Implement `sniff()` (cheap header check) and `parse()` (emit native-frame primitives + `SourceRef` + `units` + `provenance` + `warnings`).
3. Register the method's `property_type`(s) in the property registry (doc 01 §5 / doc 08) if new.
4. `@register` (or declare an entry-point for an out-of-tree plugin).
5. Add a format round-trip unit test (OVERVIEW §11 verification): known file → expected primitives/units/bbox.

No core, storage, viewer, or transform code changes — that's the contract OVERVIEW §4 promises.

---

## 10. Decisions locked in

1. **One `IngestionAdapter` protocol**; `parse(raw) → ParseResult{observations, propertyModels, features, source, units, provenance, warnings}`. Adapters declare native format only; they never reproject or unit-convert.
2. **Parse / normalize split.** All CRS, datum, and unit handling is delegated to doc 01's pipeline; all storage choice to the writer. Adapters are pure format readers.
3. **Per-method library mapping is fixed** (§2): `segyio`, `lasio`/`dlisio`, `obspy`, `mtpy`, `pygimli` readers, `rasterio`/`xarray`, `geopandas`, `pandas`+`verde`/`pykrige`. New methods add rows, don't change the framework.
4. **Raw stays raw**: observations are never auto-gridded into property models; gridding is an explicit, provenance-recorded modeling step (`verde` default, kriging when uncertainty wanted, IDW fallback).
5. **1D soundings / 2D sections are placed in 3D as columns/curtains** and only resampled to the fused grid non-destructively; AEM/TEM/MT stitching is the standard 1D→3D pattern with DOI masking.
6. **Idempotency = content hash + adapter version + normalization params**; raw files deduped by sha256; provenance links every primitive back to exact raw bytes.
7. **Fail loud on ambiguous geometry/units, degrade on partial data.** Missing units → canonical-unit assumption with high-severity warning; missing CRS in a georef project → `failed`; bad records skipped and counted; **>10%** of records dropped escalates the whole file to `failed` (fixed default, overridable per upload).
8. **Registration via decorator + setuptools entry-points** under the shared `geosim.plugins` entry-point group (doc 08), enabling out-of-tree adapter plugins.

### Resolved decisions

- **Partial-file failure threshold** — fixed at **>10%** of records dropped → `failed`, overridable per upload (DECISIONS, doc 03).
- **Pre-inverted vs raw** — files that arrive already inverted emit a `PropertyModel`; raw-only files emit `Observation`s and wait for an inversion plugin. Gridding raw → volume is always a separate, user-initiated, provenance-recorded step (raw stays raw).
- **Plugin entry-point group** — unified `geosim.plugins` (doc 08), not `geosim.adapters`.

---

## Dependencies on doc 02 (data model) — RESOLVED ✓

Doc 02 is final; the names this doc assumed now bind to it:
- **Primitives:** adapters emit `ObservationSet` / `PropertyModel` / `GeologicalFeature` (doc 02 §3–5). The pre-normalization `Raw*` twins map onto these.
- **Support / geometry vocabulary:** observations classify by **`geometryKind`** ∈ `points | soundings | profile2d | traces | raster2d | wellcurve | tensor` (doc 02 §3); property-model **`support.kind`** ∈ `volume | grid2d | section | mesh` (doc 02 §4, where `section`/`SectionSupport` is the native form for ERT and 2D-seismic vertical curtains). Use these exact tokens — the earlier ad-hoc list (point/line/section-string/well_path/…) is superseded.
- **Uncertainty:** per-observation **paired sigma columns** (`role:"sigma"`, `errorFor:"<value>"`) on tables, `methodData.errorModel` for tensors/traces, default noise floor in provenance when absent (doc 02 §3); for property models a co-registered per-cell **1σ** array `<property>_sigma` (+ optional DOI/kernel), doc 02 §6. Adapters that ingest already-inverted models pass σ through when present.
- **Time-axis encoding:** `TimeAxis` at the Dataset level, **leading `t` axis**, **explicit ISO-8601 UTC epochs** (doc 02 §1, §8) — not project-epoch offsets.
- **Provenance:** this doc populates the doc 02 §7 `Provenance` DAG — `SourceFile` (raw sha256, format, originalCrs/Unit), `Step` (op/params/code), and **reversible `Transform`** records for every CRS/unit/datum change.
- **Versioning / re-ingest:** doc 02 §9 — observations **immutable**; a corrected re-import is a *new* dataset with provenance linking to the prior (not an in-place supersede).
