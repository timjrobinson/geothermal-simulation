# Ingestion — raw files to primitives

> **What you'll learn / why it matters.** Every survey method speaks its own file
> format — a gravity survey is a CSV of points, a seismic survey is a binary SEG-Y cube,
> a well log is a LAS text file. Before any of them can be *compared*, they must be parsed,
> dragged into one coordinate system, one set of units, and one shared data model. This
> page is the **import layer**: the contract every file reader obeys, the pipeline that
> turns bytes into the three [normalized primitives](data-model.md), and the careful rules
> that keep raw data raw while still letting you build continuous fields from scattered
> points. If you have ever written an ETL pipeline (extract → transform → load), you
> already have the right mental model — this is ETL for the subsurface, with provenance and
> scientific honesty bolted on.

Ingestion is the boundary between the messy outside world (real industry file formats,
arbitrary coordinate systems, arbitrary units) and the clean inside world (one
[Engineering Frame](spatial-framework.md), canonical units, three primitive types). Think
of it as the **parser + deserializer + validator** at the edge of a system: untrusted,
heterogeneous input comes in; trusted, uniform, queryable objects come out — and a
complete audit trail (**[provenance](glossary.md)** — the record of *where every number
came from*) is stamped on everything.

---

## 1. The mental model: a typed data pipeline

Every file, regardless of method, flows through the **same seven-stage pipeline**. Only
the very first parsing step is method-specific; everything after it is shared.

```
 upload ─▶ store-raw ─▶ detect ─▶ parse ─▶ normalize ─▶ write ─▶ register
 (bytes)   (verbatim    (sniff)   (job,    (CRS+units    (Zarr/   (catalog
            + sha256)              format-  +placement    COG/raw)  + provenance)
                                   specific) — SHARED)
```

The single most important design rule, repeated everywhere below, is:

!!! abstract "The headline rule: *parse is dumb, normalize is shared*"
    A method's file reader (the **adapter**) knows **only its file format**. It does
    **not** reproject coordinates, it does **not** convert units, it does **not** choose
    storage, and it does **not** touch the database. It reads bytes and emits *native*
    values tagged with *what frame and units they are in*. **All** the reconciliation —
    coordinate-system math, unit conversion, placing 1-D/2-D data into 3-D — happens
    afterwards in **one shared normalizer** used by every method.

For a programmer the payoff is obvious. If every adapter did its own coordinate math, you
would have a dozen subtly different, subtly buggy reprojection implementations. Instead the
hard, error-prone logic lives in exactly one place (the [spatial framework](spatial-framework.md)
and the units registry), is tested once, and every new method inherits it for free. Adding
a new survey method is *one file* — a new adapter — and **zero changes to core**.

---

## 2. The `IngestionAdapter` contract

An **adapter** is the plugin every survey method implements. It is a Python class
(structurally, a `Protocol` — Python's version of an interface/trait) that does exactly two
things: **declare what it can handle**, and **parse raw bytes into a `ParseResult`**.

```python
# backend/geosim/ingestion/base.py
@runtime_checkable
class IngestionAdapter(Protocol):
    method: str                # canonical method key, e.g. "gravity" | "ert" | "mt"
    submethod: str | None      # subtype, e.g. "reflection" (seismic) — never a new top method
    name: str                  # unique adapter id, e.g. "gravity-csv-v1"
    version: str               # adapter version — part of the idempotency key (§7)
    extensions: Sequence[str]  # [".csv", ".txt"]
    media_types: Sequence[str] # optional MIME hints

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence in [0,1] that THIS adapter handles the file.
        Cheap — read magic bytes / the header only."""

    def parse(self, source: RawSource) -> ParseResult:
        """Full parse. Coords/units stay NATIVE; declare them via ParseResult.source
        so the shared pipeline can normalize. The adapter does NOT reproject or
        unit-convert itself."""
```

- **`sniff`** is content-based format detection, like the Unix `file` command or a magic-byte
  check. It peeks at the first chunk of bytes (and the filename) and returns a confidence
  score in $[0,1]$. The detector runs `sniff` over every registered adapter and the
  **highest score wins**; a tie or an all-low result prompts the user to choose.
- **`parse`** does the full read. It can be slow (a multi-gigabyte SEG-Y cube), so it runs as
  a background **job** (see [storage & serving](architecture.md) for the job machinery).

### What `parse` returns: the `ParseResult`

`parse` emits a `ParseResult` — a bundle of *pre-normalization* objects plus the metadata
the normalizer needs to finish the job.

```python
@dataclass
class ParseResult:
    observations:    list[RawObservation]    # raw measured points/curves/traces
    property_models: list[RawPropertyModel]  # already-inverted/gridded fields
    features:        list[RawFeature]        # discrete interpreted shapes
    source:  SourceRef            # the NATIVE crs/datum/units of all coords below
    units:   dict[str, str]       # property_type -> source unit, e.g. {"gravity_anomaly": "mGal"}
    provenance: Provenance        # what the adapter knows about where this came from
    warnings: list[IngestWarning] # structured, non-fatal problems
    records_total: int            # rows seen — feeds the >10%-bad-rows rule (§6)
    records_dropped: int          # rows skipped
```

The three list fields map onto the three primitives of the [data model](data-model.md):

| `Raw*` twin | Becomes the primitive | Plain meaning |
|---|---|---|
| `RawObservation` | **Observation** | what was measured, where (raw, immutable) |
| `RawPropertyModel` | **Property Model** | a continuous field of one physical property |
| `RawFeature` | **Geological Feature** | a discrete interpreted shape (fault, well path) |

The `Raw*` classes are the **pre-normalization twins** of the real primitives: same shape,
but their coordinates are still in the file's native frame and their values are still in the
file's native units. The normalizer (§5) rewrites them *in place* into the canonical doc-02
primitives.

### The crucial `SourceRef`: "where do these numbers live?"

The adapter cannot reproject — but it *must* tell the pipeline how to. It does that with a
`SourceRef`: a small descriptor of the native frame and units.

```python
@dataclass
class SourceRef:
    crs: str | None            # EPSG code / WKT2 / None if unknown or local
    vertical_datum: str | None # EPSG / "ellipsoidal" / "local" / None
    horizontal_unit: str       # "m" | "deg" | "ft"
    z_convention: str          # how the Z column is meant:
                               #   "elevation_up" | "depth_below_surface"
                               #   | "depth_below_datum" | "MD"
```

A **CRS** (Coordinate Reference System) is the recipe that says what a coordinate pair
*means* on the real Earth — e.g. `EPSG:32612` is "UTM zone 12N, metres." A **vertical
datum** is the analogous recipe for the Z axis — what "zero elevation" means. The
`z_convention` flag is subtle and important: the same number `1500` could mean *1500 m
above sea level*, *1500 m below the ground surface*, or *1500 m of drilled cable
(measured depth) down a crooked borehole*. The adapter declares which; the normalizer
resolves it. (See [coordinates, depth & units](spatial-framework.md) for the full story.)

### Registration — adapters are plugins

Adapters register themselves into a global registry at import time via a decorator, and
third-party adapters can register out-of-tree through Python setuptools entry-points under
the shared `geosim.plugins` group. This is what makes "new method = one file, zero core
changes" literally true.

```python
@adapter            # ← registers GravityCsvAdapter into the registry
class GravityCsvAdapter:
    method = "gravity"
    name   = "gravity-csv-v1"
    ...
```

---

## 3. A real adapter, annotated: gravity CSV

The cleanest worked example in the codebase is the gravity-station CSV reader
(`backend/geosim/ingestion/adapters/gravity_csv.py`). **Gravity** here is the survey method
that measures tiny variations in the Earth's gravitational pull at the surface; denser rock
pulls fractionally harder, so the **gravity anomaly** (the measured pull minus the
predicted "boring Earth" pull) is a proxy for buried density. See
[gravity & magnetics](survey-methods/potential-fields.md) for the physics — here we only
care about the *file*.

### The native format

A gravity CSV is about as simple as a survey file gets: a header row of column names, then
one row per measurement station, with optional `#`-comment metadata lines.

```csv
# crs: EPSG:32612        (1) source coordinate reference system
# unit: mGal             (2) the gravity value's native unit
x,y,z,gravity_anomaly,sigma          (3) header: aliases accepted (easting/northing/elev/…)
587230.0,4509110.0,1432.5,-12.4,0.3  (4) one station: easting, northing, elevation, value, 1σ
587280.0,4509110.0,1431.9,-11.8,0.3
587330.0,4509110.0,1430.2, -9.6,0.3
587230.0,4509160.0,1433.1,-13.1,0.4
```

1. `# crs:` — a comment line the adapter scans to discover the source CRS. UTM zone 12N
   here, so coordinates are eastings/northings in metres.
2. `# unit:` — the native unit of the value column. A **milligal (mGal)** is the standard
   gravity unit ($1\ \text{mGal} = 10^{-5}\ \text{m/s}^2$); the canonical unit the platform
   stores in may differ, which is exactly why declaring it matters.
3. The header is matched case-insensitively against alias sets, so `easting`/`east`/`x` all
   map to the X column, `gravity`/`ga`/`mgal`/`anomaly` all map to the value column, etc.
4. Each data row is one **station**: a single point with a measured value (and, here, a
   per-reading **$\sigma$** — the one-standard-deviation uncertainty of the measurement).

### What the adapter does (and pointedly does *not* do)

```python
def parse(self, source: RawSource) -> ParseResult:
    text  = (source.data or b"").decode("utf-8", errors="replace")
    meta  = _scan_comments(text)                 # pulls "# crs:" / "# unit:" lines
    rows  = list(csv.reader(io.StringIO(_strip_comments(text))))
    # ... locate x / y / z / gravity / sigma columns by alias ...
    for n, raw in enumerate(rows[1:], start=2):
        total += 1
        try:
            x = float(raw[idx["x"]]); y = float(raw[idx["y"]]); g = float(raw[idx["g"]])
            ...
        except (ValueError, IndexError):
            dropped += 1                          # a bad row is SKIPPED and COUNTED…
            warnings.append(IngestWarning("bad_row", Severity.LOW, f"unparseable row {n}", f"row {n}"))
            continue                              # …not a crash (§6 partial-success policy)
        # accumulate xs, ys, zs, gs, sg ...
    obs = RawObservation(
        geometry_kind="points",                  # doc-02 geometry vocabulary
        coords=np.column_stack([xs, ys, zs]),    # NATIVE frame — NOT reprojected
        values={"gravity_anomaly": np.asarray(gs)},
        sigma ={"gravity_anomaly": np.asarray(sg)},
        primary_property="gravity_anomaly",
    )
    return ParseResult(
        observations=[obs],
        source=SourceRef(crs=meta.get("crs"), horizontal_unit="m",
                         z_convention="elevation_up"),
        units={"gravity_anomaly": meta.get("unit", "mGal")},  # declare native unit, don't convert
        records_total=total, records_dropped=dropped,
    )
```

Note what is **absent**: no call to `pyproj`, no `* 1e-5` unit math, no database write. The
coordinates leave the adapter still in UTM metres; the value leaves still in mGal. The
adapter's *only* contribution to the eventual transform is the declarative `SourceRef` and
`units` dict. That is "parse is dumb" in code.

### The normalized result

After the shared normalizer (§5) runs, this file becomes a single **Observation** with
`geometryKind: "points"` (see the [data model](data-model.md) for the geometry vocabulary):
coordinates reprojected into the Engineering Frame (metres, X-East / Y-North / Z-Up),
`gravity_anomaly` converted to its canonical unit, a co-registered $\sigma$ column, and a
full provenance edge back to the raw bytes. The raw points are now a first-class,
queryable, immutable record — but still just *points*, not a field. Turning them into a
field is a separate, deliberate step (§4).

---

## 4. Gridding: scattered points → a continuous field (a modeling choice)

A gravity survey gives you a few hundred *points*. To compare it with, say, a resistivity
*volume* you eventually need it as a continuous **field** — a value at every location, not
just at the stations. Producing that field is called **gridding** (also "interpolation" or,
in 3-D, "stitching"). This is where the platform draws a bright line:

!!! warning "Raw stays raw — gridding is never silent"
    Gridding is a **modeling choice**, not a fact about the file. Interpolating between
    stations *invents* values where you never measured. So the pipeline **never**
    auto-converts an Observation into a Property Model. The raw points remain an immutable
    Observation; gridding is a **separate, user-initiated, parameter-recorded** step that
    produces a **new derived Property Model**. The originals are never mutated.

A CS analogy: the raw Observation is your source data; a gridded field is a *materialized
view* computed from it with explicit parameters. You keep both, and the view records the
exact query that built it so it is reproducible.

### Why three different gridders?

There is no single "correct" interpolation — the right method depends on the data shape and
whether you need an uncertainty estimate. The pipeline ships the doc-03 default table
(`backend/geosim/ingestion/gridding.py`):

| Input shape | Default method | Library | Gives uncertainty? |
|---|---|---|---|
| 2-D scattered points (gravity/mag stations, geochem) | bias-corrected **spline** (Green's-function) | `verde` | yes (prediction-variance proxy) |
| 2-D, geostatistics wanted | ordinary **kriging** | `pykrige`/`gstatsim` | yes (variogram-based) |
| sparse / quick preview | **IDW** (inverse-distance weighting) | `verde`/`scipy` | no native σ (uses a default noise floor) |
| 1-D soundings (TEM/MT/CDI) → 3-D | per-sounding 1-D model, then lateral interpolation | `verde`/`scipy` | yes |

**IDW** is the simplest to understand: a grid cell's value is the average of nearby data
points, weighted by $1/d^{\,p}$ where $d$ is distance and $p$ the power (default 2). It is
fast but carries no real uncertainty. **Spline** (the default) fits a smooth bias-corrected
surface and synthesizes an honest, distance-growing $\sigma$. **Kriging** is the
geostatistical gold standard that gives a principled variance — used when you specifically
want an uncertainty surface.

### Footprint honesty in gridding

A gridder must never paint values where there is no data to support them. Every gridder
masks cells beyond a search radius (`max_distance`) to **NaN** (not-a-number, the explicit
"no data" sentinel) — never zero, never a nearest-edge bleed:

```python
# from grid_points_2d() — cells farther than max_distance from any station become NaN
if params.max_distance is not None:
    outside = dist > params.max_distance
    grid[outside]  = np.nan
    sigma[outside] = np.nan
```

For 1-D soundings stitched into a 3-D volume, the same honesty applies in depth: each
sounding has a **depth-of-investigation (DOI)** — the depth below which it simply cannot
see — and cells below the local DOI are masked to NaN. This NaN-not-zero discipline is the
same principle that governs [fusion](fusion.md); it is worth internalizing now.

Every gridding run records its parameters (method, spacing, search radius, region) in
provenance, so the derived field is exactly reproducible — and it writes a `*.provenance.json`
sidecar next to the Zarr output for good measure.

---

## 5. Normalization: the shared transform stage

After `parse`, the `ParseResult` enters the **one normalizer every method shares**
(`backend/geosim/ingestion/normalize.py`). It runs three sub-stages, all delegated to the
spatial framework and units registry — never reinvented per method.

### (a) Coordinates → the Engineering Frame

For each coordinate array, resolve the vertical convention, then reproject:

- `elevation_up` is already canonical — pass through.
- `depth_below_datum` → negate the Z (down becomes negative elevation).
- `depth_below_surface` → resolve against the surface model (with a flat surface this is
  also a negate, plus a low-severity warning to supply a real surface to drape onto).
- `MD` (measured depth, drilled cable length) → needs a well **deviation survey** to
  resolve to true elevation; flagged as medium severity, treated as a vertical well until
  the survey arrives.

Then the horizontal reprojection is delegated to `frame.to_engineering(pts, src_crs=…)`.
The mode of the project matters: in a **local-mode** project (no real-world georeferencing)
coordinates are assumed already-Engineering and a stray source CRS is ignored with a
warning; in a **georeferenced** project a missing source CRS is a **hard error** — the
platform refuses to guess a real-world position.

### (b) Units → canonical

For each value array, `to_canonical(values, source_unit, property_type)` converts to the
property type's canonical unit (the conversion is `pint`-backed in the units registry). The
*source* unit is retained in provenance. Two honesty rules bite here:

- An **unregistered property type** is a hard error — you cannot ingest a quantity the
  platform has never heard of; register it first.
- A **missing source unit** is the most dangerous case (a silently-wrong unit is the worst
  failure mode of the whole system). The normalizer falls back to the canonical-unit
  *assumption* but emits a **high-severity** warning recorded in provenance — loud, never
  silent.

```python
if source_unit is None:
    warnings.append(IngestWarning(
        "missing_unit", Severity.HIGH,
        f"no source unit for {property_type!r}; assuming canonical {pt.canonical_unit!r} "
        "(silent wrong-unit is the worst failure mode, doc 03 §6)",
        f"property:{property_type}"))
```

Uncertainty is handled here too: if a value column arrives with no paired $\sigma$, the
normalizer applies the property type's **default noise floor** (a relative $\sigma$ from the
registry) and records the substitution — measured data is **never** silently treated as
error-free.

### (c) 1-D / 2-D → 3-D placement

Lower-dimensional data must still live in a 3-D world, but is *not* force-fit into the voxel
grid at ingest:

- **1-D sounding** → a vertical column at its $(x,y)$.
- **2-D profile / section** (ERT, 2-D seismic) → kept as the native **`section`** support —
  a vertical "curtain" following the survey line — *not* rasterized into voxels until the
  user explicitly resamples it during [fusion](fusion.md).
- **Well log** → curves vs. measured depth become a `wellcurve` Observation, joined by
  `wellId` to a separate `wellPath` Geological Feature.

The raw measurements always stay raw; placement only *positions* them in 3-D.

---

## 6. Validation, warnings, and the >10%-bad-rows policy

Ingestion never throws away problems silently. Every run produces a structured
**`IngestReport`** — primitive counts, a list of `IngestWarning`s (each with a code,
severity, and locus like `"row 41"`), the drop ratio, and a terminal status of
`ok | ok_with_warnings | failed`.

The guiding principle is **partial success over hard failure**:

| Situation | Outcome |
|---|---|
| One unparseable CSV row / one corrupt trace | skipped, counted, warned; the rest ingests |
| A corrupt **header** (can't establish geometry/units for the file) | `failed` |
| File unreadable / method undetectable | `failed` |
| Property type not registered | `failed` |
| Missing CRS in a georeferenced project | `failed` |
| Missing unit | `ok_with_warnings` (high-severity warning) |
| **More than 10% of records dropped** | escalated to **`failed`** |

That last row is the **>10%-bad-rows rule**. A few bad rows is normal field data; losing a
tenth of the file means something is structurally wrong and you should not trust the import.
The threshold is a fixed default, overridable per upload:

```python
# from IngestReport.finalize()  (backend/geosim/ingestion/base.py)
if self.records_total > 0 and self.drop_ratio > thr:   # thr defaults to 0.10
    self.status = IngestStatus.FAILED
    self.message = (f"{self.records_dropped}/{self.records_total} records dropped "
                    f"({self.drop_ratio:.0%} > {thr:.0%}) — escalated to failed")
```

---

## 7. Idempotency and provenance

### Idempotent re-ingest via content hash

Re-importing the same file should **not** create a duplicate dataset. The pipeline is
content-addressed: the raw bytes are stored under their `sha256` (so identical bytes
de-duplicate automatically), and the **idempotency key** is

$$
\text{key} = \mathrm{sha256}\big(\text{raw bytes}\big)
            \;\Vert\; \text{adapter.name}
            \;\Vert\; \text{adapter.version}
            \;\Vert\; \text{normalization params}.
$$

If that exact key already exists in the catalog, the pipeline skips the parse entirely and
returns the existing dataset with `reused=True`. Re-ingesting with *different* parameters
(say, a finer gridding spacing, or a now-supplied CRS) yields a new key → new derived
primitives, but still the **same raw file** (shared by content hash). This is what makes the
synthetic generator's repeated runs cheap and makes re-running the pipeline safe.

```python
def idempotency_key(sha256, adapter_name, adapter_version, params) -> str:
    payload = json.dumps({"sha256": sha256, "adapter": adapter_name,
                          "version": adapter_version, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

### Provenance: every number is traceable

Every emitted primitive carries a **provenance** edge recording (a) the raw file hash and
path, (b) the adapter name and version, (c) the source CRS / datum / units *before*
normalization, and (d) the normalization parameters (gridding method, variogram, search
radius, …). This makes every transform auditable and lets a future re-projection *replay
from the raw bytes*. Observations are immutable: a corrected re-import is a **new** dataset
linked to the prior one, never an in-place overwrite. (The full provenance schema lives in
the [data model](data-model.md).)

---

## 8. End-to-end: a gravity CSV becomes an Observation + a gridded Property Model

Putting it all together, here is the full journey of the example file from §3:

1. **Upload** — the CSV bytes land; the user may attach a CRS or method hint.
2. **Store-raw** — written verbatim to `raw/<sha256>/<name>`; the `sha256` is computed.
3. **Detect** — `GravityCsvAdapter.sniff()` sees `x,y,gravity_anomaly` columns and returns
   `0.9`; it wins.
4. **Parse** — the adapter emits one `RawObservation(geometry_kind="points")` with native
   UTM coords and mGal values, plus `SourceRef(crs="EPSG:32612")` and
   `units={"gravity_anomaly": "mGal"}`. Bad rows are counted, not fatal.
5. **Normalize** — coords reprojected into the Engineering Frame; mGal → canonical unit;
   the points stay an Observation (no auto-gridding).
6. **Write + Register** — the Observation is persisted, a catalog row is created, the
   provenance edge is stamped, and the `IngestReport` is stored. The dataset becomes
   layerable in the viewer.
7. **(Optional, later) Grid** — the user calls `grid_points_2d("gravity_anomaly", …)` with
   explicit parameters. This produces a **new derived Property Model** (`support: grid2d`)
   with a co-registered $\sigma$ array and NaN beyond the survey footprint — reproducible
   from its recorded provenance, and entirely separate from the immutable raw stations.

The output of step 6 is the immutable evidence; the output of step 7 is a deliberate,
reproducible model built from it. That separation — and the provenance that ties them
together — is the whole point of the ingestion layer.

---

## Key takeaways

- **Parse is dumb, normalize is shared.** Adapters read *only* their file format and emit
  native values tagged with a `SourceRef` + `units`; all CRS, datum, and unit reconciliation
  happens once, in one shared normalizer.
- **One pipeline, seven stages:** upload → store-raw → detect → parse → normalize → write →
  register. Only `parse` is method-specific.
- **The three primitives** (Observation, Property Model, Geological Feature) are the only
  things ingestion produces; `Raw*` twins carry native coords/units until normalization.
- **Raw stays raw.** Scattered points are never silently gridded; gridding is a separate,
  user-initiated, parameter-recorded step that produces a *new* derived Property Model.
- **Footprint honesty** starts at ingest: gridders mask beyond coverage / DOI to **NaN**,
  never zero, never extrapolate.
- **Partial success over hard failure:** bad rows are skipped and counted; >10% dropped
  escalates the whole file to `failed`. Missing units warn loudly; missing CRS in a
  georeferenced project fails hard.
- **Idempotent + provenanced:** content-hash + adapter version + params is the idempotency
  key; every primitive traces back to the exact raw bytes it came from.

## Where this lives in the code

| Concern | Module |
|---|---|
| Adapter contract, `ParseResult`, `Raw*` twins, `IngestReport` | `backend/geosim/ingestion/base.py` |
| The seven-stage orchestrator + idempotency key | `backend/geosim/ingestion/pipeline.py` |
| Format detection / adapter registry | `backend/geosim/ingestion/registry.py` |
| Shared normalizer (CRS + units + placement) | `backend/geosim/ingestion/normalize.py` |
| Gridding / sounding-stitching (scattered → field) | `backend/geosim/ingestion/gridding.py` |
| Per-method adapters (worked example: gravity CSV) | `backend/geosim/ingestion/adapters/` |
| Persisting primitives + catalog rows | `backend/geosim/ingestion/writer.py` |

Next: once every method's data is normalized into one frame and one set of units, the
[fusion engine](fusion.md) resamples them all onto a single shared 3-D grid so you can
finally compare them cell-by-cell.
