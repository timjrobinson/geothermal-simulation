# 04 — Storage & Serving

> Parent: `OVERVIEW.md` §10 row 4, §5. This doc defines **where bytes live on disk**
> (catalog DB + chunked-array store + raw store) and **how the FastAPI layer serves
> them to the 3D viewer** (chunks, slices, samples, queries, jobs).
>
> **Cross-doc contract:** the *logical* schemas for `Observation`, `PropertyModel`,
> `Feature`, and `FusedGrid` are owned by **doc 02** (parallel). This doc references
> their primitive names per `OVERVIEW.md §2` and stores their **catalog rows + bulk
> arrays**; it does not redefine their internal field semantics. **Coordinates,
> CRS, units, and Engineering-Frame conventions come from doc 01** — every stored
> array is in Engineering coordinates (doc 01 §1, decision #2). Things this doc
> *needs from doc 02* are flagged inline with **[NEEDS-02]**.

> ### ⚠️ Revision — user decisions applied (see `DECISIONS.md`)
> Two drafted defaults below were **overridden by the user** and are authoritative:
> 1. **Catalog DB = PostgreSQL + PostGIS from the start** (not SQLite+SpatiaLite).
>    The SQLAlchemy portability layer and Engineering-metre R-Tree/GiST bbox index
>    still stand; we simply target PostGIS as the primary engine. A project is still
>    a self-contained directory for its *array/raw/cache* stores, but catalog rows
>    live in a Postgres database (one schema or DB per project) — export/zip bundles
>    the array stores + a `pg_dump`. SQLite remains an optional lightweight path.
> 2. **Async jobs = RQ + Redis from the start** (not FastAPI BackgroundTasks).
>    The job contract (table + endpoints + WS) is unchanged; the executor is RQ
>    workers against Redis from day one, giving crash-isolation and parallel ingest.
> Resolved secondary items: **slice default = raw float32 to client** (GPU transfer
> function); **artifact versioning = keep all versions** (content-addressed bulk).
> Where prose below still says "SQLite default" / "BackgroundTasks default," read it
> as the documented *fallback/portability* path, not the chosen default.

> ### 🔗 Reconciliation with doc 02 (now final — these resolve every `[NEEDS-02]`)
> The data model is locked; this doc aligns to it:
> - **IDs are ULIDs**, kind-prefixed (`ds_ pm_ obs_ feat_ fem_ prov_ well_ ver_ run_`) —
>   *not* UUIDv7. (Doc 02 §1/§9 owns identity.)
> - **`Dataset.kind` ∈** `observation | propertyModel | feature | fusedModel` (doc 02 §2).
> - **`PropertyModel.support.kind` ∈** `volume | grid2d | mesh` (doc 02 §4) — the
>   `property_models.support` column uses these, not `regular/octree/unstructured`.
> - **Observation classification = `geometryKind` ∈** `points | soundings | profile2d |
>   traces | raster2d | wellcurve | tensor` (doc 02 §3) — replaces the ad-hoc `obs_type`.
> - **`feature_type` = `featureKind` ∈** `surface | fault | unitSolid | wellPath |
>   pointCloud | fractureNetwork | polyline` (doc 02 §5).
> - **Uncertainty array** = a co-registered sibling Zarr array named `<property>_sigma`
>   (+ optional `<property>_doi`), variance-correct pyramid (doc 02 §6, §10.3).
> - **Open `meta_json` / `props_json` columns** map to doc 02's `methodData` /
>   `attributes` blobs — open JSONB, no schema owed.
> - **Directory layout in §3 is authoritative** over doc 02 §10.1's sketch; dataset
>   files are named by their ULID id (which already encodes kind).

---

## 0. Design constraints (inherited)

- **Local-first, single-user**, grows to hosted/multi-user (`OVERVIEW.md` §1, §5).
- Browser viewer streams **multiresolution Zarr bricks → GPU 3D textures**, ray-marched (`OVERVIEW.md` §5, §7).
- Arrays are **always Engineering-frame** (doc 01 #2) → georeferencing never touches bulk bytes.
- Three primitives + fused grid (`OVERVIEW.md` §2); **observations immutable**, property models/fused derived, features vector.
- 4D (time) is first-class (`OVERVIEW.md` §2).

---

## 1. Storage tiers — the three stores

> **This document is the authoritative API + storage contract** (the catalog
> physical schema, the on-disk store layout, brick/slice/sample/query endpoints,
> content-addressing, GC). Other docs (02 logical schema, 03 ingestion, 06 viewer,
> 07 fusion, 09 drilling) **reference this doc** for those concerns; where this doc
> and another disagree on a *physical* detail, this doc wins. Logical field
> semantics remain owned by doc 02.

| Tier | What | Tech | Why |
|---|---|---|---|
| **Catalog DB** | metadata, geometry index, provenance, layer defs, job state | **PostgreSQL + PostGIS** (the choice) — SQLite+SpatiaLite optional lightweight fallback | relational, spatially queryable, transactional, concurrent |
| **Array store** | bulk N-D arrays + 2D rasters + meshes + vectors + point clouds | **Zarr** (3D/4D), **COG** (2D), glTF/VTK (mesh), GeoJSON (vector), LAZ/3D-Tiles (points) | chunked, lazy, web-streamable, format-appropriate |
| **Raw store** | original survey files, verbatim | filesystem, content-addressed | provenance, re-ingest, audit (`OVERVIEW.md` §5) |

The catalog DB is the **index/source-of-truth for metadata**; it never holds bulk
samples — only pointers (URIs) into the array/raw stores plus bounding boxes,
shapes, units, and stats. This keeps the DB tiny and the heavy bytes
memory-mappable / range-servable.

---

## 2. Catalog DB

### 2.1 Engine choice — PostgreSQL + PostGIS (the choice)

| | **PostgreSQL + PostGIS** *(the choice)* | **SQLite + SpatiaLite** *(optional fallback)* |
|---|---|---|
| Setup | a service to run (Docker/local) | zero — a file in the project dir |
| Concurrency | many writers; survives parallel RQ ingest | single-writer |
| Spatial | full GiST + rich 3D GIS (`box3d`, `geometry(...,3D)`) | R-Tree + SpatiaLite functions |
| JSON | `jsonb` (indexable, rich) | `JSON1` (basic) |
| Fit | **the primary engine, from the start** | embedded demo / portability only |

**Decision:** ship **PostgreSQL + PostGIS from the start** (`DECISIONS.md` doc-04).
It is the catalog engine for both local-first and hosted use — chosen because RQ
workers (§9.4) ingest in parallel and need a real multi-writer transactional store
with a GiST 3D bbox index. One Postgres **schema (or database) per project** holds
that project's catalog rows; the project *directory* (§3) holds only the bulk
array/raw/cache stores. Access via **SQLAlchemy** with a thin spatial helper layer.

> **Portability note:** all schema DDL and hot-path queries target SQLAlchemy core
> over the intersection of PG and SQLite semantics, and bbox indexing uses the
> portable index pattern in §2.5, so an embedded **SQLite + SpatiaLite** build
> remains an optional lightweight fallback (single-file demo / no-service path) —
> not the default. PostGIS-specific SQL is confined behind a capability flag.

### 2.2 Bounding boxes are in **Engineering metres** (doc 01)

Every spatial row stores its bbox/geometry in **Engineering-frame metres** (doc
01 §1), *not* lat/lon. This is what the viewer queries in (the viewer lives in the
Engineering frame), and it's anchor-independent so georeferencing a project never
rewrites the index. A *derived* lat/lon bbox can be computed on demand from the
`SpatialFrame` for map-extent display.

### 2.3 Logical (doc 02) → Physical (this doc) mapping table

Doc 02 §2 owns the **logical `Dataset`**; this table pins every required logical
field to its **physical home** (a column, a JSONB path, or a derived value), so the
two docs cannot drift. `meta_json` is the open `methodData`/`attributes` blob.

| doc-02 `Dataset` field | doc-02 req | Physical home (this doc) |
|---|---|---|
| `id` | ✓ | `datasets.id` (TEXT ULID PK) |
| `projectId` | ✓ | `datasets.project_id` FK |
| `name` | ✓ | `datasets.name` |
| `kind` | ✓ | `datasets.kind` (`observation\|propertyModel\|feature\|fusedModel`) |
| `method` | ✓ | `datasets.method` (canonical MethodKey) |
| `submethod` | optional | `datasets.submethod` (canonical subtype, doc 02 §2) |
| `extent` (Aabb) | ✓ | `datasets.extent_json` (Engineering m) **+** GiST/R-Tree bbox index (§2.5) |
| `time` (TimeAxis) | optional | `datasets.time_extent_json` (null ⇒ static) |
| `spatialFrameId` | ✓ | `datasets.spatial_frame_id` FK → `spatial_frame.project_id` |
| `originCrs` | optional | `datasets.origin_crs` (mirror of provenance roots) |
| `provenanceId` | ✓ | `datasets.provenance_id` FK → `provenance.id` (**NOT NULL**) |
| `version` (VersionInfo) | ✓ | `datasets.version_root_id` / `version_seq` / `version_parent_id` (doc 02 §9) |
| `tags` | ✓ | `datasets.tags_json` (string[]) |
| `createdAt` | ✓ | `datasets.created_at` (epoch-ms) |
| `createdBy` | ✓ | `datasets.created_by` |
| `payload` | ✓ | the typed child row: `property_models` / `observations` / `features` / `fused_models`(+`fused_layers`), joined on `dataset_id` |

> The four `payload` kinds become the four typed child tables below; `meta_json` /
> `props_json` / `style_json` carry doc 02's open `methodData` / `attributes` /
> `display` blobs verbatim (no schema owed — the R&D-plugin requirement).

### 2.4 Schema

ID convention: `TEXT` **ULID** (time-ordered, sortable, kind-prefixed per doc 02 §1:
`ds_ pm_ obs_ feat_ fem_ ...`) PKs. `created_at`/
`updated_at` epoch-ms on every table. JSON columns hold open/extensible metadata
(the R&D-plugin requirement, `OVERVIEW.md` §4) so new survey methods add fields
without migrations.

```sql
-- ───────────────────────── projects ─────────────────────────
CREATE TABLE projects (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  description   TEXT,
  storage_root  TEXT NOT NULL,        -- abs/relative path to project dir (§3)
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ── spatial_frame: 1:1 with project; the doc-01 SpatialFrame, serialized ──
-- Owned conceptually by doc 01; persisted here. JSON mirrors doc 01 §2 object.
CREATE TABLE spatial_frame (
  project_id    TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
  mode          TEXT NOT NULL,        -- 'georeferenced' | 'local'
  horizontal_crs TEXT,                -- EPSG / WKT2, null in local
  vertical_datum TEXT,
  anchor_json   TEXT,                 -- {easting,northing,elevation} | null
  rotation_deg  REAL NOT NULL DEFAULT 0,
  axis_convention TEXT NOT NULL DEFAULT 'ENU',
  length_unit   TEXT NOT NULL DEFAULT 'm',
  roi_json      TEXT NOT NULL,        -- {xmin,xmax,ymin,ymax} Engineering m
  depth_range_json TEXT NOT NULL,     -- {zmin,zmax} Engineering elev m
  surface_model TEXT,                 -- 'dem:...' | 'flat:0' | 'synthetic:...'
  frame_json    TEXT NOT NULL         -- full doc-01 SpatialFrame blob (canonical)
);

-- ───────────────────────── datasets ─────────────────────────
-- A dataset = one ingested survey product (a method's output bundle).
-- Groups the primitives produced by one ingestion run (OVERVIEW §3 adapter).
CREATE TABLE datasets (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,
  method        TEXT NOT NULL,        -- canonical MethodKey (doc 02 §2): 'gravity'|'ert'|'mt'|'seismic'|'insar'|...
  submethod     TEXT,                 -- canonical subtype under method (doc 02 §2), e.g. 'reflection'
  kind          TEXT NOT NULL,        -- 'observation'|'propertyModel'|'feature'|'fusedModel' (doc 02 §2)
  status        TEXT NOT NULL,        -- 'ingesting'|'ready'|'error'
  extent_json   TEXT NOT NULL,        -- {xmin..zmax} Engineering m — doc-02 Dataset.extent (spatial index source §2.5)
  time_extent_json TEXT,              -- {t0,t1,n} for 4D, null for static (doc-02 Dataset.time)
  spatial_frame_id TEXT NOT NULL REFERENCES spatial_frame(project_id),  -- doc-01 frame this is expressed in (doc 02)
  origin_crs    TEXT,                 -- CRS source arrived in, pre-reprojection (mirror of provenance)
  provenance_id TEXT NOT NULL REFERENCES provenance(id),  -- doc 02: EVERY dataset has exactly one provenance
  version_root_id   TEXT NOT NULL,    -- doc-02 VersionInfo.rootId — stable identity across versions (§9 doc 02)
  version_seq       INTEGER NOT NULL DEFAULT 1,  -- VersionInfo.seq
  version_parent_id TEXT,             -- VersionInfo.parent (null for v1)
  tags_json     TEXT,                 -- doc-02 Dataset.tags (string[])
  meta_json     TEXT,                 -- method-specific blob (doc 02 methodData/acquisition)
  created_by    TEXT NOT NULL,        -- doc-02 Dataset.createdBy ('system:fusion'|'system:synthetic'|user)
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);
-- NOTE: provenance_id is NOT NULL — there is no dataset without provenance (doc 02 §7).
-- (provenance.id is created first within the same ingest/fusion transaction.)

-- ─────────────────── property_models (3D/4D continuous fields) ───────────────────
-- Catalog row + array pointer for a doc-02 PropertyModel. The doc-02 schema
-- defines support geometry / uncertainty semantics; we store the array + index.
CREATE TABLE property_models (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  property      TEXT NOT NULL,        -- 'resistivity'|'density'|'velocity'|... (doc 01 §5 registry)
  canonical_unit TEXT NOT NULL,       -- from doc 01 units registry
  support       TEXT NOT NULL,        -- 'volume'|'grid2d'|'mesh' (doc 02 §4 support.kind)
  store_uri     TEXT NOT NULL,        -- relative path to .zarr group (§4)
  store_format  TEXT NOT NULL DEFAULT 'zarr',
  shape_json    TEXT NOT NULL,        -- [nz,ny,nx] or [nt,nz,ny,nx]
  spacing_json  TEXT,                 -- voxel size m, regular grids
  origin_json   TEXT,                 -- grid origin, Engineering m
  bbox_json     TEXT NOT NULL,        -- {xmin..zmax} Engineering m (index source)
  has_time      INTEGER NOT NULL DEFAULT 0,
  pyramid_levels INTEGER NOT NULL DEFAULT 1,  -- # multiresolution levels (§5)
  stats_json    TEXT,                 -- {min,max,mean,p1,p99,histogram} for transfer fn
  uncertainty_uri TEXT,               -- sibling zarr array '<property>_sigma' (doc 02 §6)
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ─────────────────── fused_models (the FusedEarthModel CONTAINER grid) ───────────────────
-- doc 02 §11: a FusedEarthModel is NOT a property_models row — it is a CONTAINER
-- (a regular-voxel grid + a TimeAxis) into which native PropertyModels are
-- RESAMPLED as referenced layers (fused_layers). The grid itself carries no single
-- property. (A FAVORABILITY volume is, by contrast, an ordinary property_models row
-- with property='favorability' — not modelled here.) A project may hold several.
CREATE TABLE fused_models (
  id            TEXT PRIMARY KEY,     -- 'fem_...'
  dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,  -- kind='fusedModel'
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  grid_type     TEXT NOT NULL DEFAULT 'regular_voxel',  -- doc 02 §11 (default)
  store_uri     TEXT NOT NULL,        -- relative path to the fused .zarr group (one array per layer + sigma)
  store_format  TEXT NOT NULL DEFAULT 'zarr',
  shape_json    TEXT NOT NULL,        -- [nz,ny,nx] or [nt,nz,ny,nx] — the VolumeSupport (doc 02 §4)
  spacing_json  TEXT NOT NULL,        -- voxel size m
  origin_json   TEXT NOT NULL,        -- grid origin, Engineering m
  bbox_json     TEXT NOT NULL,        -- {xmin..zmax} Engineering m (index source)
  has_time      INTEGER NOT NULL DEFAULT 0,
  time_extent_json TEXT,              -- TimeAxis if any resampled layer is 4D (doc 02 §11)
  pyramid_levels INTEGER NOT NULL DEFAULT 1,
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ── fused_layers: each native PropertyModel resampled INTO the fused grid (doc 02 §11) ──
-- Originals are NEVER overwritten — a layer references its source pm id + pinned version.
CREATE TABLE fused_layers (
  id                       TEXT PRIMARY KEY,  -- layerId
  fused_model_id           TEXT NOT NULL REFERENCES fused_models(id) ON DELETE CASCADE,
  source_property_model_id TEXT NOT NULL REFERENCES property_models(id),  -- native original (read-only)
  source_version           TEXT NOT NULL,     -- pinned doc-02 version resampled from (provenance)
  property                 TEXT NOT NULL,     -- PropertyTypeKey of this layer (doc 01 §5 registry)
  resample_op_json         TEXT NOT NULL,     -- {method:'trilinear'|'nearest'|'conservative'|'kriging'|'idw', params} (doc 07)
  sigma_array              TEXT,              -- path to resampled 1σ inside the fused Zarr group (doc 02 §11)
  valid_mask               TEXT,              -- path to coverage mask: which cells this layer informs
  created_at    INTEGER NOT NULL
);

-- ─────────────────── observations (raw measured survey data) ───────────────────
-- Immutable (OVERVIEW §2). Geometry can be points/lines; bulk values may be
-- inline (small) or in an array file (e.g. SEG-Y traces, log curves).
CREATE TABLE observations (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  geometry_kind TEXT NOT NULL,        -- 'points'|'soundings'|'profile2d'|'traces'|'raster2d'|'wellcurve'|'tensor' (doc 02 §3)
  primary_property TEXT,              -- main measured PropertyTypeKey (null for raw traces/tensors)
  geometry_wkb  BLOB,                 -- Engineering-frame geometry (small)
  values_uri    TEXT,                 -- array file if bulk (else inline)
  values_json   TEXT,                 -- inline values for small obs
  bbox_json     TEXT NOT NULL,
  acquired_at   INTEGER,              -- time of measurement (4D)
  meta_json     TEXT,                 -- doc 02 methodData/acquisition blob
  created_at    INTEGER NOT NULL
);

-- ─────────────────── features (vector geological interpretation) ───────────────────
CREATE TABLE features (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT REFERENCES datasets(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  feature_type  TEXT NOT NULL,        -- 'surface'|'fault'|'unitSolid'|'wellPath'|'pointCloud'|'fractureNetwork'|'polyline' (doc 02 §5 featureKind)
  store_uri     TEXT,                 -- glTF/VTK mesh, GeoJSON, or LAZ (§4)
  store_format  TEXT NOT NULL,        -- 'gltf'|'vtk'|'geojson'|'laz'|'3dtiles'
  geometry_wkb  BLOB,                 -- simplified geom for picking/index (optional)
  bbox_json     TEXT NOT NULL,
  has_time      INTEGER NOT NULL DEFAULT 0,
  props_json    TEXT,                 -- per-feature attributes (doc 02 §5 AttributeSpec + detail)
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ─────────────────── provenance (lineage DAG) ───────────────────
-- Edges: this artifact was DERIVED FROM these inputs by this process.
-- Covers ingest (raw→primitive) and fusion/transform (primitives→derived).
CREATE TABLE provenance (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  target_kind   TEXT NOT NULL,        -- 'dataset'|'propertyModel'|'feature'|'observation' (doc 02 §2 kinds)
  target_id     TEXT NOT NULL,
  process       TEXT NOT NULL,        -- 'ingest:ert-stg'|'fuse:resample'|'transform:rockphys'|...
  process_version TEXT,
  params_json   TEXT,                 -- args (CRS used, units, kernel, etc.)
  source_crs    TEXT,                 -- original CRS before doc-01 reprojection
  source_unit   TEXT,                 -- original unit before doc-01 canonicalization
  raw_file_id   TEXT REFERENCES raw_files(id),  -- if from a raw upload
  created_at    INTEGER NOT NULL
);
CREATE TABLE provenance_inputs (      -- many inputs per provenance record (DAG)
  provenance_id TEXT NOT NULL REFERENCES provenance(id) ON DELETE CASCADE,
  input_kind    TEXT NOT NULL,
  input_id      TEXT NOT NULL,
  PRIMARY KEY (provenance_id, input_kind, input_id)
);

-- ─────────────────── raw_files (raw store index) ───────────────────
CREATE TABLE raw_files (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  filename      TEXT NOT NULL,        -- original name
  rel_path      TEXT NOT NULL,        -- raw/<sha256>/<filename> (§3)
  sha256        TEXT NOT NULL,        -- content address / dedupe
  bytes         INTEGER NOT NULL,
  media_type    TEXT,                 -- detected format
  uploaded_at   INTEGER NOT NULL
);

-- ─────────────────── layers / views (viewer presentation state) ───────────────────
-- A layer = a renderable binding of one artifact + display settings.
CREATE TABLE layers (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,
  source_kind   TEXT NOT NULL,        -- 'propertyModel'|'feature'|'observation'|'fusedModel' (doc 02 §2 kinds)
  source_id     TEXT NOT NULL,
  render_type   TEXT NOT NULL,        -- 'volume'|'slice'|'isosurface'|'mesh'|'points'|'tubes'
  visible       INTEGER NOT NULL DEFAULT 1,
  z_index       INTEGER NOT NULL DEFAULT 0,
  style_json    TEXT,                 -- colourmap, transfer fn, opacity, iso value, clip box
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);
-- A view = a saved camera + layer-set + time snapshot (shareable scene state).
CREATE TABLE views (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,
  camera_json   TEXT,                 -- pose, target, fov
  layer_set_json TEXT,               -- [{layer_id, overrides}]
  time_json     TEXT,                 -- current t for 4D
  created_at    INTEGER NOT NULL
);

-- ─────────────────── jobs (async work) ───────────────────
CREATE TABLE jobs (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL,        -- 'ingest'|'fuse'|'isosurface'|'transform'|'pyramid'|'export'|'import'|'gc'
  status        TEXT NOT NULL,        -- 'queued'|'running'|'succeeded'|'failed'|'cancelled'
  progress      REAL NOT NULL DEFAULT 0,  -- 0..1
  message       TEXT,                 -- latest human-readable status
  params_json   TEXT NOT NULL,        -- inputs
  result_json   TEXT,                 -- produced artifact ids on success
  error_json    TEXT,
  created_at    INTEGER NOT NULL,
  started_at    INTEGER,
  finished_at   INTEGER
);
```

### 2.5 Spatial indexing of bounding boxes (portable)

Hot-path spatial queries ("which artifacts intersect this clip box / this
slice?") use a 3D bbox index. On the primary engine this is a **PostGIS GiST
index** on a `box3d` / `geometry(...,3D)` column built from `extent_json`; the
portable bbox-table pattern below is the SQLite-fallback mirror so the same
SQLAlchemy code path runs on both:

```sql
-- Portable bbox index (PostGIS GiST primary; SQLite R-Tree virtual table mirror).
CREATE VIRTUAL TABLE artifact_rtree USING rtree(
  rowid,                 -- maps to artifact via artifact_rtree_map
  xmin, xmax, ymin, ymax, zmin, zmax   -- Engineering metres
);
CREATE TABLE artifact_rtree_map (
  rowid INTEGER PRIMARY KEY,
  kind  TEXT NOT NULL,   -- 'propertyModel'|'feature'|'observation'|'fusedModel' (doc 02 §2 kinds)
  id    TEXT NOT NULL
);
```

On PostgreSQL a bbox-intersection query hits the GiST 3D index directly; on the
SQLite fallback it reads `artifact_rtree`, joins the map, then the typed table.
The SQLAlchemy spatial helper picks the backend. The 3D bbox (incl. `zmin/zmax`)
means depth-clipping the scene also prunes candidates.

---

## 3. On-disk project layout

The **project directory holds the bulk stores** (`arrays/ grids/ meshes/ vectors/
points/ raw/ cache/`); the **catalog lives in PostgreSQL** (one schema/DB per
project, §2.1), *not* in a file under this directory. The directory + a `pg_dump`
of the project's catalog rows together form the portable **project bundle** (§3.1)
— that bundle, not "copy one folder," is the copy/share/backup unit.

```
<storage_root>/<project_id>/        # bulk stores only — catalog is in PostgreSQL (§2.1)
├── frame.json                     # cached SpatialFrame (doc 01), DB is canonical
├── arrays/                        # Zarr volumes (property models + fused)
│   ├── pm_<id>.zarr/              #   one zarr group per property model — layout owned by doc 02 §10.2
│   │   ├── zarr.json              #     v3 group metadata (multiscales spec, frame ref)
│   │   ├── <property>/0/ 1/ 2/ ...#     per-property multiscale subgroup; 0 = full res (doc 02 §10.2)
│   │   └── <property>_sigma/      #     co-registered 1σ sibling array (doc 02 §6, §10.2)
│   └── fem_<id>.zarr/             #   fused-model volumes — SAME layout (§7), doc 02 §10.2
├── grids/                         # 2D rasters → COG
│   └── g_<id>.tif                 #   surface grids, anomaly maps, InSAR scenes
├── meshes/                        # surfaces & solids
│   ├── m_<id>.glb                 #   horizons, faults, unit solids (glTF)
│   └── m_<id>.vtu                 #   unstructured/VTK when richer attrs needed
├── vectors/                       # lightweight vector features
│   └── f_<id>.geojson             #   well paths, outlines, fracture traces
├── points/                        # point clouds
│   └── pc_<id>.laz                #   microseismic; → 3D-Tiles tileset when large
├── raw/                           # raw store — verbatim originals (content-addressed)
│   └── <sha256>/<original_name>   #   immutable; provenance.raw_file_id → here
└── cache/                         # derived/transient (safe to delete; §8)
    ├── slices/  isosurfaces/  tiles/
```

**Format choice per primitive** (`OVERVIEW.md` §5):

| Primitive / data | Format | Notes |
|---|---|---|
| 3D/4D volumes (property models, fused) | **Zarr** (v3) | chunked, lazy, pyramided, HTTP-streamable (§4–5) |
| 2D grids/rasters (surface, anomaly, InSAR) | **COG** | range-readable tiled GeoTIFF; Engineering grid + frame georef |
| Surfaces / solids / fault meshes | **glTF (.glb)** default; **VTK (.vtu)** when unstructured + rich attrs | glTF streams straight to Three.js |
| Lightweight vectors (well paths, traces) | **GeoJSON** | Engineering coords; small |
| Point clouds (microseismic) | **LAZ** small; **3D Tiles** when large | 4D via time attribute → time-filtered fetch |
| Raw originals | as-uploaded | content-addressed, never mutated |

### 3.1 Export / Import — the project bundle

Because the catalog now lives in PostgreSQL (not in a file inside the project
dir), "copy/share/backup a project" is **not** "zip one folder." A **project
bundle** is the two halves packaged together:

```
<project_id>.bundle/   (or a .tar/.zip of it)
├── stores/            # the project directory: arrays/ grids/ meshes/ vectors/ points/ raw/ cache?/
│                      #   (cache/ is omitted by default — fully derivable, §8)
├── catalog.sql        # pg_dump of THIS project's catalog rows (its schema/DB, §2.1)
└── bundle.json        # manifest: project_id, schema/format version, content hashes, created_at
```

- **Export:** `pg_dump` the project's catalog schema/rows → `catalog.sql`, copy
  the bulk `stores/` (skip `cache/`), write `bundle.json` (records the array/raw
  content hashes from §8.1 for integrity). Exposed as an `export` job (§9.4) →
  `GET /projects/{pid}/export` streams the bundle.
- **Import:** `pg_restore`/`psql` `catalog.sql` into a fresh per-project schema,
  drop the bulk `stores/` under a new `<storage_root>/<project_id>/`,
  rewrite `projects.storage_root`, verify content hashes. Exposed as
  `POST /projects:import` (multipart bundle) → `{job_id}`.
- **Copy / share / backup** all go through this bundle. A backup is a periodic
  bundle export; sharing is sending the bundle; "duplicate project" is
  export-then-import with a new `project_id`. The bulk halves are
  content-addressed (§8.1), so de-dup and incremental backup are natural.

---

## 4. Zarr volume store (the core bulk format)

### 4.1 Group/array structure — **owned by doc 02 §10.2**

The on-disk Zarr group layout is **authoritatively specified in doc 02 §10.2** —
this doc does **not** restate a different layout. The older doc-04 sketch
(`pm_<id>.zarr/0/1/2/...`) is **superseded** by doc 02 §10.2 and removed.

Per doc 02 §10.2, a PropertyModel (or fused model) is **one Zarr v3 sharded
group** whose members are **per-property multiscale subgroups**; the canonical
on-disk path to a chunk is:

```
<datasetId>.zarr/<property>/<level>/c/<bz>/<by>/<bx>
```

with sibling arrays `<property>_sigma` (1σ), `<property>_classes` (categorical
probabilities), and `<property>_doi` (DOI surface), all per doc 02 §10.2. The
key conventions this doc's storage + serving rely on — all defined there:

- **Axis order** `[z,y,x]` (3D) / `[t,z,y,x]` (4D), Z-up, x fastest (doc 02 §10.2).
- **`origin` + `spacing` in array attrs, Engineering metres, REQUIRED** for regular
  grids (never CRS coords — doc 01 #2); explicit CF coord arrays only for irregular.
- **`fill_value` = NaN**; masked/outside-DOI cells are NaN, never 0.
- **OME-Zarr `multiscales`** block per property subgroup describes the pyramid so
  any Zarr reader (Python xarray *and* the JS client) discovers levels the same way.
- **dtype:** `float32` canonical for rendering; `int`/`uint` for hard-label
  categorical arrays (with a `categories` attr table, doc 02 §10.2).

This doc's brick addressing (§6) maps **1:1** onto the doc-02 chunk path
`<property>/<level>/c/<bz>/<by>/<bx>`.

### 4.2 Chunking strategy — chunks **are** bricks

The chunk *is* the unit we stream to the GPU. **Cubic chunks** so a chunk maps
directly to a 3D-texture **brick** for ray-marching (`OVERVIEW.md` §5, §7):

| Param | Default | Rationale |
|---|---|---|
| Chunk shape (3D) | **64³** voxels | 64³×float32 = **1 MiB** — good HTTP payload, one brick upload |
| Chunk shape (4D) | **1 × 64³** (time outermost, size 1) | fetch one timestep's brick without pulling the series |
| Compression | **Blosc(zstd, level 3, shuffle)** | fast decode in browser; ~3–8× on smooth geophysics fields |
| Brick = chunk | 1:1 | chunk address = brick address (§6) — no server re-tiling |

A 64³ float32 brick is 1 MiB raw / typically 150–350 KiB compressed — sized so
the viewer pulls dozens in parallel over HTTP/2 and uploads each as a
`Data3DTexture` sub-brick with no server-side stitching.

### 4.3 Compression

**Blosc meta-codec with the zstd codec, shuffle filter** as default (great
ratio + very fast multithreaded decode; a WASM blosc/zstd decoder runs in the
browser worker). `zstd` level is tunable per dataset; lossless throughout (we
never lossy-compress measured geophysics). Quantized display copies *may* use a
coarser dtype but the canonical array stays lossless.

---

## 5. Multiresolution pyramid (LOD)

Every volume is stored as a **power-of-two pyramid** so the viewer streams
coarse→fine (`OVERVIEW.md` §5 "octree LOD").

- **Levels:** `0` = full res; each level halves each spatial axis (×⅛ voxels),
  built by **mean** downsampling (block-average; preserves field values) for
  continuous properties, **mode/nearest** for labelled volumes. Time axis is not
  downsampled by default.
- **Depth:** build until the coarsest level fits in **≤ 1–2 chunks** (i.e. a
  ~64³ thumbnail of the whole volume). `pyramid_levels` recorded in the catalog.
- **Build:** a `pyramid` job (§9) on ingest/fusion; cheap, parallel per level.
- **Same layout for fused volumes** — derived grids get pyramids too (§7), so
  the viewer treats ingested and fused identically.

**Why pyramids and not a true sparse octree (for now):** dense per-level pyramids
are dead-simple to build, address, and stream over plain HTTP, and they match the
viewer's brick-upload model. A sparse octree (skip empty bricks) is a later
optimization for very large/sparse volumes — flagged in open questions. The
addressing scheme (§6) is octree-compatible so we can upgrade without changing
the client contract.

---

## 6. Brick / tile addressing scheme

A brick is addressed by **`(artifact_id, property, level, t, bz, by, bx)`**:

| Field | Meaning |
|---|---|
| `artifact_id` | property-model / fused-model dataset id (`<datasetId>.zarr`) |
| `property` | the property subgroup name (doc 02 §10.2) — e.g. `resistivity`, `resistivity_sigma` |
| `level` | pyramid level (0 = finest) |
| `t` | time index (0 for static) |
| `bz,by,bx` | chunk/brick indices within that level |

This maps **1:1 onto the doc-02 §10.2 chunk path
`<property>/<level>/c/<bz>/<by>/<bx>`** inside `<datasetId>.zarr` — so **brick
address == Zarr chunk path** and the server serves a brick by mapping the URL
straight to a chunk object (or proxying the chunk store). It is also
octree-compatible: a node `(level, bz,by,bx)` has children
`(level-1, 2b{x,y,z}+{0,1})`, so a future sparse octree reuses the same URLs.

**LOD flow (viewer ↔ server):**
1. Viewer fetches group metadata → learns shape, chunks, levels, stats, frame.
2. Renders the **coarsest level** immediately (1–2 bricks) for instant context.
3. As the camera/clip-box settles, computes which finer bricks are **visible &
   within the screen-space error budget**, requests them by address, uploads
   each into the `Data3DTexture` brick atlas, refines.
4. Bricks are cached client-side (and server-side, §8); evicted by LOD/visibility.

Detailed screen-space-error LOD math is **doc 06**'s concern; this doc guarantees
the **addressing + transport contract** that makes it possible.

---

## 7. Derived / fused volumes — stored & served identically

Two distinct things get the identical Zarr layout (§4–5) and serving path, and
must not be conflated (doc 02 §11):

**(a) A `FusedEarthModel` — a CONTAINER grid, *not* a property model.** It is a
regular-voxel grid into which native PropertyModels are **resampled as referenced
layers**; the grid carries no single property. It is written as an
`fem_<id>.zarr` group (one array per layer + sigma, doc 02 §10.2) and gets:
- a `datasets` row with `kind='fusedModel'` (a project may hold several),
- a **`fused_models`** row (the container grid/time) **+ one `fused_layers` row
  per resampled native property** (§2.4) — *not* a `property_models` row,
- a `provenance` record linking it to its input artifacts (the fusion DAG),
- pyramids built by the same `pyramid` job.

**(b) A favorability / rock-physics derived volume — an ordinary PropertyModel.**
A favorability volume (`OVERVIEW.md` §6) is a single-property field, so it is an
ordinary **`property_models`** row (`property='favorability'`) with its own Zarr
group — exactly like any ingested volume — even when it was computed *from* a
fused grid. The fused grid is the container; a favorability volume is a product.

**Consequence:** the **same `/bricks`, `/slice`, `/sample`, `/isosurface`
endpoints serve fused-model layers and favorability volumes alike** with zero
special-casing — each layer/array is just a `<property>` subgroup (doc 02 §10.2)
in a Zarr volume. Storage & serving stay uniform; only the catalog rows differ.

---

## 8. Caching & derived-on-demand

| Cache | Where | Key | Eviction |
|---|---|---|---|
| **Zarr chunks (HTTP)** | server `cache/tiles/` + browser | brick address + ETag | LRU by size; immutable content |
| **Slices** | `cache/slices/` | `(artifact,level,plane,position,t)` hash | LRU; cheap to recompute |
| **Isosurfaces** | `cache/isosurfaces/` | `(artifact,level,isovalue,t)` hash | LRU |
| **Stats/histograms** | catalog `stats_json` | computed once at ingest | persistent |
| **Pyramids** | `arrays/.../<level>/` | persistent | rebuilt only on source change |

- **Immutable chunks** ⇒ aggressive HTTP caching: `Cache-Control: immutable` +
  content-hash ETags. Editing a volume writes a new artifact id (versioning),
  never mutates a served chunk.
- `cache/` is **fully derivable** and safe to delete — a cache-clear just forces
  recompute. Never holds source-of-truth.
- Slices/isosurfaces are computed on first request and memoized; this is why the
  same endpoint shape works whether the result is cached or freshly computed.

### 8.1 Content-addressing & garbage collection

"**Keep all versions cheaply**" (`DECISIONS.md`; doc 02 §9) rests on a precise
content-addressing scheme — versions share unchanged bytes instead of copying:

| Object | Address | Immutability |
|---|---|---|
| **Raw file** | whole-file **sha256** → `raw/<sha256>/<name>` (§3) | immutable; re-upload of identical bytes de-dups |
| **Zarr chunk** | content hash of the chunk object, exposed as its **ETag** | immutable; written once, never mutated in place |
| **Artifact version** | a **manifest hash** = hash over (array/group metadata + the ordered list of its chunk **content references**) | a version *is* its manifest hash; identical content ⇒ identical hash |

- **Versioning is structural sharing.** A new version (re-fuse, edit, re-anchor,
  doc 02 §9) writes only the chunks whose content changed and a **new manifest**
  referencing both new and unchanged chunks. Unchanged chunks are referenced, not
  duplicated — so "keep all versions" costs only the delta, not a full copy.
- **Chunks are immutable + content-ETagged**, which is exactly what makes the §8
  HTTP caching (`Cache-Control: immutable`) and the S3+CDN path (§10) safe.
- **Garbage collection = mark-and-sweep from the catalog.** *Mark:* walk every
  live `datasets`/`property_models`/`fused_models`/`fused_layers`/`raw_files` row
  (and every retained version) and collect the manifest hashes + chunk + raw
  `sha256` references they reach. *Sweep:* any chunk object or `raw/<sha256>/`
  file **not** in the live set is unreferenced and deletable. This is the concrete
  mechanism doc 02 §9.5 delegates here ("deletable when no live version
  references its `sha256`"). GC runs as a `gc` job (§9.4); `cache/` is exempt
  (fully derivable, deleted freely).
- **A bundle export (§3.1)** records these content hashes in `bundle.json`, so
  import can verify integrity and de-dup against an existing store.

---

## 9. Serving API (FastAPI)

### 9.1 REST vs WebSocket — the split

| Transport | Used for | Why |
|---|---|---|
| **REST + HTTP range / Zarr-over-HTTP** | bricks, slices, samples, queries, CRUD | cacheable, parallel (HTTP/2), CDN-able later, dead-simple client |
| **WebSocket** | **job progress** + live cancel | server-push of `0..1` progress without polling |
| (SSE alt for jobs) | one-way progress | simpler than WS if no client→server mid-job msgs |

**Decision:** **REST for all data movement** (bricks/slices/samples/queries) —
they are cacheable, range-friendly, and parallel; **WebSocket only for job
progress/cancel**. We do **not** stream volume bricks over WebSocket: that would
defeat HTTP caching and HTTP/2 multiplexing. Brick streaming = many small cached
GETs.

### 9.2 Endpoints

```
# ── Projects ─────────────────────────────────────────────
GET    /projects                         → [ProjectSummary]
POST   /projects                         {name, frame?} → Project   (creates dir + catalog)
GET    /projects/{pid}                   → Project (incl. SpatialFrame)
PATCH  /projects/{pid}                   {name?, frame?}            (frame edit = doc 01 georeference)
DELETE /projects/{pid}
GET    /projects/{pid}/export            → project bundle (stores/ + pg_dump catalog.sql, §3.1)  [async: {job_id}]
POST   /projects:import                  (multipart bundle)  → {job_id} → Project              (§3.1)

# ── Datasets / upload / ingest ───────────────────────────
GET    /projects/{pid}/datasets          → [Dataset]
POST   /projects/{pid}/uploads           (multipart)  → {raw_file_id, sha256}
POST   /projects/{pid}/datasets:ingest   {raw_file_id, method, options} → {job_id}   (async §9.4)
GET    /datasets/{did}                    → Dataset (+ child artifact ids)
DELETE /datasets/{did}

# ── Artifact catalog / discovery ─────────────────────────
GET    /projects/{pid}/artifacts?bbox=&kind=&method=&property=&t=   → [ArtifactSummary]
                                          (bbox in Engineering m → GiST/R-Tree query §2.5)
GET    /property-models/{id}              → PropertyModel meta (shape,levels,stats,frame)
GET    /features/{id}                     → Feature meta
GET    /observations/{id}                 → Observation meta (+inline or values_uri)

# ── Volume bricks (the hot path) ─────────────────────────
GET    /property-models/{id}/zarr/{path}  → raw Zarr object (group/array meta or chunk)
          # Zarr-over-HTTP: the JS Zarr client reads the store directly.
          # supports If-None-Match / Range; Cache-Control: immutable for chunks.
GET    /property-models/{id}/bricks/{property}/{level}/{t}/{bz}/{by}/{bx}  → brick bytes (+headers §6)
          # convenience alias resolving to the doc-02 §10.2 chunk path
          # <property>/<level>/c/<bz>/<by>/<bx>. Fused models: GET /fused-models/{id}/bricks/...

# ── Slices (arbitrary plane) ─────────────────────────────
POST   /property-models/{id}/slice        SliceRequest  → SliceResponse (PNG|raw f32|npy)

# ── Point / line sampling ────────────────────────────────
POST   /property-models/{id}/sample       SampleRequest → SampleResponse
POST   /projects/{pid}/sample             MultiSampleRequest → cross-property samples (fusion L2)

# ── Isosurface extraction ────────────────────────────────
POST   /property-models/{id}/isosurface   IsoRequest    → {job_id}  OR inline glTF if small

# ── Features / vectors / points ──────────────────────────
GET    /features/{id}/geometry            → glTF | VTK | GeoJSON | 3D-Tiles tileset.json
GET    /features/{id}/points?bbox=&t0=&t1= → filtered point cloud (microseismic 4D)

# ── Layers / views (viewer state) ────────────────────────
GET/POST/PATCH/DELETE  /projects/{pid}/layers ; /projects/{pid}/views

# ── Jobs ─────────────────────────────────────────────────
GET    /jobs/{jid}                        → Job (status, progress, result)
POST   /jobs/{jid}:cancel                 → 202
WS     /jobs/{jid}/progress               ← server pushes {status, progress, message}
GET    /projects/{pid}/jobs               → [Job]
```

### 9.3 Key request/response shapes

```jsonc
// ── SliceRequest ── arbitrary plane through a volume (OVERVIEW §7 slices/fences)
SliceRequest {
  "plane":  "x" | "y" | "z" | "arbitrary",
  "position": 1234.5,                 // Engineering m, for axis-aligned
  "origin":   [x,y,z], "normal": [nx,ny,nz],  // for arbitrary plane
  "level":    0,                      // pyramid level (LOD)
  "t":        0,                      // time index
  "bounds":   {xmin,xmax,...} | null, // crop (Engineering m)
  "encoding": "png" | "f32" | "npy",  // png=colour-mapped server-side; f32=raw for client TF
  "colormap": "viridis" | null,       // only if png
  "range":    [min,max] | null        // transfer-fn clamp; default from stats
}
SliceResponse {                       // (body is the image/array; this is the JSON header variant)
  "width": 512, "height": 384,
  "dx": 10.0, "dy": 10.0,             // sample spacing m
  "plane_basis": {origin:[..], u:[..], v:[..]},  // place it in the 3D scene
  "encoding": "f32", "dtype": "float32",
  "data_uri": "/cache/slices/<hash>.bin"   // or inline base64 for small
}

// ── SampleRequest ── value at point(s), along a straight line, or along a CURVED PATH (OVERVIEW §6 L2)
SampleRequest {
  "mode": "points" | "line" | "path",
  "points": [[x,y,z], ...],           // points mode
  "line":   {"from":[x,y,z],"to":[x,y,z],"n":200},  // line mode (straight segment)
  // path mode: a POLYLINE / curved trajectory — e.g. a deviation-survey well path (doc 09).
  // Sampled piecewise along the vertices; this is what the well planner uses to read a
  // predicted log along a deviated borehole (NOT just a straight from→to).
  "path":   {"vertices":[[x,y,z], ...], "n": 500, "spacing": null} | null,
  //   vertices = Engineering-m polyline; sample N points evenly by arc-length (or fixed `spacing` m).
  "level": 0, "t": 0,
  "interp": "nearest" | "trilinear",
  "withUncertainty": false            // true ⇒ include co-registered 1σ in the response
}
SampleResponse {
  "property": "resistivity", "unit": "ohm.m",
  "values": [12.3, 14.1, ...],        // null where outside volume
  "sigma":  [2.1, 2.4, ...] | null,   // co-registered 1σ (doc 02 §6) when requested & available
  "positions": [[x,y,z], ...],        // echoes the sampled XYZ (esp. for line/path mode)
  "distance":  [0.0, 12.5, ...] | null // cumulative arc-length m along line/path (well-log MD axis, doc 09)
}
// SampleRequest/MultiSampleRequest accept "withUncertainty": true to include sigma
// (doc 09's well-planner needs value + uncertainty along the curved trajectory).

// ── MultiSampleRequest ── sample SEVERAL volumes at shared points → cross-plot (L2)
MultiSampleRequest {
  "property_model_ids": ["pm_a","pm_b"],
  "mode": "points"|"line"|"path", "points":[...] | "line":{...} | "path":{vertices:[...],n},
  "level": 0, "t": 0, "interp": "trilinear", "withUncertainty": false
}
// → { "samples": [ {id, property, unit, values[], sigma?[]}, ... ], "positions":[...] }

// ── IsoRequest ── marching-cubes surface (OVERVIEW §7)
IsoRequest {
  "isovalue": 100.0,
  "level": 0, "t": 0,
  "bounds": {...} | null,
  "format": "gltf" | "vtk",
  "simplify": 0.0..1.0                 // optional decimation
}
// small → inline glTF; large → {job_id} then GET /features/{id}/geometry
```

### 9.4 Job / async model

**Inline for fast ops; an RQ + Redis worker queue for everything else, from the
start** (`DECISIONS.md` doc-04):

| Tier | Mechanism | When |
|---|---|---|
| **Inline** | compute in the request | fast ops: small slice, point sample, small isosurface |
| **RQ + Redis** *(the choice)* | external RQ worker pool against Redis, writing the `jobs` table | ingest, fusion, pyramid build, big isosurface, export/import, GC — parallel, crash-isolated, survives API restart |
| **BackgroundTasks** *(lightweight fallback)* | FastAPI `BackgroundTasks` + a `jobs` row + WS progress | only the no-service embedded build (paired with the SQLite fallback, §2.1) |

**Decision:** run **RQ + Redis workers from day one** as the async tier (user
decision) — chosen over Celery because it's lighter-weight and Redis-only
(matching local-first), and over BackgroundTasks because parallel RQ ingest needs
real crash-isolation and survives an API restart. The `jobs` row is the durable
source of truth, so a job survives a page reload (the client reconnects to
`/jobs/{jid}/progress` or polls `GET /jobs/{jid}`). **FastAPI BackgroundTasks
remains a documented lightweight fallback** for the embedded no-service build
only. The **job contract (table + endpoints + WS) is identical** across both, so
the executor is a swap behind `JobRunner`, not an API change.

**Job pattern:** `POST ...:ingest` → create `jobs` row (`queued`) → enqueue on RQ
→ return `{job_id}` immediately → worker updates `progress`/`message` and pushes
over WS → on success writes artifact rows + `result_json` → client refetches.

---

## 10. Scaling / hosted path (non-MVP, kept open)

The local-first design upgrades without an API rewrite:

| Local-first (now) | Hosted (later) | Trigger |
|---|---|---|
| PostgreSQL + PostGIS (single instance) | managed/clustered Postgres | many concurrent users |
| RQ workers + Redis (local) | RQ workers + managed Redis, more workers | parallel/isolated jobs at scale |
| Filesystem stores | S3-compatible object store (Zarr/COG are object-native) | shared/remote access |
| FastAPI serves chunks | CDN in front of object store (chunks are immutable) | scale-out reads |

(PostgreSQL+PostGIS and RQ+Redis are the engines **from the start** — see §2.1,
§9.4; the hosted path scales them rather than swapping them in.)

Because Zarr/COG are **object-store-native and HTTP-range-served**, and chunks
are immutable & content-addressed, moving bulk bytes to S3+CDN is a URL/driver
change, not a format change.

---

## Decisions locked in

1. **Catalog DB = PostgreSQL + PostGIS from the start** *(user decision)*, accessed
   via SQLAlchemy with a backend-portable spatial helper; **SQLite + SpatiaLite**
   remains the optional lightweight fallback. No engine-specific SQL in hot paths,
   so either engine runs the same code.
2. **Catalog in PostgreSQL; bulk in the project directory.** The project dir holds
   only the bulk stores (`arrays/ grids/ meshes/ vectors/ points/ raw/ cache/`);
   catalog rows live in a per-project Postgres schema. Copy/share/backup is the
   **project bundle** = bulk `stores/` + a `pg_dump` of the catalog (§3.1), not
   "copy one folder."
3. **Bounding boxes / geometry indexed in Engineering metres** (doc 01), via a
   **PostGIS GiST** 3D bbox index (portable R-Tree mirror on the SQLite fallback) —
   anchor-independent, so georeferencing never rewrites the index.
4. **3D/4D volumes = Zarr v3**, axis order `[t,]z,y,x`, **float32 canonical**,
   **64³ cubic chunks (= 1 MiB bricks)**, **Blosc+zstd+shuffle** lossless
   compression. **Chunk == GPU brick == addressable unit.**
5. **Power-of-two multiresolution pyramid** per volume (mean downsample, built to
   a ~64³ thumbnail); LOD streams coarse→fine. Addressing is **octree-compatible**
   (`(id, property, level, t, bz, by, bx)` == doc 02 §10.2 chunk path
   `<property>/<level>/c/<bz>/<by>/<bx>`) so a sparse octree is a future
   non-breaking upgrade.
6. **Format per primitive:** Zarr (volumes) · COG (2D grids) · glTF/VTK (meshes) ·
   GeoJSON (vectors) · LAZ/3D-Tiles (point clouds) · verbatim content-addressed
   raw store. (`OVERVIEW.md` §5.)
7. **Fused models vs derived volumes are modelled distinctly** (doc 02 §11): a
   **`FusedEarthModel` is a container grid** (`fused_models` + `fused_layers`, §2.4,
   §7), *not* a `property_models` row; a **favorability volume is an ordinary
   PropertyModel**. Both use the **identical Zarr layout & endpoints** as ingested
   volumes — no special-casing in storage or serving, only in the catalog rows.
8. **REST + HTTP-range / Zarr-over-HTTP for all data** (bricks/slices/samples/
   queries — cacheable, parallel, CDN-able); **WebSocket only for job
   progress/cancel.**
9. **Jobs:** inline for fast ops; **RQ + Redis workers from the start** *(user
   decision)* as the async tier (crash-isolation + parallel ingest), against the
   `jobs` table; FastAPI BackgroundTasks remains the documented lightweight
   fallback. Same job contract (table + endpoints + WS) throughout.
10. **Content-addressing + mark-and-sweep GC (§8.1):** raw files = whole-file
    sha256; Zarr chunks = immutable content/ETag objects; an artifact version = a
    manifest hash over metadata + chunk references; GC marks from live catalog
    references and sweeps the rest. This is what makes **keep-all-versions cheap**
    (unchanged bytes shared) and HTTP/S3+CDN caching safe. `cache/` is fully
    derivable & deletable and exempt from GC.

### Resolved (was: open questions)

- **Catalog DB** → PostgreSQL + PostGIS *(user decision)*.
- **Async executor** → RQ + Redis from the start *(user decision)*.
- **Slice colour-mapping locus** → **raw-f32 to client** is the default (GPU transfer
  function, consistent with volume rendering); PNG is an export/thumbnail option.
- **Artifact versioning depth** → **keep all versions** *(user decision)*; cheap via
  content-addressed bulk sharing + manifest-hash versions + mark-and-sweep GC
  (§8.1; doc 02 §9).
- **`[NEEDS-02]` confirmations** → all resolved against doc 02 (see the
  *Reconciliation with doc 02* banner above): `support`∈volume/grid2d/mesh;
  uncertainty = `<property>_sigma` sibling; `geometryKind`/`featureKind`
  vocabularies; `FusedEarthModel` = regular voxel, **multiple permitted per project**.

### Still open (genuinely deferred, non-blocking)

- **Sparse octree vs dense pyramid timing** — dense pyramids ship in MVP; revisit
  sparse-brick skipping only if continental-AEM / large seismic cubes demand it
  (likely Phase 4+).
```
