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

---

## 0. Design constraints (inherited)

- **Local-first, single-user**, grows to hosted/multi-user (`OVERVIEW.md` §1, §5).
- Browser viewer streams **multiresolution Zarr bricks → GPU 3D textures**, ray-marched (`OVERVIEW.md` §5, §7).
- Arrays are **always Engineering-frame** (doc 01 #2) → georeferencing never touches bulk bytes.
- Three primitives + fused grid (`OVERVIEW.md` §2); **observations immutable**, property models/fused derived, features vector.
- 4D (time) is first-class (`OVERVIEW.md` §2).

---

## 1. Storage tiers — the three stores

| Tier | What | Tech | Why |
|---|---|---|---|
| **Catalog DB** | metadata, geometry index, provenance, layer defs, job state | **SQLite + SpatiaLite** (default) → PostgreSQL + PostGIS (hosted) | small, relational, spatially queryable, transactional |
| **Array store** | bulk N-D arrays + 2D rasters + meshes + vectors + point clouds | **Zarr** (3D/4D), **COG** (2D), glTF/VTK (mesh), GeoJSON (vector), LAZ/3D-Tiles (points) | chunked, lazy, web-streamable, format-appropriate |
| **Raw store** | original survey files, verbatim | filesystem, content-addressed | provenance, re-ingest, audit (`OVERVIEW.md` §5) |

The catalog DB is the **index/source-of-truth for metadata**; it never holds bulk
samples — only pointers (URIs) into the array/raw stores plus bounding boxes,
shapes, units, and stats. This keeps the DB tiny and the heavy bytes
memory-mappable / range-servable.

---

## 2. Catalog DB

### 2.1 Engine choice — SQLite+SpatiaLite default, PG+PostGIS growth path

| | **SQLite + SpatiaLite** *(default)* | **PostgreSQL + PostGIS** *(growth)* |
|---|---|---|
| Setup | zero — a file in the project dir | a service to run |
| Concurrency | single-writer (fine: single-user) | many writers |
| Spatial | R-Tree + SpatiaLite functions | full GiST + rich GIS |
| JSON | `JSON1` (good enough) | `jsonb` (richer) |
| Fit | **local-first MVP** | hosted/multi-user later |

**Decision:** ship **SQLite + SpatiaLite**. It's a single file (`catalog.sqlite`)
living *inside the project directory* (§3), so a project is a self-contained,
copyable, zippable folder — exactly the local-first contract. Access via
**SQLAlchemy** with a thin spatial helper layer so the **PostGIS swap is a
connection-string + dialect change**, not a rewrite. We deliberately avoid
SpatiaLite-only SQL in app code; bbox indexing uses the portable R-Tree pattern
below.

> **Portability rule:** all schema DDL and queries target the SQLAlchemy core /
> the intersection of SQLite and PG semantics. Geometry columns use WKB + an
> explicit bbox-index table (§2.4) rather than engine-specific spatial types in
> hot paths, so the same code runs on both. SpatiaLite/PostGIS functions are used
> only for richer offline analysis, behind a capability flag.

### 2.2 Bounding boxes are in **Engineering metres** (doc 01)

Every spatial row stores its bbox/geometry in **Engineering-frame metres** (doc
01 §1), *not* lat/lon. This is what the viewer queries in (the viewer lives in the
Engineering frame), and it's anchor-independent so georeferencing a project never
rewrites the index. A *derived* lat/lon bbox can be computed on demand from the
`SpatialFrame` for map-extent display.

### 2.3 Schema

ID convention: `TEXT` UUIDv7 (time-ordered, sortable) PKs. `created_at`/
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
  method        TEXT NOT NULL,        -- 'gravity'|'ert'|'mt'|'seismic'|'insar'|... (OVERVIEW §3)
  kind          TEXT NOT NULL,        -- 'observation'|'property_model'|'feature'|'fused' (provenance origin)
  status        TEXT NOT NULL,        -- 'ingesting'|'ready'|'error'
  time_extent_json TEXT,              -- {t0,t1,n} for 4D, null for static
  meta_json     TEXT,                 -- method-specific (acquisition params, [NEEDS-02])
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ─────────────────── property_models (3D/4D continuous fields) ───────────────────
-- Catalog row + array pointer for a doc-02 PropertyModel. The doc-02 schema
-- defines support geometry / uncertainty semantics; we store the array + index.
CREATE TABLE property_models (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  property      TEXT NOT NULL,        -- 'resistivity'|'density'|'velocity'|... (doc 01 §5 registry)
  canonical_unit TEXT NOT NULL,       -- from doc 01 units registry
  support       TEXT NOT NULL,        -- 'regular'|'octree'|'unstructured'  [NEEDS-02]
  store_uri     TEXT NOT NULL,        -- relative path to .zarr group (§4)
  store_format  TEXT NOT NULL DEFAULT 'zarr',
  shape_json    TEXT NOT NULL,        -- [nz,ny,nx] or [nt,nz,ny,nx]
  spacing_json  TEXT,                 -- voxel size m, regular grids
  origin_json   TEXT,                 -- grid origin, Engineering m
  bbox_json     TEXT NOT NULL,        -- {xmin..zmax} Engineering m (index source)
  has_time      INTEGER NOT NULL DEFAULT 0,
  pyramid_levels INTEGER NOT NULL DEFAULT 1,  -- # multiresolution levels (§5)
  stats_json    TEXT,                 -- {min,max,mean,p1,p99,histogram} for transfer fn
  uncertainty_uri TEXT,               -- sibling zarr array (optional)  [NEEDS-02]
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ─────────────────── observations (raw measured survey data) ───────────────────
-- Immutable (OVERVIEW §2). Geometry can be points/lines; bulk values may be
-- inline (small) or in an array file (e.g. SEG-Y traces, log curves).
CREATE TABLE observations (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  obs_type      TEXT NOT NULL,        -- 'gravity_station'|'ert_pseudosection'|'mt_edi'|'las_curve'|... [NEEDS-02]
  geometry_kind TEXT NOT NULL,        -- 'point'|'multipoint'|'line'|'profile'|'volume'
  geometry_wkb  BLOB,                 -- Engineering-frame geometry (small)
  values_uri    TEXT,                 -- array file if bulk (else inline)
  values_json   TEXT,                 -- inline values for small obs
  bbox_json     TEXT NOT NULL,
  acquired_at   INTEGER,              -- time of measurement (4D)
  meta_json     TEXT,                 -- [NEEDS-02]
  created_at    INTEGER NOT NULL
);

-- ─────────────────── features (vector geological interpretation) ───────────────────
CREATE TABLE features (
  id            TEXT PRIMARY KEY,
  dataset_id    TEXT REFERENCES datasets(id) ON DELETE CASCADE,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  feature_type  TEXT NOT NULL,        -- 'horizon'|'fault'|'unit_solid'|'well_path'|'fracture'|'microseismic' [NEEDS-02]
  store_uri     TEXT,                 -- glTF/VTK mesh, GeoJSON, or LAZ (§4)
  store_format  TEXT NOT NULL,        -- 'gltf'|'vtk'|'geojson'|'laz'|'3dtiles'
  geometry_wkb  BLOB,                 -- simplified geom for picking/index (optional)
  bbox_json     TEXT NOT NULL,
  has_time      INTEGER NOT NULL DEFAULT 0,
  props_json    TEXT,                 -- per-feature attributes [NEEDS-02]
  created_at    INTEGER NOT NULL,
  updated_at    INTEGER NOT NULL
);

-- ─────────────────── provenance (lineage DAG) ───────────────────
-- Edges: this artifact was DERIVED FROM these inputs by this process.
-- Covers ingest (raw→primitive) and fusion/transform (primitives→derived).
CREATE TABLE provenance (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  target_kind   TEXT NOT NULL,        -- 'dataset'|'property_model'|'feature'|'observation'
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
  source_kind   TEXT NOT NULL,        -- 'property_model'|'feature'|'observation'|'fused'
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
  kind          TEXT NOT NULL,        -- 'ingest'|'fuse'|'isosurface'|'transform'|'pyramid'|'export'
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

### 2.4 Spatial indexing of bounding boxes (portable)

Hot-path spatial queries ("which artifacts intersect this clip box / this
slice?") use an **R-Tree index** that works identically on both engines:

```sql
-- SQLite R-Tree virtual table (mirrors PostGIS GiST on the same bbox).
CREATE VIRTUAL TABLE artifact_rtree USING rtree(
  rowid,                 -- maps to artifact via artifact_rtree_map
  xmin, xmax, ymin, ymax, zmin, zmax   -- Engineering metres
);
CREATE TABLE artifact_rtree_map (
  rowid INTEGER PRIMARY KEY,
  kind  TEXT NOT NULL,   -- 'property_model'|'feature'|'observation'
  id    TEXT NOT NULL
);
```

A bbox-intersection query reads `artifact_rtree`, joins the map, then the typed
table. On PostgreSQL the same logical query is a GiST index on a `box3d` /
`geometry(...,3D)` column; the SQLAlchemy spatial helper picks the backend.
3D bbox (incl. `zmin/zmax`) so depth-clipping the scene also prunes candidates.

---

## 3. On-disk project layout

A **project is a self-contained directory** — the catalog DB plus all stores
nested under it, so it copies/zips/backs-up as one unit (local-first contract).

```
<storage_root>/<project_id>/
├── catalog.sqlite                 # the catalog DB (§2)
├── frame.json                     # cached SpatialFrame (doc 01), DB is canonical
├── arrays/                        # Zarr volumes (property models + fused)
│   ├── pm_<id>.zarr/              #   one zarr group per property model
│   │   ├── zarr.json              #     v3 group metadata
│   │   ├── 0/  1/  2/ ...         #     pyramid levels (§5); 0 = full res
│   │   └── uncertainty/           #     optional sibling array [NEEDS-02]
│   └── fused_<id>.zarr/           #   fused/derived volumes — SAME layout (§7)
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

---

## 4. Zarr volume store (the core bulk format)

Authoritative Zarr/array conventions belong to **doc 02**; this section pins the
**storage + serving** details the viewer depends on.

### 4.1 Group/array structure (Zarr v3)

```
pm_<id>.zarr/                       # a zarr GROUP = one property model
  zarr.json                         # group metadata (multiscale spec, units, frame ref)
  0/                                # level 0 = full resolution ARRAY
    zarr.json                       #   array meta: shape, chunks, dtype, codecs
    c/0/0/0   c/0/0/1   ...         #   chunk objects (one file per chunk)
  1/  2/  ...                       # coarser pyramid levels (§5)
  uncertainty/                      # optional parallel multiscale array [NEEDS-02]
```

- **Axis order:** `[z, y, x]` (3D) or `[t, z, y, x]` (4D), C-order, matching the
  Engineering ENU frame (doc 01). Grid origin + spacing live in `zarr.json`
  attrs **in Engineering metres** (never CRS coords — doc 01 #2).
- **dtype:** `float32` canonical for rendering (GPU 3D-texture native). `int16`/
  `uint8` allowed for quantized/labelled volumes; transfer fn handles scaling.
- **multiscale metadata:** OME-Zarr-style `multiscales` block in group attrs
  listing levels + their downsample factors, so any Zarr reader (Python xarray
  *and* the JS client) discovers the pyramid the same way.

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

A brick is addressed by **`(artifact_id, level, t, bz, by, bx)`**:

| Field | Meaning |
|---|---|
| `artifact_id` | property-model / fused id |
| `level` | pyramid level (0 = finest) |
| `t` | time index (0 for static) |
| `bz,by,bx` | chunk/brick indices within that level |

This is exactly a Zarr chunk key, so **brick address == Zarr chunk path** — the
server can serve it by mapping the URL straight to a chunk object (or proxy the
chunk store). It is also octree-compatible: a node `(level, bz,by,bx)` has
children `(level-1, 2b{x,y,z}+{0,1})`, so a future sparse octree reuses the same
URLs.

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

A fused or transform-derived volume (`OVERVIEW.md` §6 L1–L3; resample-to-fused,
rock-physics outputs, favorability, uncertainty) is written as a **`fused_<id>.zarr`
group with the identical layout** (§4–5): same chunking, compression, pyramids,
multiscale metadata. It gets:
- a `datasets` row with `kind='fused'`,
- a `property_models` row (it *is* a property model in catalog terms),
- a `provenance` record linking it to its input artifacts (the fusion DAG),
- pyramids built by the same `pyramid` job.

**Consequence:** the **same `/bricks`, `/slice`, `/sample`, `/isosurface`
endpoints serve fused volumes** with zero special-casing. The fused grid (the
canonical model grid, `OVERVIEW.md` §2 / `FusedGrid` **[NEEDS-02]**) is just
another Zarr volume to storage & serving.

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

# ── Datasets / upload / ingest ───────────────────────────
GET    /projects/{pid}/datasets          → [Dataset]
POST   /projects/{pid}/uploads           (multipart)  → {raw_file_id, sha256}
POST   /projects/{pid}/datasets:ingest   {raw_file_id, method, options} → {job_id}   (async §9.4)
GET    /datasets/{did}                    → Dataset (+ child artifact ids)
DELETE /datasets/{did}

# ── Artifact catalog / discovery ─────────────────────────
GET    /projects/{pid}/artifacts?bbox=&kind=&method=&property=&t=   → [ArtifactSummary]
                                          (bbox in Engineering m → R-Tree query §2.4)
GET    /property-models/{id}              → PropertyModel meta (shape,levels,stats,frame)
GET    /features/{id}                     → Feature meta
GET    /observations/{id}                 → Observation meta (+inline or values_uri)

# ── Volume bricks (the hot path) ─────────────────────────
GET    /property-models/{id}/zarr/{path}  → raw Zarr object (group/array meta or chunk)
          # Zarr-over-HTTP: the JS Zarr client reads the store directly.
          # supports If-None-Match / Range; Cache-Control: immutable for chunks.
GET    /property-models/{id}/bricks/{level}/{t}/{bz}/{by}/{bx}  → brick bytes (+headers §6)
          # convenience alias resolving to the same chunk object.

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

// ── SampleRequest ── value at point(s) or along a line (OVERVIEW §6 L2)
SampleRequest {
  "mode": "points" | "line",
  "points": [[x,y,z], ...],           // points mode
  "line":   {"from":[x,y,z],"to":[x,y,z],"n":200},  // line mode (e.g. along a well)
  "level": 0, "t": 0,
  "interp": "nearest" | "trilinear"
}
SampleResponse {
  "property": "resistivity", "unit": "ohm.m",
  "values": [12.3, 14.1, ...],        // null where outside volume
  "positions": [[x,y,z], ...]         // echoes (esp. for line mode)
}

// ── MultiSampleRequest ── sample SEVERAL volumes at shared points → cross-plot (L2)
MultiSampleRequest {
  "property_model_ids": ["pm_a","pm_b"],
  "mode": "points"|"line", "points":[...] | "line":{...},
  "level": 0, "t": 0, "interp": "trilinear"
}
// → { "samples": [ {id, property, unit, values[]}, ... ], "positions":[...] }

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

**Sync-first, escalate to a queue** (`OVERVIEW.md` §5 "background tasks first;
Celery/RQ when needed"):

| Tier | Mechanism | When |
|---|---|---|
| **Inline** | compute in the request | fast ops: small slice, point sample, small isosurface |
| **BackgroundTasks** *(default for jobs)* | FastAPI `BackgroundTasks` + a `jobs` row + WS progress | ingest, fusion, pyramid build, big isosurface — single-user, one box |
| **RQ (Redis Queue)** *(growth)* | external worker pool, same `jobs` table | multi-user / parallel / crash-isolation; survives API restart |

**Decision:** default to **FastAPI BackgroundTasks** backed by the `jobs` table
(so a job survives a page reload — the client reconnects to `/jobs/{jid}/progress`
or polls `GET /jobs/{jid}`). Escalate to **RQ over Celery** when we need real
worker isolation/parallelism — **RQ** because it's lighter-weight and Redis-only,
matching local-first; Celery only if routing/scheduling complexity demands it.
Either way the **job contract (table + endpoints + WS) is identical**, so the
escalation is an executor swap behind `JobRunner`, not an API change.

**Job pattern:** `POST ...:ingest` → create `jobs` row (`queued`) → enqueue →
return `{job_id}` immediately → worker updates `progress`/`message` and pushes
over WS → on success writes artifact rows + `result_json` → client refetches.

---

## 10. Scaling / hosted path (non-MVP, kept open)

The local-first design upgrades without an API rewrite:

| Local-first (now) | Hosted (later) | Trigger |
|---|---|---|
| SQLite + SpatiaLite | PostgreSQL + PostGIS | multi-user / concurrent writes |
| Filesystem stores | S3-compatible object store (Zarr/COG are object-native) | shared/remote access |
| BackgroundTasks | RQ workers + Redis | parallel/isolated jobs |
| FastAPI serves chunks | CDN in front of object store (chunks are immutable) | scale-out reads |

Because Zarr/COG are **object-store-native and HTTP-range-served**, and chunks
are immutable & content-addressed, moving bulk bytes to S3+CDN is a URL/driver
change, not a format change.

---

## Decisions locked in

1. **Catalog DB = SQLite + SpatiaLite** (one file per project), accessed via
   SQLAlchemy with a backend-portable spatial helper; **PostgreSQL + PostGIS** is
   the drop-in hosted upgrade. No engine-specific SQL in hot paths.
2. **A project is a self-contained directory** (`catalog.sqlite` + `arrays/`,
   `grids/`, `meshes/`, `vectors/`, `points/`, `raw/`, `cache/`) — copyable as one
   unit.
3. **Bounding boxes / geometry indexed in Engineering metres** (doc 01), via a
   portable **R-Tree** (SQLite) / GiST (PG) 3D bbox index — anchor-independent, so
   georeferencing never rewrites the index.
4. **3D/4D volumes = Zarr v3**, axis order `[t,]z,y,x`, **float32 canonical**,
   **64³ cubic chunks (= 1 MiB bricks)**, **Blosc+zstd+shuffle** lossless
   compression. **Chunk == GPU brick == addressable unit.**
5. **Power-of-two multiresolution pyramid** per volume (mean downsample, built to
   a ~64³ thumbnail); LOD streams coarse→fine. Addressing is **octree-compatible**
   (`(id, level, t, bz, by, bx)` == Zarr chunk path) so a sparse octree is a future
   non-breaking upgrade.
6. **Format per primitive:** Zarr (volumes) · COG (2D grids) · glTF/VTK (meshes) ·
   GeoJSON (vectors) · LAZ/3D-Tiles (point clouds) · verbatim content-addressed
   raw store. (`OVERVIEW.md` §5.)
7. **Derived/fused volumes use the identical Zarr layout & endpoints** as ingested
   ones — no special-casing in storage or serving.
8. **REST + HTTP-range / Zarr-over-HTTP for all data** (bricks/slices/samples/
   queries — cacheable, parallel, CDN-able); **WebSocket only for job
   progress/cancel.**
9. **Sync-first jobs:** inline for fast ops; **FastAPI BackgroundTasks + a `jobs`
   table** as the default async tier; **RQ** (over Celery) as the growth executor —
   same job contract throughout.
10. **`cache/` is fully derivable & deletable;** served chunks are **immutable +
    content-ETagged** (edits create new artifact ids), enabling aggressive HTTP
    caching and a clean S3+CDN path.

### Open questions for you

See **QUESTIONS FOR USER** (returned alongside this doc) for the highest-leverage
forks. Secondary items parked here:

- **Sparse octree vs dense pyramid timing** — dense pyramids ship in MVP; do we
  expect volumes large/sparse enough (e.g. continental AEM, big seismic cubes) to
  need sparse-brick skipping in an early phase, or is that safely Phase 4+?
- **Artifact versioning depth** — edits create new ids (immutability). Do we keep
  full version history (undo/branching) or just latest + provenance? Affects DB
  growth.
- **Slice colour-mapping locus** — server-side PNG slices (simple, cached) vs
  always raw-f32 to the client with GPU transfer functions (consistent with volume
  TF, more client work). Likely both; which is the default for the section/fence
  panels?
- **[NEEDS-02 confirmations]:** exact `support` enum + uncertainty-array
  convention for `PropertyModel`; `obs_type`/`feature_type` vocabularies;
  `FusedGrid` identity (is it 1 canonical grid per project, or many?). Storage is
  ready for any of these; just needs the doc-02 names pinned.
```
