# 01 — Spatial Framework & Coordinate Handling

> Parent: `OVERVIEW.md` §2. This doc defines the coordinate, datum, and units
> foundation that every other doc (data model, storage, viewer, fusion) binds to.
> Nothing downstream invents its own coordinate handling — it all flows through here.

## Goals & requirements

- **Anchor to real geographic locations** (real CRS, real terrain/DEM, real basemaps) **when location is known**.
- **Also support location-agnostic data** — synthetic datasets or surveys whose real-world position is unknown or irrelevant — in the *same* internal representation.
- A local/floating dataset can be **georeferenced later** by assigning an anchor, without re-processing its bulk arrays.
- Correct, unit-safe handling of the geoscience coordinate mess: lat/lon vs projected, elevation vs depth, MD vs TVD, multiple vertical datums.
- GPU-friendly: coordinates small enough for float32 rendering without jitter.

---

## 1. The central idea: one internal frame, optional geo-anchor

Everything inside the system — storage grids, the fused model, the 3D viewer, all geometry — lives in a single **Engineering Frame**:

> **Engineering Frame** = local right-handed Cartesian, **ENU** axes (X = East, Y = North, Z = Up), **meters**, origin at a project-chosen anchor point. Z is **elevation** (positive up).

Georeferencing is *just an optional rigid transform* from the Engineering Frame to a real-world CRS. The two anchoring modes differ only in whether that transform is set:

| | **Georeferenced mode** | **Local mode** |
|---|---|---|
| Horizontal CRS | real projected CRS (e.g. UTM zone) | none |
| Anchor point | real (easting, northing, elevation) in CRS | none (origin is just `(0,0,0)`) |
| Terrain/basemap | real DEM + map tiles | flat or synthetic surface |
| Everything else | **identical** | **identical** |

Because the rest of the system only ever sees the Engineering Frame, **a synthetic resistivity cube and a real MT inversion render and fuse through exactly the same code path.** Geo-anchoring only matters at three boundaries: (a) **ingest** (transform incoming real-world coords into the frame), (b) **terrain/basemap** loading, and (c) **export** (transform back out).

### Why a local frame instead of working in CRS coordinates directly

- **Float32 precision.** UTM coordinates are large (e.g. `E=412,300  N=4,517,800`). On the GPU (float32, ~7 sig figs) that leaves ~0.5 m of jitter — visible wobble on a 3D model. Subtracting a local origin keeps coordinates in the ±tens-of-km range → millimetre precision. This is the standard "floating origin" pattern.
- **Depth is natural.** Subsurface work is all about depth below surface; a Z-up elevation frame with a known surface makes depth↔elevation trivial.
- **Local datasets fall out for free.** They simply never set the anchor.

---

## 2. The `SpatialFrame` object (per project)

Each **project** owns exactly one `SpatialFrame`. It is small metadata (stored in the catalog DB), not bulk data.

```jsonc
SpatialFrame {
  "mode": "georeferenced" | "local",

  // --- georeferencing (null in local mode) ---
  "horizontalCRS": "EPSG:32612" | "<WKT2 string>" | null,  // projected CRS
  "verticalDatum":  "EPSG:3855"  | "ellipsoidal" | "local" | null, // e.g. EGM2008 geoid
  "anchor": { "easting": 412300.0, "northing": 4517800.0, "elevation": 1620.0 } | null,
  "rotationDeg": 0.0,   // azimuth of Engineering +X axis CW from CRS East (about Z). Usually 0.

  // --- always present ---
  "axisConvention": "ENU",          // X=East, Y=North, Z=Up
  "lengthUnit": "m",
  "roi":        { "xmin": -5000, "xmax": 5000, "ymin": -5000, "ymax": 5000 }, // Engineering metres
  "depthRange": { "zmin": -8000, "zmax": 2000 },  // Engineering elevation metres (zmax≈surface top)
  "surfaceModel": "dem:copernicus-30m" | "flat:0" | "synthetic:<id>" | null
}
```

**Transform (Engineering ⇄ CRS), georeferenced mode:**

```
crs_easting  = anchor.E + ( x·cos θ − y·sin θ )
crs_northing = anchor.N + ( x·sin θ + y·cos θ )
crs_elev     = anchor.elevation + z          (θ = rotationDeg)
```

and the inverse for ingest. `lat/lon` for basemaps comes from `pyproj.Transformer(horizontalCRS → EPSG:4326)`. In local mode the transform is identity (`Engineering == world`).

**Promoting local → georeferenced later:** set `mode`, `horizontalCRS`, `verticalDatum`, `anchor`, `rotationDeg`. Bulk arrays are untouched — only the frame metadata changes, because arrays were always stored in Engineering coordinates. Terrain/basemap can then load. (This is why we never bake CRS coordinates into stored arrays.)

---

## 3. Choosing the horizontal CRS (georeferenced mode)

Typical projects are **single-site / pad scale (1–10 km)** — small enough that projection distortion is negligible with a standard local projection — but the system must **also support larger basin/regional areas** when needed.

**Strategy (recommended default):**
1. On project creation from a real location, take the ROI centroid.
2. **Auto-select the UTM zone** containing it → that's `horizontalCRS` (e.g. `EPSG:326xx` N / `327xx` S). At 1–10 km extent distortion is negligible, the code is a standard EPSG that every external tool understands, and most DEMs/datasets align cleanly.
3. **Escalation for larger / awkward areas**, surfaced to the user rather than silently mishandled:
   - **ROI straddles a UTM zone boundary, or is basin/regional scale (≳ 50 km)** → generate a **custom Transverse Mercator centred on the ROI** (WKT2 with central meridian = centroid longitude, latitude-of-origin = centroid latitude). Conformal distortion stays under ~1/1000 across a few hundred km with no zone seam. Stored as inline WKT2.
   - **High latitude (>84°N / <80°S)** → polar stereographic (UPS or custom).
   - **Antimeridian crossing** → custom TM centred on the ROI.
4. The anchor (origin) defaults to the **ROI centroid at surface elevation**, keeping Engineering coordinates centred on zero. Even at a few-hundred-km extent the floating origin keeps coordinates within ~1 cm float32 precision.

We **never** use Web Mercator (EPSG:3857) for the model itself — its area/distance distortion is unacceptable for measurement. It's fine only as a *basemap tile scheme*, handled separately by the viewer.

Incoming datasets in any CRS (lat/lon, a different UTM zone, a national grid) are reprojected to the project CRS on ingest via `pyproj`; the original CRS is kept in provenance so the transform is auditable/reversible.

---

## 4. Vertical: elevation, depth, and datums

This is where geoscience data most often goes wrong, so we standardize hard.

**Internal canonical vertical = orthometric elevation, metres, Z-up, relative to the project `verticalDatum`** (default **EGM2008 geoid / MSL**, since DEMs and surface elevations are usually orthometric). Ellipsoidal height supported when a dataset needs it (geoid separation via `pyproj`).

Everything else is a **derived view** computed on demand, never the source of truth:

| Concept | Definition | Conversion |
|---|---|---|
| **Elevation (Z)** | metres above vertical datum, + up | canonical |
| **Depth (from surface)** | metres below local ground surface | `depth = surface_elev(x,y) − z` |
| **Depth (from datum / TVDSS)** | metres below MSL/datum | `depth = −z` |
| **MD** (measured depth) | length along a borehole | needs the well's deviation survey |
| **TVD** | true vertical depth from a well reference | integrate deviation survey → Δelev |
| **Reference data** | KB, GL, MSL, ellipsoid | each well stores its reference elevation |

Helper functions (backend, unit-checked):
`elevation_to_depth(z, ref)`, `depth_to_elevation(d, ref)`, `md_to_tvd(md, deviation_survey)`, `tvd_to_elevation(tvd, well_ref)`. Boreholes store a **deviation survey** (MD, inclination, azimuth) so MD↔TVD↔Engineering-XYZ is well-defined.

---

## 5. Units registry

Every numeric quantity carries an explicit unit; nothing is dimensionless-by-assumption.

- **Backend:** `pint` unit registry. Convert to **canonical internal units** on ingest, store the canonical unit in metadata, keep the original unit in provenance.
- **Canonical internal units per property:**

  | Property | Canonical unit |
  |---|---|
  | length / coordinates | m |
  | resistivity | Ω·m |
  | conductivity | S/m |
  | density | kg/m³ |
  | magnetic susceptibility | SI (dimensionless) |
  | seismic velocity | m/s |
  | chargeability | mV/V (or ms) |
  | temperature | °C |
  | gravity anomaly | mGal |
  | magnetic field | nT |
  | deformation (InSAR) | mm (with time axis) |

- **Display units** are a UI concern: the viewer/panels can show ft, °F, etc. via the same registry, without touching stored values.
- A **property type registry** (feeds doc 02 & 08) pins, per property: canonical unit, default colourmap, default log/linear scaling, and sensible display range — so a new survey method declares its property once and the whole stack knows how to handle it.

---

## 6. Terrain & basemap (georeferenced mode)

- On georeferencing, fetch a **DEM** over the ROI (default **Copernicus GLO-30** / SRTM fallback), reproject to the project CRS, convert to Engineering elevation, and store as a surface grid (`surfaceModel`). The viewer renders it as the ground surface; subsurface volumes hang beneath it.
- **Basemap imagery** (satellite/topo tiles) is draped on the surface by the viewer using the CRS→lat/lon transform; tiles are a Web-Mercator concern handled only at render time, isolated from the model's measurement CRS.
- **Local mode:** `surfaceModel = flat:0` (or a synthetic surface emitted by the data generator). No tiles.

---

## 7. Public API surface (backend, used by ingest/viewer/export)

```
project.spatial_frame                      → SpatialFrame
frame.to_engineering(points_xyz, src_crs, src_vertical)   → Engineering XYZ (m)
frame.from_engineering(points_xyz)         → (lat, lon, elev) and/or CRS coords
frame.engineering_to_crs(points_xyz)       → CRS easting/northing/elev
frame.depth_to_elevation(d, ref) / .elevation_to_depth(z, ref)
frame.georeference(crs, vertical, anchor, rotationDeg)     # promote local → georef
units.to_canonical(value, unit, property_type)  /  units.to_display(...)
```

All transforms are pure functions of the `SpatialFrame` + `pyproj`; they record source CRS/datum/unit in provenance so any conversion is reversible and auditable.

---

## 8. Decisions locked in

1. Single internal **Engineering Frame** (ENU, metres, Z-up, floating origin); georeferencing is an optional rigid transform. Local vs georeferenced data share one code path.
2. Bulk arrays are **always stored in Engineering coordinates** → local datasets can be georeferenced later with zero array reprocessing.
3. Horizontal CRS = **auto-selected UTM by default** (single-site 1–10 km is the common case); **custom ROI-centred Transverse Mercator** escalation for zone-straddling / basin-scale areas; polar stereographic at high latitude. Never Web Mercator for the model.
4. Vertical canonical = **orthometric elevation, metres, Z-up** (default **EGM2008 geoid / MSL**); depth/MD/TVD are derived views.
5. **`pint`-based units registry**; everything normalized to canonical SI-ish units on ingest, display units handled at the edges.
6. Default terrain source = **Copernicus GLO-30** DEM; default temperature display = **°C** (°F toggle).

### Resolved with user
- Typical extent: **single-site / pad, 1–10 km** (default) → UTM-by-default; **larger basin/regional areas supported** via custom-TM escalation (decision #3).
- Vertical datum default: **EGM2008 geoid / MSL**.
- DEM source: **Copernicus GLO-30**.
```
