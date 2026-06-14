# Coordinates, depth & units

!!! abstract "What you'll learn / why it matters"
    Before you can fuse a gravity survey with a borehole temperature log, you have to make
    them *agree on where they are* and *what their numbers mean*. That sounds trivial — it is
    the single most error-prone thing in all of geoscience software. This page is the
    foundation everything else binds to: how the platform represents **position** (latitude/
    longitude vs projected grids vs a local metre grid), why it deliberately works in a
    **local "Engineering Frame"** instead of real-world coordinates, the genuinely confusing
    **vertical** mess (elevation vs depth vs TVD vs MD vs TVDSS), how a crooked borehole's
    geometry is reconstructed, and the **units registry** that stops a temperature in °C from
    silently corrupting your math. Think of it as defining the coordinate system and the type
    system for the whole codebase.

If you are a programmer, here is the mental model up front: geoscience data arrives in a dozen
incompatible "encodings" of space and a dozen incompatible units. This layer is a
**normalization / canonicalization pass** — like decoding every input into UTF-8 and a single
struct layout before any business logic runs. Get it right once, here, and nothing downstream
has to think about it again.

---

## 1. The geoscience coordinate mess, briefly

The Earth is a lumpy, slightly-squashed ball. To put a number on "where is this rock," people
have invented several incompatible schemes, and real data files use all of them — often without
saying which.

!!! note "Term: CRS (Coordinate Reference System)"
    A **CRS** is a precise, named recipe for turning a position on Earth into numbers (and back).
    It bundles a model of the Earth's shape (the *datum*/*ellipsoid*) with a *coordinate system*
    (angles, or metres on a flat grid). Every CRS has an **EPSG code** — a stable integer ID from
    a public registry (e.g. `EPSG:4326`). Treat an EPSG code like a well-known type ID: it tells
    you *exactly* how to interpret the numbers. A pair of numbers with **no** CRS is like a byte
    buffer with no charset — meaningless and dangerous.

The schemes you will meet:

| Scheme | What the numbers are | Example | CS analogy |
|---|---|---|---|
| **Geographic (lat/lon)** | angles on the ellipsoid, degrees | `lat 38.65, lon -118.10` (`EPSG:4326`) | polar-ish coordinates; degrees aren't metres |
| **Projected (UTM, national grids)** | metres on a flat map (a *projection* of the curved Earth) | `E 412300, N 4517800` (`EPSG:32612`) | the curved surface "rasterized" flat with controlled distortion |
| **Web Mercator** | metres-ish, tuned for *web map tiles* | `EPSG:3857` | a *lossy display format*, not a measurement format |
| **Local engineering grid** | metres from a chosen local origin | `x 300, y -150, z -2000` | coordinates relative to a struct's base pointer |

!!! warning "Web Mercator (`EPSG:3857`) is banned for measurement"
    Web Mercator is the projection Google Maps and slippy map tiles use. It is *conformal but
    wildly area- and distance-distorting* — a kilometre near the poles measures very differently
    from a kilometre at the equator. It exists to make square 256×256 tiles line up, not to
    measure anything. The platform **never** stores or computes geometry in Web Mercator. It is
    allowed in exactly one place: as a *tile scheme for draping satellite imagery on the surface
    at render time*, handled entirely by the viewer and isolated from the model's measurement CRS.
    See [the survey methods overview](survey-methods/index.md) for why measurement fidelity matters
    per method.

**Projection** (turning the curved Earth into flat metres) is itself a lossy operation — exactly
like any map of a sphere on flat paper, you cannot preserve angles, areas, and distances all at
once. UTM (Universal Transverse Mercator) chops the world into 60 north–south zones, each 6°
wide, and accepts a tiny, bounded distortion inside each zone. For a typical single geothermal
site (1–10 km across) that distortion is utterly negligible, which is why **UTM is the default
projected CRS** the platform auto-selects.

---

## 2. The Engineering Frame — one internal coordinate system

Everything *inside* the system — storage grids, the [fused model](fusion.md), the
[3-D viewer](visualization.md), every piece of geometry — lives in **one** coordinate system,
the **Engineering Frame**:

!!! note "Definition: the Engineering Frame"
    A **local, right-handed Cartesian frame** with **ENU axes** (X = **E**ast, Y = **N**orth,
    Z = **U**p), in **metres**, with the origin at a project-chosen anchor point. **Z is
    elevation, positive up.** It is the single internal frame; everything resamples, renders, and
    fuses in it.

Georeferencing — tying the model to a real place on Earth — is then *just an optional rigid
transform* from the Engineering Frame out to a real CRS. There are two project modes, and they
**share one code path**:

| | **Georeferenced mode** | **Local mode** |
|---|---|---|
| Horizontal CRS | a real projected CRS (e.g. a UTM zone) | none |
| Anchor point | real (easting, northing, elevation) | none — origin is just `(0,0,0)` |
| Terrain / basemap | real DEM + map tiles | flat or synthetic surface |
| **Everything else** | **identical** | **identical** |

The payoff: **a synthetic resistivity cube and a real magnetotelluric inversion render and fuse
through exactly the same code.** Geo-anchoring matters at only three boundaries: **ingest**
(transform incoming real-world coordinates *into* the frame), **terrain/basemap** loading, and
**export** (transform back *out*). Internally, nothing knows or cares whether the data is real.

### 2.1 Why a local frame? The float32 jitter math

This is the part a CS person will appreciate. The GPU renders in **float32** — about 7 significant
decimal digits. A UTM easting is a big number like `E = 412,300 m`. Ask float32 to represent that
and the *spacing between representable values* near that magnitude is roughly:

$$
\varepsilon \;\approx\; 2^{\lfloor \log_2 (412300) \rfloor - 23}
\;=\; 2^{18 - 23}
\;=\; 2^{-5}
\;\approx\; 0.03\ \text{m}.
$$

That is per-axis quantization just from the *magnitude* of the coordinate, before any math. In
practice, accumulated GPU arithmetic on coordinates that large produces **visible sub-metre
wobble** — vertices jitter as the camera moves, surfaces shimmer, two things that should touch
separate. This is the classic "large world coordinates" bug every game/GIS engine hits.

The fix is the standard **floating-origin** pattern: subtract a local origin so coordinates stay
in the ±tens-of-kilometres range. At `x ≈ ±50,000 m` the float32 spacing drops to:

$$
\varepsilon \approx 2^{15 - 23} = 2^{-8} \approx 0.004\ \text{m} \;=\; 4\ \text{mm},
$$

and for a typical ±5 km site you are at **sub-millimetre** precision. Same data, same GPU — the
only change is keeping the numbers small. So the platform stores **all bulk arrays in Engineering
metres** and never ships giant UTM coordinates to the GPU.

!!! tip "Two more reasons the local frame wins"
    - **Depth is natural.** Subsurface work is all about depth below the surface. A Z-up elevation
      frame with a known surface makes depth↔elevation a one-line subtraction (see §4).
    - **Local datasets are free.** A synthetic or position-unknown dataset simply never sets an
      anchor. It is *already* in the Engineering Frame, so it uses the identical pipeline.

### 2.2 The `SpatialFrame` object

Each project owns exactly one `SpatialFrame`. It is small metadata (a catalog row), not bulk data.
Here is the real shape, lightly annotated:

```python
@dataclass
class SpatialFrame:                       # geosim/spatial/frame.py
    mode: FrameMode = FrameMode.LOCAL     # "georeferenced" | "local"

    # --- georeferencing (None in local mode) ---
    horizontal_crs: str | None = None     # e.g. "EPSG:32612" or an inline WKT2 string
    vertical_datum: str | None = None     # e.g. "EPSG:3855" (EGM2008 geoid), "ellipsoidal"
    anchor: Anchor | None = None          # real (easting, northing, elevation) of the origin
    rotation_deg: float = 0.0             # azimuth of Engineering +X CW from CRS East. Usually 0.

    # --- always present ---
    axis_convention: Literal["ENU"] = "ENU"   # X=East, Y=North, Z=Up
    length_unit: str = "m"
    roi: Aabb                              # region of interest, Engineering metres
    depth_range: DepthRange                # Engineering elevation metres (zmax ≈ surface top)
    surface_model: str | None = "flat:0"  # "dem:copernicus-30m" | "flat:0" | "synthetic:<id>"
    georef_status: GeorefStatus = GeorefStatus.ASSUMED_LOCAL   # the QUALITY flag (see §2.4)
```

The Engineering ⇄ CRS transform (the actual code in `frame.py`) is a 2-D rotation by
`rotation_deg` = $\theta$ plus the anchor translation, with elevation passed straight through:

$$
\begin{aligned}
\text{easting}  &= \text{anchor}.E + (x\cos\theta - y\sin\theta)\\
\text{northing} &= \text{anchor}.N + (x\sin\theta + y\cos\theta)\\
\text{elev}     &= \text{anchor.elevation} + z
\end{aligned}
$$

In **local mode** this is the identity (`Engineering == world`). For basemaps,
lat/lon comes from chaining the CRS→`EPSG:4326` transform with `pyproj` (the standard Python
projection library — think of it as the reference implementation of "convert between any two
CRSs").

### 2.3 Choosing the horizontal CRS (georeferenced mode)

When a project is created from a real location, the platform auto-selects the CRS:

1. Take the ROI centroid.
2. **Auto-select the UTM zone** containing it (`utm_epsg_for_lonlat` in `frame.py` returns
   `326##` north / `327##` south). At 1–10 km extent the distortion is negligible and the code is
   a standard EPSG every external tool understands.
3. **Escalation**, surfaced to the user rather than handled silently:
    - ROI straddles a UTM zone boundary, or is basin/regional scale (≳ 50 km) → a **custom
      Transverse Mercator centred on the ROI** (stored as inline WKT2). Keeps distortion under
      ~1/1000 across a few hundred km with no zone seam.
    - High latitude (> 84°N / < 80°S) → polar stereographic.
    - Antimeridian crossing → custom TM centred on the ROI.
4. The anchor defaults to the **ROI centroid at surface elevation**, keeping Engineering
   coordinates centred on zero (back to the jitter math in §2.1).

Incoming datasets in *any* CRS are reprojected to the project CRS on ingest via `pyproj`; the
original CRS is kept in [provenance](data-model.md#6-provenance-a-dag-of-how-every-number-was-made) so the transform is auditable.

### 2.4 `georefStatus` — quality is not the same as mode

`mode` only says *whether* a transform to a real CRS exists. It does **not** say whether that
transform is *trustworthy*. That is a separate flag, `georef_status`:

| `georefStatus` | Meaning |
|---|---|
| `unknown` | no spatial info at all |
| `assumed_local` | local/synthetic; coordinates are Engineering by fiat, not a real place |
| `anchored` | a rigid transform to a real CRS was **assigned** (not checked) |
| `validated` | the anchor was checked against control (DEM tie, known coordinates) within tolerance |
| `survey_controlled` | positions come from surveyed GNSS/ground-control points — authoritative |

!!! warning "Re-anchoring is cheap, but it does NOT make data 'real'"
    Because bulk arrays are *always* stored in Engineering coordinates, you can **promote a local
    dataset to georeferenced later with zero array reprocessing** — `frame.georeference(...)` just
    sets the metadata. But "zero reprocessing" means *the array bytes never change*, **not** that
    the data is now physically correct in the world. Assigning an anchor sets
    `georef_status = "anchored"`, **never** `"validated"`. A synthetic dataset authored against a
    flat surface and then anchored to a real DEM has correct *elevations* (elevation is canonical)
    but its *derived depth-below-surface* shifts when the surface model changes — see §4. The UI
    must always show `georefStatus` and never imply that anchoring turned synthetic data into a
    real measurement.

---

## 3. Worked example: the same point, four ways

Suppose a project is anchored at a real site, `anchor = (E 412300, N 4517800, elev 1620)` in UTM
zone 12N (`EPSG:32612`), `rotation_deg = 0`. A measurement sits 300 m east, 150 m south, and 80 m
above the anchor: Engineering `(x, y, z) = (300, -150, 80)`.

```python
from geosim.spatial import SpatialFrame, Anchor, FrameMode

frame = SpatialFrame(
    mode=FrameMode.GEOREFERENCED,
    horizontal_crs="EPSG:32612",
    anchor=Anchor(easting=412300.0, northing=4517800.0, elevation=1620.0),
)
frame.engineering_to_crs([[300, -150, 80]])   # → [[412600.0, 4517650.0, 1700.0]]
```

| Representation | Value |
|---|---|
| Engineering (internal, GPU-safe) | `(300, -150, 80)` m |
| Projected CRS (UTM 12N) | `(E 412600, N 4517650, elev 1700)` m |
| Geographic (for basemaps) | `(lon, lat)` via `frame.to_lonlat(...)` |
| Elevation | `+1700 m` above the vertical datum |

Everything stored and rendered uses the first row. The others are computed only at the edges.

---

## 4. The vertical mess: elevation, depth, TVD, MD, TVDSS

Horizontal coordinates are merely fiddly. **Vertical** is where geoscience data most often goes
silently wrong, because "depth" means at least four different things and files rarely say which.
The platform standardizes hard.

!!! note "Canonical vertical = orthometric elevation, metres, Z-up"
    Internally, the one true vertical coordinate is **orthometric elevation** (height above the
    project's `verticalDatum`, default **EGM2008 geoid / mean sea level**), in metres, **positive
    up**. Everything else is a **derived view** computed on demand — never the source of truth.

!!! note "Term: orthometric vs ellipsoidal height, geoid"
    The Earth's gravity field is lumpy, so "sea level" is not a smooth ellipsoid — it is a bumpy
    equipotential surface called the **geoid**. **Orthometric height** is height above the geoid
    (what a surveyor's level and most DEMs report — "height above sea level"). **Ellipsoidal
    height** is height above the smooth mathematical ellipsoid (what raw GPS reports). They differ
    by the *geoid separation*, tens of metres. The platform stores orthometric by default and
    converts via `pyproj` when a dataset needs ellipsoidal.

Here are the vertical concepts you must keep straight. Let $z$ be canonical elevation and
$\text{surface\_elev}(x,y)$ the ground elevation at that map position:

| Concept | Plain meaning | Conversion |
|---|---|---|
| **Elevation ($z$)** | metres above the vertical datum, + up | canonical (the source of truth) |
| **Depth (from surface)** | metres **below the local ground** | $depth = surface\_elev - z$ |
| **Depth from datum / TVDSS** | metres below MSL/datum | $depth = -z$ |
| **MD** (measured depth) | length **along** a borehole | needs the well's deviation survey (§5) |
| **TVD** (true vertical depth) | straight-down depth from a well reference | integrate the deviation survey |
| **Reference data** (KB, GL, MSL, ellipsoid) | the zero a well measures from | each well stores its reference elevation |

These conversions are real one-line functions in `geosim/spatial/vertical.py`:

```python
elevation_to_depth(z, surface_elev)   # depth = surface_elev - z
depth_to_elevation(depth, surface_elev)
elevation_to_tvdss(z)                 # TVDSS = -z   (depth below MSL/datum)
tvd_to_elevation(tvd, ref_elev)       # z = ref_elev - tvd
```

!!! example "Why TVDSS and depth-from-surface differ"
    A well on a 1620 m plateau hits a hot zone at elevation $z = -380$ m (380 m below sea level).
    - **TVDSS** = $-z = 380$ m below the datum.
    - **Depth from surface** = $surface\_elev - z = 1620 - (-380) = 2000$ m of drilling.

    Same rock, two correct "depths" that differ by 1620 m. Confusing the two is a classic,
    expensive geoscience bug. Storing only canonical elevation and *deriving* each depth on demand
    makes the bug structurally impossible.

!!! note "Terms: KB, GL"
    A borehole measures depth from a physical reference. **KB** = *kelly bushing*, a point on the
    rig floor a few metres above the ground. **GL** = *ground level*. **MSL** = mean sea level.
    A log that says "1000 m" means 1000 m below *that well's* reference, so each well stores its
    reference `kind` and `elevation` (see the [`wellPath` feature](data-model.md#4-primitive-3-geologicalfeature-discrete-interpreted-geometry)).

---

## 5. Deviation surveys & minimum curvature

A borehole is almost never vertical. Modern wells are deliberately *deviated* — steered to hit a
target off to the side. So the position of a point in the well is **not** simply "straight down by
its measured depth." You have to reconstruct the actual 3-D path from a **deviation survey**.

!!! note "Term: deviation survey"
    A **deviation survey** is a table of stations down the hole, each giving:
    `(MD, inclination, azimuth)` — the **measured depth** (length of pipe to that point), the
    **inclination** (angle off vertical: 0° = straight down, 90° = horizontal), and the **azimuth**
    (compass bearing of the hole, clockwise from North). It is a *sparse, sampled signal* of the
    trajectory; the full path is reconstructed by integration. A CS analogy: it is a list of
    `(arc-length, direction)` samples, and you integrate to get the polyline.

The integration method everyone uses is **minimum curvature**: between two adjacent stations,
assume the hole follows a circular arc (the smoothest curve through both directions) rather than
two straight tangents. The platform implements it in `min_curvature_positions`
(`geosim/spatial/vertical.py`) — the *shared* integrator used by both ingested wells
([the well path feature](data-model.md#4-primitive-3-geologicalfeature-discrete-interpreted-geometry)) and the
[well planner](well-planning.md).

For an interval between stations 1 and 2 with inclinations $I_1, I_2$ and azimuths $A_1, A_2$,
separated by $\Delta\text{MD}$:

$$
\cos\beta = \cos(I_2 - I_1) - \sin I_1 \sin I_2 \,\bigl(1 - \cos(A_2 - A_1)\bigr)
$$

$$
\text{RF} = \frac{2}{\beta}\tan\!\frac{\beta}{2}\quad(\text{the ratio factor};\ \text{RF}\to 1\ \text{as}\ \beta\to 0)
$$

$$
\begin{aligned}
\Delta N &= \tfrac{\Delta\text{MD}}{2}\,(\sin I_1\cos A_1 + \sin I_2\cos A_2)\,\text{RF} &&(+\text{North} = +Y)\\
\Delta E &= \tfrac{\Delta\text{MD}}{2}\,(\sin I_1\sin A_1 + \sin I_2\sin A_2)\,\text{RF} &&(+\text{East} = +X)\\
\Delta V &= \tfrac{\Delta\text{MD}}{2}\,(\cos I_1 + \cos I_2)\,\text{RF} &&(+\text{Down})
\end{aligned}
$$

**Symbols:** $\beta$ is the *dogleg angle* — the total change in direction over the interval (the
angle between the two unit direction vectors). $\text{RF}$ (ratio factor) corrects the simple
straight-segment average so the path follows the circular arc; as the hole goes straight,
$\beta \to 0$ and $\text{RF}\to 1$ (the code guards $\beta < 10^{-7}$ to avoid dividing by zero).
$\Delta N, \Delta E, \Delta V$ accumulate into Engineering XYZ — note $\Delta V$ is **downward**,
so the code does `enu[i].z = enu[i-1].z - d_v` (Z is up). **TVD** accumulates $\Delta V$, and
**dogleg severity (DLS)** is reported as $\beta \cdot (30/\Delta\text{MD})$ — degrees of bend per
30 m, the industry standard for "how sharply is this hole turning." DLS feeds directly into
[well planning](well-planning.md) drillability checks.

```python
from geosim.spatial import min_curvature_positions

# stations: (MD, inclination°, azimuth°)
survey = [(0, 0, 0), (500, 0, 0), (1000, 30, 90), (1500, 60, 90)]
res = min_curvature_positions(survey, wellhead=(300, -150), kb_elev=1622.0)
res.enu   # (N,3) Engineering XYZ per station
res.tvd   # true vertical depth below KB
res.dls   # dogleg severity per interval, °/30 m
```

---

## 6. The units registry

Every numeric quantity carries an explicit unit; **nothing is dimensionless-by-assumption**. The
backend uses a single [`pint`](https://pint.readthedocs.io/) registry (`geosim/spatial/units.py`)
and **converts to canonical internal units on ingest**, storing the canonical unit in metadata and
the original unit in [provenance](data-model.md#6-provenance-a-dag-of-how-every-number-was-made). Think of `to_canonical()` as the
parser that turns any external unit into your one internal representation, and `to_display()` as
the formatter at the UI edge.

The canonical unit per property (the real `CANONICAL_UNITS` dict):

| Property | Canonical unit |
|---|---|
| length / coordinates | `m` |
| [resistivity](survey-methods/electrical.md) | `Ω·m` (`ohm*m`) |
| conductivity | `S/m` |
| density | `kg/m³` |
| magnetic susceptibility | `dimensionless` (SI) |
| [seismic velocity](survey-methods/seismic.md) (P, S) | `m/s` |
| chargeability (time-domain) | `chargeability_time_ms` — `ms` |
| chargeability (frequency-domain) | `chargeability_mv_v` — `mV/V` |
| IP phase | `phase_mrad` — `mrad` |
| **temperature (absolute)** | **`K` (kelvin)** — *not* °C internally |
| temperature **gradient** | `K/km` |
| temperature **uncertainty / Δ** | `K` |
| [gravity anomaly](survey-methods/potential-fields.md) | `mGal` |
| [magnetic field](survey-methods/potential-fields.md) | `nT` |
| [deformation (InSAR)](survey-methods/insar.md) | `mm` (with a time axis) |

`mGal` and `nT` are not in `pint`'s default registry, so the code defines them
(`Gal = cm/s²`, `mGal = 1e-3 Gal`; `nT` resolves via the SI prefix on tesla). A
companion **property-type registry** (`property_types.py`) pins, *per property*: canonical unit,
default colourmap, log/linear scaling, a sensible display range, and the *interpolation space* used
by [fusion](fusion.md) (resistivity interpolates in $\log_{10}$ because it spans orders of
magnitude). A new survey method declares its property once and the entire stack — units, storage
metadata, colour mapping, viewer defaults, fusion — knows how to handle it. [Plugins](architecture.md)
register new keys here.

### 6.1 Gotcha #1: temperature is canonical in **kelvin**, displayed in °C

This is the subtle one. `pint` treats `degC` as an **offset unit**: an *absolute* temperature and
a *temperature difference* are different quantities that happen to share the symbol. 20 °C as an
absolute temperature is 293.15 K, but a *change* of 20 °C is a change of 20 K. If you store a
standard deviation, a gradient, or any *difference* as `degC`, ordinary arithmetic silently
corrupts it (it adds the 273.15 offset where it must not).

```python
from geosim.spatial import to_canonical, to_display

to_canonical(150.0, "degC", "temperature")        # 423.15  (K — absolute)
to_display(423.15, "temperature")                  # 150.0   (°C — for the UI only)
```

So the rule, enforced by storing distinct `PropertyTypeKey`s:

- **absolute temperature** → store in **K** (`temperature`),
- **temperature gradient** → store in **K/km** (`temperature_gradient`),
- **temperature uncertainty / Δ** → store in **K** (`temperature_sigma`),
- convert to **°C only for display** (the `DISPLAY_UNITS` table maps `temperature → degC`).

This carries straight into [uncertainty handling](uncertainty.md) and the
[data model](data-model.md): a temperature field's `_sigma` array is in K, not °C.

### 6.2 Gotcha #2: "chargeability" is three different measurements

!!! note "Term: chargeability / IP"
    **Induced polarization (IP)** measures how much the ground *temporarily stores charge* (like a
    leaky capacitor) — a proxy for clay and disseminated sulphide minerals. See
    [electrical methods](survey-methods/electrical.md).

The catch: there is no single "chargeability" unit. Time-domain IP, frequency-domain IP, and IP
phase are *distinct physical measurements* with incompatible units. Collapsing them into one
canonical unit would be a category error — like storing duration and frequency in the same column.
So they get **distinct property keys**, never merged:

| Measurement | Property key | Canonical unit |
|---|---|---|
| time-domain IP | `chargeability_time_ms` | `ms` |
| frequency-domain IP | `chargeability_mv_v` | `mV/V` |
| IP phase | `phase_mrad` | `mrad` |

---

## Key takeaways

- The platform has **one internal coordinate system**, the **Engineering Frame**: local ENU
  metres, **Z = elevation, up**, origin at a project anchor. Local and georeferenced data share
  one code path.
- Working locally is not cosmetic: it is required for **float32 GPU precision** (large UTM
  coordinates jitter at ~3 cm; a local origin gets you to millimetres). This is the floating-origin
  pattern.
- **Web Mercator is banned** for measurement; UTM (auto-selected) is the default projected CRS,
  with escalation to custom Transverse Mercator for large areas.
- `mode` (georeferenced/local) is separate from `georefStatus` (the *quality* of that anchoring);
  re-anchoring is byte-free but does **not** make synthetic data "real."
- **Vertical** is the danger zone: canonical = orthometric **elevation** (Z-up, m); depth, TVDSS,
  TVD, and MD are all *derived views* with explicit one-line conversions.
- Crooked boreholes are reconstructed from a **deviation survey** via **minimum curvature** — the
  same integrator serves ingest and planning, and yields TVD and dogleg severity.
- The **`pint` units registry** canonicalizes on ingest. Two gotchas: **temperature is canonical
  in kelvin** (offset-unit trap), and **chargeability is three distinct property keys**, never one.

## Where this lives in the code

| Concern | Module |
|---|---|
| `SpatialFrame`, Engineering⇄CRS transforms, UTM auto-select, promotion | `backend/geosim/spatial/frame.py` |
| Elevation/depth/TVDSS/TVD conversions + `min_curvature_positions` | `backend/geosim/spatial/vertical.py` |
| `pint` registry, `CANONICAL_UNITS`, `to_canonical` / `to_display`, temperature/IP gotchas | `backend/geosim/spatial/units.py` |
| Property-type registry (unit + colormap + scaling + interp space per property) | `backend/geosim/spatial/property_types.py` |
| `SpatialFrame` persisted as a catalog row (`spatial_frame` table) | `backend/geosim/catalog/models.py` |

Design source of truth: `design/01-spatial-framework.md` (and §4.3 of `design/09-drilling-well-planning.md`
for minimum curvature). Next, see [the data model](data-model.md) for the primitives these
coordinates and units live inside.
