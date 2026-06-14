# Gravity & Magnetics — the potential fields

> **What you'll learn / why it matters.** Gravity and magnetics are the two cheapest,
> fastest, most regional reconnaissance methods in geophysics — you can fly them over a
> whole prospect in a day. They are also the two most *humbling*: both are **potential
> fields**, which makes them smooth, deep-reaching, and badly **non-unique**. This page
> teaches the physics of both from Newton's law and induced magnetization, defines their
> units (**mGal** and **nT**), shows you the real CSV / GeoTIFF files the platform ingests,
> and explains the single most useful geothermal trick of the whole pair: the **magnetic
> low** that hydrothermal alteration carves into the magnetic map. Read
> [how to read these pages](index.md) first if you haven't — it defines *potential field*,
> *DOI*, *resolution kernel*, and *non-uniqueness*, which this page leans on constantly.

## Why "potential field"? (and why it dooms you to blur)

A **potential field** is a field that can be written as the gradient of a scalar
**potential** function — and gravity and magnetism are the two classic examples. The
gravitational potential of a mass and the magnetic potential of a magnetized body both obey
**Laplace's equation** in the empty air above the ground:

$$
\nabla^2 \phi = 0 \quad\text{(in source-free space)}
$$

where $\phi$ is the potential and $\nabla^2$ is the Laplacian (the sum of second spatial
derivatives). You don't need to solve this; you need its **consequence**. Harmonic fields
(solutions of Laplace's equation) are extraordinarily **smooth**: they cannot have sharp
local features in the air, and as you move *away* from the sources, the field is **low-pass
filtered** more and more heavily. Moving the sensor up — or, equivalently, the source down —
acts like a blur kernel whose width grows with distance. This single fact explains
everything annoying about potential fields:

- **They are smooth.** Sharp boundaries underground show up as gentle gradients at the
  surface.
- **They are deep-reaching but depth-blind.** The field sums contributions from *all*
  mass / magnetization at once; it has essentially **no inherent depth resolution**.
- **They are badly non-unique.** A small shallow body and a big deep body can produce an
  identical surface anomaly. (This is the textbook example of
  [non-uniqueness](index.md#non-uniqueness).)

!!! abstract "The lossy-compression analogy"
    "Source down = blur up" is mathematically the same operation as **upward continuation**,
    a literal low-pass filter applied to the field. Burying a feature is like applying a
    stronger Gaussian blur to an image. By the time the signal of a deep, small body reaches
    the surface, it has been *lossily compressed* into a faint, broad smudge — and you cannot
    invert that smudge back to a unique shape without extra assumptions.

Gravity and magnetics differ only in **what the source is** (mass for gravity, magnetization
for magnetics) and **what property controls it** (density for gravity, magnetic
susceptibility for magnetics). The blur is the same.

---

# Part 1 — Gravity

## The physics: Newton, summed over the ground

Gravity surveying measures tiny variations in the strength of Earth's gravitational pull,
caused by **lateral changes in rock density**. Denser rock pulls a little harder; less dense
rock (a sedimentary basin, a fractured or altered zone) pulls a little less.

Newton's law of gravitation gives the vertical acceleration $g_z$ that a small excess mass
$\Delta m$ at distance $r$ produces at the sensor. Summed over all the rock voxels in the
ground, the platform's forward model uses the far-field **point-mass kernel**:

$$
g_z \;=\; G \sum_{\text{voxels}} \frac{\Delta m \,\cdot\, \Delta z}{r^{3}}
\;=\; G \sum_{\text{voxels}} \frac{\Delta\rho \, \cdot V \,\cdot\, \Delta z}{r^{3}}
$$

- $G = 6.674\times10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}}$ — the gravitational constant.
- $\Delta\rho$ — the **density anomaly**: the rock's density *minus the background density at
  its depth* (kg/m³). Only *contrasts* produce an anomaly; a uniform layered earth produces
  none.
- $V$ — the volume of the voxel, so $\Delta m = \Delta\rho\,V$ is the excess mass.
- $\Delta z$ — vertical distance from voxel to sensor; the $\Delta z/r^3$ factor extracts the
  *vertical* component of the pull (a gravimeter only senses "down").
- $r$ — the 3-D distance from voxel to sensor.

The $1/r^3$ falloff (for the vertical component) is gentle — that is exactly why gravity
reaches deep but blurs. You can read this kernel almost verbatim in
`backend/geosim/synthgen/forward/potential_field.py` (`GravityForward.simulate`); the
rigorous tier (`GravityRigorousForward`) swaps the point mass for the **exact Newtonian
attraction of a rectangular prism** (the Nagy formula, via the `harmonica` library), which
is identical in the far field but correct close to a finite body.

!!! note "What the background subtraction does"
    The code computes $\Delta\rho$ as density minus the **per-depth median** density. This is
    why a flat, layered earth shows a flat gravity map: lining up the layers cancels. Only
    *lateral* mass contrasts — a dense intrusion, a light fault gouge, a basin of low-density
    fill — survive into the anomaly. That subtracted product is the **Bouguer anomaly**.

### Unit: the milligal (mGal)

Gravity anomalies are minute. The unit is the **gal** (after Galileo), where
$1\ \mathrm{gal} = 1\ \mathrm{cm/s^2} = 0.01\ \mathrm{m/s^2}$. Surveys report the
**milligal**:

$$
1\ \mathrm{mGal} = 10^{-3}\ \mathrm{gal} = 10^{-5}\ \mathrm{m/s^2}
$$

For scale: Earth's gravity is about $9.8\ \mathrm{m/s^2} \approx 980{,}000\ \mathrm{mGal}$,
and a meaningful exploration anomaly might be **a few mGal** — about one part in a million.
A modern gravimeter resolves a few **hundredths** of a mGal. The synthetic forward adds
~0.03 mGal of Gaussian noise to match that.

### The "Bouguer anomaly"

What surveys actually publish is the **Bouguer anomaly**: the raw reading after stripping out
predictable effects (latitude, the sensor's elevation, and the gravitational pull of the
topographic mass between the sensor and the datum — the "Bouguer slab"). What remains is the
signal of **subsurface density contrasts** — the thing you actually want. In this codebase
the synthetic forward emits the Bouguer anomaly directly (the `bouguer_mgal` column).

## What gravity CAN and CAN'T see (geothermal)

| Gravity is good at | Gravity is blind to |
|---|---|
| **Basin shape** — thick low-density sediment reads as a gravity low | **Depth of a feature** — depth-blind, badly non-unique |
| **Faults** — density step across a fault offsets the field | **Fine/sharp detail** — everything is smoothed |
| **Intrusions** — a dense igneous body (possible heat source) reads as a high | **Heat directly** — density barely changes with temperature |
| **Density contrasts of any kind, to depth** | **Fluid/permeability** — only seen if they change bulk density |

For geothermal specifically, gravity is a **structural** tool: it maps the basin geometry and
the faults that might channel fluids, and it can flag a dense intrusion that could be the
heat source. It does **not** see heat or fluid directly.

## DOI & resolution

- **Depth of investigation:** effectively "everything, blurrily." Gravity has **no intrinsic
  depth resolution** — the field is the sum over all depths at once. Practical depth
  sensitivity is limited by your ability to separate a deep, broad anomaly from a shallow,
  broad one (you can't, without other constraints).
- **Lateral resolution:** set by **station spacing** (a sampling-rate problem) and by the
  inherent smoothing. The forward applies an explicit Gaussian low-pass to the station grid
  to model this. Closer stations → finer resolvable features, up to the physics limit.

## Native file format (annotated)

The platform ingests gravity as a **pair**: a CSV of station readings and a Bouguer-anomaly
**grid** (GeoTIFF). The synthetic generator writes both
(`potential_field.py` → `_emit_gravity`); real surveys arrive the same way (columnar CSV +
`.grd`/GeoTIFF).

### Station CSV — `gravity_stations.csv`

```csv
# unit: mGal              # (1) optional metadata comment — sets the source unit
# crs: local              # (2) coordinate reference; "local" = Engineering frame, no projection
station,x,y,elev,bouguer_mgal   # (3) the column header
0,0.00,0.00,1205.0,-1.83        # (4) one reading per row
1,250.00,0.00,1205.0,-1.71
2,500.00,0.00,1205.0,-1.42
3,750.00,0.00,1205.0,-0.95
4,1000.00,0.00,1205.0,-0.31
```

1. A `# key: value` comment line tells the adapter the **source unit** (mGal). Missing units
   trigger a loud warning — silently mis-scaled data is the worst failure mode.
2. The CRS line; `local` (or absent) means the X/Y are already Engineering-frame **metres**,
   so no reprojection is needed.
3. The columns the adapter recognizes (it accepts aliases like `easting`/`northing`,
   `gravity`, `mgal`, and an optional `sigma`/`error` column):
   - `station` — an id for the reading.
   - `x`, `y` — horizontal position (metres east/north here).
   - `elev` — the sensor elevation in metres (Z-up).
   - `bouguer_mgal` — **the measurement**: the Bouguer anomaly in mGal.
4. Each row is one gravity station: *what was measured (the anomaly) and where (x, y, elev).*

### Bouguer grid — `gravity_bouguer.tif`

The companion **GeoTIFF** is a single-band raster of the gridded anomaly — already
interpolated onto a regular 2-D grid. It is a small binary file, but its essential structure
is:

```text
Driver:        GTiff                 # (1) GeoTIFF raster
Band 1 Type:   Float32               # (2) one band of floating-point mGal values
Size:          41 x 41 (cols x rows) # (3) the grid dimensions (nx, ny)
NoData Value:  nan                   # (4) gaps are NaN, not a sentinel number
Affine transform (pixel -> ground):  # (5) maps pixel (col,row) -> Engineering metres
   | 250.0   0.0    -125.0 |         #   dx = 250 m east per pixel; x-origin
   |  0.0  -250.0   5125.0 |         #   dy = 250 m north per pixel (negative: row 0 = north)
CRS:           (none — local frame)  # (6) no projection; ground coords are Engineering metres
```

1. **GeoTIFF** is just a TIFF image with georeferencing baked in — the workhorse format for
   2-D geophysical grids (the platform stores them as Cloud-Optimized GeoTIFFs, COGs).
2. The pixel values *are* the anomaly in mGal (`Float32`).
3. The grid shape — a regular raster, unlike the scattered CSV stations.
4. `NoData = nan` marks gaps (areas with no coverage) so they don't get plotted as zero.
5. The **affine transform** is the key: it's a 2×3 matrix turning a pixel `(col, row)` into a
   ground `(x, y)` in Engineering metres. Note row 0 is the **northmost** row (the
   convention `rasterio` uses, which is why the writer flips the array). This is the raster
   equivalent of an origin + spacing.
6. No CRS — these are local Engineering metres, read straight off the affine.

## The normalized primitives gravity becomes

Ingestion (`backend/geosim/ingestion/adapters/gravity.py`,
`gravity_csv.py`) turns the pair into two different primitives:

- **The CSV → an `Observation`** with `geometryKind: "points"`, carrying the
  `gravity_anomaly` property (canonical unit mGal), one value per station, plus any `sigma`
  column as paired uncertainty. This is the **immutable raw record** — *what was measured
  where.* It is **not** auto-gridded; raw stays raw.
- **The GeoTIFF → a `PropertyModel`** with `support.kind: "grid2d"`, carrying
  `gravity_anomaly` as a continuous 2-D field embedded in 3-D (a single layer at the
  observation elevation). This is the *already-gridded* product.

To go from gravity to a **3-D density volume** you must run an **inversion** (the inverse
problem) — and because gravity is non-unique, that inversion needs regularization and,
ideally, constraints from other methods. The result is a `PropertyModel` with
`support.kind: "volume"` carrying `density`. See [inversion](../inversion.md).

```text
gravity_stations.csv ──▶ Observation(points, gravity_anomaly)        # raw, immutable
gravity_bouguer.tif  ──▶ PropertyModel(grid2d, gravity_anomaly)      # gridded product
        (later, via inversion) ──▶ PropertyModel(volume, density)    # the inverse problem
```

---

# Part 2 — Magnetics

## The physics: induced magnetization of susceptibility

Magnetic surveying measures tiny variations in Earth's magnetic field caused by **magnetic
minerals in the rock** — overwhelmingly **magnetite**. Earth's field acts as a giant ambient
magnet $B_0$; rock containing magnetic minerals becomes **induced** into magnetization $M$
proportional to its **magnetic susceptibility** $\chi$ (chi, a dimensionless "how easily
magnetized" number):

$$
M = \chi H = \frac{\chi\, B_0}{\mu_0}
$$

where $H$ is the magnetizing field, $B_0$ is the ambient field strength (Earth's field, about
$50{,}000\ \mathrm{nT}$), and $\mu_0$ is the permeability of free space. Each magnetized
voxel then acts like a tiny **dipole**, and the survey sums the vertical-field anomaly of all
of them. The platform's forward model (`MagneticsForward` in
`potential_field.py`) uses the dipole anomaly:

$$
\Delta B_z \;=\; \frac{1}{4\pi}\sum_{\text{voxels}} \chi\, B_0\, V \,\frac{3\,\Delta z^{2} - r^{2}}{r^{5}} \times 10^{9}\ \ [\mathrm{nT}]
$$

- $\chi$ — **magnetic susceptibility** (dimensionless); the property magnetics actually
  senses.
- $B_0 = 50{,}000\ \mathrm{nT}$ — Earth's ambient inducing field (the code's value).
- $V$ — voxel volume; $\Delta z, r$ — geometry as in gravity.
- The $3\Delta z^2 - r^2$ shape is the dipole kernel; $\times 10^9$ converts tesla to nT.

Magnetics is *also* a potential field: same Laplace smoothness, same depth-blindness, same
non-uniqueness as gravity. The difference is the property ($\chi$, not $\rho$) and the dipole
(rather than monopole) source.

!!! note "Reduced-to-pole (RTP)"
    Because a dipole's anomaly is *lopsided* (skewed by the field's inclination at your
    latitude), raw magnetic maps put the anomaly peak *off* to one side of the body that
    caused it. A processing step called **reduction to the pole (RTP)** mathematically
    re-poses the data as if measured at the magnetic pole (vertical field), so anomalies sit
    *directly over* their sources — much easier to interpret. The synthetic forward emits an
    RTP grid (`product: "RTP"`), and uses a vertical inducing field for exactly this reason.

### Unit: the nanotesla (nT)

Magnetic field strength is measured in **tesla (T)**; survey anomalies are in
**nanoteslas**:

$$
1\ \mathrm{nT} = 10^{-9}\ \mathrm{T}
$$

Earth's field is roughly $50{,}000\ \mathrm{nT}$. Exploration anomalies range from a few nT
to a few thousand nT. A modern magnetometer resolves well under 1 nT; the forward adds ~2 nT
of noise plus a small **per-line leveling drift** (a real artifact of flying long survey
lines).

## The geothermal headline: the *magnetic low* over alteration

Here is the single most important geothermal fact about magnetics, and it is delightfully
counter-intuitive.

Hot geothermal fluids chemically attack the rock they flow through — **hydrothermal
alteration**. One thing this alteration does is **destroy magnetite**, converting it to
non-magnetic minerals. Destroying the magnetite drops the rock's susceptibility $\chi$ toward
zero. So over an active upflow zone, magnetics sees **not a high, but a magnetic *low*** — a
"hole" in the magnetic field where the rock has been demagnetized by the very fluids you're
hunting.

!!! tip "Why this is diagnostic"
    Most methods see heat or fluid only *indirectly*. The magnetic low is a **chemical
    fingerprint of past or present hot-fluid flow** — it marks where alteration has happened.
    The synthetic earth bakes this in: in `potential_field.py` the docstring notes that *"the
    altered plume has $\chi \to \approx 0$, so the magnetics sees a low over the upflow, not
    heat."* This is the [resolution-kernel / "only sees what it could"](index.md#resolution-kernel)
    principle made physical — magnetics can't see heat, but it *can* see the scar heat leaves.

| Magnetics is good at | Magnetics is blind to |
|---|---|
| Mapping **magnetic basement** & magnetic intrusions | Depth of a feature (depth-blind, non-unique) |
| **Hydrothermal alteration** as a magnetic *low* (demagnetization) | Heat & fluid *directly* |
| Faults (offsets in magnetic units) | Non-magnetic structure (no susceptibility contrast = invisible) |
| Fast, cheap **airborne** reconnaissance over huge areas | Sharp/fine detail (smoothed, worse with flight altitude) |

## DOI & resolution

- **DOI:** like gravity, depth-blind and non-unique — the field sums all magnetization.
- **Resolution:** dominated by **flight altitude** (for airborne) and **line spacing**.
  Higher and farther apart = blurrier. The forward models this directly: it applies an
  upward-continuation Gaussian low-pass whose width is **proportional to flight altitude**
  (`sigma_cells = mag_altitude / ... `). Flying lower and tighter buys sharper maps — a
  literal sampling-and-blur tradeoff.

## Native file format (annotated)

Magnetics also arrives as a pair: a **flight-line `.xyz`** of along-line readings and an
**RTP GeoTIFF** grid (`MagneticsForward` → `aeromag_lines.xyz` + `mag_rtp.tif`).

### Flight-line file — `aeromag_lines.xyz`

A whitespace-delimited text file, one reading per row, grouped by survey **line**:

```text
LINE X Y ALT TMI_RTP_nT          # (1) header: line id, position, altitude, the measurement
0 0.00 0.00 1305.00 52.1430      # (2) line 0, first sample
0 31.25 0.00 1305.00 53.8021     # (3) dense ALONG the line (small X step)...
0 62.50 0.00 1305.00 55.2096
1 0.00 250.00 1305.00 48.7711    # (4) line 1 is 250 m NORTH — coarse ACROSS lines
1 31.25 250.00 1305.00 47.9902
```

1. The header names the columns the adapter recognizes (aliases like `tmi`, `rtp`,
   `magnetic_field`, `mag` are also accepted):
   - `LINE` — the flight-line id (kept in `meta` so per-line leveling stays inspectable).
   - `X`, `Y` — position in Engineering metres.
   - `ALT` — sensor altitude (flight elevation).
   - `TMI_RTP_nT` — **the measurement**: Total Magnetic Intensity, reduced to pole, in nT.
2–3. Samples are **dense along each line** (small X step) — that's the high-resolution
   direction.
4. The next **line** is a full line-spacing (here 250 m) to the north — sampling is
   **coarse across lines**. This anisotropy (fine along-line, coarse across-line) is a
   defining feature of airborne data and a real resolution limit.

### RTP grid — `mag_rtp.tif`

Structurally identical to the gravity Bouguer GeoTIFF (single-band `Float32`, NaN nodata,
affine pixel→metres, no CRS), but the band values are **nT** and the product is **RTP**. The
adapter disambiguates "is this the gravity grid or the mag grid?" purely by **filename** (a
bare float grid carries no clue otherwise) — `mag`/`rtp`/`tmi`/`aeromag` in the name claims
it for magnetics.

## The normalized primitives magnetics becomes

Ingestion (`backend/geosim/ingestion/adapters/magnetics.py`) mirrors gravity:

- **The `.xyz` → an `Observation`** with `geometryKind: "points"`, carrying
  `magnetic_field` (canonical unit nT), one value per sample; the `line` id rides along in
  `meta`.
- **The GeoTIFF → a `PropertyModel`** with `support.kind: "grid2d"`, carrying
  `magnetic_field` as a 2-D field.
- **Inverted (later) → a `PropertyModel`** with `support.kind: "volume"` carrying
  `susceptibility`.

```text
aeromag_lines.xyz ──▶ Observation(points, magnetic_field)         # raw, immutable
mag_rtp.tif       ──▶ PropertyModel(grid2d, magnetic_field)       # gridded RTP product
   (later, via inversion) ──▶ PropertyModel(volume, susceptibility)
```

## Where gravity & magnetics are strong for geothermal — together

These two are the **reconnaissance layer** of a geothermal program: cheap, fast, regional.
You fly them early to draw the structural skeleton (basins, faults, intrusions from gravity;
magnetic basement and *especially* the alteration low from magnetics) and to decide where to
spend money on the expensive, sharper, deeper methods like
[MT](electromagnetic.md), [ERT](electrical.md), and [seismic](seismic.md). On their own they
are blurry and non-unique; fused with electrical/EM evidence for fluid and seismic for
structure, they become powerful constraints — see [fusion](../fusion.md).

## Key takeaways

- Gravity and magnetics are **potential fields**: smooth, deep-reaching, **depth-blind**,
  and **non-unique** — burying a source = blurring its surface signal (upward continuation).
- **Gravity senses density contrasts** (mGal; $1\ \mathrm{mGal}=10^{-5}\,\mathrm{m/s^2}$);
  the **Bouguer anomaly** is the cleaned signal of subsurface density. Good for basins,
  faults, intrusions; blind to heat and depth.
- **Magnetics senses susceptibility** (nT; $1\ \mathrm{nT}=10^{-9}\,\mathrm{T}$) via induced
  magnetization; **RTP** re-centers anomalies over their sources.
- The flagship geothermal signal is the **magnetic low** over hydrothermal alteration:
  hot fluids destroy magnetite, demagnetizing the rock — a chemical fingerprint of fluid
  flow.
- Both ingest as a **CSV/`.xyz` `Observation(points)`** plus a **GeoTIFF
  `PropertyModel(grid2d)`**; a 3-D **density/susceptibility volume** only comes from a
  (regularized, non-unique) **inversion**.
- They are the cheap **reconnaissance** layer — best used to target the sharper, deeper
  methods, and best interpreted **fused** with them.

## Where this lives in the code

- Forward models (earth → data simulators): `backend/geosim/synthgen/forward/potential_field.py`
  — `GravityForward` (point-mass sum), `GravityRigorousForward` (Nagy prism via
  `harmonica`), `MagneticsForward` (dipole sum + RTP + upward-continuation low-pass), and
  `write_local_geotiff` (the GeoTIFF writer).
- Ingestion adapters (native files → primitives):
  `backend/geosim/ingestion/adapters/gravity.py` (synthgen CSV + Bouguer GeoTIFF),
  `gravity_csv.py` (generic gravity station CSV), and `magnetics.py` (aeromag `.xyz` + RTP
  GeoTIFF; reuses the gravity GeoTIFF reader).
- Design references: `design/OVERVIEW.md` §3 (rows 1–2), `design/03-ingestion-adapters.md`
  §2 (gravity/magnetics rows), and design doc 05 §4 (the forward-model contract).
