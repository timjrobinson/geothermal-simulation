# InSAR — ground deformation from space

> **What you'll learn / why it matters.** InSAR (Interferometric Synthetic Aperture
> Radar) lets a satellite measure how the ground **moves** — swelling up or sinking
> down — to a precision of **millimetres**, over whole regions, without anyone setting
> foot on site. Unlike every other method in this course, it doesn't image the rock
> directly: it watches the **surface** and lets you *infer* what's happening kilometres
> below. For geothermal work that surface wobble is a fingerprint of **pressure changes
> in the reservoir** — inject water and the ground bulges; produce fluid and it
> subsides. And because a satellite passes over the same spot again and again, InSAR is
> **4D**: a *time-series of images*, not one snapshot. That makes it the canonical "time
> axis" dataset in our model, and the page where we nail down how time is encoded.

---

## 1. The one-sentence version, then the physics

> **InSAR measures the change in distance between a satellite and the ground, between
> two radar passes, by comparing the *phase* of the reflected radar wave.**

A radar satellite (Sentinel-1, for example) flies overhead and shouts a microwave pulse
at the ground. The ground reflects it; the satellite records the **echo**. Two things
come back in that echo:

1. **Amplitude** — *how much* energy bounced back (bright = buildings/rock, dark =
   water/smooth surfaces). This is the ordinary "radar image."
2. **Phase** — *where in its cycle* the returning wave is. A radar wave is a sinusoid
   with a known wavelength $\lambda$ (for Sentinel-1, $\lambda \approx 56\ \text{mm}$).
   The phase tells you the round-trip distance **modulo one wavelength**.

Phase is the magic ingredient. If you fly over the same place **twice** and the ground
has moved even a few millimetres *toward or away from the satellite* between the two
passes, the round-trip path length changed, and so the **phase changed**. Subtract the
two phase images and you get an **interferogram**: a map of how much each point moved
along the satellite's viewing direction.

!!! note "Analogy: phase is a sub-sample clock"
    Think of sampling a sine wave (a [signal](../glossary.md) you already understand).
    The *amplitude* is the sample value. The *phase* is "how far into the current cycle
    are we" — a fractional position between 0 and $2\pi$. Two recordings of the same
    signal that are shifted by a tiny time offset look identical in amplitude but differ
    in phase. InSAR reads that phase offset and converts it to a distance offset. It is
    a ruler whose smallest tick is a *fraction of a wavelength*, which is why it can see
    millimetres with a sensor hundreds of kilometres away.

### What "interferometry" buys you, conceptually

A single radar pass can locate a point on the ground to maybe a few **metres**. That's
useless for measuring millimetre creep. But the *difference* of two phase measurements
cancels almost everything that's common to both passes (the bulk distance to the
ground, the satellite's nominal orbit) and leaves only what **changed** — the tiny
deformation. This is the same trick as a [diff](../glossary.md) in version control: the
absolute files are huge and mostly identical; the *delta* is small and is the only thing
you care about.

The deformation $d_\text{LOS}$ relates to the measured phase difference
$\Delta\phi$ by:

$$
d_\text{LOS} = \frac{\lambda}{4\pi}\,\Delta\phi
$$

- $\lambda$ — radar wavelength (m), a known constant of the satellite.
- $\Delta\phi$ — phase difference between the two passes (radians).
- $d_\text{LOS}$ — displacement **along the line of sight** (m). The factor of $4\pi$
  (not $2\pi$) is because the wave makes a **round trip** — out and back — so a given
  ground movement changes the path by *twice* that movement.

!!! warning "Phase wrapping — the lossy part"
    Phase is only known **modulo $2\pi$** (it "wraps" every wavelength), exactly like
    an angle that resets at 360°. The raw interferogram is therefore *wrapped*: a smooth
    deformation shows up as repeating colour fringes. Recovering the true continuous
    deformation requires **phase unwrapping** — counting how many whole cycles you
    crossed. This is a hard, error-prone inverse step (think: reconstructing an
    integer-overflowed counter from its low bits). The platform ingests the **already
    unwrapped, processed** product (a deformation raster in millimetres), not raw phase
    — but it is worth knowing the number you receive is the output of a non-trivial,
    sometimes-wrong reconstruction.

---

## 2. Line of sight — the most important caveat

InSAR does **not** measure vertical movement. It measures movement **along the
satellite's line of sight (LOS)** — the straight line from the ground point up to the
radar.

> **Line of sight (LOS)** — the unit vector pointing from a ground point to the
> satellite. The satellite looks down at a slant (typically 30–45° off vertical), so the
> LOS is a *mix* of vertical and horizontal directions.

A LOS vector in our Engineering Frame (X = East, Y = North, Z = Up; see
[Coordinates, depth & units](../spatial-framework.md)) is three numbers,
$\hat{l} = (l_E, l_N, l_U)$. The number InSAR reports is the **projection** of the true
3D ground displacement $\vec{u} = (u_E, u_N, u_U)$ onto that vector:

$$
d_\text{LOS} = \vec{u}\cdot\hat{l} = u_E\,l_E + u_N\,l_N + u_U\,l_U
$$

This is just a dot product — a [projection](../glossary.md) of a 3D vector onto a 1D
axis, the same operation as taking one component of a vector. It is **lossy**: you get
one scalar per pixel, not the full 3D motion. A purely horizontal slide and a purely
vertical lift can produce the *same* LOS reading. To recover full 3D motion you need
**multiple geometries** — e.g. an *ascending* track (satellite heading north) and a
*descending* track (heading south), whose LOS vectors differ, then solve the small
linear system. Our synthetic generator emits a single track with an explicit LOS
vector (e.g. `los: [0.6, -0.1, 0.79]` in `acquisition.jsonc`, doc 05 §4.3), so the
vertical component $l_U \approx 0.79$ dominates but is not the whole story.

!!! example "Why the dot product matters for geothermal"
    A geothermal reservoir inflating from injection pushes the surface **straight up**
    ($u_U > 0$, $u_E = u_N \approx 0$ directly above it). With $l_U = 0.79$, an actual
    10 mm uplift shows up as only $10 \times 0.79 = 7.9$ mm in the LOS image. If you
    forget the projection and read 7.9 mm as the real uplift, you underestimate the
    reservoir pressure change by 20%. The LOS vector is therefore stored with the data
    and must travel into any inversion.

---

## 3. Why it's 4D — a time-series of rasters

A satellite revisits the same ground track on a fixed cadence (Sentinel-1: every
**12 days**). Each pass yields one deformation map. Stack them in acquisition order and
you have a **movie of the ground**: for each pixel $(x, y)$, a curve of
displacement-versus-time.

In data-structure terms, an InSAR product is a **3D array** indexed
`[t, y, x]`:

- `t` — epoch index (which satellite pass).
- `y, x` — raster row/column (a regular grid of ground pixels).

This is a 2D [raster](../glossary.md) (an image) with a **leading time axis** — exactly
a video. The platform treats time as **always the first array axis** when present (see
[the data model](../data-model.md) §8), and stores the **explicit calendar dates** of
each pass as ISO-8601 UTC strings, because satellite passes are *irregular in time*
(orbits get skipped, weather kills some interferograms) — you can never assume a fixed
$\Delta t$ between frames.

### What causes geothermal deformation

The ground over a geothermal field moves because **pressure and temperature change the
volume of the rock and pore fluid** at depth:

| Cause | Surface effect | Why |
|---|---|---|
| **Fluid injection** (e.g. EGS stimulation, reinjection) | **Uplift** (ground bulges up) | Raising pore pressure inflates the reservoir; the overburden lifts. |
| **Fluid production** (pumping out hot water/steam) | **Subsidence** (ground sinks) | Dropping pore pressure compacts the reservoir; the surface settles. |
| **Thermal contraction/expansion** | slow uplift or subsidence | Cooling injected zones contract; heated zones expand. |

A simple, classic model for the surface bowl produced by a small pressurised source at
depth is the **Mogi source** — a point pressure change in an elastic half-space. It
predicts a radially symmetric uplift bowl whose vertical displacement is:

$$
u_U(r) = \frac{(1-\nu)\,\Delta V}{\pi}\cdot\frac{D}{\left(r^2 + D^2\right)^{3/2}}
$$

- $r$ — horizontal distance from the point above the source (m).
- $D$ — depth of the source (m).
- $\Delta V$ — volume change of the source (m³).
- $\nu$ — Poisson's ratio of the rock (dimensionless, ~0.25).

The shape is a smooth bell curve centred over the source — wide and gentle for a deep
source, tight and peaked for a shallow one. **This is itself an inverse problem**: from
the *surface* bowl you infer the *depth and volume* of the buried pressure change, the
same "infer the hidden cause from the visible effect" structure as every geophysical
method in this course. Our synthetic InSAR forward model literally builds a Mogi-like
Gaussian uplift bowl that grows over time and projects it to LOS
(`backend/geosim/synthgen/forward/surface.py`, `InSARForward`).

---

## 4. Noise sources — what makes InSAR lie

InSAR is precise but not clean. The phase you measure is contaminated by things that
have nothing to do with ground motion:

- **Atmospheric phase screen (APS).** Radar waves slow down passing through wet air. If
  the troposphere's water content differs between the two passes, that adds a spurious
  phase that looks exactly like deformation. It is *spatially correlated* (smooth
  blobs), so it mimics a real deformation signal — the hardest noise to remove. Our
  forward model injects a smoothed correlated noise field to mimic this (doc 05 §4).
- **Decorrelation.** Phase comparison only works if the ground *scatters radar the same
  way* on both passes. Vegetation growing, snow, ploughed fields, or just a long time
  gap **decorrelate** the signal — the phase becomes random noise. Cities and bare rock
  stay coherent; forests and farmland fall apart.
- **Orbital / DEM error.** Imperfect knowledge of the satellite's orbit or of the
  ground elevation model adds smooth ramps and tilts across the scene.

The standard answer to decorrelation is **Persistent Scatterers (PS)**: instead of using
every pixel, you keep only the points that stay radar-bright and phase-stable across the
*whole* time-series (rock outcrops, buildings, corner reflectors). PS InSAR gives you a
**sparse set of points**, each with a clean millimetre-level deformation time-series —
delivered as a **CSV table of points**, not a raster. So InSAR arrives in *two* shapes:
dense rasters (one per epoch) and sparse PS point time-series.

---

## 5. The native file formats (annotated)

### 5.1 GeoTIFF time-series — one raster per epoch

The dense product is a **stack of GeoTIFFs**, one file per satellite pass, named in
epoch order. Each is a single-band float32 raster of LOS deformation in millimetres.
This is exactly what the synthetic generator writes (`los_00.tif … los_NN.tif` in one
directory) and what the adapter reads back.

```text
insar/
  los_00.tif    # epoch 0  (e.g. 2026-01-01)  ── single-band float32, units = mm (LOS)
  los_01.tif    # epoch 1  (e.g. 2026-01-13)  ── +12 days, Sentinel-1-like 12-day repeat
  los_02.tif    # epoch 2  (e.g. 2026-01-25)
  ...
  los_NN.tif    # last epoch
```

A GeoTIFF is a [raster](../glossary.md) with an embedded **geotransform** — the affine
mapping from pixel `(col, row)` to ground `(x, y)`. The fields that matter to ingestion
(read with `rasterio`, doc 03):

```python
# rasterio reads these from each los_NN.tif header:
transform.a   #  +dx  : ground metres per pixel column (e.g. 50.0)  → pixel width
transform.e   #  -dy  : metres per row; NEGATIVE because raster row 0 is at the NORTH (top)
transform.c   #  x0   : x-coordinate of the top-left pixel corner (Engineering metres)
transform.f   #  y_top: y-coordinate of the top-left pixel corner (Engineering metres)
r.read(1)     #  the float32 band: LOS deformation in mm, shape [ny, nx]
r.crs         #  CRS — None in local/synthetic mode (coords ARE Engineering metres)
```

!!! note "The north-up flip — a real gotcha"
    Raster row 0 is the **northernmost** row (image convention: top row first), so
    `transform.e` is **negative** (`y` *decreases* as `row` increases). But our
    Engineering Frame is **Z-up / Y-north ascending**. The InSAR adapter therefore
    `flipud`s each band so the array's first row is the **southernmost** (minimum `y`)
    before stacking, giving a clean ascending `[t, y, x]` cube. This is the same
    image-row-order mismatch that bites everyone moving between screen coordinates
    (top-left origin) and math coordinates (bottom-left origin).

### 5.2 CSV — Persistent Scatterer points

PS products are tabular: one row per stable scatterer, with its position and a
deformation **value per epoch** (a time-series spread across columns):

```csv
ps_id,x,y,los_mean_mm_yr,coherence,d_20260101,d_20260113,d_20260125,...
PS00001,-1240.5,820.3,12.4,0.91,0.0,1.1,2.3,...     # an uplifting point: +mm per epoch
PS00002,640.0,-310.2,-3.1,0.88,0.0,-0.4,-0.9,...    # a subsiding point
# x,y           : Engineering metres (East, North)
# los_mean_mm_yr: average LOS velocity (mm/year) — the headline "is it moving?" number
# coherence     : 0..1 phase-stability score (1 = rock-solid scatterer)
# d_<date>      : LOS displacement (mm) at that epoch, relative to epoch 0
```

A PS point is a [point Observation](../data-model.md) carrying a time-series — the dot
product of §2 is baked into each `d_<date>` column (it is already a LOS scalar).

---

## 6. What it becomes in the model

After ingestion ([Ingestion — raw files to primitives](../ingestion.md)), InSAR lands as
two possible normalized primitives — both defined in [the data model](../data-model.md):

| Native shape | Normalized primitive | Support / geometry |
|---|---|---|
| GeoTIFF time-series | **`PropertyModel`** (`property: "deformation"`) | `support.kind = "grid2d"` with a **leading `t` axis** → `[t, y, x]` |
| PS CSV | **`ObservationSet`** (`geometryKind: "points"`) | per-point XYZ + a `TimeAxis` of epochs |

The dense raster becomes a `PropertyModel` of property type `deformation` (canonical
unit **mm**, colormap `RdBu` — red/blue divergence so uplift and subsidence read at a
glance; see `backend/geosim/spatial/property_types.py`). Its support is a 2D grid
(`grid2d`) carrying `origin`, `spacing`, and the all-important **`TimeAxis`** of explicit
ISO-8601 UTC epochs. Concretely, the adapter emits:

```python
RawPropertyModel(
    property="deformation",
    values=cube,                 # float32 array, shape [t, y, x]  (t leads)
    origin=(0.0, y0, x0),        # (z, y, x): z spacing is 0 — a grid2d has no z extent
    spacing=(0.0, dy, dx),
    support="grid2d",
    meta={
        "timeAxis": {"epochs": ["2026-01-01T00:00:00Z", ...], "unit": "ISO-8601-UTC"},
        "los": "line_of_sight",  # the LOS vector concept rides into the model
        "leading_axis": "t",
    },
)
```

The LOS vector and "this number is a *projection*, not vertical motion" travel with the
dataset so downstream consumers (fusion, an inflation-source inversion) never mistake LOS
for true 3D displacement. The 4D time axis is what drives the **time slider** in
[the 3D viewer](../visualization.md).

---

## Key takeaways

- **InSAR measures surface deformation along the satellite line of sight (LOS)** in
  millimetres, by differencing the **phase** of two radar passes — a "diff of two
  signals" that cancels everything except what moved.
- The reading is a **dot product** of the true 3D motion onto the LOS vector: **lossy**
  (one scalar per pixel), so it can't distinguish horizontal from vertical without
  multiple viewing geometries. **The LOS vector must travel with the data.**
- It is intrinsically **4D**: a time-series of rasters `[t, y, x]` with **explicit,
  irregular ISO-8601 UTC epochs** — never an assumed fixed cadence.
- Geothermal deformation comes from **reservoir pressure changes** — injection → uplift,
  production → subsidence — modellable as a **Mogi source** (itself an inverse problem:
  surface bowl → buried source depth/volume).
- Noise is dominated by the **atmosphere** (correlated, mimics signal), **decorrelation**
  (vegetation/time), and **orbit/DEM error**; **Persistent Scatterers** trade dense
  coverage for clean points.
- Normalized form: dense → **`PropertyModel(deformation, grid2d)` with a leading time
  axis**; PS → **point `Observation`s** carrying a time-series.

## Where this lives in the code

- Adapter (GeoTIFF time-series → `PropertyModel(deformation, grid2d)`):
  `backend/geosim/ingestion/adapters/insar.py` (`InsarGeotiffAdapter`).
- Synthetic forward (Mogi-like uplift bowl → LOS GeoTIFF series):
  `backend/geosim/synthgen/forward/surface.py` (`InSARForward`).
- The `deformation` property type (unit `mm`, colormap `RdBu`):
  `backend/geosim/spatial/property_types.py`.
- Vertical / coordinate handling the adapter normalizes into:
  `backend/geosim/spatial/vertical.py`, `backend/geosim/spatial/frame.py`.
