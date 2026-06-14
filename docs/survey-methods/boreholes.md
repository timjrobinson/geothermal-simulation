# Boreholes — well logs, temperature & heat-flow

> **What you'll learn / why it matters.** Every other method in this course is **remote
> inference**: you stand on the surface (or fly, or orbit) and *guess* at the rock from
> physics. A borehole is the exception — it is the **only place we touch the rock
> directly**. We lower instruments down a hole and read resistivity, density, sound
> speed and temperature *at the rock itself*. That makes wells the **ground truth**, the
> **calibration anchor** that tells the remote methods whether their guesses are right.
> The catch: a well is **1D** — a single thread of perfect data through a 3D earth that
> everything else samples coarsely but everywhere. This page covers the logging tools and
> what each measures, how a crooked well's true path is reconstructed (minimum
> curvature), how temperature logs give you the geothermal gradient and heat-flow, and
> the LAS file format that carries it all.

---

## 1. The mental model: a 1D probe through a 3D world

Picture every other survey as a **lossy, low-resolution sensor** that covers the whole
volume: gravity sees the whole basin but blurrily; MT sees deep but smooth. A well log is
the opposite — **infinite resolution, zero coverage**. It is one vertical (or curved)
**line** through the earth along which you know the rock properties almost exactly, every
few centimetres.

!!! note "Analogy: ground-truth labels for a noisy model"
    In machine-learning terms, the remote surveys are a **model** predicting rock
    properties over a 3D grid, and a well is a column of **ground-truth labels**. You
    can't label the whole volume (drilling is expensive — millions of dollars per hole),
    but the few labels you *do* have **calibrate and validate** the model. When the MT
    inversion says "20 Ω·m at 1500 m here" and the well actually logs 18 Ω·m, you trust
    the inversion a little more. When it says 200 Ω·m, you've found a problem. This is
    exactly the **calibration** role wells play in [rock physics](../rock-physics.md).

Because a well is the cleanest data, our synthetic generator treats it as such — it
**samples the truth volumes directly along the well path** with only tiny noise and the
tool's vertical resolution applied (`backend/geosim/synthgen/forward/borehole.py`,
`WellLogForward`). Everything else gets heavily degraded; the well barely.

---

## 2. Logging tools — the curves and what they tell you

A **logging tool** is a sensor lowered down the borehole on a cable (a "wireline"). As it
moves it records a continuous **curve** of one property versus depth. Each curve is a
[signal](../glossary.md) sampled along the depth axis. The standard suite:

| Curve (mnemonic) | Measures | What it tells a geothermal explorer |
|---|---|---|
| **Resistivity** (`RES`) | electrical [resistivity](electrical.md) of the rock, $\Omega\cdot\text{m}$ | Low resistivity = hot, salty, porous, clay-rich → the geothermal signature. Directly ties remote ERT/MT to truth. |
| **Gamma-ray** (`GR`) | natural radioactivity, API units | Shales/clays are radioactive; clean sands/carbonates are not. A clay/alteration indicator. (No canonical property key — see §6.) |
| **Density** (`DEN`/`RHOB`) | bulk density, $\text{kg/m}^3$ | Ties to [gravity](potential-fields.md); porosity reduces density. |
| **Sonic** (`VP`/`DT`) | P-wave velocity $V_p$, m/s (often as slowness $\Delta t$, µs/ft) | Ties to [seismic](seismic.md); soft/fractured/fluid-filled rock is slow. |
| **Temperature** (`TEMP`) | rock/fluid temperature | **The geothermal prize** — the gradient and heat-flow come from here (§4). |

> **Resistivity** — how strongly a rock opposes electrical current
> ($\Omega\cdot\text{m}$); the inverse of conductivity. Defined fully on the
> [electrical methods page](electrical.md).
>
> **P-wave velocity ($V_p$)** — the speed of a compressional ("push") sound wave through
> the rock; faster in hard, dense, unfractured rock. See [seismic](seismic.md).

Each of these maps to a **canonical property type** in the registry (doc 01 §5) so its
units and colourmap resolve automatically: `RES → resistivity`, `DEN/RHOB → density`,
`VP/DT → velocity_p`, `TEMP → temperature`. Gamma-ray has *no* canonical key, so it is
carried along as method-specific metadata rather than a first-class property
(`backend/geosim/ingestion/adapters/welllog.py`, `_CURVE_TO_PROPERTY`).

!!! tip "Why wells are the calibration anchor for rock physics"
    [Rock physics](../rock-physics.md) is the set of equations that convert geophysical
    properties (resistivity, $V_p$) into the things engineers care about (temperature,
    porosity, fluid). Those equations have free parameters. A well gives you *both* the
    geophysical property *and* (via the temperature log) the geothermal answer in the
    **same** rock — so you can **fit** the rock-physics parameters to reality instead of
    using textbook guesses. No wells = uncalibrated rock physics = numbers you can't
    trust.

---

## 3. Where is the well, really? MD vs TVD and the deviation survey

Wells are rarely straight down. Modern geothermal wells **deviate** — they curve to hit a
target off to the side, or to cross a fault at a useful angle. This creates the single
most important coordinate subtlety in borehole data, first introduced on the
[spatial framework page](../spatial-framework.md):

- **MD — Measured Depth.** Length of cable spooled out: distance **along the hole**. The
  logging tool only knows MD — that's literally how much cable is out.
- **TVD — True Vertical Depth.** Straight-line **vertical** depth below the wellhead. This
  is what actually matters for "how deep is the reservoir" and for placing the data in 3D.

!!! warning "MD ≠ TVD for any deviated well"
    A well that drills 2000 m of hole at 60° from vertical has reached only
    $2000 \times \cos 60° = 1000$ m TVD. If you naively plot the log at "2000 m depth"
    you've placed it 1000 m too deep and in the wrong horizontal spot. **MD is the
    measurement axis; TVD/XYZ is the geometry — and you cannot get from one to the other
    without the deviation survey.**

### The deviation survey

A **deviation survey** is a small table recording the well's orientation at intervals
along the hole:

| Column | Meaning | Units |
|---|---|---|
| **MD** | measured depth at this station | m |
| **INC** (inclination) | angle from vertical (0° = straight down, 90° = horizontal) | degrees |
| **AZI** (azimuth) | compass heading of the hole (0° = North, 90° = East) | degrees |

From this you **reconstruct the full 3D path**: at each station you know how far along the
hole you are and which way it's pointing, and you integrate those directions into a
trajectory of $(x, y, z)$ positions in the [Engineering Frame](../spatial-framework.md).

### Minimum curvature — reconstructing the path

The industry-standard integrator is **minimum curvature**, which assumes the hole follows
a **circular arc** between two survey stations rather than two straight segments — a
realistic, smooth path. For an interval between stations 1 and 2 with inclinations
$I_1, I_2$ and azimuths $A_1, A_2$ and measured-depth step $\Delta\text{MD}$:

$$
\cos\beta = \cos(I_2 - I_1) - \sin I_1 \sin I_2\,\bigl(1 - \cos(A_2 - A_1)\bigr)
$$

$$
\text{RF} = \frac{2}{\beta}\tan\!\frac{\beta}{2}\quad(\text{RF}\to 1\ \text{as}\ \beta\to 0)
$$

$$
\Delta E = \tfrac{\Delta\text{MD}}{2}\,(\sin I_1 \sin A_1 + \sin I_2 \sin A_2)\,\text{RF}
$$
$$
\Delta N = \tfrac{\Delta\text{MD}}{2}\,(\sin I_1 \cos A_1 + \sin I_2 \cos A_2)\,\text{RF}
$$
$$
\Delta V = \tfrac{\Delta\text{MD}}{2}\,(\cos I_1 + \cos I_2)\,\text{RF}
$$

- $\beta$ — the **dogleg angle**: the total change in direction across the interval
  (radians). A bigger $\beta$ means a sharper bend.
- $\text{RF}$ — the **ratio factor**, the correction that turns the chord (straight line)
  into the arc length. As the bend vanishes ($\beta\to 0$) it goes to 1 and the formula
  reduces to a simple average of directions.
- $\Delta E,\Delta N$ — eastward/northward steps (added to $x$/$y$).
- $\Delta V$ — vertical drop, the **TVD increment**; in the Z-up Engineering Frame this is
  *subtracted* from elevation ($z \mathrel{-}= \Delta V$).

A related output is **dogleg severity (DLS)** — the dogleg angle normalised to a standard
length, $\text{DLS} = \beta \cdot (30 / \Delta\text{MD})$ in degrees per 30 m. It's the
well's "how sharply is it turning" rate, and the well planner uses it as a drillability
limit (see [well planning](../well-planning.md)). The same routine that places ingested
wells also drives planned wells — it lives in one place,
`backend/geosim/spatial/vertical.py` (`min_curvature_positions`), referenced from the
[spatial framework](../spatial-framework.md).

!!! note "If there's no deviation survey"
    Many old LAS files ship without one. The ingestion adapter then emits a
    **vertical-well assumption** warning and treats $\text{MD} = \text{TVD}$ straight down
    from the wellhead — explicit and flagged, never a silent guess (doc 03 §5/§6). The
    wellhead position $(x, y, \text{elevation})$ is still mandatory to place the well at
    all.

---

## 4. Temperature: gradient, bottom-hole temperature, and heat-flow

Temperature is the whole point of geothermal, and a well measures it directly.

- **Bottom-hole temperature (BHT)** — the temperature at the deepest point. A single
  number, but the headline "how hot does it get" figure. (Caveat: drilling cools the
  rock near the hole, so raw BHT readings are usually *too low* and need a correction —
  our synthetic generator models that as a BHT-correction error,
  `HeatFlowForward` in `borehole.py`.)
- **Geothermal gradient** — how fast temperature rises with depth. From a continuous
  temperature log it's just the slope:

$$
\nabla T = \frac{dT}{dz} \approx \frac{T_\text{bottom} - T_\text{surface}}{\text{depth}}
$$

  measured in **K/km** (canonical) and often quoted in °C/km. A "normal" continental
  gradient is ~25–30 °C/km; a Basin-&-Range geothermal play like our flagship scenario
  runs **~45 °C/km** (doc 05 §2.3) — that elevated gradient is *the* first-order signal
  that there's heat to exploit.

- **Surface heat-flow** — the rate heat escapes the earth per unit area, combining the
  gradient with how well the rock conducts heat (its thermal conductivity $k$):

$$
q = k\,\nabla T
$$

  - $q$ — heat-flow (mW/m²).
  - $k$ — thermal conductivity of the rock (W/m·K).
  - $\nabla T$ — geothermal gradient (K/m).

  High heat-flow over a region is the classic regional "drill here" indicator.

!!! warning "Temperature is stored in kelvin, not °C"
    The platform stores **absolute temperature in kelvin** internally and only displays
    °C. This is not pedantry: the `pint` units library treats °C as an *offset* unit, so
    a *temperature* and a *temperature difference* are different quantities — storing a
    gradient or an uncertainty as "°C" silently corrupts arithmetic (is "5 °C" a
    temperature or a 5-degree change?). Kelvin has no offset, so $\Delta T$ and $T$
    behave consistently (doc 01 §5). The synthetic well writes a `TEMP` curve in °C in
    the file; ingestion canonicalises it to kelvin.

---

## 5. The native file formats (annotated)

### 5.1 LAS — Log ASCII Standard

LAS is the universal text format for well logs: a few **header sections** describing the
well and the curves, then a block of **columnar ASCII data** (one row per depth sample).
Sections begin with `~`. Here is a fully annotated LAS 2.0 file matching what our
generator writes (`WellLogForward`):

```text
~VERSION INFORMATION                         # section: file format version
 VERS.   2.0 : CWLS LOG ASCII STANDARD       #   LAS 2.0 (also 1.2 / 3.0 exist)
 WRAP.   NO  : ONE LINE PER DEPTH STEP       #   data rows are not wrapped

~WELL INFORMATION                            # section: WHICH well + index range
 STRT.m    0.0     : START DEPTH             #   first index value (here MD/DEPT = 0)
 STOP.m  2300.0    : STOP DEPTH              #   last index value
 STEP.m    0.5     : STEP                    #   sample spacing (0.5 m → tool resolution)
 NULL.   -999.25   : NULL VALUE             #   sentinel for missing samples
 WELL.   GT-1      : WELL                    #   well identity → the join key (wellId)
 FLD.    great-basin-v1 : FIELD             #   field / scenario id
 SRC.    synthgen  : SOURCE                  #   marks this as synthetic, not real
 XCOORD. 0.0       : WELLHEAD X (m)          #   wellhead Engineering X (East)
 YCOORD. 0.0       : WELLHEAD Y (m)          #   wellhead Engineering Y (North)
 EKB.    1600.0    : KB ELEVATION (m)        #   kelly-bushing (MD datum) elevation

~CURVE INFORMATION                           # section: declares each DATA column, in order
 DEPT.m         : 1  DEPTH (TVD below KB)    #   column 1 = TVD (true vertical depth)
 MD  .m         : 2  MEASURED DEPTH          #   column 2 = MD (the measurement axis)
 RES .ohm.m     : 3  RESISTIVITY            #   → property_type "resistivity"
 GR  .gAPI      : 4  GAMMA (alteration)      #   → no canonical key (carried as methodData)
 DEN .kg/m3     : 5  BULK DENSITY            #   → property_type "density"
 VP  .m/s       : 6  P VELOCITY (SONIC)      #   → property_type "velocity_p"
 TEMP.degC      : 7  TEMPERATURE            #   → property_type "temperature" (→ kelvin)

~PARAMETER INFORMATION                        # optional: run parameters, mud, etc.
 RUN .  1 : RUN NUMBER

~ASCII                                        # the data block: rows of the curves above
#   DEPT       MD      RES     GR      DEN      VP     TEMP
   0.00     0.00    45.2    52.1   2050.0   1820.0   15.3
   0.50     0.50    44.8    53.0   2048.0   1835.0   15.5
   1.00     1.00    43.9    51.7   2055.0   1841.0   15.7
   ...      ...      ...     ...     ...      ...      ...
 1500.00  1732.10    8.1   168.4   2670.0   5480.0  112.6   # deep: low RES + high GR + hot
```

Read the `~ASCII` block like a CSV with no commas: each column is a curve declared in
`~CURVE`, each row is one depth sample. Note how **MD and DEPT(TVD) diverge** as you go
deep — that's the deviation showing up in the numbers (parsed with `lasio`, doc 03).

### 5.2 Deviation survey CSV

Shipped alongside the LAS as `<well>_deviation.csv`, joined to it by the file stem:

```csv
MD,INC,AZI
0.0,0.0,90.0          # surface: vertical, heading east
328.6,5.0,90.0        # starting to build angle
657.1,10.0,90.0
...
2300.0,35.0,90.0      # bottom: 35° inclination, heading due east
# MD  : measured depth (m)
# INC : inclination from vertical (deg)
# AZI : azimuth / compass heading (deg, 0=N, 90=E)
```

The adapter finds this sibling file automatically and feeds it to `min_curvature_positions`
to build the trajectory (`welllog.py`, `_load_deviation_survey`).

---

## 6. What it becomes in the model

A LAS file normalizes into **two** primitives joined by `wellId` — see
[the data model](../data-model.md) §3/§5:

| Part | Normalized primitive | Why split |
|---|---|---|
| The curves vs MD | **`ObservationSet`** (`geometryKind: "wellcurve"`) | The **immutable measured record** — never edited. |
| The borehole path | **`GeologicalFeature`** (`featureKind: "wellPath"`) | A (re-editable) interpreted trajectory; carries the deviation survey. |

This split matters: the *measurement* (the curve) is sacred and immutable, while the
*geometry* (the path) can be re-computed if a better deviation survey arrives — they are
kept apart and linked by the well's identity. There is deliberately **no "well_path"
support kind**; the path is a `Feature`, not a `PropertyModel` (doc 03 §3d).

Concretely the adapter emits:

```python
RawObservation(                          # 1. the curves
    geometry_kind="wellcurve",
    coords=coords,                       # each MD sample placed on the trajectory (XYZ)
    values={"resistivity": ..., "density": ..., "velocity_p": ..., "temperature": ...},
    meta={"wellId": "GT-1", "md": [...], "methodData": {"GR": {...}}},  # GR rides here
)
RawFeature(                              # 2. the path
    feature_type="wellPath",
    geometry={"type": "LineString", "coordinates": path_coords},  # min-curvature XYZ
    props={"wellId": "GT-1", "trajectory": "deviation_survey", "wellhead": [...]},
)
```

Each curve sample is placed at its true 3D position by interpolating MD onto the
min-curvature trajectory, so `RES` at MD 1732 m sits at the correct $(x,y,z)$ — possibly
hundreds of metres east of the wellhead. A continuous **temperature log** ingests the
same way (a `wellcurve` Observation); sparse bottom-hole-temperature or spring points
ingest as point temperature Observations in kelvin (the `heatflow` method, doc 03 §2).
The viewer ([3D viewer](../visualization.md)) then drapes each curve as a colour-mapped
tube along the trajectory.

---

## Key takeaways

- Boreholes are the **only direct, ground-truth** measurement — perfect resolution along
  a **1D** line — which makes them the **calibration anchor** for [rock physics](../rock-physics.md)
  and the validation check on every remote inversion.
- Standard curves: **resistivity, gamma-ray, density, sonic ($V_p$), temperature** — each
  ties a remote method to truth (and maps to a canonical property type; gamma-ray is
  carried as metadata).
- **MD ≠ TVD** for any deviated well. The **deviation survey (MD/INC/AZI)** plus
  **minimum-curvature** integration reconstructs the true 3D path; **dogleg severity**
  measures how sharply it bends.
- **Temperature** gives the **bottom-hole temperature**, the **geothermal gradient**
  ($dT/dz$, K/km), and with conductivity the **surface heat-flow** ($q = k\,\nabla T$) —
  the first-order "is there heat here" signal. Stored in **kelvin**, displayed in °C.
- Native format = **LAS** (`~VERSION`/`~WELL`/`~CURVE`/`~ASCII` sections) + a
  **deviation CSV**, joined by well id.
- Normalized form = an immutable **`Observation(wellcurve)`** of the curves **plus** a
  **`wellPath` Feature** of the trajectory, joined by `wellId`.

## Where this lives in the code

- Adapter (LAS + deviation → `wellcurve` Observation + `wellPath` Feature):
  `backend/geosim/ingestion/adapters/welllog.py` (`WellLogLasAdapter`).
- Minimum-curvature trajectory integration + DLS:
  `backend/geosim/spatial/vertical.py` (`min_curvature_positions`).
- Synthetic forward (sample truth along the path → LAS; BHT points → CSV):
  `backend/geosim/synthgen/forward/borehole.py` (`WellLogForward`, `HeatFlowForward`).
- Property types (`resistivity`, `density`, `velocity_p`, `temperature`):
  `backend/geosim/spatial/property_types.py`.
