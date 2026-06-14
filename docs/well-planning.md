# Drilling & well planning

!!! abstract "What you'll learn / why it matters"
    This is the **top of the ladder** — everything in the rest of the docs exists so an
    engineer can stand here and answer one question: *where do I drill, and what will the well
    hit?* You'll learn how the platform turns the [fused earth model](fusion.md) into a
    concrete plan: pick a **target** (the favorable hot fractured zone), design a **trajectory**
    (the curved path the bit follows, as a deviation survey), compute its geometry with the
    industry-standard **minimum-curvature** method, sample every volume *along that curve* to
    produce a **predicted log** (what the well will encounter, with uncertainty), derive the
    **geothermal outputs** (bottom-hole temperature, pay length, fracture intersections), score
    a transparent **risk**, and **export** the result as CSV and WITSML. Every concept is defined
    from first principles; the math is real and every symbol is explained.

This is a **geometric, model-driven** planner. It is deliberately *not* a drilling-mechanics
simulator (no torque-and-drag, no hydraulics) — that lives behind a plugin. What it does
exceptionally well is connect the fused model to a drillable plan and tell you, honestly, what
you'd find.

---

## 0. The vocabulary, up front

A few drilling terms recur. Define them once:

!!! note "Drilling glossary (see also [glossary](glossary.md))"
    - **MD (measured depth)** — distance *along the borehole* from the surface datum. A curved
      2 km well has 2 km of MD even if it only reaches 1.4 km straight down. Think arc length.
    - **TVD (true vertical depth)** — straight-down depth below the datum. The vertical drop.
    - **Inclination** — angle of the borehole from vertical: 0° = straight down, 90° = horizontal.
    - **Azimuth** — compass heading of the borehole (0° = North, 90° = East).
    - **Deviation survey** — the borehole's shape, stored as a list of `(MD, inclination, azimuth)`
      stations. This is the *source of truth*; everything else (TVD, XYZ, dogleg) is derived.
    - **Wellhead / KB** — the surface slot and its **kelly bushing** elevation, the datum where
      MD = 0.
    - **KOP (kick-off point)** — the MD where the well starts turning away from vertical.
    - **BHT (bottom-hole temperature)** — temperature at the deepest point. For geothermal, the
      headline number.
    - **EGS (enhanced geothermal system)** — hot dry rock made productive by engineered
      fractures; drilled with long horizontals to maximise fracture intersection (Fervo/FORGE
      style).

A planned well is a [Feature](data-model.md) (the discrete-geometry primitive), persisted in
the project catalog and rendered through the normal feature path in the [3-D viewer](visualization.md).
Critically, **a planned trajectory and an ingested real well share the same deviation-survey
representation** — so a plan promotes to "as-drilled" with zero schema change.

---

## 1. The target — where we want to go

A **drill target** describes the subsurface volume we want the well to reach. It comes in two
flavours sharing one schema:

- **Point target** — a single Engineering-Frame bullseye + a tolerance radius (the acceptable
  miss).
- **Zone target** — a volume: an isosurface-bounded blob (e.g. `favorability ≥ 0.7`), a
  geological unit solid, or a hand-drawn box.

The *primary* workflow is "drill at the hottest, most fractured, highest-favorability blob":
show a [favorability](rock-physics.md) isosurface in the viewer, click it, and the ray-hit
point becomes the target. At creation the target is **enriched** by a single `sample` call at
the bullseye — temperature, favorability, lithology, nearest-fault distance, each with its
[uncertainty](uncertainty.md) — stamped onto the target so it reads like:

```jsonc
DrillTarget {
  "kind": "point",
  "location":  { "x": -120.0, "y": 340.0, "z": -2650.0 },   // Engineering metres (z is +up)
  "tolerance": { "radius_m": 50.0, "tvd_window_m": 25.0 },   // acceptable miss
  "desiredTemperatureC": 200.0, "minTemperatureC": 175.0,    // a hard floor for viability
  "sampled": {                                               // snapshot at creation
    "temperatureC": { "value": 203.0, "sigma": 14.0, "confidence": 0.71 },
    "favorability": { "value": 0.82,  "sigma": 0.09 },
    "lithology": "granite", "depthTVD_m": 2670.0, "nearestFault_m": 1850.0,
    "modelVersion": "fused_v7"
  }
}
```

The `sampled` block is a snapshot tied to `modelVersion`. If the fused model is re-derived, the
target shows a "stale — re-sample" badge rather than silently drifting. Target logic lives in
`backend/geosim/planning/targets.py`.

---

## 2. The trajectory — a deviation survey + design solvers

The planned well stores the **deviation survey** as its source of truth — `(MD, inc, azi)`
rows — plus the wellhead datum. TVD, Engineering XYZ, north/east offsets, and dogleg are all
*derived*:

```python title="backend/geosim/planning/trajectory.py — PlannedWell (excerpt)"
@dataclass
class PlannedWell:
    wellhead: tuple[float, float]      # Engineering XY of the slot
    kb_elev_m: float                   # MD datum (MD = 0) elevation, Engineering Z (+up)
    deviation_survey: np.ndarray       # (N,3) (MD, inc°, azi°) — the source of truth
    constraints: TrajectoryConstraints # max DLS, max inc
```

### 2.1 Geometry families

Designers don't hand-type surveys; they declare *intent* and a **solver** emits the survey.
The supported families (`solve_survey` in `trajectory.py`):

| Family | Shape | Geothermal use |
|---|---|---|
| **Vertical** | inc ≈ 0 throughout | shallow hydrothermal / observation wells |
| **Build-hold-land** | vertical → KOP → build a curved arc → straight tangent landing in the target | reach an offset target; the standard directional well |
| **S-curve** | build to a tangent, then drop back toward vertical | thread between hazards / hit stacked targets |
| **Horizontal / EGS** | build to ~90° and hold a long lateral in the reservoir | **Fervo/FORGE-style**: maximise fracture intersection in hot basement |

Take **build-hold-land** as the worked example. The horizontal heading is the closed-form
wellhead→target bearing. With the build rate clamped to the dogleg ceiling, it becomes a
**2-parameter solve** over (KOP, landing inclination θ): for each candidate it computes, in
closed form, where a build arc of radius $R = \frac{180}{\pi}\cdot\frac{30}{\text{build rate}}$
followed by a straight tangent would *land*, and keeps the pair with the smallest miss:

```python title="backend/geosim/planning/trajectory.py — _build_hold_geometry (the closed form)"
radius   = (180.0 / math.pi) * (DLS_COURSE_M / build_rate_deg30m)   # arc radius for the build rate
v_build  = kop + radius * math.sin(theta)        # vertical gained building 0→θ
h_build  = radius * (1.0 - math.cos(theta))      # horizontal gained building 0→θ
tangent  = (vert - v_build) / math.cos(theta)    # straight section length to hit target depth
h_total  = h_build + tangent * math.sin(theta)   # total horizontal reach
return abs(h_total - horiz), tangent             # (horizontal miss, tangent length)
```

Every solver output is validated against the constraints (max DLS, max inclination); a
violation is reported per-interval (it turns the offending tube segment red in the viewer and
feeds the [risk score](#5-risk-a-transparent-glass-box)) but the survey is still returned —
the caller decides whether to block export.

### 2.2 Minimum-curvature: the math that turns a survey into a path

A deviation survey is just angles at MD stations. To get an actual 3-D curve we need to
*interpolate between stations*. The industry standard is the **minimum-curvature method**: it
fits a **circular arc** between consecutive stations (not a straight chord), which is both the
smoothest and the most physically faithful interpolation a bending drill string can follow.

!!! note "Why an arc, not a straight line?"
    A drill string bends *continuously*. Connecting survey stations with straight segments
    would underestimate the true path length and misplace the bit. Minimum curvature assumes
    the tangent direction rotates uniformly along a great circle between the two stations —
    the gentlest curve consistent with both endpoints' headings.

For each interval between station 1 `(MD₁, I₁, A₁)` and station 2 `(MD₂, I₂, A₂)`, with
$\Delta MD = MD_2 - MD_1$:

$$
\cos\beta = \cos(I_2 - I_1) - \sin I_1 \sin I_2 \,\bigl(1 - \cos(A_2 - A_1)\bigr)
$$

$\beta$ is the **dogleg angle** — the total angular change over the interval (by the spherical
law of cosines). Then the **ratio factor** smooths the straight chord into the arc (and → 1 as
$\beta \to 0$):

$$
RF = \frac{2}{\beta}\tan\!\frac{\beta}{2}
$$

The incremental displacements in the Engineering frame (East = +X, North = +Y, and a downward
TVD increment $\Delta V$):

$$
\begin{aligned}
\Delta E &= \tfrac{\Delta MD}{2}\,(\sin I_1 \sin A_1 + \sin I_2 \sin A_2)\,RF \\
\Delta N &= \tfrac{\Delta MD}{2}\,(\sin I_1 \cos A_1 + \sin I_2 \cos A_2)\,RF \\
\Delta V &= \tfrac{\Delta MD}{2}\,(\cos I_1 + \cos I_2)\,RF
\end{aligned}
$$

Accumulate from the wellhead, remembering Z is **+up** so we *subtract* the downward $\Delta V$:

$$
x = x_{wh} + \textstyle\sum \Delta E, \quad
y = y_{wh} + \textstyle\sum \Delta N, \quad
z = z_{KB} - \textstyle\sum \Delta V, \quad
TVD = \textstyle\sum \Delta V
$$

Every symbol: $I$ = inclination (rad), $A$ = azimuth (rad), $\beta$ = dogleg angle (rad), $RF$
= dimensionless ratio factor, $\Delta E/\Delta N/\Delta V$ = east/north/down increments (m),
$x_{wh},y_{wh}$ = wellhead XY, $z_{KB}$ = kelly-bushing elevation.

This is implemented **once**, in a shared spatial module, so an ingested real well and a
planned well use the exact same integrator:

```python title="backend/geosim/spatial/vertical.py — min_curvature_positions (the loop)"
cos_beta = math.cos(i2 - i1) - math.sin(i1)*math.sin(i2)*(1.0 - math.cos(a2 - a1))
beta = math.acos(max(-1.0, min(1.0, cos_beta)))
rf   = 1.0 if beta < 1e-7 else (2.0 / beta) * math.tan(beta / 2.0)
d_e = (d_md/2.0)*(math.sin(i1)*math.sin(a1) + math.sin(i2)*math.sin(a2)) * rf
d_n = (d_md/2.0)*(math.sin(i1)*math.cos(a1) + math.sin(i2)*math.cos(a2)) * rf
d_v = (d_md/2.0)*(math.cos(i1) + math.cos(i2)) * rf
enu[i, 2] = enu[i-1, 2] - d_v               # Z is +up; ΔV is downward
dls[i] = math.degrees(beta) * (30.0 / d_md)  # dogleg severity, °/30 m
```

### 2.3 Dogleg severity (DLS)

**Dogleg severity** is how *sharply* the well bends, normalised to a standard course length:

$$
DLS = \beta \cdot \frac{30}{\Delta MD} \quad [\text{degrees per 30 m}]
$$

The platform stores DLS canonically in **degrees per 30 metres** (the metric analogue of the
field unit °/100 ft, which the UI can display instead via the units registry). A high DLS means
a tight turn — hard or impossible to drill — so it is a hard constraint that flags in the
viewer and can block export.

### 2.4 The crude drillability flag (sanity check, not engineering)

Before claiming a geometry is sensible, the planner runs a **deliberately crude** `ok`/`warn`
gate (never `fail`). It does **no** torque-and-drag, hydraulics, or buckling analysis — those
are a later plugin. It just catches obviously-impractical geometry early via five transparent
checks (`backend/geosim/planning/drillability.py`):

| Check | Rule |
|---|---|
| **DLS exceedance** | max per-interval DLS vs `max_dls_deg30m` (already computed) |
| **Build/turn rate** | inclination- and azimuth-change rate per 30 m vs limits |
| **MD/TVD ratio** | total MD ÷ TVD vs a ceiling (step-out / horizontal-reach proxy) |
| **Max inclination** | peak inclination vs `max_inc_deg` |
| **Lithology-hardness** | a sustained hard-rock interval along the path → slow-ROP warning |

Each emits its value, limit, and the offending MD interval. A `warn` is advisory metadata on
the plan; unlike the DLS-vs-constraint export gate, it does **not** block export. (Note the
turn-rate check weights azimuth change by `sin(inc)` — heading is meaningless near vertical,
so a near-vertical azimuth flip doesn't spuriously warn.)

---

## 3. The predicted log — what the well will hit

This is the payoff. Given a resolved trajectory, sample **every relevant fused volume along
it** to produce a synthetic log of what the well would encounter — *before drilling a metre*.

The subtlety: a horizontal/EGS lateral is *curved*, so you must sample along the **real
wellbore**, not its straight chord. The planner therefore:

1. **Densifies** the survey to a fixed MD step (default 5 m) *along the minimum-curvature arc*.
   Each densified station's `(inc, azi)` is **SLERP-interpolated** (spherical-linear, on the
   great circle of the arc) so it lies *on* the curve:

    ```python title="backend/geosim/planning/trajectory.py — densify_survey (arc-faithful)"
    # Between two stations the unit tangent rotates on a great circle (the min-curvature arc);
    # SLERP the tangent unit vectors and convert back to (inc, azi) → the densified station
    # lies ON the curve, not on the chord.
    ```

2. **Integrates** those vertices to Engineering XYZ via the shared min-curvature integrator.

3. **Batch-samples** the fused layers at the curved vertices — *with σ* — reusing the
   [fusion](fusion.md) layers fusion already wrote. **Temperature and favorability are never
   re-derived here**; they are read from the model:

    ```python title="backend/geosim/planning/predict.py — the sampling core"
    dense   = densify_survey(well.deviation_survey, md_step_m)        # curved-arc vertices
    pos     = well_positions(dense, well.wellhead, well.kb_elev_m)    # → Engineering XYZ
    sampled = sample_layers_with_sigma(session, fem, pts_zyx,
                  properties=["temperature","favorability","resistivity",
                              "lithology_class","fracture_density","water_saturation"])
    ```

Each station becomes a `PredictedStation` carrying, per property, `{value, sigma, confidence}`
plus hazards, fault distance, and a composite risk:

```jsonc
PredictedStation {
  "md": 1500.0, "tvd": 1402.3, "z": 224.7, "x": -40.1, "y": 88.0,   // Engineering
  "values": {
    "temperatureC": { "value": 168.0, "sigma": 12.0, "confidence": 0.68 },
    "favorability": { "value": 0.55, "sigma": 0.10 }
  },
  "lithology": "granodiorite",
  "hazards": { "lostCirculation_proxy": 0.2 },
  "distToNearestFault_m": 410.0,
  "risk": 0.31, "riskDrivers": { /* see §5 */ }
}
```

!!! tip "Confidence from σ — a transparent mapping"
    The platform never invents confidence. It maps a value's σ to a `[0,1]` confidence via
    `confidence = 1 − clip(σ / scale, 0, 1)`, where `scale` is the property's registry display
    range — so confidence is dimensionless and comparable across properties, and σ ≈ 0 reads as
    ~1 while a large σ reads as low confidence. If σ is unavailable it returns `None` (a genuine
    unknown, not a silent 1.0). See `sigma_to_confidence` in `predict.py`.

The log renders in the [viewer](visualization.md) two ways: as colour-mapped curves painted
along the well tube, and as 2-D log tracks (temperature, favorability, lithology fill, hazard,
risk) with uncertainty bands shaded.

!!! note "Temperature is kelvin internally"
    Like all properties, temperature is carried in canonical SI (**kelvin**) through sampling
    and storage, and converted to °C only for the display/export fields (`*_C`). The math is
    always SI-canonical; field units (°F, ft, °/100 ft) are a presentation choice via the
    [units registry](spatial-framework.md).

---

## 4. Geothermal outputs

The predicted log reduces to the numbers a geothermal engineer judges a plan by
(`_geothermal_summary` in `predict.py`):

| Output | Definition |
|---|---|
| **Predicted BHT** | temperature at TD (the deepest station), with σ/confidence; compared to the target's `desiredTemperatureC` |
| **Max temperature along path** | peak temperature and its MD/TVD (a shallower hot zone can beat TD) |
| **Target intersection length ("pay")** | contiguous MD where the path is above the favorability threshold — horizontals maximise this |
| **Reservoir intersection length** | contiguous MD inside the target tolerance solid |
| **Productive fracture intersections** | count + MD list of rising-edge crossings where fracture density exceeds threshold (an EGS productivity proxy) |
| **In-window fraction** | % of pay length within both `minTemperatureC` *and* the favorability threshold — a single feasibility number |

Each carries its propagated uncertainty, so a plan reads like:
**"BHT 203 ± 14 °C, 640 m pay above fav 0.7, 11 fracture intersections, 92 % in-window."**

---

## 5. Risk — a transparent glass box

Engineers distrust opaque scores, so risk is a **simple, visible, user-tunable** weighted
blend in `[0,1]`, always shown with its driver breakdown:

$$
\text{risk} = w_T\,(1 - \text{tempConfidence}) + w_H\,\text{hazard}
            + w_D\,\text{dlsExceedance} + w_U\,\text{structuralUncertainty}
$$

Defaults: $w_T=0.40,\; w_H=0.30,\; w_D=0.10,\; w_U=0.20$ (the drilling-feasibility view;
weights are per-project tunable and re-normalised to sum 1). Symbols: **tempConfidence** =
temperature confidence at the station; **hazard** = max of the sampled drilling-hazard
likelihoods (lost circulation, overpressure, instability — or a flagged proxy); **dlsExceedance**
= how far DLS exceeds the ceiling, normalised; **structuralUncertainty** = mean
`1 − confidence` of the key properties plus a fault-interpretation-uncertainty bump.

```python title="backend/geosim/planning/predict.py — the driver breakdown (always returned)"
drivers = {
  "tempConfidence":        weights.temp_confidence * (1.0 - t_conf),
  "hazard":                weights.hazard * hazard_lvl,
  "dlsExceedance":         weights.dls_exceedance * float(dls_exceed[i]),
  "structuralUncertainty": weights.structural_uncertainty * struct_unc,
}
risk = float(np.clip(sum(drivers.values()), 0.0, 1.0))
```

### 5.1 Why fault proximity is split into four channels

Naïvely, "risk goes up near a fault." But that conflates effects pulling in **opposite
directions** — and for EGS the best targets *are* near fractured fault rock. So fault proximity
is an *input*, not "the risk", and it feeds four independent channels:

| Channel | Sign | Where it goes |
|---|---|---|
| **(a) Productivity opportunity** — enhanced permeability in the damage zone | **raises favorability** | folded into the [geothermal outputs](#4-geothermal-outputs), **not** penalised as risk |
| **(b) Drilling hazard** — lost circulation / instability in fractured rock | raises hazard | the `hazard` term ($w_H$) |
| **(c) Induced-seismicity hazard** — slip on a stressed fault under injection | raises hazard | a **later `RiskPlugin`** — not in the core score |
| **(d) Structural-interpretation uncertainty** — the fault position is itself uncertain | raises uncertainty | the `structuralUncertainty` term ($w_U$) |

The proximity itself is `faultProx = clamp(1 − dist/influence_radius, 0, 1)` (default influence
radius 250 m). Collapsing all four into one "risk up near faults" scalar would mislabel the
*best* EGS targets as the worst — so the platform refuses to.

Risk aggregates per well into mean and peak, and a risk-by-depth profile.

---

## 6. Planning UX & multi-well pads

In the [viewer](visualization.md): drag the target and the active solver re-solves the survey
live (coarse step while dragging, fine on release); BHT, pay length, fracture count, and risk
update in real time. Alternative paths are saved as named scenarios and compared in a table
(best-in-column highlighted) — see `frontend/src/ui/ScenarioTable.tsx`,
`PlanningPanel.tsx`, `RiskReadout.tsx`, `PredictedLogTracks.tsx`. A **pad** groups wells sharing
a surface location (the Fervo/FORGE multi-well workflow) with inter-well clearance checks and
EGS lateral-spacing intent.

---

## 7. Export — handing the plan to real tools

The plan exports in formats real planning/drilling tools already read. Exports transform back
*out* of the Engineering Frame into the real-world CRS when the project is georeferenced, and
carry a provenance block (`modelVersion`, design method/constraints, sampling step, timestamp)
so they are auditable and reproducible.

| Format | Content | Priority |
|---|---|---|
| **CSV — deviation survey** | `MD, Inc, Azi, TVD, N, E, DLS` (+ optional lat/lon/elev, TVDSS) | **P0, in scope** |
| **CSV — predicted log** | `MD, TVD, Temperature(+σ), favorability, lithology, resistivity, fractureDensity, hazards, risk` | **P0, in scope** |
| **WITSML `trajectory`** | trajectoryStation objects + MD-datum/CRS metadata | **P1, in scope** |
| LAS, Compass `.dev`, GeoJSON/glTF, PDF report | — | deferred |

### 7.1 The CSV deviation survey, annotated

```csv title="exported deviation survey (georeferenced)"
# geothermal-simulator deviation survey export
# well: Pad-A / W-01    modelVersion: fused_v7
# constraints: maxDLS=5.0 deg/30m, maxInc=92.0 deg
# CRS: EPSG:26912 (project CRS) — coordinates re-georeferenced from the Engineering Frame
MD_m,Inc_deg,Azi_deg,TVD_m,N_m,E_m,DLS_deg30m,Elev_m,TVDSS_m,Lat_deg,Lon_deg
0.0,0.0,0.0,0.0,0.0,0.0,0.0,1627.0,-1627.0,39.0001,-112.0003   # MD 0 at the KB datum
800.0,0.0,0.0,800.0,0.0,0.0,0.0,827.0,-827.0,39.0001,-112.0003 # vertical section
1500.0,35.0,95.0,1402.3,-40.1,88.0,1.5,224.7,-224.7,39.0005,-112.0001 # building
2400.0,88.0,95.0,1980.0,-150.0,820.0,0.0,-353.0,353.0,39.0012,-111.9990 # ~horizontal in target
```

`MD/Inc/Azi` are the survey's source-of-truth values; `TVD/N/E/DLS` are the **derived**
min-curvature outputs; `Lat/Lon` appear only when the project is georeferenced. Writer:
`backend/geosim/planning/export/csv_export.py`.

### 7.2 WITSML

[WITSML](glossary.md) is the industry XML interchange for well data. The exporter targets
**WITSML 2.0** (Energistics ETP-aligned) by default with **1.4.1.1** as a legacy alternative
behind the same writer, emitting the trajectory-focused minimum objects (`Well`, `Wellbore`,
`Trajectory`, `TrajectoryStation[]`), every quantity carrying a `uom` (unit of measure), plus
the MD datum and CRS so a consumer re-georeferences identically. There is a mandatory
**export → re-import round-trip test**: re-read the emitted XML and assert each station's
`(MD, inc, azi)` and derived `(TVD, N, E)` survive within tolerance. Writer + reader:
`backend/geosim/planning/export/witsml.py` (units in `units.py`).

---

## Key takeaways

- A planned well is a **deviation survey** (MD/inc/azi) — identical to an ingested real well;
  TVD/XYZ/DLS are derived, never stored.
- **Minimum curvature** fits a circular arc between survey stations; **DLS** measures bend
  sharpness in °/30 m and is a hard constraint.
- The **predicted log** samples every fused volume *along the curved arc* (densified + SLERP),
  reusing fusion's layers — temperature/favorability are never re-derived — and carries σ
  through to every output.
- **Geothermal outputs**: BHT, pay length, reservoir length, fracture intersections, in-window
  fraction — each with uncertainty.
- **Risk is a glass box**: a transparent weighted blend with its drivers always shown; **fault
  proximity splits into four channels** so productivity near faults isn't mislabelled as risk.
- **Export**: CSV survey + CSV log (P0) and WITSML trajectory (P1), CRS-round-tripped and
  provenance-stamped.

## Where this lives in the code

| Concern | File(s) |
|---|---|
| Targets | `backend/geosim/planning/targets.py` |
| Trajectory model + design solvers | `backend/geosim/planning/trajectory.py` |
| Shared minimum-curvature integrator | `backend/geosim/spatial/vertical.py` (`min_curvature_positions`) |
| Predicted log + geothermal outputs + risk | `backend/geosim/planning/predict.py` |
| Crude drillability flag | `backend/geosim/planning/drillability.py` |
| Along-path sampling | `backend/geosim/planning/_sampling.py` |
| Export (CSV / WITSML / units) | `backend/geosim/planning/export/csv_export.py`, `witsml.py`, `units.py` |
| API surface | `backend/geosim/api/planning.py` |
| Frontend planning + UI | `frontend/src/lib/planning.ts`, `planningApi.ts`; `frontend/src/ui/PlanningPanel.tsx`, `ScenarioTable.tsx`, `RiskReadout.tsx`, `PredictedLogTracks.tsx` |
