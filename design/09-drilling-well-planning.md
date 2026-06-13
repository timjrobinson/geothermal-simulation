# 09 — Drilling Target & Well-Path Planning

> Parent: `OVERVIEW.md` §6 (level 6), §10 row 9. This is the top of the fusion
> ladder: everything below it (a co-registered, fused, interpreted earth model
> with favorability/temperature/uncertainty volumes) exists so an engineer can
> **pick a target, plan a well to it, see what the well will hit, score the risk,
> and export** a deviation survey a real planning tool can consume.
>
> **Cross-doc contract (this doc consumes, does not redefine):**
> - Coordinates, MD/TVD/elevation, units, `SpatialFrame` — **doc 01** (locked).
> - Property models, **features** (faults, units, fracture networks, well paths),
>   provenance, on-disk schema — **doc 02**.
> - **`sample_along_line` / `sample_at_points`** query, fused-grid storage, tile/slice API — **doc 04**.
> - **Viewer** scene graph, picking, gizmos, well-tube rendering — **doc 06**.
> - **Favorability / temperature / uncertainty volumes** and rock-physics derivations — **doc 07**.
>
> Where a parallel doc owns an interface, this doc **references it, states the
> assumption it needs, and flags the need** rather than re-specifying it.

> ### ⚠️ Revision — user decisions applied (see `DECISIONS.md`)
> - **Trajectory fidelity:** geometric (min-curvature + DLS) **plus a crude
>   drillability flag in core** — a lightweight build-rate / DLS-vs-feasibility
>   sanity check that warns on likely-undrillable geometry. Full torque-and-drag /
>   hydraulics / BHA mechanics remain a later `TrajectoryPlugin`.
> - **Exports:** **CSV deviation survey + CSV predicted log + WITSML-trajectory**
>   are the supported set (WITSML promoted into scope; Compass `.dev`/named-tool
>   still deferred until a specific downstream tool is in the loop).
> - **Induced-seismicity risk:** stays a later `RiskPlugin`; the core risk score
>   remains the simple transparent weighted form — confirmed.

---

## 1. Scope & non-goals

**In scope.** A *geometric, model-driven* well planner:

1. **Target definition** — pick a point/zone in the fused 3D model (typically a high-favorability hot fractured volume from doc 07), capture target metadata.
2. **Trajectory model** — a planned wellbore as a deviation survey (MD/inc/azi); vertical, deviated, horizontal/EGS geometries; minimum-curvature math; dogleg-severity (DLS) constraints.
3. **Model intersection** — sample every relevant volume along the trajectory → a **predicted log** of what the well will encounter.
4. **Risk / uncertainty along path** — surface doc-07 uncertainty (temperature confidence, fault proximity, lost-circulation / hazard zones) into a simple per-station risk score.
5. **Geothermal outputs** — predicted BHT, reservoir intersection length, expected productive fracture intersections.
6. **Planning UX** — interactive trajectory editing, alternative-path comparison, multi-well pad layouts.
7. **Export** — trajectory + predicted logs to CSV, deviation-survey, and WITSML-trajectory formats.

**Non-goals (explicit, with extension paths).**

| Not doing now | Why | Extension path |
|---|---|---|
| Torque & drag / hydraulics / BHA mechanics | This is a *planning & feasibility* tool, not an engineering-of-record drilling simulator | §11 — `TrajectoryPlugin` can carry a mechanics back-end (e.g. wrap an open T&D solver) |
| Anti-collision against *real* offset wells (formal SF-based) | No real well DB yet; pad wells are planned, not surveyed | §7.3 ships a geometric clearance check; promote to ellipsoid-of-uncertainty SF when real surveys land |
| Geosteering / real-time trajectory updates from LWD | No live data; all data is simulated/static | 4D hook: re-sample a re-inverted model along the same MD axis |
| Drilling-cost / time (AFE, ROP modeling) | Out of platform scope | optional cost plugin keyed off predicted lithology log |
| Casing/cement/completion design | Downstream of trajectory | predicted log already gives the inputs (hazard depths, temperature) |

Fidelity stance: **trajectory geometry is industry-standard (minimum curvature, DLS).** Everything *mechanical* is deferred behind a plugin. The model-intersection and risk parts are as good as the doc-07 volumes feeding them.

---

## 2. Where this sits in the architecture

```
          doc 07  favorability / temperature / uncertainty / hazard volumes
          doc 02  features: faults, unit solids, fracture networks
                         │   (all in the Engineering Frame, doc 01)
                         ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  PLANNING DOMAIN (this doc)                                   │
   │  Target ── Trajectory(min-curv) ── Intersection ── Risk ──Export│
   └───────────────▲───────────────────────────┬──────────────────┘
        viewer pick/gizmo (doc 06)   sample_along_line (doc 04)
```

Planning objects are **features** (doc 02 primitive #3 — discrete vector geometry). A planned well is a first-class, persisted feature; a target is a small feature with metadata. They live in the project catalog alongside everything else and render through the normal feature path in the viewer. **A planned trajectory and an ingested real well path share the same deviation-survey representation** (doc 01 §4) — so a plan can later be promoted to "as-drilled" with zero schema change.

---

## 3. Target definition

### 3.1 What a target is

A `DrillTarget` is a feature describing *what subsurface volume we want the well to reach/penetrate*. Two flavors share one schema:

- **Point target** — a single Engineering-Frame point (a "geological target" / landing point). Bullseye + tolerance radius.
- **Zone target** — a volume: an isosurface-bounded blob (e.g. favorability ≥ 0.7), a unit solid, a fault-damage envelope, or a hand-drawn box. Stored as a reference to the producing feature/isosurface plus a bounding solid.

### 3.2 How the user picks one (viewer — doc 06 owns the mechanics)

Three picking modes, all resolving to Engineering XYZ via doc-06 picking + doc-01 frame:

1. **Click-on-isosurface** — with a favorability/temperature isosurface shown (doc 07 + doc 06), click it; the ray-hit point and the local property values become the target. *Primary workflow* — "drill at the hottest, most fractured, highest-favorability blob."
2. **Click-in-volume + threshold grow** — click inside a volume; backend flood-fills the connected region above a threshold (reusing doc-04 sampling) and returns its centroid + bounding solid as a zone target.
3. **Manual entry** — type/drag an Engineering XYZ (or lat/lon/elev, or MD/TVD against an existing well) for precise targets.

The target is then **enriched** by a single `sample_at_points` call (doc 04) at the bullseye: temperature, favorability, lithology, key properties, and their uncertainties (doc 07) are stamped onto the target metadata at creation time.

### 3.3 `DrillTarget` schema

```jsonc
DrillTarget {
  "id": "tgt_01H...",
  "name": "FORGE-style hot fractured zone A",
  "kind": "point" | "zone",

  // geometry — Engineering Frame, metres (doc 01)
  "location":   { "x": -120.0, "y": 340.0, "z": -2650.0 },   // bullseye (zone: centroid)
  "tolerance":  { "radius_m": 50.0, "tvd_window_m": 25.0 },   // acceptable miss
  "zoneRef":    { "featureId": "iso_fav_0.7", "boundingSolid": "solid_..." } | null,

  // intent (engineer's goals)
  "desiredTemperatureC": 200.0,
  "minTemperatureC":     175.0,            // hard floor for a viable geothermal target
  "geologicalUnit":      "granitic basement",   // ref to a doc-02 unit solid id when known
  "rationale":           "peak joint favorability×temperature; >2 km from mapped fault",

  // enrichment — sampled from the model at creation (doc 04 + doc 07), cached
  "sampled": {
    "temperatureC":      { "value": 203.0, "sigma": 14.0, "confidence": 0.71 },
    "favorability":      { "value": 0.82,  "sigma": 0.09 },
    "lithology":         "granite",
    "depthTVD_m":        2670.0,           // derived view from z (doc 01 §4)
    "nearestFault_m":    1850.0,           // distance to nearest doc-02 fault feature
    "sampledAt":         "2026-06-13T...", "modelVersion": "fused_v7"
  },
  "provenance": { "createdBy": "viewer:click-isosurface", "isosurfaceThreshold": 0.7 }
}
```

Notes:
- `z` is **canonical** (Engineering elevation, doc 01); `depthTVD_m` is a derived view, cached for display only.
- `sampled` is a **snapshot** tied to `modelVersion`. If the fused model is re-derived, the target shows a "stale — re-sample" badge rather than silently drifting.
- `geologicalUnit`/`zoneRef`/`nearestFault` are **references to doc-02 features**, not copies.

---

## 4. Well trajectory model

### 4.1 Representation — a deviation survey (binds to doc 01 §4)

A planned well is stored as the **same deviation survey** doc 01 mandates for boreholes: ordered stations of **(MD, inclination, azimuth)**. Everything else (TVD, Engineering XYZ, northing/easting offsets, dogleg) is **derived** from the survey + the wellhead, never stored as source of truth.

```jsonc
PlannedWell {
  "id": "well_plan_01H...",
  "name": "Pad-A / W-01",
  "status": "planned",                    // planned → permitted → as-drilled (future)
  "padId": "pad_A" | null,
  "targetIds": ["tgt_01H..."],            // ordered targets the path must honor

  // wellhead / reference (doc 01 §4: each well stores its reference elevation)
  "wellhead": {
    "x": 0.0, "y": 0.0,                   // Engineering XY of the slot
    "groundElev_m": 1620.0,               // z of GL at the slot (from surfaceModel, doc 01 §6)
    "kbElev_m": 1627.0,                   // kelly bushing / rotary table; MD datum (= MD 0)
    "depthReference": "KB"
  },

  // the survey: the source of truth
  "deviationSurvey": [
    { "md": 0.0,    "inc": 0.0,  "azi": 0.0 },
    { "md": 800.0,  "inc": 0.0,  "azi": 0.0 },     // vertical section
    { "md": 1500.0, "inc": 35.0, "azi": 95.0 },    // build
    { "md": 2400.0, "inc": 88.0, "azi": 95.0 }     // landed ~horizontal in target
    // ...
  ],

  // how the survey was generated (so it can be re-solved when a target moves)
  "design": {
    "method": "build-hold-land" | "S-curve" | "vertical" | "catenary" | "manual",
    "kop_md_m": 800.0,                    // kick-off point
    "buildRate_deg30m": 3.0,              // build-up rate (°/30 m)
    "maxDLS_deg30m": 5.0                  // dogleg-severity ceiling (constraint)
  },
  "constraints": { "maxInc_deg": 92.0, "maxDLS_deg30m": 5.0, "minMD_m": 0 }
}
```

> **DLS unit convention (locked):** **degrees per 30 m** (the SI/metric analogue of °/100 ft). Stored metric; the UI may *display* °/100 ft via the units registry (doc 01 §5) for users who think in field units.

### 4.2 Geometry families supported

| Family | Shape | Geothermal use |
|---|---|---|
| **Vertical** | inc ≈ 0 throughout | shallow/hydrothermal wells, observation wells |
| **Deviated (build-hold)** | vertical → KOP → build → tangent/hold to TD | reach an offset target; standard directional |
| **S-curve** | build then drop back toward vertical | thread between hazards / hit stacked targets |
| **Horizontal / EGS** | build to ~90° and hold a long lateral in the reservoir | **Fervo/FORGE-style**: maximize stimulated-fracture intersection in hot basement |
| **Catenary / curved-land** | continuously curving land (no long tangent) | smoother DLS profile into deep laterals |

### 4.3 Trajectory math — minimum curvature (the industry standard)

Given the survey, compute position by the **minimum-curvature method**, which fits a circular arc between consecutive stations (vs the cruder balanced-tangential or radius-of-curvature methods).

**Per interval** between station 1 `(MD₁, I₁, A₁)` and station 2 `(MD₂, I₂, A₂)`:

```
ΔMD = MD₂ − MD₁

# dogleg angle β (the total angular change over the interval), spherical law:
cos β = cos(I₂−I₁) − sin I₁ · sin I₂ · (1 − cos(A₂−A₁))

# ratio factor (smooths a straight segment to a circular arc); →1 as β→0:
RF = (2 / β) · tan(β / 2)          # use RF = 1 when β < 1e-6 rad (limit)

# incremental displacements (Engineering ENU, doc 01):
ΔN = (ΔMD/2) · ( sin I₁·cos A₁ + sin I₂·cos A₂ ) · RF      # +North = +Y
ΔE = (ΔMD/2) · ( sin I₁·sin A₁ + sin I₂·sin A₂ ) · RF      # +East  = +X
ΔV = (ΔMD/2) · ( cos I₁        + cos I₂        ) · RF      # +Down (TVD increment)

# dogleg severity over the interval, normalized to the metric course length:
DLS = β · (30 / ΔMD)              # degrees per 30 m  (β in degrees)
```

**Accumulate** from the wellhead to get each station's position. Mapping into doc-01 canonical coordinates:

```
x_eng(MD) = wellhead.x + Σ ΔE
y_eng(MD) = wellhead.y + Σ ΔN
z_eng(MD) = kbElev_m   − Σ ΔV          # ΔV is downward; Engineering Z is +up
TVD(MD)   = Σ ΔV                       # below KB (depthReference)
TVDSS(MD) = −z_eng(MD)                 # below datum/MSL (doc 01 §4 derived view)
```

This is exactly `md_to_tvd` / `tvd_to_elevation` from doc 01 §4, applied station-by-station. Between survey stations we interpolate **along the same minimum-curvature arc** (not linearly) so the rendered tube and any MD sample are geometrically faithful.

> **Reuse:** the survey→position integrator is the *same backend routine* doc 01 §4 needs for ingested wells. This doc owns the spec; the implementation is shared. **Flag to doc 01/02 owners:** put `min_curvature_positions(deviation_survey, wellhead)` in the shared spatial/well module, not in the planner.

### 4.4 Design solvers (survey ← intent)

Designers don't hand-type surveys; they state intent and a solver emits the survey:

- **Vertical:** trivial — straight down to target TVD.
- **Build-hold-land:** given wellhead, target XYZ, KOP, build rate, target landing inclination → solve the build-up arc + tangent that lands inside `tolerance`. Closed-form for the planar case; 2-parameter numeric solve (KOP, build rate) when DLS-constrained.
- **S-curve / catenary:** parametric, same constraint check.
- **Manual:** the user edits stations directly (§8); solver off.

**Every solver output is validated against `constraints`** (max DLS, max inc). DLS violations are highlighted per-interval in the UI (red tube segment) and block export unless the user overrides with a flagged note.

### 4.5 Extension path (mechanics)

The geometric survey is the input a torque-and-drag or hydraulics model needs. §11's `TrajectoryPlugin` interface exposes the resolved survey + predicted lithology log (§5) so a future plugin can compute side forces, T&D, ECD, etc. — **no change to the planning core**.

---

## 5. Model intersection — the predicted log

The payoff: given a resolved trajectory, **sample every relevant volume along it** and produce a synthetic log of what the well will encounter — *before drilling*.

### 5.1 Sampling (reuses doc 04)

Densify the trajectory into sample stations at a fixed **MD step** (default **5 m**, configurable; finer near targets). Each station has an Engineering XYZ from §4.3. Then issue **one batched `sample_along_line` / `sample_at_points` query (doc 04)** across the relevant volumes:

```
points_eng   = min_curvature_positions(survey, wellhead) densified @ mdStep
predicted    = storage.sample_at_points(
                 points_eng,
                 volumes=[temperature, favorability, resistivity,
                          lithology, fractureDensity, hazard_LCZ, ...],
                 with_uncertainty=True)        # doc 07 sigma/confidence bands
```

> **Assumption / flag to doc 04:** `sample_at_points` accepts a list of volume ids and returns aligned arrays (value + optional `sigma`/`confidence`) per volume, with out-of-ROI stations flagged `null`. If doc 04's signature differs we adapt here; the planner only needs *"give me these properties at these points, with uncertainty."*

### 5.2 Predicted-log schema

```jsonc
PredictedLog {
  "wellId": "well_plan_01H...",
  "modelVersion": "fused_v7",
  "mdStep_m": 5.0,
  "stations": [
    {
      "md": 1500.0, "tvd": 1402.3, "z": 224.7,
      "x": -40.1, "y": 88.0,                        // Engineering (doc 01)
      "temperatureC": { "value": 168.0, "sigma": 12.0, "confidence": 0.68 },
      "favorability": { "value": 0.55, "sigma": 0.10 },
      "resistivity_ohmm": 240.0,
      "lithology": "granodiorite",                  // categorical (doc 07 / geomodel)
      "fractureDensity": 0.8,                       // P32-ish, from doc-02 fracture net / doc-07
      "hazards": { "lostCirculation": 0.2, "overpressure": 0.05 },
      "distToNearestFault_m": 410.0,                // §7.2
      "risk": 0.31                                  // §7.4 composite
    }
    // ... one per MD step
  ],
  "summary": { /* §6 geothermal outputs */ }
}
```

Renders in the viewer (doc 06) as **color-mapped curves along the well tube** and as **2D log tracks** (Observable Plot, OVERVIEW §5) — temperature track, favorability track, lithology fill, hazard track, risk track, with uncertainty bands shaded.

---

## 6. Geothermal-specific outputs

Derived from the predicted log; these are the numbers a geothermal engineer judges a plan by.

| Output | Definition | Notes |
|---|---|---|
| **Predicted BHT** | temperature at TD (max MD) station | with σ/confidence band (doc 07). Compared to target's `desiredTemperatureC` |
| **Max temperature along path** | peak temperature & its MD/TVD | a hotter shallower zone may beat TD |
| **Target intersection length** | contiguous MD where the path is inside the target zone solid / above favorability threshold | the "pay" length; horizontal/EGS wells maximize this |
| **Reservoir intersection length** | contiguous MD inside the reservoir unit solid (doc 02) | distinct from favorability pay |
| **Productive fracture intersections** | count & MD list where the trajectory crosses a fracture feature (doc 02) or fractureDensity > threshold | EGS productivity proxy; weighted by aperture/favorability when available |
| **Cumulative reservoir-volume sampled** | swept volume of the lateral × drainage radius assumption | rough deliverability proxy; assumption flagged |
| **In-window fraction** | % of pay length within `minTemperatureC` *and* favorability threshold | single feasibility number |

Each output carries its uncertainty (propagated from per-station σ), so a plan reads as e.g. **"BHT 203 ± 14 °C, 640 m pay above fav 0.7, 11 fracture intersections, 92% in-window."**

---

## 7. Risk & uncertainty along path

We don't invent uncertainty — we **surface doc-07's** and convert it to actionable per-station risk plus a few geometric hazards we can compute here.

### 7.1 Temperature / property confidence (from doc 07)

Each sampled property may carry `sigma` and/or `confidence` (doc 07 uncertainty volumes). Surfaced directly as shaded bands on the log and folded into the score. A target whose temperature confidence is low is a *known unknown*, not a silent guess.

### 7.2 Fault proximity (geometric, from doc-02 features)

For each station, distance to the nearest **fault feature** (doc 02). Drilling near/through faults is both opportunity (permeability) and hazard (losses, instability, seismicity). We compute a **fault-proximity factor** that *rises* close to a fault:

```
faultProx = clamp( 1 − dist_to_fault / influence_radius , 0, 1 )   # influence_radius default 250 m
```

Fault **intersections** (path crosses a fault surface) are listed explicitly with MD and the fault id.

### 7.3 Drilling hazards (from doc-07 hazard volumes, when present)

If doc 07 / the synthetic generator emits hazard likelihood volumes — **lost-circulation zones (LCZ)**, overpressure, instability, high-temperature-mud zones — they're sampled like any property and contribute to risk. Where a dedicated volume is absent, **proxy rules** fill in (flagged as proxy, low weight): e.g. LCZ proxy = high fracture density ∩ high permeability proxy; overpressure proxy = sealing-unit + temperature gradient anomaly.

A lightweight **clearance check** vs other planned wells on the pad (§8.3): min center-to-center distance per MD; flags potential collisions. (Not formal anti-collision — see §1 non-goals.)

### 7.4 Composite risk score (simple, transparent)

Per station, a weighted blend in [0,1] — deliberately interpretable, not a black box:

```
risk =  w_T · (1 − tempConfidence)          # geological/temperature uncertainty
      + w_F · faultProx                      # fault proximity/intersection
      + w_H · hazardLikelihood               # LCZ / overpressure / instability (max of)
      + w_D · dlsExceedance                  # geometry: DLS over the ceiling
      + w_U · propUncertainty                # avg normalized σ of key properties
   (weights default 0.30/0.25/0.25/0.10/0.10, sum 1; user-tunable per project)
```

Aggregated per well into **mean** and **peak** risk, plus a **risk-by-depth profile**. The score is advisory and the formula/weights are visible and editable — engineers distrust opaque scores, so we keep it a glass box. The driver breakdown ("what's driving risk at this depth") is always shown alongside the number.

---

## 8. Planning UX

### 8.1 Interactive trajectory editing (viewer — doc 06)

- **Target-pull:** drag the target; the active design solver (§4.4) re-solves the survey live; predicted log + outputs recompute (debounced; coarse `mdStep` while dragging, fine on release).
- **Control handles:** drag KOP, build rate, landing inclination, or individual survey stations via gizmos (doc 06 owns gizmo mechanics; this doc supplies the parameter set). Manual mode lets users grab any station.
- **Constraint feedback:** tube segments exceeding `maxDLS` render red with a per-interval DLS readout; out-of-ROI segments dim; in-target segments highlight.
- **Live panel:** BHT, pay length, fracture count, mean/peak risk update as the path moves — the core feedback loop ("is this a better well?").

> **Flag to doc 06:** needs (a) ray-pick returning Engineering XYZ + hit feature id, (b) draggable handle gizmos with change callbacks, (c) tube geometry with per-station vertex colors + per-segment color override for DLS flags. The planner provides geometry & colors; doc 06 provides the interaction substrate.

### 8.2 Comparing alternative well paths

- Multiple `PlannedWell`s per target as **named scenarios** (e.g. "W-01 vertical", "W-01 horizontal-N", "W-01 horizontal-E").
- **Comparison table**: BHT, pay length, reservoir length, fracture intersections, mean/peak risk, max DLS, total MD — one row per scenario, best-in-column highlighted.
- **Overlay mode**: render candidates together, color-coded; ghost the inactive ones.
- **Diff on logs**: overlay predicted temperature/favorability/risk tracks for two scenarios.

### 8.3 Multi-well pad layouts

A `Pad` groups wells sharing a surface location (Fervo/FORGE-style multi-well pads):

```jsonc
Pad {
  "id": "pad_A",
  "surfaceLocation": { "x": 0.0, "y": 0.0, "groundElev_m": 1620.0 },
  "slots": [ { "id":"slot_1","dx":0,"dy":0 }, { "id":"slot_2","dx":8,"dy":0 } ], // slot offsets (m)
  "wellIds": ["well_plan_01H...", "well_plan_02H..."],
  "spacingPolicy": { "minSeparation_m": 6.0, "targetLateralSpacing_m": 100.0 }    // EGS frac-spacing intent
}
```

- Slot wellheads inherit the pad surface location + slot offset.
- **Pad-level checks**: inter-well clearance (§7.3); lateral spacing of horizontals vs `targetLateralSpacing_m` (EGS stimulation spacing); shared targets vs independent targets.
- **Pad summary**: total pay length across wells, aggregate fracture intersections, worst-case clearance.

### 8.4 Persistence & versioning

Targets, wells, and pads are **doc-02 features** in the project catalog. Each carries `modelVersion`; predicted logs are cached and invalidated when the fused model is re-derived (stale badge → one-click re-sample). Scenarios are immutable snapshots once exported, so an exported plan always matches its file.

---

## 9. Export

Goal: hand a real planning/drilling tool a **trajectory + predicted logs** in formats it already reads. Export transforms back **out** of the Engineering Frame using doc 01 §7 (`engineering_to_crs`, `from_engineering`) so coordinates land in real-world CRS when the project is georeferenced.

| Format | Content | Consumer | Priority |
|---|---|---|---|
| **CSV — deviation survey** | MD, Inc, Azi, TVD, N, E, DLS, + (optional) lat/lon/elev & TVDSS | universal; spreadsheet, Compass import, scripts | **P0** |
| **CSV — predicted log** | MD, TVD, temperature(+σ), favorability, lithology, resistivity, fractureDensity, hazards, risk | analysis, plotting, hand-off | **P0** |
| **LAS** | predicted log as 1D curves vs MD (mirrors how doc 03 ingests *real* logs) | log viewers; round-trips with ingestion | **P1** |
| **WITSML `trajectory`** (1.4.1.1 / 2.0) | trajectoryStation objects (MD, incl, azi, tvd, N/E, dls) + CRS metadata | drilling data platforms; the industry interchange | **P1** |
| **Survey export (Compass/`.dev`/`.wl`)** | plain MD/Inc/Azi survey | directional-drilling tools (Landmark Compass etc.) | **P1** |
| **GeoJSON / glTF** | trajectory geometry as a feature | GIS / other viewers (doc 02 export path) | **P2** |
| **Plan report (PDF/MD)** | targets, survey, outputs, risk, comparison table | human review / permitting packet | **P2** |

Rules:
- **Units honored** via doc 01 §5 registry — export metric (canonical) or field units (ft, °F, °/100 ft) per the export dialog, with units written into headers.
- **CRS round-trip:** georeferenced exports carry the project CRS + vertical datum (doc 01) so a downstream tool re-georeferences identically. Local-mode exports state "local frame, no CRS."
- **Provenance block** in every export: `modelVersion`, design method/constraints, sampling step, generation timestamp — the plan is auditable and reproducible.
- WITSML/LAS reuse the same writers the ingestion adapters (doc 03) need for round-trip tests — **flag to doc 03:** share the WITSML/LAS I/O module.

---

## 10. Backend API surface (sketch)

```
# targets
POST   /projects/{p}/targets                 {pick payload}      → DrillTarget (enriched)
POST   /projects/{p}/targets/{t}/resample                        → refreshed sampled snapshot

# wells / design
POST   /projects/{p}/wells                   {intent|survey}     → PlannedWell (survey resolved)
POST   /wells/{w}/solve                       {design params}     → deviationSurvey + DLS report
GET    /wells/{w}/positions                                       → min-curvature XYZ/TVD per station
POST   /wells/{w}/predict                     {mdStep, volumes}   → PredictedLog (+ geothermal summary, risk)

# pads / comparison
POST   /projects/{p}/pads                                         → Pad (+ clearance report)
GET    /projects/{p}/wells/compare?ids=...                        → comparison table

# export
GET    /wells/{w}/export?fmt=csv|las|witsml|dev|geojson&units=…   → file (CRS round-trip via doc 01)
```

Core compute lives in Python (reuses doc 04 sampling, doc 01 transforms); `predict` is the heaviest call (batched sampling) — background-task it for very long horizontals (OVERVIEW §5).

---

## 11. Plugin hook (doc 08)

Per OVERVIEW's R&D mandate, planning is extensible without core changes:

- **`TrajectoryPlugin`** — alternative path generators (catenary, designer-of-record imports) and the **mechanics extension** (T&D, hydraulics): consumes the resolved survey + predicted log, returns extra per-station channels (side force, ECD…) that flow into the log/risk like any other.
- **`RiskPlugin`** — swap/extend the §7.4 scoring (e.g. an induced-seismicity model keyed off fault proximity + injection intent) without touching the planner.
- **`ExportPlugin`** — register a new export format (a specific drilling platform's schema) via doc 03's I/O registry.

Each registers like any other plugin (doc 08): declare inputs (survey, log, features), outputs (channels/score/file), no core edits.

---

## 12. Decisions locked in

1. **A planned well is a deviation survey** (MD/Inc/Azi) — identical representation to ingested real wells (doc 01 §4). TVD/Engineering-XYZ are derived, never stored. A plan promotes to "as-drilled" with no schema change.
2. **Minimum-curvature method** for all trajectory math; arc-faithful interpolation between stations. The survey→position integrator is shared with doc 01/02 (flagged), not planner-private.
3. **DLS stored in °/30 m** (metric canonical), displayed in °/100 ft on request via the units registry (doc 01 §5). DLS ceiling is a hard constraint that flags in-viewer and gates export (overridable with a logged note).
4. **Fidelity = geometric planning only.** Torque-and-drag / hydraulics / BHA mechanics are explicitly out, behind a `TrajectoryPlugin` extension path (§4.5, §11).
5. **Targets and wells are doc-02 features**, persisted in the catalog, rendered through the normal feature path, versioned against `modelVersion` with stale-detection.
6. **Predicted log = batched `sample_along_line` (doc 04) across doc-07 volumes**, with uncertainty bands carried through to outputs. This doc never re-implements sampling or re-derives favorability/temperature.
7. **Risk is a transparent, user-tunable weighted score** (temperature confidence + fault proximity + hazard likelihood + DLS exceedance + property σ), always shown with its driver breakdown — a glass box, not a black box.
8. **Geothermal outputs** = predicted BHT, target/reservoir pay length, productive-fracture intersection count, in-window fraction — each with propagated uncertainty.
9. **Export priorities: CSV deviation survey + CSV predicted log are P0**; LAS, WITSML-trajectory, and Compass survey are P1; all round-trip through doc-01 CRS and the doc-03 I/O writers (flagged for sharing).
10. **Multi-well pads are first-class** (slots, clearance, EGS lateral-spacing intent), matching the Fervo/FORGE multi-well workflow.

---

## 13. Open questions for you

> *(The 2–3 highest-leverage of these are escalated to the parent for an immediate decision; the rest are here for the record.)*

1. **Trajectory-planning fidelity ceiling.** Confirmed plan is *geometric only* (min-curvature + DLS), mechanics behind a plugin. Is there any near-term mechanical check you'd want *in core* before the plugin (e.g. a crude DLS-vs-build-rate feasibility, or a simple T&D "drag risk" flag), or is pure geometry right for the R&D phase?
2. **Export formats that actually matter to you.** P0 = CSV survey + CSV log; P1 = LAS, WITSML-trajectory, Compass `.dev`. Which (if any) of WITSML / Compass / LAS is must-have for *your* downstream workflow, and is there a specific tool (Landmark Compass, Petrel, a Fervo internal format, openwell) whose import schema we should target precisely?
3. **Risk model scope.** §7.4 is a transparent weighted blend. Do you want **induced-seismicity potential** treated as a first-class risk channel now (relevant for EGS / your KB), keyed off fault proximity + injection intent — or is that a later `RiskPlugin`?
4. **Target picking primary workflow.** Assumed primary = click-on-favorability-isosurface (doc 07). Confirm that's the main path vs threshold-grow-a-zone or manual MD/TVD entry, so doc 06 prioritizes the right gizmo.
5. **EGS productivity proxy.** "Productive fracture intersections" + "reservoir-volume sampled" are rough EGS deliverability proxies. Is a simple intersection-count enough for planning, or do you want a slightly richer proxy (aperture/favorability-weighted, or a stimulated-rock-volume estimate) given EGS is the headline use case?
```
