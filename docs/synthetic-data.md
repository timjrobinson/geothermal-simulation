# The synthetic data generator

> **What you'll learn / why it matters.** Every other page in these docs talks about *fusing* many survey methods into one earth model. But how do you know the fusion is *correct*? With real field data you can't — nobody has ever dug up a cubic kilometre of crust to check. The synthetic data generator solves this by building a **fake planet whose every property is known exactly**, then forward-modelling what each survey *would* measure over it. Because we kept the truth, we can score how well fusion and inversion recover it. This page explains the "one geology → all properties" rule, the three universal degradations that make each simulated method realistically limited, the two fidelity tiers, the flagship scenario, the on-disk folder layout, and the CLI that builds it all.

If you're a programmer, here's the one-sentence framing: **the generator is a property-based test harness for geophysics.** You write a declarative spec (the "input"), the system computes a known-correct ground truth, runs many lossy encoders over it (the survey forward models), and later you assert that the decoders (inversion/fusion) reconstruct the original within tolerance. The ground truth is your test oracle.

---

## 1. Why a fake planet?

Real geophysics is an **inverse problem** (defined in the [glossary](glossary.md)): you measure effects at the surface and try to infer the cause (the rock) underground. Inverse problems are *non-unique* — many different earths produce nearly the same surface measurements, the same way many different source files can compress to nearly the same lossy JPEG. So when a fusion pipeline outputs "there is a hot conductive body at 2 km depth," you have a credibility problem:

- With **real data**, the true earth is unknown, so you can never compute the error. You can only check internal consistency.
- With a **synthetic earth**, you authored the truth, so the error is a subtraction.

!!! abstract "The generator plays three roles at once"
    1. **Day-one data.** Until real surveys are loaded, *every pixel* in the [3D viewer](visualization.md) and *every* [fusion](fusion.md) result traces back to this generator. It is what makes the whole pipeline runnable on a laptop with no field campaign.
    2. **End-to-end exercise of ingestion.** The generator emits the *same native file formats real instruments emit* (SEG-Y, EDI, LAS, GeoTIFF, QuakeML…). Those files go through the **normal** [ingestion adapters](ingestion.md) — there is no back-door into the data store. If a format round-trips through synthgen and ingestion, the contract is proven.
    3. **The scoring oracle.** Because the true property volumes are saved alongside the measured files, a validation tool can resample any fused or inverted result onto the truth grid and compute recovery metrics (RMS error, structural similarity, anomaly-detection rate).

> **Where it sits in the data flow.** The generator is *just another data source* feeding ingestion — not a special privileged path. It happens to know the truth; ingestion never sees the truth.
>
> ```
> SceneSpec (you author)  ──▶  TruthEarth (known property volumes)
>                                   │
>                       forward models (3 degradations)
>                                   ▼
>                          measured/ native files  ──▶  ingestion  ──▶  fused model
>                                   │                                        │
>                          truth/ zarr (kept aside) ───────── score ────────┘
> ```

---

## 2. The core trick: one geology → all properties

The single most important design rule is **"one geology → all properties"** (locked decision #1 of the synthetic-data design). Properties are **never authored independently.** You do *not* hand-paint a resistivity cube, then a density cube, then a temperature cube and hope they agree. You author **one** geology, and every geophysical property is *derived* from it through rock physics.

Why does this matter so much? Because the whole point of fusion is that the methods agree where they should. In the real earth, a hot, saline, fractured, clay-altered zone is **simultaneously**:

- more **conductive** (low resistivity) — salt water and clay conduct electricity,
- less **dense** (low density) — it's porous and fractured,
- magnetically **quiet** (low susceptibility) — hydrothermal alteration chemically destroys magnetite,
- seismically **slow** (low Vp) — cracks and fluid slow down sound.

If you authored those four cubes separately, nothing would force the low-resistivity blob to sit exactly where the low-density blob sits. Fusion would then be testing your painting skill, not the algorithm. By deriving everything from one source, the co-location is *guaranteed by construction* — exactly as physics guarantees it in the field.

### 2.1 The two intermediate fields: L and S

The generator does not rasterise each property directly. It first builds **two** intermediate fields on a high-resolution **truth grid** (finer than any survey can resolve — typically 50 m laterally, 20 m vertically), then maps them to every property.

| Field | Type | Meaning (programmer analogy) |
|---|---|---|
| **Lithology** $L(x,y,z)$ | integer label per voxel | *Which rock type* fills this voxel — an enum / categorical layer (alluvium, volcanics, carbonate, granite…). Built by stacking layers, cutting with faults, inserting intrusions. |
| **State** $S(x,y,z)$ | continuous struct per voxel | *The condition* of that rock — a record of `temperature`, `porosity`, `water_saturation`, `salinity`, `clay/alteration_fraction`, `fracture_density`. Continuous fields that modulate the rock physics *within* a lithology. |

!!! tip "L vs S — the conceptual split"
    A **fault** that juxtaposes two rock units changes $L$ (the categorical map). A **hydrothermal upflow** that heats, alters, and fractures whatever rock it passes through changes $S$ (the continuous condition). The geothermal target is *primarily a state perturbation* — hot + altered + porous/fractured + saline — layered on top of whatever lithology it occupies, and steered up a conduit fault.

The compilation order is deterministic:

```
surface DEM  →  stack layers  →  cut faults / insert intrusions  →  L(x,y,z)
geotherm     →  blend anomaly + clay-cap perturbations           →  S(x,y,z)
rock physics:  (L, S)  →  { ρ, χ, resistivity, η, Vp, Vs, T, φ }
```

Small **seeded** correlated random fields add texture so the volumes aren't cartoon-flat. Everything is reproducible from `(spec, seed)` — the same seed gives byte-identical output (see [glossary: seed](glossary.md)).

### 2.2 The authored property set

These are the canonical properties the truth earth emits (units per the [spatial framework](spatial-framework.md) registry):

| Property | Symbol | Unit | Primary driver | Consumed by method |
|---|---|---|---|---|
| density | $\rho$ | kg/m³ | lithology, porosity | [gravity](survey-methods/potential-fields.md) |
| magnetic susceptibility | $\chi$ | SI (dimensionless) | lithology, alteration | [magnetics](survey-methods/potential-fields.md) |
| resistivity / conductivity | $\rho_r$ / $\sigma$ | Ω·m / S·m⁻¹ | porosity, saturation, salinity, clay, T | [ERT](survey-methods/electrical.md), [EM](survey-methods/electromagnetic.md), [MT](survey-methods/electromagnetic.md) |
| chargeability | $\eta$ | mV/V | clay + sulphide/alteration | [IP](survey-methods/electrical.md) |
| P- and S-wave velocity | $V_p$, $V_s$ | m/s | lithology, porosity, fracture, saturation | [seismic](survey-methods/seismic.md), [microseismic](survey-methods/seismic.md) |
| temperature | $T$ | °C | state field (the target!) | [heat-flow, well temp](survey-methods/boreholes.md) |
| porosity | $\phi$ | fraction | state field | ties many others |

### 2.3 The rock-physics rules (where L, S become numbers)

A **named ruleset** (`default-v1`) maps `(lithology unit, state) → each property`. The rules are deliberately *simple, textbook petrophysics* — physically plausible and internally consistent, not research-grade. Each is a pure function applied per voxel. The full treatment of these equations (and how they're re-used at fusion time) is in [rock physics & favorability](rock-physics.md); here are the load-bearing ones:

**Resistivity — a modified Archie's law with a clay term.** [Archie's law](glossary.md) is the foundational petrophysics relation linking a rock's resistivity to how much (salty) water fills its pores:

$$
\frac{1}{\rho_r} \;=\; \underbrace{\frac{\phi^{\,m}\, S_w^{\,n}}{a\,\rho_w}}_{\text{Archie (pore-fluid path)}} \;+\; \underbrace{\text{clayCond}(\text{alterationFrac},\,T)}_{\text{surface-conduction path}}
$$

- $\rho_r$ — bulk resistivity of the rock (Ω·m); its inverse $1/\rho_r$ is conductivity.
- $\phi$ — porosity (fraction of the rock that is pore space).
- $S_w$ — water saturation (fraction of the pore space filled with water vs gas/steam).
- $\rho_w$ — resistivity of the pore *water*, which drops sharply with salinity and temperature (computed via an Arps-style relation, see [glossary: brine](glossary.md)).
- $m$ — cementation exponent, $n$ — saturation exponent, $a$ — tortuosity constant: empirical Archie parameters (typically $m\approx 2$, $n\approx 2$, $a\approx 1$).
- the **clay term** adds a parallel conduction path because clay surfaces conduct independently of the pore fluid. This is why the geothermal **clay cap** (defined below) is such a strong, shallow conductor.

The upshot: **hot + saline + porous + clay-altered → very conductive**. That is *the* electromagnetic signature of a geothermal system.

**Density — porosity mixing:**

$$\rho \;=\; (1-\phi)\,\rho_{\text{grain}}(\text{unit}) \;+\; \phi\,\rho_{\text{fluid}}$$

solid grains of the rock unit, diluted by light pore fluid. Intrusions are dense; alluvium is light.

**Susceptibility — alteration suppression:**

$$\chi \;=\; \chi_{\text{base}}(\text{unit}) \,\cdot\, (1 - \text{alterationFrac})$$

hydrothermal alteration chemically **destroys magnetite**, so the upflow reads as a *magnetic low* — a real, diagnostic effect. The magnetics method therefore sees a hole, not heat, and fusion must *infer* the heat from the joint pattern.

**Seismic velocity — porosity & fracture softening:**

$$V_p \;=\; V_{p,\text{base}}\,\big(1 - k_\phi\,\phi - k_{\text{fr}}\,\text{fractureDensity}\big)$$

saturation raises $V_p$ but barely moves $V_s$, so the ratio $V_p/V_s$ flags fluid.

The per-unit base values live in a small shipped library (`DEFAULT_UNIT_LIBRARY` in `backend/geosim/synthgen/rockphysics.py`), e.g.:

```jsonc
"alluvium":         { "rho": 2050, "chi": 0.0005, "Vp": 1800, "phi": 0.30 },
"volcanics":        { "rho": 2450, "chi": 0.02,   "Vp": 3400, "phi": 0.12 },
"carbonate":        { "rho": 2680, "chi": 0.0001, "Vp": 5200, "phi": 0.05 },
"basement_granite": { "rho": 2670, "chi": 0.005,  "Vp": 5600, "phi": 0.01 },
"young_intrusive":  { "rho": 2750, "chi": 0.03,   "Vp": 5900, "phi": 0.01 }
// resistivity / chargeability / Vs are DERIVED, never authored directly
```

---

## 3. The three universal degradations

Authoring a sharp, complete truth volume is only half the job. A real survey **never** sees the truth at full sharpness — it sees a blurry, partial, noisy projection of it. The realism of the generator comes from applying **three degradations in *every* forward model** (this is what makes each method "only see what it physically could"). In CS terms: each forward model is a **lossy encoder** with a method-specific loss profile.

| # | Degradation | What it models | CS analogy |
|---|---|---|---|
| 1 | **Acquisition geometry** | the survey layout — station spacing, flight-line spacing, electrode array, well path, frequency/period band — which limits *where* you sample. | irregular, sparse **sampling** of a continuous signal; aliasing where you under-sample. |
| 2 | **Resolution / depth-of-investigation (DOI)** | each method's physical blur: smoothing kernels, depth-dependent sensitivity decay, frequency→depth mapping, footprint averaging. The truth is *never* emitted at full sharpness. | a **low-pass filter** whose cutoff frequency worsens with depth; lossy compression that throws away high spatial frequencies. |
| 3 | **Noise** | additive/multiplicative noise with method-appropriate statistics (Gaussian, % of reading, correlated drift), drawn from a seeded RNG. | adding a controlled, reproducible **noise signal** to the samples. |

!!! note "DOI — depth of investigation, defined"
    **DOI** is the depth below which a method effectively can't see — its signal has decayed into the noise. It's the geophysical analogue of a sensor's range. ERT loses depth fast (DOI ≈ 0.15–0.2 × the array length). MT, using long-period natural signals, reaches kilometres deep but only *smoothly*. The shared machinery applies a depth-dependent DOI mask plus a depth-widening Gaussian low-pass, so deep features come out blurrier than shallow ones — exactly as in the field.

### 3.1 Worked examples — "only sees what it could"

These three are the canonical demonstrations that the degradations produce *physically correct disagreement* between methods (and therefore make fusion a real test, not a rubber stamp):

- **Magnetic low over the upflow.** Alteration drives $\chi \to 0$ in the plume, so the magnetics forward emits a **low**, never the temperature. Fusion must infer heat from the *joint* pattern — exactly the field situation.
- **MT vs ERT depth split.** The shallow **clay cap** is sharply imaged by ERT but only smoothly by MT; the deep reservoir conductor is *below ERT's DOI* (invisible to it) and is MT's domain. Their disagreement at depth is *correct*, and becomes a fusion test case.
- **Seismic sees structure, not fluid.** Reflectivity comes from impedance contrasts (layer contacts, the fault), so seismic images the faulted *geometry* but is nearly blind to the temperature field. The fluid cue comes from $V_p/V_s$ via logs/refraction.

The shared degradation machinery (separable Gaussian low-pass, DOI mask, trilinear truth sampling, seeded noise) lives in `backend/geosim/synthgen/forward/base.py`; the per-method physics lives in sibling modules (`potential_field.py`, `electrical.py`, `em_mt.py`, `seismic.py`, `borehole.py`, `surface.py`).

---

## 4. Two fidelity tiers behind one contract

Both tiers implement the **same** `ForwardModel` interface, so upgrading is a drop-in swap — no ingestion, storage, or viewer changes. This is the [plugin architecture](architecture.md) at work: a forward model is just one of the six extension points.

```python
# backend/geosim/synthgen/forward/base.py (paraphrased)
class ForwardModel(Protocol):
    method: str                 # canonical key: "gravity", "mt", ... (see glossary)
    submethod: str | None
    fidelity: Literal["plausible", "rigorous"]
    def simulate(self, truth: TruthEarth, acq: Acquisition,
                 rng: Generator) -> list[Artifact]:
        ...   # Artifact = (native_file_path, format, provenance)
```

| Tier | What it is | Cost | Build priority |
|---|---|---|---|
| **T0 — plausible** | *degrade-the-truth*: take the truth volume, apply the resolution kernel + DOI mask + noise, project to the acquisition geometry, write the native format. Plus analytic forwards for potential fields. Captures coverage, smoothing, and noise without solving PDEs. | hours of dev per method; ms–s to run | **first, for every method** — gets data through the whole pipeline on day one. |
| **T1 — rigorous** | a real forward solver from the geoscience stack (`harmonica` for grav/mag, `SimPEG`/`PyGIMLi` for DC/IP/EM/MT, `empymod` for 1-D EM/MT, `devito`/`pykonal` for seismic), behind the *same* contract. | days per method; seconds–minutes to run | **incrementally, per method**, starting with the ones whose realism most gates fusion: **MT, gravity, seismic**. |

!!! abstract "Why this order"
    T0-everything-first proves the **vertical slice** end-to-end (synth → ingest → store → view → fuse) before any expensive solver exists. Then T1 is promoted where it matters most: MT and gravity are the deep, smooth, *non-unique* hard cases for fusion; seismic provides the sharp structure everything else is hung on. Potential-field T1 (the analytic prism sum is already close to rigorous) is the cheap high-value first promotion. The currently-shipped forwards are all **T0** (`fidelity="plausible"`).

---

## 5. The shippable scenarios

A **scenario** is a named, self-contained synthetic survey play: a declarative `SceneSpec` (the earth) plus an `Acquisition` (what gets collected). Two ship today (in `backend/geosim/synthgen/scenarios/`):

| Scenario | Role | Earth |
|---|---|---|
| **`unit-cube-v1`** | CI smoke test | a single conductive cube in a layered halfspace — coarse + tiny so a build runs in seconds; round-trip + scoring asserts. |
| **`great-basin-v1`** ⭐ | **flagship** | a Basin-&-Range extensional hydrothermal play (below). |

### 5.1 Flagship: `great-basin-v1`

A **Basin & Range / Great Basin extensional geothermal play** — the canonical Western-US hydrothermal setting, aligned with the project's Nevada / Milford-FORGE interest (high heat flow, extensional tectonics, accessible hot crystalline basement). It makes *every* method earn its keep and creates genuine non-uniqueness challenges, while being a real, recognisable play type rather than a toy.

- **Structure.** Alluvium-filled valley over volcanics / carbonate over granite basement; a **range-front normal fault** (60° dip, ~700 m throw) is both the master structure *and* the fluid conduit.
- **Geothermal target.** A **fault-controlled hydrothermal upflow** rising along the range-front fault — hot (~220 °C), conductive (saline + altered), with a shallow **clay-cap** conductor and a deeper **propylitically-altered, fractured reservoir**. This produces the textbook joint signature: an MT/EM conductor + a magnetic low + the gravity expression of the basin/fault + seismic-imaged structure + a hot well.

Here is the actual flagship scene spec (the real `scene.jsonc`, annotated):

```jsonc
{
  "id": "great-basin-v1",
  "seed": 42,                              // (1) deterministic root seed → sub-streams

  "frame": {                               // (2) becomes the scenario SpatialFrame (doc 01)
    "mode": "local",                       //     local ENU metres; geo-anchorable to FORGE
    "roi":   { "xmin": -6000, "xmax": 6000, "ymin": -6000, "ymax": 6000 },
    "depthRange": { "zmin": -6000, "zmax": 1700 },   // Engineering elevation (m)
    "truthGrid": { "dx": 50, "dy": 50, "dz": 20 }    // truth resolution (finer than any survey)
  },
  "surface": {                             // (3) synthetic DEM → surfaceModel: synthetic:<id>
    "kind": "fractal", "baseElev": 1600, "relief": 250, "roughness": 0.7
  },

  // --- LITHOLOGY: build L(x,y,z) ---
  "layers": [                              // (4) top→down; each fills below the previous contact
    { "unit": "alluvium",         "top": "surface",     "thickness": [200, 500] },  // ranged → noisy contact
    { "unit": "volcanics",        "top": "conformable", "thickness": [300, 900] },
    { "unit": "carbonate",        "top": "conformable", "thickness": [800, 1500] },
    { "unit": "basement_granite", "top": "conformable", "thickness": "fill" }       // fills to base
  ],
  "intrusions": [
    { "unit": "young_intrusive", "shape": "stock",
      "center": [1500, -500, -3500], "radiusXY": 1200, "radiusZ": 1800 }
  ],
  "faults": [                              // (5) cut L, offset blocks, act as conduits
    { "id": "range-front", "kind": "normal", "dip": 60, "dipAzimuth": 90,
      "trace": [[-6000, -3000], [6000, 1000]], "throw": 700, "isConduit": true }
  ],

  // --- STATE: build S(x,y,z) ---
  "geotherm": { "surfaceTemp": 15, "gradient": 45 },   // (6) background conductive geotherm (°C, °C/km)
  "anomalies": [                           // (7) the geothermal target = a STATE perturbation
    { "id": "upflow", "kind": "hydrothermal-plume",
      "controlledBy": "range-front",       //     rises along the conduit fault
      "footprint": { "center": [800, 200], "radiusXY": 1500 },
      "topElev": -200, "bottomElev": -4500,
      "perturb": {                         //     overrides/blends into S
        "tempPeak": 220, "alterationFrac": 0.6,        // hot + altered
        "porosityBoost": 0.04, "salinityTDS": 8000,    // porous + saline (conductive)
        "fractureDensity": 0.5                          // permeable reservoir
      },
      "clayCap": { "topElev": -150, "thickness": 250 } // shallow conductive "smile"
    }
  ],
  "rockPhysics": "default-v1"              // (8) named ruleset (§2.3); maps (L,S) → all properties
}
```

1. The root seed; `numpy.random.SeedSequence` spawns one independent sub-stream per method, so methods are reproducible *and* independent.
2. The ROI box and depth range define the engineering frame; the **truth grid** is finer than any survey can resolve.
3. A fractal DEM gives the surface relief; the scenario's surface model becomes `synthetic:great-basin-v1`.
4. Layers stack top-down; a *ranged* thickness (`[200, 500]`) produces a noisy, realistic contact, not a flat plane.
5. The range-front fault both offsets the layers (changing $L$) and is marked `isConduit` so the upflow rises along it.
6. The conductive **geotherm** — 15 °C surface, 45 °C/km gradient (Basin & Range runs hot) — sets the background $T$ in $S$.
7. The **anomaly** is the geothermal target, applied as a *state* perturbation: hot, altered, porous, saline, fractured.
8. The named ruleset turns every `(L, S)` voxel into the full property set, guaranteeing cross-method consistency.

The companion `acquisition.jsonc` (decoupled, so the same earth can be surveyed densely or sparsely) declares the survey layouts — gravity station spacing, aeromag line spacing & altitude, MT period band, ERT/seismic line geometry, well paths and which logs to run, InSAR track geometry, the microseismic array, etc.

!!! tip "egs-granite-v1 and layered-cake-v1 (designed, not yet shipped)"
    The design envisions two more scenarios: **`egs-granite-v1`** (hot low-permeability granite + a stimulated fracture set + injection — the 4-D microseismic + InSAR EGS story, the Fervo/FORGE drilling case) and **`layered-cake-v1`** (flat layers, one fault, one body — the cleanest possible cross-method teaching/validation case). `great-basin-v1` is the one that exercises the full breadth first.

---

## 6. The output: a self-contained scenario folder

A scenario build produces one self-contained directory. The critical structural rule: **`measured/` is the only thing ingestion reads; `truth/` is never ingested** (it would be cheating — truth is the answer key).

```text
scenarios/great-basin-v1/
  scene.jsonc            # the authored spec (input, kept for provenance)
  acquisition.jsonc      # the survey plan (decoupled from the earth)
  frame.json             # the scenario SpatialFrame (doc 01) — one frame for the whole scenario
  measured/              # ← native-format files; the ONLY thing ingestion reads
    gravity_stations.csv   gravity_bouguer.tif
    aeromag_lines.xyz      mag_rtp.tif
    mt/ST001.edi … ST030.edi
    ert/lineAA.stg         ip/lineAA.stg
    tem/soundings.xyz
    seismic/lineAA.segy    horizons.geojson
    microseismic.quakeml   catalog.csv
    insar/los_timeseries/*.tif
    wells/GT-1.las         wells/GT-1_deviation.csv
    temperature_points.csv
  truth/                 # ← ground truth, NEVER ingested; validation only
    properties.zarr        # ρ, χ, resistivity, η, Vp, Vs, T, φ on the truth grid (Engineering coords)
    lithology.zarr  state.zarr
    features.geojson       # true faults / horizons / anomaly solids
  manifest.json          # seed, versions, checksums, per-file provenance
```

- **`measured/`** files are byte-compatible with what the [ingestion adapters](ingestion.md) parse from real instruments — proving the ingestion contract. Each carries provenance noting it is synthetic (`source: synthgen`, scene id, seed) so it can never be mistaken for real data.
- **`truth/`** is stored as **Zarr** (chunked N-D arrays — see [glossary: Zarr](glossary.md)) in Engineering coordinates, following the [data model](data-model.md) conventions, so a validation tool can resample any fused/inverted [PropertyModel](data-model.md) onto the truth grid and compute recovery metrics.
- **`frame.json`** lets a project ingest the whole scenario into one [SpatialFrame](spatial-framework.md) — local by default, geo-anchorable to (e.g.) the Milford/FORGE area to exercise georeferenced mode with a real DEM/basemap.
- **`manifest.json`** records the seed, library versions, per-file checksums, and provenance — so a build is reproducible and auditable.

---

## 7. How to build one (the CLI)

The generator is a standalone Python package, `geosim.synthgen`, with a small CLI. The easiest entry points are the Makefile targets:

```bash
# list the registered scenarios (id + title)
make scenarios

# build the flagship to scenarios/great-basin-v1/
make scenario SCENARIO=great-basin-v1

# build the tiny CI smoke-test scenario instead
make scenario SCENARIO=unit-cube-v1
```

Under the hood those call the module directly:

```bash
# from backend/, in the project venv
python -m geosim.synthgen list
python -m geosim.synthgen build great-basin-v1 --out ../scenarios/great-basin-v1
python -m geosim.synthgen build great-basin-v1 --overwrite   # rebuild existing truth zarr
```

`build` does three things, in order (see `backend/geosim/synthgen/scenarios/__init__.py` and `builder.py`):

1. **Compile the earth** — `compile_scene(spec)` builds $L$, $S$, and all property volumes into a `TruthEarth`.
2. **Run every registered T0 forward** over that earth (`FORWARD_MODELS`), each applying the three degradations and writing its native file(s) into `measured/`.
3. **Write the scenario folder** — the truth Zarr bundle + `features.geojson`, `frame.json`, `manifest.json` (with seed, versions, and per-file checksums).

Because the build is deterministic from `(spec, seed)`, re-running with the same seed yields byte-identical output — handy for CI diffs and for the `unit-cube-v1` round-trip assertions.

---

## Key takeaways

- The synthetic generator exists to give you **ground truth**, which real data can never provide — it is the test oracle for fusion and inversion of an inherently non-unique [inverse problem](glossary.md).
- **One geology → all properties:** author a single lithology field $L$ and state field $S$, then *derive* every geophysical property through rock physics. This guarantees the methods co-locate the way physics demands (low-resistivity = low-density = magnetically-quiet = seismically-slow = hot, all in the same voxel).
- The geothermal target is primarily a **state perturbation** (hot + altered + porous/fractured + saline) steered up a conduit fault.
- **Three universal degradations** — acquisition geometry, resolution/DOI, noise — make every simulated method realistically limited, "only seeing what it physically could," so the methods *correctly disagree* and fusion is a real test.
- **Two fidelity tiers** (T0 degrade-the-truth, T1 rigorous solvers) sit behind one `ForwardModel` contract; T1 is a drop-in upgrade. The shipped forwards are T0.
- The flagship **`great-basin-v1`** is a Basin-&-Range hydrothermal play that exercises every method; **`unit-cube-v1`** is the tiny CI smoke test.
- A build produces a self-contained folder where **`measured/` is the only thing ingestion reads** and **`truth/` is the never-ingested scoring oracle**, both in Engineering coordinates.

## Where this lives in the code

| Concern | Path |
|---|---|
| Package + public API | `backend/geosim/synthgen/__init__.py` |
| CLI (`list`, `build`) | `backend/geosim/synthgen/__main__.py` |
| Scene spec dataclasses (`SceneSpec`, layers, faults, anomalies…) | `backend/geosim/synthgen/scene.py` |
| Compiler: `(spec) → L, S, all property volumes` | `backend/geosim/synthgen/compiler.py` |
| Rock-physics ruleset + unit library (`default_v1`, `DEFAULT_UNIT_LIBRARY`) | `backend/geosim/synthgen/rockphysics.py` |
| Truth earth + truth-bundle writer | `backend/geosim/synthgen/truth.py` |
| Forward-model contract + the three degradations | `backend/geosim/synthgen/forward/base.py` |
| Per-method T0 physics | `backend/geosim/synthgen/forward/{potential_field,electrical,em_mt,seismic,borehole,surface}.py` |
| Scenario registry + `build_scenario` | `backend/geosim/synthgen/scenarios/` |
| Shipped scene/acquisition specs | `backend/geosim/synthgen/scenarios/{great-basin-v1,unit-cube-v1}/` |
