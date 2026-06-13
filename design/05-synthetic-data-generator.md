# 05 — Synthetic Data Generator

> Parent: `OVERVIEW.md` §8 (and §10 row 5). This doc defines the **ground-truth
> synthetic earth** and the **per-method forward models** that turn it into
> realistic datasets in native formats. It feeds the **ingestion adapters** (doc 03)
> the same way real data does — the generator is just another data *source*, not a
> special path.
>
> **Cross-doc bindings (not redefined here):**
> - Emitted geometry/coordinates → **Engineering Frame** (ENU, m, Z-up, floating
>   origin) per doc 01 §1. Synthetic earths run in **local mode** by default
>   (`surfaceModel: synthetic:<id>`), optionally geo-anchored to a real site.
> - Canonical units / property registry → doc 01 §5.
> - Primitive schemas (observations / property models / features / provenance) →
>   doc 02. We emit *files*; ingestion produces the primitives.
> - Native formats → OVERVIEW §3 (the format-contract table).
> - We do **not** invent inversion here — forward only. Inversion is doc 10 (later).

> ### ⚠️ Revision — user decisions applied (see `DECISIONS.md`)
> - **Forward-model fidelity:** build **rigorous (T1) physics first for the 2–3 key
>   methods — MT, gravity, and seismic** — rather than T0-plausible-for-all-first.
>   Other methods still use the T0 "degrade-the-truth"/analytic path initially and
>   upgrade to T1 later. (The two-tier `ForwardModel` contract is unchanged; only the
>   *build order/priority* changes — rigorous solvers come first for MT/gravity/seismic.)
> - **Flagship scenario = `great-basin-v1`** (Basin & Range / Nevada hydrothermal play) — confirmed.
> - **Rock-property values = defensible textbook defaults now**, refined later — confirmed.
> - **Extra properties required:** because doc 07 adopts the *full* rock-physics table
>   (alteration index, microseismic→fracture density, Waxman-Smits/dual-water,
>   permeability proxies), the ground-truth earth must also emit the supporting
>   property/state fields those transforms consume (clay/alteration fraction,
>   fracture density, salinity, porosity) so fusion outputs can be validated against truth.

---

## 1. What this module is (and is not)

**Is:** a standalone, deterministic, seedable Python package (`synthgen`) that
(a) builds a co-located, mutually-consistent ground-truth property earth from a
declarative scene spec, (b) forward-models what each survey *would* measure over it
with realistic noise / resolution / depth-of-investigation (DOI) degradation, and
(c) writes **native-format files** (SEG-Y, EDI, LAS, GeoTIFF, CSV…) plus a
**ground-truth bundle** kept aside for validation/scoring.

**Is not:** an inversion engine, a physics research code, or a real-time service. It
runs offline to produce a scenario folder. Fidelity is explicitly tiered (§6) so
"plausible" datasets exist on day one and "rigorous" ones come later behind the same
interface.

**Why it's central:** until real surveys are loaded, *every* pixel in the viewer and
every fusion result traces back to this generator. It is also the **scoring oracle**:
because we kept the ground truth, we can quantify how well fusion/inversion recovers
the true earth.

### Design rules

| Rule | Consequence |
|---|---|
| **One geology → all properties** | Properties are never authored independently; they derive from a shared lithology/state field via rock-physics (§3). Guarantees cross-method consistency (resistivity low *and* Vp low *and* hot in the same voxel). |
| **Deterministic + seedable** | Every scenario reproducible from `(spec, seed)`. Noise realizations are seeded sub-streams so re-runs are byte-identical. |
| **Forward only, native out** | Generator emits the *same file formats real instruments emit*, so the ingestion path is exercised end-to-end. No back-door into the primitive store. |
| **Ground truth retained** | Truth volumes + the scene spec are saved alongside the "measured" files for validation. Truth is never an input to ingestion. |
| **Frame-correct** | All emitted geometry is in (or transformable to) the Engineering Frame via doc 01. The generator owns the scenario's `SpatialFrame`. |

---

## 2. Ground-truth earth specification

The earth is authored as a **declarative scene spec** (JSONC), compiled into a stack
of **co-located property volumes** on a high-resolution **truth grid** (finer than any
survey can resolve — typically 25–50 m laterally, 10–25 m vertically over the ROI).

### 2.1 The truth grid & state model

The compiler does **not** rasterize each property by hand. It builds two intermediate
fields first, then maps them to all geophysical properties through rock-physics (§3):

1. **Lithology field** `L(x,y,z)` — integer label per voxel (unit id), built by
   stacking layers, cutting with faults, and inserting intrusions/anomaly bodies.
2. **State field** `S(x,y,z)` — continuous per-voxel state that modulates rock
   physics within a lithology: `temperature`, `porosity`, `water_saturation`,
   `salinity (TDS)`, `clay/alteration_fraction`, `fracture_density`. The geothermal
   anomaly is *primarily a state perturbation* (hot + altered + porous/fractured +
   saline) layered on top of whatever lithology it occupies.

> This split is the core trick: a fault that juxtaposes two units changes `L`; a
> hydrothermal upflow that heats and alters rock changes `S`. Both then propagate
> into density, resistivity, Vp, etc. consistently.

### 2.2 Authored property set (canonical units, doc 01 §5)

| Property | Unit | Primary driver | Used by |
|---|---|---|---|
| density ρ | kg/m³ | lithology, porosity | gravity |
| magnetic susceptibility χ | SI | lithology, alteration (destroys magnetite) | magnetics |
| resistivity / conductivity | Ω·m / (S/m) | porosity, saturation, salinity, clay, T | ERT, EM, MT |
| chargeability η | mV/V | clay + sulphide/alteration fraction | IP |
| Vp, Vs | m/s | lithology, porosity, fracture, saturation | seismic, microseismic |
| temperature T | °C | state field (the geothermal target) | heat-flow, well temp |
| porosity φ | frac | state field | (ties many others) |

### 2.3 Scene spec (JSONC)

```jsonc
SceneSpec {
  "id": "great-basin-v1",
  "seed": 42,
  "frame": {                       // becomes the scenario SpatialFrame (doc 01 §2)
    "mode": "local",               // or "georeferenced" w/ anchor for a real site
    "roi":   { "xmin": -6000, "xmax": 6000, "ymin": -6000, "ymax": 6000 }, // m
    "depthRange": { "zmin": -6000, "zmax": 1700 },   // Engineering elevation (m)
    "truthGrid": { "dx": 50, "dy": 50, "dz": 20 }    // truth resolution
  },
  "surface": {                     // synthetic DEM → surfaceModel: synthetic:<id>
    "kind": "fractal",             // "flat" | "fractal" | "tilted-block"
    "baseElev": 1600, "relief": 250, "roughness": 0.7
  },

  // --- LITHOLOGY: build L(x,y,z) ---
  "layers": [                      // top→down; each fills below the previous contact
    { "unit": "alluvium",  "top": "surface", "thickness": [200, 500] }, // ranged → noisy contact
    { "unit": "volcanics", "top": "conformable", "thickness": [300, 900] },
    { "unit": "carbonate", "top": "conformable", "thickness": [800, 1500] },
    { "unit": "basement_granite", "top": "conformable", "thickness": "fill" }
  ],
  "intrusions": [
    { "unit": "young_intrusive", "shape": "stock",
      "center": [1500, -500, -3500], "radiusXY": 1200, "radiusZ": 1800 }
  ],
  "faults": [                      // cut L, offset blocks, act as conduits for anomaly
    { "id": "range-front", "kind": "normal", "dip": 60, "dipAzimuth": 90,
      "trace": [[-6000,-3000],[6000,1000]], "throw": 700, "isConduit": true }
  ],

  // --- STATE: build S(x,y,z) ---
  "geotherm": {                    // background conductive geotherm
    "surfaceTemp": 15, "gradient": 45              // °C, °C/km (Basin&Range ~ high)
  },
  "anomalies": [                   // the geothermal target = a STATE perturbation
    { "id": "upflow", "kind": "hydrothermal-plume",
      "controlledBy": "range-front",               // rises along the conduit fault
      "footprint": { "center": [800, 200], "radiusXY": 1500 },
      "topElev": -200, "bottomElev": -4500,
      "perturb": {                                  // overrides/blends into S
        "tempPeak": 220, "tempHalo": "gaussian",
        "alterationFrac": 0.6,                      // clay cap up high, propylitic deep
        "porosityBoost": 0.04, "salinityTDS": 8000, // ppm → conductive
        "fractureDensity": 0.5                       // permeable reservoir
      },
      "clayCap": { "topElev": -150, "thickness": 250 } // shallow conductive smile
    }
  ],

  "rockPhysics": "default-v1",     // named ruleset (§3); per-unit overrides allowed
  "units": { /* per-unit base props + variability, see §3.2 */ }
}
```

**Compilation order:** surface → layers → intrusion/fault cuts → `L`; geotherm →
anomaly/clay-cap blends → `S`; then rock-physics maps `(L,S) → {ρ, χ, res, η, Vp, Vs}`.
Small correlated random fields (seeded) add texture so volumes aren't cartoon-flat.

---

## 3. Rock physics — one geology, mutually-consistent properties

A **named ruleset** maps `(lithology unit, state)` → each geophysical property. Rules
are deliberately *simple, well-known petrophysics* — enough to be physically
plausible and internally consistent, not research-grade. Each rule is a pure function
the compiler applies per voxel.

### 3.1 Core relationships (`default-v1`)

| Property | Relationship | Notes |
|---|---|---|
| Resistivity | **Modified Archie + clay term** | `1/ρ = (φ^m · Sw^n / (a·ρw)) + clayCond(alterationFrac, T)`. `ρw` from salinity & T (Arps). Hot + saline + porous + clay → very conductive (the geothermal signature). |
| Density | **φ-mixing** | `ρ = (1−φ)·ρ_grain(unit) + φ·ρ_fluid`. Intrusions dense; alluvium light. |
| Susceptibility | **unit base, alteration-suppressed** | `χ = χ_base(unit) · (1 − alterationFrac)`. Hydrothermal alteration *destroys magnetite* → magnetic low over upflow (real, diagnostic). |
| Vp / Vs | **unit base − φ & fracture softening** | `Vp = Vp_base·(1 − k_φ·φ − k_fr·fractureDensity)`; saturation raises Vp, barely moves Vs → Vp/Vs flags fluid. |
| Chargeability | **clay + sulphide** | `η = η0·(clayFrac + sulphideFrac)`; alteration haloes are chargeable. |
| Temperature | **direct from S** | conductive geotherm + advective plume blended in §2.3. |

### 3.2 Per-unit property library

Each lithology unit declares base values + variability; rock-physics modulates them.
Shipped library (editable) covers Basin-&-Range lithologies:

```jsonc
"alluvium":        { "rho": 2050, "chi": 0.0005, "Vp": 1800, "phi": 0.30 },
"volcanics":       { "rho": 2450, "chi": 0.02,   "Vp": 3400, "phi": 0.12 },
"carbonate":       { "rho": 2680, "chi": 0.0001, "Vp": 5200, "phi": 0.05 },
"basement_granite":{ "rho": 2670, "chi": 0.005,  "Vp": 5600, "phi": 0.01 },
"young_intrusive": { "rho": 2750, "chi": 0.03,   "Vp": 5900, "phi": 0.01 }
// resistivity/η/Vs are DERIVED, never authored directly
```

> Defensible defaults from standard petrophysics tables; values are tunable and not
> claimed to be site-exact. This library is the single place to adjust "what the rocks
> are like." (See **Open questions** on sourcing a citable property table.)

---

## 4. Per-method forward models

Each method is a **forward-model plugin** with a uniform contract:

```python
class ForwardModel(Protocol):
    method: str                  # "gravity", "mt", ...
    fidelity: Literal["plausible","rigorous"]
    def simulate(truth: TruthEarth, acq: Acquisition, rng: Generator) -> list[Artifact]
    # Artifact = (native_file_path, format, provenance)
```

Three degradations are applied **in every model** (this is what makes it realistic):

1. **Acquisition geometry** — a simulated survey layout (station spacing, line
   spacing, electrode array, well path, frequency band) that limits coverage.
2. **Resolution / DOI** — each method only "sees" what it physically could: smoothing
   kernels, depth-dependent sensitivity decay, frequency→depth mapping, footprint
   averaging. The truth is *never* emitted at full sharpness.
3. **Noise** — additive/multiplicative noise with method-appropriate statistics
   (Gaussian, % of reading, correlated drift), seeded per method.

> **Plausible tier** = degrade-the-truth: take the truth volume, apply the method's
> resolution kernel + DOI mask + noise, project to the acquisition geometry, write
> native format. Captures *coverage, smoothing, and noise* (the things that matter for
> a fusion/visualization platform) without solving Maxwell/elastic PDEs.
> **Rigorous tier** = a real forward solver from the geoscience stack. Same output
> contract, swappable per method (§6).

### 4.1 Method table

| Method | Plausible forward | Rigorous forward (lib) | Simulated geometry | DOI / resolution | Noise | Native out (OVERVIEW §3) |
|---|---|---|---|---|---|---|
| **Gravity** | analytic prism/voxel sum of ρ-anomaly (Nagy formula) → Bouguer grid | same physics, full mesh (`harmonica`) | station grid, spacing s | smooth, deep, non-unique; low-pass ∝ depth | 0.02–0.05 mGal Gaussian + drift | CSV stations + GeoTIFF anomaly grid |
| **Magnetics** | voxel-sum of χ (Poisson/RTP) over flight lines | `harmonica`/`SimPEG.PF` | aeromag lines @ altitude h, line-spacing | upward-cont. low-pass ∝ h | 1–3 nT + line leveling error | CSV/`.xyz` lines + GeoTIFF RTP grid |
| **ERT** | apparent-resistivity pseudosection from res field via sensitivity kernels | `PyGIMLi`/`SimPEG.DC` (Poisson solve) | dipole-dipole/W-Schlumberger line, n electrodes @ spacing a | DOI ≈ 0.15–0.2·array length; loses depth fast | 2–5 % reading | AGI `.stg` / Res2DInv / UBC |
| **IP** | chargeability pseudosection (same kernels as ERT) | `PyGIMLi`/`SimPEG.IP` | co-located with ERT | as ERT | 5–10 % + worse at depth | AGI / UBC |
| **EM/TEM** | 1-D conductivity-depth (smoke-ring DOI) per sounding | `SimPEG.EM.TDEM`/`empymod` (1-D layered) | airborne/ground soundings on grid | DOI ∝ √(t·ρ/μ); diffusive smear | 3–8 % + late-time floor | ASEG-GDF / USF / `.xyz` |
| **MT** | 1-D/quasi-2-D app-res & phase from res vs period (skin depth) | `SimPEG.NSEM` / `MARE2DEM` (2-D/3-D) | station grid, period band 0.001–1000 s | skin depth δ=503√(ρT); deep but smooth | 2–5 % + static shift + dead band | **EDI** (impedance tensor) |
| **Seismic reflection** | convolve reflectivity (from impedance contrasts) with wavelet; 1-D per CMP | `devito`/`PySIT` acoustic FD | 2-D line, src/rec spacing, fold | vertical res ≈ λ/4; band-limited | band-limited noise + multiples (opt) | **SEG-Y** + horizon/fault GeoJSON |
| **Seismic refraction** | first-break traveltimes from Vp via eikonal | `pykonal` eikonal | refraction spread | shallow velocity model only | pick error ±2–5 ms | SEG-Y + CSV picks |
| **Microseismic** | sample events on stimulated fault planes; magnitude–freq (Gutenberg-Richter); locate w/ Vp | ray-trace/FD traveltimes + location error ellipsoid | downhole/surface array; time window | location σ grows w/ distance from array | location error ellipsoid + Mc cutoff | **QuakeML** + CSV catalog |
| **InSAR** | project modeled surface deformation (e.g. injection-driven uplift) to LOS | poroelastic/Mogi/Okada source → LOS | sat track geometry, LOS vector, revisit | atmospheric + decorrelation noise; 4-D stack | atmos phase screen + DEM error | **GeoTIFF** time-series + CSV |
| **Well logs** | sample truth volumes along a well path (the cleanest data) | + tool response/borehole effects | deviated well path (MD survey, doc 01 §4) | tool vertical res (0.1–0.5 m); invaded zone (opt) | small Gaussian per curve | **LAS** (+ optional DLIS) |
| **Temp / heat-flow** | sample T along wells + surface heat-flow points (∇T·k) | conductive/advective FE (later) | well bottom-hole temps, BHT points, springs | sparse points → big interp uncertainty | BHT correction error | CSV + LAS (temp curve) |
| **Geology map** | export contacts/faults at surface from L | — | — | mapped-surface only | interpretive uncertainty tag | Shapefile / GeoJSON |

### 4.2 Worked examples of the "only-sees-what-it-could" principle

- **Magnetic low over upflow:** alteration sets `χ→~0` in the plume; the magnetics
  forward sees a *low*, not the temperature directly — so fusion must *infer* heat
  from the joint pattern, exactly as in the field.
- **MT vs ERT depth split:** the clay cap (shallow conductor) is sharply imaged by ERT
  but only smoothly by MT; the deep reservoir conductor is invisible to ERT (below
  DOI) and is the MT's domain. Their disagreement at depth is *physically correct* and
  becomes a fusion test.
- **Seismic sees structure, not fluid:** reflectivity comes from impedance contrasts
  (layer contacts, fault), so the seismic "sees" the faulted geometry but is nearly
  blind to the temperature field — Vp/Vs from logs/refraction carry the fluid cue.

### 4.3 Acquisition spec (per scenario)

Each scenario ships an `acquisition.jsonc` describing what gets collected — decoupled
from the earth, so the *same* earth can be surveyed densely or sparsely:

```jsonc
{
  "gravity":  { "spacing": 250, "footprint": "roi" },
  "magnetics":{ "lineSpacing": 200, "altitude": 80, "heading": 90 },
  "mt":       { "stations": "grid:1000", "periods": [0.001, 1000], "nPeriods": 30 },
  "ert":      [{ "line": "A-A'", "array": "dipole-dipole", "n": 64, "a": 25 }],
  "seismic":  [{ "line": "A-A'", "srcSpacing": 50, "recSpacing": 25, "fold": 30 }],
  "wells":    [{ "id": "GT-1", "path": "deviation:GT-1.csv",
                 "logs": ["res","gr","den","vp","temp"] }],
  "insar":    { "track": "asc", "los": [0.6,-0.1,0.79], "revisitDays": 12, "span": "2y" },
  "microseismic": { "array": "downhole:GT-1", "window": "stim-2026" }
}
```

---

## 5. Outputs

A scenario run produces a **self-contained folder**:

```
scenarios/great-basin-v1/
  scene.jsonc            # the authored spec (input, kept for provenance)
  acquisition.jsonc
  frame.json            # the scenario SpatialFrame (doc 01 §2)
  measured/             # ← native-format files; the ONLY thing ingestion reads
    gravity_stations.csv  gravity_bouguer.tif
    aeromag_lines.xyz     mag_rtp.tif
    mt/ST001.edi ... ST030.edi
    ert/lineAA.stg        ip/lineAA.stg
    tem/soundings.xyz
    seismic/lineAA.segy   horizons.geojson
    microseismic.quakeml  catalog.csv
    insar/los_timeseries/*.tif
    wells/GT-1.las        wells/GT-1_deviation.csv
    temperature_points.csv
  truth/                # ← ground truth, NEVER ingested; validation only
    properties.zarr     # ρ, χ, res, η, Vp, Vs, T, φ on the truth grid (Engineering coords)
    lithology.zarr  state.zarr
    features.geojson    # true faults/horizons/anomaly solids
  manifest.json         # seed, versions, checksums, per-file provenance
```

- **`measured/`** files are byte-compatible with what doc 03 adapters parse from real
  instruments — proving the ingestion contract. Each file carries provenance noting it
  is synthetic (`source: synthgen`, scene id, seed) so it's never mistaken for real.
- **`truth/`** is the **scoring oracle**: stored as Engineering-coordinate Zarr (doc 02
  conventions) so a validation tool can resample any fused/inverted result onto the
  truth grid and compute recovery metrics (RMS error, structural similarity, anomaly
  detection rate) — the basis of OVERVIEW's "verify against known ground truth."
- **`frame.json`** lets a project ingest the whole scenario into one `SpatialFrame`
  (local by default; geo-anchorable to e.g. the Milford/FORGE area to test
  georeferenced mode + real DEM/basemap).

---

## 6. Fidelity tiers & build order

| Tier | What it is | Cost | Build |
|---|---|---|---|
| **T0 — plausible** | degrade-the-truth: resolution kernel + DOI mask + noise + geometry projection; analytic forwards for potential fields | hours of dev/method, ms–s to run | **first, for all methods** |
| **T1 — rigorous** | real forward solvers (`harmonica`, `SimPEG`, `PyGIMLi`, `empymod`, `devito`, `pykonal`) behind the same `ForwardModel` contract | days/method, seconds–minutes to run | **incrementally, per method, prioritised by which fusion result it gates** |

**Recommended:** ship **T0 for every method first** (gets data into the whole
pipeline on day one and exercises ingestion → storage → viewer → fusion end-to-end),
then upgrade to **T1 method-by-method**, starting with the methods whose realism most
affects fusion validation — **MT and gravity** (deep, smooth, non-unique: the hard
fusion cases) and **seismic** (provides the sharp structure everything else is hung
on). ERT/IP/EM T1 follow. Because the contract is identical, a T1 upgrade is a drop-in
swap with no ingestion/viewer changes.

> Potential-field T1 (gravity/mag) is cheap and high-value — the analytic prism sum is
> *already* close to rigorous, so promote those first as a quick win.

---

## 7. Shippable scenarios

| Scenario | Role | Earth | Methods emphasised |
|---|---|---|---|
| **`unit-cube-v1`** | smoke test / CI | single conductive cube in halfspace | all — round-trip + scoring asserts |
| **`great-basin-v1`** ⭐ | **flagship** | Basin-&-Range extensional hydrothermal play (below) | full multi-method suite |
| **`egs-granite-v1`** | EGS / 4-D | hot low-perm granite + stimulated fracture set + injection | microseismic + InSAR + wells (4-D) |
| **`layered-cake-v1`** | teaching/validation | flat layers, one fault, one body | clean cross-method comparison |

### 7.1 Flagship: `great-basin-v1` (recommended)

A **Basin & Range / Great Basin extensional geothermal play**, the canonical Western-US
hydrothermal setting and directly aligned with the project's Nevada/Milford-FORGE
interest (KB: Cape Station, FORGE — "eastern margin of the Basin and Range… high heat
flow, extensional tectonics, accessible hot crystalline basement").

- **Structure:** alluvium-filled valley over volcanics/carbonate over granite basement;
  a **range-front normal fault** (60° dip, ~700 m throw) as the master structure and
  **fluid conduit**.
- **Geothermal target:** a **fault-controlled hydrothermal upflow** rising along the
  range-front fault — hot (~220 °C), conductive (saline + altered), with a shallow
  **clay-cap conductor** and a deeper **propylitically-altered, fractured reservoir**.
  Produces the textbook joint signature: **MT/EM conductor + magnetic low + gravity
  expression of the basin/fault + seismic-imaged structure + a hot well**.
- **Why flagship:** it makes *every* method earn its keep and creates genuine
  fusion/non-uniqueness challenges (the whole point of the platform), while being a
  real, recognisable play type rather than a toy. Optionally geo-anchor it near Milford
  to test georeferenced mode with a real DEM/basemap.

`egs-granite-v1` is the natural second scenario (covers 4-D microseismic + InSAR during
stimulation — the Fervo/FORGE EGS story), but `great-basin-v1` is the one that
exercises the full breadth first.

---

## 8. Implementation notes (libraries & structure)

- Package `synthgen` (Python), CLI: `synthgen build scenarios/great-basin-v1`.
- Truth grid + property volumes: `xarray`/`numpy`, written `truth/*.zarr` (doc 02).
- Geometry/CRS: doc 01 `SpatialFrame` + `pyproj` (only if geo-anchored).
- Writers reuse the **same format libraries as ingestion** (doc 03) for guaranteed
  round-trip symmetry: `segyio` (SEG-Y), `lasio` (LAS), `rasterio` (GeoTIFF),
  `ObsPy`/`quakeml` (microseismic), `pandas` (CSV), custom EDI/`.stg` writers.
- T1 solvers: `harmonica` (grav/mag), `SimPEG` + `PyGIMLi` (DC/IP/EM/MT),
  `empymod` (1-D EM/MT), `devito`/`pykonal` (seismic) — added per §6, optional deps.
- Seeding: one root seed → per-method sub-streams (`numpy.random.SeedSequence`) so
  methods are independent yet reproducible.

---

## 9. Decisions locked in

1. **One geology → all properties.** Properties are *derived* from a shared
   `(lithology, state)` model via a named rock-physics ruleset — never authored
   independently. Guarantees cross-method consistency.
2. **The geothermal anomaly is primarily a state perturbation** (hot + altered +
   porous/fractured + saline) layered onto lithology, controlled by a conduit fault —
   yielding the realistic joint signature (conductor + magnetic low + hot well).
3. **Forward only; native formats out.** The generator emits the *same file formats
   real instruments emit* and feeds them through the normal doc 03 ingestion path. No
   back-door into the primitive store.
4. **Three universal degradations** — acquisition geometry, resolution/DOI, noise —
   applied in every forward model, so each method "only sees what it physically could."
5. **Two fidelity tiers behind one `ForwardModel` contract:** **T0 plausible**
   (degrade-the-truth + analytic potential-field) **for all methods first**, **T1
   rigorous** (real solvers) upgraded method-by-method. T1 is a drop-in swap.
6. **Ground truth retained & frame-correct.** Truth volumes/features saved as
   Engineering-coordinate Zarr (doc 01/02) as the **scoring oracle**; never an input to
   ingestion.
7. **Deterministic & seedable** — every scenario reproducible from `(spec, seed)`.
8. **Flagship scenario = `great-basin-v1`** (Basin-&-Range extensional hydrothermal
   play), local mode by default, geo-anchorable to Milford/FORGE.

---

## 10. Open questions for you

1. **Forward-model fidelity tier to build first?** *Why it matters:* sets the entire
   build cost and how "real" early demos look. **Options:** (a) **T0-plausible for all
   methods first, T1 later per method** *(recommended)* — fastest to a full end-to-end
   pipeline; (b) T1-rigorous from the start for a 2–3 key methods (MT, gravity,
   seismic), skip the rest initially — fewer methods but publication-grade; (c) buy
   nothing, only the cheapest analytic forwards forever. *Default: (a).*

2. **Flagship scenario?** *Why it matters:* it's the demo everyone sees and the earth
   all fusion is validated against. **Options:** (a) **`great-basin-v1` Basin-&-Range
   hydrothermal play** *(recommended — matches the Nevada/FORGE interest and exercises
   every method)*; (b) `egs-granite-v1` EGS-in-granite with 4-D microseismic/InSAR
   (closer to the Fervo drilling story, but lighter on the classic potential-field/MT
   fusion); (c) a generic layered-cake first (simplest to validate, least compelling).
   *Default: (a), with (b) as the immediate second.*

3. **Rock-physics property library — how authoritative?** *Why it matters:* the
   per-unit base values (§3.2) determine whether anomalies look credible to a
   geophysicist and whether scoring means anything. **Options:** (a) **ship defensible
   textbook defaults now, refine later** *(recommended)*; (b) digitise a specific
   citable petrophysical reference / a named Nevada field's published properties up
   front (slower, more credible); (c) make every value a tunable knob with no shipped
   defaults (maximally flexible, nothing works out-of-the-box). *Default: (a).*
```
