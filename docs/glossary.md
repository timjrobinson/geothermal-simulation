# Glossary

> **What you'll learn / why it matters.** This is the A-Z of every geoscience, geophysics, drilling, and file-format term used across these docs, each defined in 1–3 plain-English sentences aimed at a programmer. Wherever a concept page first uses one of these words, it links here. If you hit a term anywhere and it's fuzzy, this is the page to jump to. Entries are sorted alphabetically; cross-references point to the page where the idea is developed in full.

!!! tip "How to read these definitions"
    Many entries lean on a CS analogy — a **survey method** as a *lossy encoder*, **inversion** as a *decoder*, **resolution** as a *sampling/low-pass* limit, a **property volume** as an *N-D array*. The analogies are aids, not exact equivalences; the linked page has the real detail.

---

## A

**Acquisition geometry.** The physical layout of a survey — station spacing, flight-line spacing, electrode array, well path, frequency/period band. It limits *where and how densely* you sample the earth, exactly like the sampling pattern of a sensor. One of the [three universal degradations](synthetic-data.md) the synthetic generator applies.

**AEM (airborne electromagnetics).** [EM](#e) surveying flown on an aircraft/helicopter, covering large areas fast. See [electromagnetic methods](survey-methods/electromagnetic.md).

**Alteration.** Chemical change of rock by hot circulating fluids (**hydrothermal alteration**). It rearranges minerals — e.g. growing conductive **clay** near the surface and destroying magnetic **magnetite** at depth — which is *why* a geothermal system has a diagnostic geophysical signature (a conductor + a magnetic low). The synthetic state field tracks an `alteration_fraction`. See [rock physics](rock-physics.md).

**Anomaly.** A local departure from the regional background of a measured field (e.g. a **gravity anomaly**, **magnetic anomaly**). Anomalies are what you hunt for — they signal something different underground. Compare [Bouguer anomaly](#b).

**Apparent resistivity.** The [resistivity](#r) a homogeneous earth *would* have to produce the voltages you actually measured. It is not the true resistivity at a point — it's a smeared, geometry-dependent proxy that [inversion](#i) must convert into a true resistivity model. See [electrical methods](survey-methods/electrical.md).

**Archie's law.** The foundational petrophysics equation linking a rock's [resistivity](#r) to its [porosity](#p), water [saturation](#s), and pore-water salinity: $\;1/\rho_r = \phi^{m} S_w^{n} / (a\,\rho_w)$. It's why salty, porous, water-filled rock conducts electricity — the basis of using [EM](#e)/[MT](#m) to find geothermal fluid. Defined with all symbols in [rock physics](rock-physics.md) and [the synthetic generator](synthetic-data.md).

## B

**Basement.** The deep, old, usually crystalline rock (often [granite](#g)) beneath the younger sedimentary/volcanic cover. In a geothermal play it's the hot, hard rock you may target. See [the synthetic generator](synthetic-data.md).

**Bouguer anomaly.** A [gravity](#g) reading corrected for the elevation of the station and the mass of rock between the station and sea level, so what remains reflects density variations *underground*. The standard processed product of a gravity survey. See [potential fields](survey-methods/potential-fields.md).

**Brine.** Salty subsurface water. High salinity (high **TDS** — total dissolved solids) makes brine highly conductive, which is the key reason geothermal reservoirs show up as electrical [conductors](#c). See [rock physics](rock-physics.md).

## C

**Chargeability (η).** How strongly rock stores and slowly releases electrical charge — measured by [IP](#i). High where there's clay or disseminated sulphide minerals, so it flags [alteration](#a) haloes. Unit: mV/V. See [electrical methods](survey-methods/electrical.md).

**Clay cap.** A shallow, conductive layer of clay minerals formed by [alteration](#a) above a geothermal upflow — it seals heat in below. Because clay conducts, it appears as a strong, shallow electrical [conductor](#c) (a "smile" in cross-section), sharply imaged by [ERT](#e) but only smoothly by [MT](#m). See [the synthetic generator](synthetic-data.md).

**COG (Cloud-Optimized GeoTIFF).** A GeoTIFF raster laid out so a client can fetch just the tiles/zoom level it needs over HTTP, without downloading the whole file. Used for 2-D grids/rasters. See [the data model](data-model.md).

**Conductivity (σ).** How easily rock passes electric current — the reciprocal of [resistivity](#r) ($\sigma = 1/\rho_r$). Unit: S/m (siemens per metre). High conductivity = low resistivity = salty/hot/clay-rich. See [electrical](survey-methods/electrical.md) / [electromagnetic](survey-methods/electromagnetic.md) methods.

**Conduit (fault).** A fault that channels hot fluid upward. In the flagship synthetic scenario the range-front normal fault is the conduit that steers the hydrothermal upflow. See [the synthetic generator](synthetic-data.md).

**CRS (Coordinate Reference System).** The definition of how coordinates map to locations on Earth — a geographic CRS (lat/lon) or a projected one (metres). Everything must agree on one CRS or layers won't line up. See [UTM](#u), [datum](#d), and [coordinates, depth & units](spatial-framework.md).

**Cross-gradient.** A joint-inversion coupling term that rewards different property models (e.g. resistivity and density) for having boundaries in the *same places*, without forcing their values to match. See [forward modeling & inversion](inversion.md).

## D

**Datum (vertical/horizontal).** The agreed reference surface for measuring position or elevation — e.g. a horizontal datum for lat/lon, or a vertical datum (sea level / the [geoid](#g)) for height. Two datasets on different datums are silently misaligned. See [coordinates, depth & units](spatial-framework.md).

**Density (ρ).** Mass per unit volume of rock (kg/m³). [Gravity](#g) surveys sense lateral density variations. Porous/fractured rock is less dense; intrusions are denser. See [potential fields](survey-methods/potential-fields.md).

**Dogleg severity (DLS).** How sharply a well bore changes direction, in degrees per 30 m (or per 100 ft) — the curvature of the trajectory. Too high and you can't run pipe through it. See [drilling & well planning](well-planning.md).

**DOI (depth of investigation).** The depth below which a method effectively can't see — its signal has decayed into the noise. The geophysical analogue of a sensor's range. ERT's DOI is shallow; [MT](#m)'s reaches kilometres. One of the [three universal degradations](synthetic-data.md).

## E

**EDI.** The standard text file format for [magnetotelluric](#m) data — it stores the frequency-dependent **impedance tensor** plus metadata. See [electromagnetic methods](survey-methods/electromagnetic.md).

**EGS (Enhanced/Engineered Geothermal System).** A geothermal approach for hot but low-[permeability](#p) rock: you *create* the cracks by injecting fluid under pressure (stimulation), then circulate water through the engineered fracture network. Monitored with [microseismic](#m) and [InSAR](#i). See [the geothermal primer](geothermal-primer.md).

**Electromagnetic (EM) methods.** Methods that use changing magnetic fields to induce currents in the ground and infer [conductivity](#c) vs depth — [TEM](#t), [AEM](#a), and (using natural fields) [MT](#m). See [electromagnetic methods](survey-methods/electromagnetic.md).

**Engineering Frame.** This project's local working coordinate frame: X-East / Y-North / Z-Up, in metres, with a floating origin. Every primitive is transformed into it on ingest so all layers share one space. See [coordinates, depth & units](spatial-framework.md).

**ERT (Electrical Resistivity Tomography).** Inject DC current through electrodes, measure voltages, and invert for a [resistivity](#r) image — sharp but shallow. See [electrical methods](survey-methods/electrical.md).

## F

**Fault.** A fracture surface in rock along which the two sides have slipped past each other. Faults offset rock units (changing the lithology map) and can act as fluid [conduits](#c). A **normal fault** is one where the upper block slid down — the dominant style in extensional settings like the Basin & Range. See [geology](survey-methods/geology-geochem.md).

**Favorability.** A derived volume scoring "how good a geothermal drilling target is this voxel" by combining the evidence for **heat + fluid + permeability** all coinciding. The output of [rock-physics transforms](rock-physics.md) and fusion. See [fusion](fusion.md).

**Forward problem.** Given a known earth, *compute* what a survey would measure — the "easy," well-posed direction (cause → effect). The opposite of the [inverse problem](#i). The synthetic generator's [forward models](synthetic-data.md) do exactly this. See [forward modeling & inversion](inversion.md).

**Fracture density.** How heavily cracked the rock is per unit volume — a proxy for [permeability](#p). Carried in the synthetic [state field](#s) and inferred from [microseismic](#m). See [the synthetic generator](synthetic-data.md).

**Fused earth model / fused grid.** The canonical 3-D grid covering the region of interest, onto which any [property model](#p) can be *resampled* so methods can be compared, overlaid, and cross-plotted cell-by-cell — without destroying the native-resolution originals. The common ground of fusion. See [fusion](fusion.md).

## G

**Geoid.** The Earth's true gravitational equipotential surface — roughly mean sea level extended under the continents — used as a vertical reference. Real elevations are measured relative to it, not to the smooth ellipsoid. See [coordinates, depth & units](spatial-framework.md).

**Geothermal gradient.** How fast temperature rises with depth, in °C/km. A normal crust is ~25–30 °C/km; the Basin & Range runs hot at ~45 °C/km, which is why it's a good geothermal target. See [the geothermal primer](geothermal-primer.md).

**Granite.** A hard, crystalline intrusive rock common in [basement](#b) and as [intrusions](#i). Often the hot rock targeted by [EGS](#e).

**Gravity.** The survey method that measures tiny variations in the Earth's gravitational pull caused by lateral [density](#d) differences underground. Deep-seeing but smooth and [non-unique](#n). See [potential fields](survey-methods/potential-fields.md).

**Gutenberg-Richter.** The empirical law that earthquake counts fall off exponentially with magnitude (many small, few large): $\log_{10} N = a - bM$. Used to model realistic [microseismic](#m) event catalogs. See [seismic & microseismic](survey-methods/seismic.md).

## H

**Heat flow.** The rate heat escapes through the Earth's surface (mW/m²), proportional to the [geothermal gradient](#g) times the rock's thermal conductivity. High heat flow marks geothermal prospectivity. See [boreholes](survey-methods/boreholes.md).

**Horizon.** An interpreted geological surface — typically a boundary between rock layers — picked from [seismic](#s) or wells, stored as a [feature](data-model.md). See [seismic](survey-methods/seismic.md).

**Hydrothermal.** Pertaining to hot water circulating in the crust. A **hydrothermal system** is the heat-driven plumbing — upflow, [reservoir](#r), [clay cap](#c) — that a conventional geothermal play exploits. See [the geothermal primer](geothermal-primer.md).

## I

**Impedance (seismic, Z).** The product of [density](#d) and seismic [velocity](#v) ($Z = \rho V_p$). Sound reflects off *contrasts* in impedance, so impedance boundaries are what [seismic reflection](#s) actually images. See [seismic](survey-methods/seismic.md).

**Impedance tensor (MT).** The frequency-dependent 2×2 complex matrix relating electric to magnetic fields in [MT](#m), from which [apparent resistivity](#a) and phase vs period are derived. Stored in [EDI](#e) files. See [electromagnetic methods](survey-methods/electromagnetic.md).

**Induced polarization (IP).** A method (usually run with [ERT](#e)) that measures how rock stores charge — [chargeability](#c) — flagging clay and sulphide [alteration](#a). See [electrical methods](survey-methods/electrical.md).

**Ingestion adapter.** A plugin that parses one native file format into the normalized [data-model](data-model.md) primitives. One of the six plugin [extension points](architecture.md). See [ingestion](ingestion.md).

**InSAR (Interferometric Synthetic Aperture Radar).** A satellite technique that compares radar phase between passes to measure millimetre-scale ground deformation (uplift/subsidence) over time — sensed only along the [LOS](#l). Used to monitor injection-driven swelling. See [InSAR](survey-methods/insar.md).

**Intrusion.** A body of molten rock that pushed into surrounding rock and solidified (e.g. a granite **stock**). Often dense and, when young, a heat source. See [the synthetic generator](synthetic-data.md).

**Inverse problem / inversion.** Given measurements, *infer* the earth that produced them (effect → cause) — the hard, ill-posed direction. Like decoding a lossy-compressed file: many earths fit the data ([non-unique](#n)), so you need [regularization](#r) to pick a sensible one. The decoder to the [forward problem](#f)'s encoder. See [forward modeling & inversion](inversion.md).

## K

**KB (Kelly Bushing).** The reference height at the rig floor from which well depths are measured — the zero of [MD](#m). To compare wells to the earth model you convert KB-referenced depths to a true elevation. See [drilling & well planning](well-planning.md).

**Kriging.** A geostatistical interpolation method that fills gaps between scattered samples while estimating the uncertainty of each interpolated value. Used in gridding/ingestion. See [ingestion](ingestion.md).

## L

**LAS (Log ASCII Standard).** The standard text format for well-log curves — depth-indexed columns of measured properties (resistivity, gamma, density, temperature…) down a borehole, with a header of metadata. See [boreholes](survey-methods/boreholes.md).

**Lithology.** The rock *type* (alluvium, volcanics, carbonate, granite…). In the synthetic generator the lithology field $L$ is an integer label per voxel — a categorical map — distinct from the continuous [state field](#s). See [the synthetic generator](synthetic-data.md).

**LOS (line of sight).** The look direction from a radar satellite to the ground. [InSAR](#i) measures only the component of ground motion *along* the LOS, not full 3-D displacement. See [InSAR](survey-methods/insar.md).

## M

**Magnetics.** The survey method that maps variations in the Earth's magnetic field caused by magnetic minerals (mainly magnetite) in rock — i.e. [susceptibility](#s). [Alteration](#a) destroys magnetite, so geothermal upflows read as magnetic *lows*. See [potential fields](survey-methods/potential-fields.md).

**MD (Measured Depth).** Distance measured *along* a (possibly curved) well bore from the [KB](#k). Contrast [TVD](#t) (straight-down depth). See [drilling & well planning](well-planning.md).

**Microseismic.** Tiny earthquakes triggered by fluid injection or rock stress, recorded to map where rock is fracturing — direct evidence of [permeability](#p)/[fracture density](#f). Stored as a 4-D event point cloud ([QuakeML](#q)). See [seismic & microseismic](survey-methods/seismic.md).

**Minimum curvature.** The standard method for reconstructing a well's 3-D path from its directional survey (a list of [MD](#m), inclination, azimuth), assuming a smooth circular arc between stations. See [drilling & well planning](well-planning.md).

**MT (Magnetotellurics).** An [EM](#e) method using *natural* electromagnetic fields over a huge period band to image [resistivity](#r) kilometres deep. Deep-seeing but smooth; the workhorse for finding deep geothermal [conductors](#c). See [electromagnetic methods](survey-methods/electromagnetic.md).

## N

**Non-uniqueness.** The defining curse of geophysical [inversion](#i): many different earths produce nearly identical surface measurements, so the data alone can't pick the true one. Worst for smooth, deep methods ([gravity](#g), [MT](#m)). It's why we fuse methods and validate against synthetic [ground truth](synthetic-data.md). See [uncertainty](uncertainty.md).

## P

**Permeability.** How easily fluid flows through rock (units of m² or darcy) — distinct from [porosity](#p) (how much space there *is*). One of the three things that must coincide for a geothermal target (heat + fluid + permeability). Hard to measure directly; inferred from [fracture density](#f)/[microseismic](#m). See [the geothermal primer](geothermal-primer.md).

**Porosity (φ).** The fraction of a rock's volume that is empty pore space (0–1). Drives [density](#d), [resistivity](#r) (via [Archie](#a)), and [velocity](#v). See [rock physics](rock-physics.md).

**Potential fields.** The collective term for [gravity](#g) and [magnetics](#m) — methods sensing static fields that obey potential theory. Both are deep, smooth, and [non-unique](#n). See [potential fields](survey-methods/potential-fields.md).

**Property model.** A [data-model](data-model.md) primitive: a continuous 3-D (or 4-D) field of *one* physical property (a resistivity cube, a density grid…), with units, support geometry, and uncertainty. In CS terms, a labelled N-D array on a grid. See [the data model](data-model.md).

**Propylitic.** A deep, hot style of [alteration](#a) (chlorite/epidote-rich) found below the [clay cap](#c) in a geothermal system — marks the [reservoir](#r) zone. See [the synthetic generator](synthetic-data.md).

**Provenance.** The recorded lineage of every artifact — which plugin/version, which inputs, which parameters, which source CRS/units produced it — so any number is reproducible and auditable. Stamped automatically. See [the data model](data-model.md).

**Pseudosection.** A raw 2-D plot of [apparent resistivity](#a) (or [chargeability](#c)) against electrode position and spacing — a quick, distorted picture *before* [inversion](#i) turns it into a true depth model. See [electrical methods](survey-methods/electrical.md).

## Q

**QuakeML.** An XML standard for describing seismic events (origin time, location, magnitude, uncertainty). The native format for [microseismic](#m) catalogs. See [seismic & microseismic](survey-methods/seismic.md).

## R

**Reflectivity.** The fraction of seismic energy bounced back at an [impedance](#i) contrast. Convolving the reflectivity series with a source wavelet gives a synthetic [seismic](#s) trace. See [seismic](survey-methods/seismic.md).

**Regularization.** Extra constraints added to an [inversion](#i) to tame [non-uniqueness](#n) — e.g. preferring smooth or small models. Mathematically it makes an ill-posed problem solvable. See [Tikhonov](#t) and [forward modeling & inversion](inversion.md).

**Reservoir (geothermal).** The permeable, fluid-bearing, hot rock volume you actually produce from — the prize. Below the [clay cap](#c), often [propylitically](#p) altered and fractured. See [the geothermal primer](geothermal-primer.md).

**Resistivity (ρ_r).** How strongly rock opposes electric current (Ω·m) — the reciprocal of [conductivity](#c). The single most important property for geothermal: hot, saline, clay-altered rock is strongly conductive (low resistivity). Imaged by [ERT](#e), [EM](#e), and [MT](#m). See [electrical methods](survey-methods/electrical.md).

**ROI (region of interest).** The bounding box (X, Y, and depth range) a project covers — the extent of the [fused grid](#f). See [coordinates, depth & units](spatial-framework.md).

**RTP (Reduction To the Pole).** A processing step that transforms a [magnetic](#m) anomaly to how it would look if measured at the magnetic pole (field vertical), so anomalies sit symmetrically over their sources and are easier to interpret. See [potential fields](survey-methods/potential-fields.md).

## S

**Saturation (water, S_w).** The fraction of a rock's [pore space](#p) filled with water (vs gas/steam), 0–1. A key [Archie's-law](#a) term controlling [resistivity](#r). See [rock physics](rock-physics.md).

**Seed.** The integer that initializes the synthetic generator's random-number streams. Because everything derives deterministically from `(spec, seed)`, the same seed gives byte-identical output — reproducibility like a fixed PRNG seed in a test. See [the synthetic generator](synthetic-data.md).

**SEG-Y.** The standard binary format for [seismic](#s) data — traces plus a textual + binary header describing geometry and sampling. See [seismic](survey-methods/seismic.md).

**Seismic (reflection/refraction).** Methods that send sound/elastic waves into the ground and time the echoes (reflection) or refracted first arrivals (refraction) to image structure and [velocity](#v). The sharpest structural method. See [seismic & microseismic](survey-methods/seismic.md).

**Skin depth (δ).** In [EM](#e)/[MT](#m), the depth at which the field has decayed to $1/e$ — it sets how deep a given frequency sees: $\delta \approx 503\sqrt{\rho_r T}$ metres (ρ_r in Ω·m, period $T$ in s). Lower frequencies (longer periods) reach deeper, which is how MT does depth. See [electromagnetic methods](survey-methods/electromagnetic.md).

**.stg.** The native text format of AGI electrical-resistivity instruments — the raw electrode geometry and measured voltages for an [ERT](#e)/[IP](#i) line. See [electrical methods](survey-methods/electrical.md).

**State field (S).** The synthetic generator's continuous per-voxel record of rock *condition* — temperature, porosity, water saturation, salinity, alteration/clay fraction, fracture density. Modulates rock physics within a [lithology](#l); the geothermal target is primarily a state perturbation. See [the synthetic generator](synthetic-data.md).

**Stimulation.** Deliberately pumping fluid to open/create fractures and raise [permeability](#p), central to [EGS](#e) — monitored by [microseismic](#m) and [InSAR](#i). See [the geothermal primer](geothermal-primer.md).

**Susceptibility (magnetic, χ).** How strongly rock becomes magnetized in an applied field (dimensionless SI) — driven by magnetite content. [Magnetics](#m) senses it; [alteration](#a) suppresses it. See [potential fields](survey-methods/potential-fields.md).

## T

**TDS (total dissolved solids).** The salinity of pore water (ppm). High TDS → conductive [brine](#b) → low [resistivity](#r). See [rock physics](rock-physics.md).

**TEM (Transient Electromagnetics).** A time-domain [EM](#e) method: pulse a transmitter, then watch the decaying "smoke-ring" of induced currents to infer [conductivity](#c) vs depth. See [electromagnetic methods](survey-methods/electromagnetic.md).

**Tikhonov regularization.** The classic [regularization](#r) scheme that adds a penalty on model size/roughness to an [inversion](#i)'s data-misfit objective, controlled by a trade-off parameter — picking a smooth, stable solution from the [non-unique](#n) many. See [forward modeling & inversion](inversion.md).

**Transfer function (rendering).** The mapping from a property's data value to colour and opacity in the 3-D viewer — the same idea as a colormap-plus-alpha-ramp in scientific visualization. Declared per property type. See [the 3D viewer](visualization.md).

**Transform (rock-physics).** A plugin that derives new property fields from existing ones (e.g. resistivity + temperature → fluid likelihood), running on the [fused grid](#f). One of the six plugin [extension points](architecture.md). See [rock physics](rock-physics.md).

**TVD / TVDSS.** **TVD** (True Vertical Depth) is straight-down depth below the [KB](#k), versus along-hole [MD](#m). **TVDSS** (TVD Sub-Sea) references that vertical depth to sea level instead, so wells can be compared to each other and to the earth model. See [drilling & well planning](well-planning.md).

**Two-way time (TWT).** In [seismic reflection](#s), the time for a wave to travel down to a reflector and back, in seconds — the native vertical axis of raw seismic before depth conversion. See [seismic](survey-methods/seismic.md).

## U

**UTM (Universal Transverse Mercator).** A common projected [CRS](#c) that slices the globe into 6°-wide zones, giving coordinates in metres (easting/northing) — convenient because distances are nearly true within a zone. The usual choice for a project's horizontal frame. See [coordinates, depth & units](spatial-framework.md).

## V

**Velocity (seismic, Vp/Vs).** The speed of P-waves ($V_p$, compressional) and S-waves ($V_s$, shear) through rock (m/s). Their ratio $V_p/V_s$ is sensitive to fluid (saturation barely moves $V_s$ but raises $V_p$), making it a fluid indicator. See [seismic](survey-methods/seismic.md).

**Voxel.** A 3-D pixel — one cell of a regular volumetric grid. A [property model](#p) is, in the simplest case, an array of voxels. See [the data model](data-model.md).

## W

**WITSML.** An XML standard for exchanging drilling/well data (trajectories, logs, operations). A well-planning export target. See [drilling & well planning](well-planning.md).

**Well log.** A depth-indexed curve of a property measured down a borehole (resistivity, gamma, density, temperature…). The *cleanest, most direct* subsurface data — ground-truth that calibrates everything else. Stored as [LAS](#l). See [boreholes](survey-methods/boreholes.md).

## Z

**Zarr.** A format for chunked, compressed N-D arrays, readable lazily and in parallel — ideal for streaming big 3-D/4-D volumes to the browser a brick at a time, and for storing the synthetic [ground-truth](synthetic-data.md) volumes. Think of it as a cloud-friendly, chunked NumPy array on disk. See [the data model](data-model.md).

---

## Key takeaways

- Geophysical methods are **lossy encoders** of the earth; [inversion](#i) is the **decoder**, and it's [non-unique](#n) — which is the whole reason this platform fuses many methods and validates against synthetic [ground truth](synthetic-data.md).
- The three things a geothermal target needs — **heat + fluid + permeability** — map onto distinct properties (temperature; resistivity/saturation/salinity; fracture density/permeability), each best sensed by *different* methods.
- Depth-seeing is governed by physics: [skin depth](#s) for [EM](#e)/[MT](#m), [DOI](#d) for [ERT](#e), [two-way time](#t) for [seismic](#s).
- When in doubt about a term anywhere in the docs, search this page — every concept page links its first use of a term back here.

## Where this lives in the code

The canonical machine-readable lists behind these human definitions live in:

| Concept | Path |
|---|---|
| Canonical method/submethod keys | `backend/geosim/plugins/methods.py` |
| Property-type registry (units, colormap, scaling) | `backend/geosim/spatial/property_types.py` |
| Units registry (`pint`) | `backend/geosim/spatial/units.py` |
| Spatial frame / CRS / datum / vertical | `backend/geosim/spatial/{frame,vertical}.py` |
| Rock-physics rules + unit library | `backend/geosim/synthgen/rockphysics.py`, `backend/geosim/fusion/rockphys/` |
