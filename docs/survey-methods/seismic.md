# Seismic & microseismic

!!! abstract "What you'll learn / why it matters"
    This page covers the methods that work with **sound in rock**. Active **seismic**
    surveys make a controlled bang and record the echoes: *reflection* seismic images
    layer boundaries and faults (the sharpest picture of subsurface *structure* you can
    get), while *refraction* seismic times the first arrivals to recover shallow velocity.
    **Microseismic** is the opposite — you *listen* for the tiny natural earthquakes that
    rock makes when it fractures, and locate them in space and time to watch a fracture
    network grow. The recurring lesson: **seismic sees structure beautifully but is nearly
    blind to fluid and temperature**, the very things geothermal cares about — so it must be
    [fused](../fusion.md) with the [electromagnetic](electromagnetic.md) and
    [borehole](boreholes.md) methods that *do* see fluid. By the end you'll know how a
    [SEG-Y](#the-seg-y-file-format) file and a [QuakeML](#the-quakeml-file-format) catalog
    are built and what [primitives](../data-model.md) they become.

If [electromagnetic](electromagnetic.md) methods are about how rock *conducts electricity*,
seismic is about how rock *carries sound*. The governing property is **seismic velocity** —
how fast a sound wave travels through the rock:

!!! note "Define: Vp, Vs, acoustic impedance"
    - **$V_p$** — the **P-wave (compressional) velocity** (m/s): the speed of a
      push-pull wave, like sound in air, the *first* arrival. Typical rock: 2,000–6,000 m/s.
    - **$V_s$** — the **S-wave (shear) velocity** (m/s): a slower, sideways-shaking wave that
      cannot travel through fluids. The $V_p/V_s$ ratio is a fluid clue.
    - **Acoustic impedance $Z = \rho \cdot V_p$**: density times P-velocity. This is the
      single most important quantity for reflection seismic — **echoes come from *changes*
      in $Z$, not from $Z$ itself.** (Analogy: a signal only reflects at an *impedance
      mismatch*, exactly like an electrical transmission line or an optical interface.)

---

## A programmer's mental model

Reflection seismic is, almost literally, a **convolution**. Take the impedance as a 1-D
array sampled down a vertical column:

```python
imp = density * vp          # acoustic impedance Z(depth), a 1-D array
refl = (imp[1:] - imp[:-1]) / (imp[1:] + imp[:-1])   # reflectivity: normalized derivative
trace = numpy.convolve(refl_in_time, wavelet)        # what the geophone records
```

1. **Reflectivity** `refl` is the normalized first difference of impedance — it spikes only
   where impedance *changes* (a layer contact or a fault). Between contacts it is ~zero.
2. You can't record an infinitely sharp spike; your source emits a band-limited pulse (a
   **wavelet**). The recorded **trace** is the reflectivity series *convolved* with that
   wavelet. The wavelet is the survey's "point spread function."
3. Because the wavelet is band-limited, the result is **blurred** in exactly the way a
   low-pass filter blurs a signal — which sets the resolution limit we'll meet below.

Microseismic is a different shape entirely: a **4-D point cloud**, $(x, y, z, t,
\text{magnitude})$ — one point per micro-earthquake, located in space and stamped in time.
Think of it as a stream of events, not a field.

---

## Seismic reflection

!!! note "Define: reflection seismology, CMP, two-way time"
    - **Reflection seismology**: bang the surface (a vibrator truck, weight drop, or small
      charge), and record at many geophones the **echoes** bouncing off subsurface impedance
      contrasts. From the echo *times* and *amplitudes* you reconstruct the layered/faulted
      structure.
    - **CMP** (Common Mid-Point): by combining many source-receiver pairs that share a
      midpoint, you build one high-quality trace per surface location. Each CMP trace is one
      "pixel column" of the final image.
    - **TWT** (Two-Way Time): the *vertical axis of a seismic section is not depth — it is
      time*, the round-trip travel time of the echo, in seconds (or ms). Down means later.

### Reflectivity, the wavelet, and the convolution model

At every interface where impedance jumps from $Z_1$ to $Z_2$, the fraction of energy
reflected (at normal incidence) is the **reflection coefficient**:

$$
r \;=\; \frac{Z_2 - Z_1}{Z_2 + Z_1}, \qquad Z = \rho\,V_p
$$

| Symbol | Meaning | Units |
|---|---|---|
| $r$ | reflection coefficient (the spike height in the reflectivity series) | dimensionless, $[-1, 1]$ |
| $Z_1, Z_2$ | acoustic impedance above / below the interface | $\text{kg}\,\text{m}^{-2}\text{s}^{-1}$ |
| $\rho$ | bulk density | $\text{kg/m}^3$ |
| $V_p$ | P-wave velocity | m/s |

Stack all the $r$ values down a column into a *reflectivity series*, convolve with a
band-limited **wavelet** $w(t)$ (the project uses a zero-phase **Ricker** wavelet — the
classic symmetric "Mexican hat" pulse), and you have the synthetic trace:

$$
\text{trace}(t) \;=\; r(t) \,\ast\, w(t) \;+\; \text{noise}
$$

To get $r$ from depth onto the time axis, you map depth to **two-way time** by integrating
slowness down the column: each layer of thickness $\Delta z$ adds $2\,\Delta z / V_p$ of
round-trip time.

### Vertical resolution — the $\lambda/4$ rule

The wavelet is band-limited, so two reflectors closer than about a quarter-wavelength merge
into one. The **vertical resolution limit** is:

$$
\Delta z_{\min} \;\approx\; \frac{\lambda}{4} \;=\; \frac{V_p}{4 f}
$$

| Symbol | Meaning | Units |
|---|---|---|
| $\lambda$ | dominant wavelength of the seismic wavelet | m |
| $V_p$ | local P-velocity | m/s |
| $f$ | dominant (peak) frequency of the wavelet | Hz |

!!! example "Plug in real numbers"
    For $V_p = 3000$ m/s and a $f = 30$ Hz wavelet: $\lambda = 100$ m, so
    $\Delta z_{\min} \approx 25$ m. Beds thinner than ~25 m can't be resolved as separate
    reflectors at that depth — they blur together. This is the seismic equivalent of a
    sampling / Nyquist resolution bound. Higher frequency = finer resolution, but high
    frequencies attenuate faster, so deep imaging is inherently lower-resolution.

### Why seismic is blind to fluid and temperature

This is the single most important conceptual point of the page, and it's why fusion exists:

!!! warning "Seismic sees structure, not fluid"
    Reflectivity comes from impedance *contrasts* — layer contacts and faults. So reflection
    seismic gives you the **geometry**: where the layers are, where the faults cut, the shape
    of a reservoir. But the temperature and pore-fluid that matter for geothermal change
    impedance only weakly compared to lithology, so seismic is **nearly blind to the fluid
    and temperature field** (doc 05 §4.2). You learn *where the container is*, not *what's in
    it or how hot it is*. The fluid/heat cue must come from [MT/EM](electromagnetic.md)
    (conductivity), [boreholes](boreholes.md) (direct $V_p/V_s$, temperature), or
    [rock physics](../rock-physics.md).

!!! example "How the synthetic generator builds a section"
    `backend/geosim/synthgen/forward/seismic.py` (`SeismicReflectionForward`, T0) does
    exactly the convolution model: down each CMP column it samples the truth impedance
    $Z=\rho V_p$, forms reflectivity at contrasts, maps depth→TWT with the interval
    velocity, convolves with a Ricker wavelet (`ricker(...)`), adds band-limited noise, and
    picks the strongest reflector as a horizon. The **rigorous (T1)**
    `SeismicReflectionRigorousForward` upgrades this to the *full* impedance series (every
    depth sample, not just hand-picked contacts), uses **amplitude-conserving linear
    placement** of each reflection at its true two-way time (instead of snapping to the
    nearest sample), and adds a first-order surface **multiple** (a delayed, polarity-flipped
    echo — a real artifact). Both emit the **same** SEG-Y + horizons GeoJSON, so ingestion
    is unchanged; only the realism of the numbers improves.

### The SEG-Y file format

**SEG-Y** is the universal seismic exchange format (SEG = Society of Exploration
Geophysicists). It is a *binary* format, so you can't `cat` it — but its structure is simple
and worth knowing:

```text title="SEG-Y structure (conceptual layout)"
┌─────────────────────────────────────────────────────────────┐
│ 1. Textual header   3200 bytes, 40 lines of 80 chars         │  ← human notes:
│    "C 1 CLIENT: GEOSIM   C 2 LINE: AA'  ..."                  │    (the "C 1" magic
│                                                              │     adapters sniff)
├─────────────────────────────────────────────────────────────┤
│ 2. Binary header    400 bytes of survey-wide fields:         │  ← sample interval dt,
│    sample_interval (dt, µs), samples_per_trace (ns),         │    #samples, data format
│    data_format (=5 means IEEE float32)                       │
├─────────────────────────────────────────────────────────────┤
│ 3. Per trace, repeated tracecount times:                     │
│    ├ Trace header  240 bytes: cdp, cdp_x, cdp_y, ns, dt, ... │  ← WHERE this trace is
│    └ Trace samples  ns × 4-byte floats: the amplitudes       │  ← the recorded wiggle
└─────────────────────────────────────────────────────────────┘
```

The fields that matter for placing the data in space:

| Field | Meaning | Why it matters |
|---|---|---|
| `dt` (sample interval) | time between samples, in microseconds | sets the **TWT axis spacing** |
| `ns` (samples per trace) | number of samples down one trace | trace length in time |
| `cdp` | the CMP/CDP number | trace identity along the line |
| `cdpx`, `cdpy` | the trace's plan coordinates (X, Y) | places the trace in the [Engineering Frame](../spatial-framework.md) |
| `data_format = 5` | IEEE 4-byte float | how to decode the sample bytes |

The generator writes traces with `segyio`: format 5 (IEEE float32), the time axis in ms from
`dt`, and per-trace `cdpx`/`cdpy` plan coordinates — see `_write_segy_section` in
`seismic.py`. The picked horizons go in a sibling **GeoJSON**:

```json title="seismic_horizons.geojson (annotated)"
{
  "type": "FeatureCollection",
  "name": "great-basin-v1-horizons",
  "features": [{
    "type": "Feature",
    "geometry": {                          // (1) the horizon traced along the line in plan
      "type": "LineString",
      "coordinates": [[1200, 1750], [1250, 1750], [1300, 1750]]
    },
    "properties": {
      "kind": "horizon",                   // (2) it's a picked reflector
      "pick": "strongest_reflector",       // (3) how it was picked
      "twt_s": [0.412, 0.418, 0.421]       // (4) two-way time per vertex (the depth proxy)
    }
  }]
}
```

1. The `LineString` is the horizon's trace in plan view (X, Y per CMP).
2. `kind: "horizon"` marks it a picked seismic reflector (vs a fault, well path, etc.).
3. `pick` records the picking rule for provenance.
4. `twt_s` carries the two-way time per vertex — the vertical position, in *time*; converting
   it to depth needs a velocity model.

### What it becomes after [ingestion](../ingestion.md)

`backend/geosim/ingestion/adapters/seismic.py` (`SeismicSegyAdapter`,
`method="seismic"`, `submethod="reflection"`) reads the SEG-Y with `segyio` and emits a
[`PropertyModel`](../data-model.md) with **`support = "section"`** — a 2-D vertical *curtain*
along the shot line (leading axis = TWT/depth, second axis = along-line distance). When
inline/crossline geometry is present it instead emits a 3-D velocity `volume`. If the sibling
`*_horizons.geojson` is found, each horizon becomes a [`Feature`](../data-model.md) of type
`horizon` (a **surface/stick**). One file, multiple primitives — returned together in one
parse result. Coordinates and the native time axis are kept as-is; the
[normalizer](../ingestion.md) canonicalizes.

---

## Seismic refraction

!!! note "Define: refraction, first break, head wave"
    Where reflection seismic uses *echoes*, **refraction seismic** uses **first breaks** —
    the earliest energy to arrive at each geophone. Part of that energy travels down to a
    fast layer, runs *along* the top of it (the **head wave**), and surfaces again. By timing
    the first arrival vs distance from the shot (**offset**), you recover the **shallow
    velocity structure** — invaluable for near-surface corrections and for shallow geothermal
    targets.

### First-break traveltimes and the eikonal equation

Plot first-break **traveltime** against **offset** and you get a *traveltime curve*. At short
offset the **direct wave** (straight through the slow top layer) arrives first; past a
crossover distance the faster **head wave** overtakes it. The slope of each segment is
$1/V$ (slowness), so the curve's increasing apparent velocity reveals a layered, faster-with-
depth earth — the basis of refraction inversion.

The exact physics is the **eikonal equation**, which governs how a wavefront's arrival-time
field $T(\mathbf{x})$ propagates through a velocity model:

$$
|\nabla T| \;=\; \frac{1}{V_p}
$$

| Symbol | Meaning |
|---|---|
| $T(\mathbf{x})$ | first-arrival traveltime to point $\mathbf{x}$ (s) |
| $\nabla T$ | spatial gradient of traveltime (its magnitude is slowness) |
| $V_p$ | P-velocity at $\mathbf{x}$ (m/s) |

In plain terms: the wavefront's travel-time gradient everywhere equals the local slowness
($1/V_p$). Solving it (a *fast-marching*-style level-set computation, conceptually like
Dijkstra's shortest path on a velocity grid) yields the traveltime to every node.

!!! example "How the synthetic generator computes refraction picks"
    `SeismicRefractionRigorousForward` (T1) in `seismic.py` solves the eikonal equation
    through the truth $V_p$ model with **pykonal** along a refraction spread (a shot at the
    line start, geophones at growing offset), giving a physically sensible first-break to
    each geophone (monotone in offset; the direct/head-wave crossover appears naturally). It
    emits a **SEG-Y** (one trace per geophone, a Ricker pulse at the picked time) plus a
    **CSV** of picks:

    ```text title="seismic_refraction_picks.csv (annotated)"
    trace,offset_m,traveltime_s   # (1) column header
    0,0.000,0.000000              # (2) the shot itself: zero offset, zero time
    1,25.000,0.012500             # (3) near geophone: direct wave, slope ~ 1/V_top
    2,50.000,0.024100             #     ...
    40,1000.000,0.221000          # (4) far geophone: head wave has overtaken (faster apparent V)
    ```

    1. The CSV is the refraction analogue of the reflection horizons file.
    2. Trace 0 is co-located with the shot.
    3. At small offsets the slope of (offset vs time) is $1/V_{\text{top}}$ — the slow layer.
    4. At large offsets the slope flattens — the fast head-wave layer dominates.

### What it becomes after [ingestion](../ingestion.md)

A refraction SEG-Y normalizes to a velocity `section` (2-D line) or `volume`
[`PropertyModel`](../data-model.md) of `velocity_p` (doc 03 §2). The picks themselves feed a
shallow-velocity inversion downstream ([inversion](../inversion.md)).

---

## Microseismic

!!! note "Define: microseismic monitoring"
    **Microseismic monitoring** records the *natural* seismic energy — tiny earthquakes,
    typically magnitude $< 0$ to $\sim 2$, far too small to feel — that rock emits when it
    fractures or slips. In geothermal, especially **EGS** (Enhanced Geothermal Systems),
    you inject high-pressure fluid to crack low-permeability hot rock; each crack pops as a
    micro-earthquake. **Locating those pops in space and time maps the growing fracture
    network — i.e. where you just created permeability.** This is a direct, 4-D proxy for the
    one geothermal ingredient nothing else measures well: *cracks for fluid to flow through.*

!!! note "Why EGS cares: the three-ingredient rule"
    Geothermal needs **heat + fluid + permeability** in the same place (see the
    [core problem](../core-problem.md)). [MT](electromagnetic.md) and
    [boreholes](boreholes.md) hint at heat and fluid; **permeability** is the hardest to
    image. Microseismic is the closest thing to watching permeability *being created*, live,
    during a stimulation. That's why it's the flagship dataset of the EGS scenario.

### Gutenberg–Richter — the magnitude distribution

Earthquakes of all sizes follow a strikingly simple power law: small ones vastly outnumber
big ones, log-linearly. This is the **Gutenberg–Richter law**:

$$
\log_{10} N \;=\; a - b\,M
$$

| Symbol | Meaning |
|---|---|
| $N$ | number of events with magnitude $\ge M$ |
| $M$ | magnitude |
| $a$ | overall productivity (how many events total) |
| $b$ | the **b-value** — the slope; typically $\approx 1$ (each unit drop in $M$ gives ~10× more events) |

A larger $b$ means relatively *more small* events (often gentle, fluid-driven cracking); a
smaller $b$ can flag larger, more worrying events. There is also a **completeness magnitude
$M_c$** below which the network is too insensitive to catch every event — a detection floor.

!!! example "How the synthetic generator makes a catalog"
    `MicroseismicForward` in `seismic.py` samples events **on the stimulated fault plane**
    (the conduit fault), draws magnitudes from Gutenberg–Richter by inverse-CDF
    ($M = M_c - \log_{10}(U)/b$ for uniform $U$), and **locates each event with an error
    that grows with depth** (deeper = farther from the surface array = fuzzier location).
    That depth-growing location $\sigma$ is the honest part: real microseismic locations are
    uncertain, and the platform carries that as [uncertainty](../uncertainty.md). It emits a
    **QuakeML** catalog plus a sibling CSV.

### The QuakeML file format

**QuakeML** is the standard XML schema for earthquake catalogs (events, origins, magnitudes).
Here is one event, annotated:

```xml title="microseismic.quakeml (annotated, one event)"
<event>                                           <!-- (1) one micro-earthquake -->
  <origin>                                        <!-- (2) where & when it happened -->
    <time>
      <value>2026-01-01T03:00:00.000000Z</value>  <!-- (3) explicit ISO-8601 UTC epoch -->
    </time>
    <latitude><value>0.0</value></latitude>       <!-- (4) zeroed: see note below -->
    <longitude><value>0.0</value></longitude>
    <depth><value>1820.0</value></depth>          <!-- (5) depth below surface, metres -->
  </origin>
  <magnitude>                                      <!-- (6) the size of the event -->
    <mag><value>-0.420</value></mag>
    <type>ML</type>                                <!-- local magnitude -->
  </magnitude>
</event>
```

1. Each `<event>` is one located micro-earthquake.
2. `<origin>` holds the hypocentre time and position.
3. `<time>` is an **explicit ISO-8601 UTC timestamp** — the data model fixes time as a
   leading `t` axis with real UTC epochs, not project-relative offsets (doc 02 §8).
4. **Important quirk:** QuakeML's lat/lon are geographic, but our events live in a local
   [Engineering Frame](../spatial-framework.md). The forward therefore *zeroes* lat/lon in
   the QuakeML and stores the real Engineering $(x, y)$ in a sibling
   `microseismic_catalog.csv`. The adapter joins the two.
5. `<depth>` is depth below surface in metres.
6. `<magnitude>` carries the value and type (`ML` = local magnitude).

The sibling CSV carries the plan coordinates the QuakeML can't:

```text title="microseismic_catalog.csv (annotated)"
id,time,x,y,elev,mag                                  # (1) header
0,2026-01-01T00:00:00,1480.20,1755.61,-1820.00,-0.420 # (2) event 0: Engineering X/Y/elev + mag
1,2026-01-01T01:00:00,1495.83,1749.12,-1640.50,0.115  #     elev is +up (negative = below surface)
```

1. The CSV provides the Engineering-frame `x`, `y`, `elev` the QuakeML lacks.
2. Joined by `id`, each row supplies the spatial location; the QuakeML supplies the canonical
   time and magnitude.

### What it becomes after [ingestion](../ingestion.md)

`backend/geosim/ingestion/adapters/microseismic.py`
(`MicroseismicQuakeMlAdapter`, `method="microseismic"`) reads the QuakeML with `obspy` for
each event's **time + magnitude**, joins the sibling CSV for Engineering $(x, y, \text{elev})$,
and emits a single [`Feature`](../data-model.md) of `feature_type = "pointCloud"`: a GeoJSON
`MultiPoint` of $[x, y, z]$ whose `props` carry the parallel **time** array (ISO-8601 UTC),
the **magnitudes**, and `dims: [x, y, z, t, mag]`. That is the **4-D point cloud** the data
model mandates for microseismic. If the CSV is missing, it falls back to QuakeML depth alone
and warns — never silently fabricating coordinates.

---

## Key takeaways

- Seismic works with **sound in rock**; the property is velocity ($V_p$, $V_s$) and, for
  reflection, **acoustic impedance $Z = \rho V_p$**. **Echoes come from changes in $Z$.**
- **Reflection seismic** = a **convolution model**: reflectivity (normalized $\Delta Z$)
  $\ast$ a band-limited wavelet, plotted in **two-way time**. Vertical resolution
  $\approx \lambda/4 = V_p/(4f)$. It images **structure** (layers, faults) sharply but is
  **nearly blind to fluid/temperature**. Native: **SEG-Y** (+ horizon GeoJSON). Primitive:
  `PropertyModel(section/volume)` of `velocity_p` + horizon/fault `Feature`s.
- **Refraction seismic** = **first-break traveltimes**; the **eikonal equation**
  $|\nabla T| = 1/V_p$ recovers shallow velocity. Native: SEG-Y + CSV picks. Primitive:
  `velocity_p` `section`/`volume`.
- **Microseismic** = *listening* for tiny earthquakes from fracturing/injection;
  **Gutenberg–Richter** $\log_{10}N = a - bM$ describes their sizes. Located in space + time
  it is a **4-D fracture/permeability proxy** — the key EGS dataset. Native: **QuakeML** (+
  CSV). Primitive: a 4-D `pointCloud` `Feature` $(x, y, z, t, \text{mag})$.
- Across all three, the project's honesty shows: **T0 vs T1** forwards (convolutional vs
  full-impedance + multiples; eikonal via `pykonal`) and explicit location/resolution
  **uncertainty** — same file formats, more truthful numbers.

## Where this lives in the code

| Concern | Path |
|---|---|
| Reflection forward (T0 convolutional) | `backend/geosim/synthgen/forward/seismic.py` — `SeismicReflectionForward` |
| Reflection forward (T1 full-impedance + multiples) | `backend/geosim/synthgen/forward/seismic.py` — `SeismicReflectionRigorousForward` |
| Refraction forward (T1 pykonal eikonal) | `backend/geosim/synthgen/forward/seismic.py` — `SeismicRefractionRigorousForward` |
| Ricker wavelet, SEG-Y writer | `backend/geosim/synthgen/forward/seismic.py` — `ricker`, `_write_segy_section` |
| Microseismic forward (Gutenberg–Richter) | `backend/geosim/synthgen/forward/seismic.py` — `MicroseismicForward` |
| Seismic SEG-Y ingestion adapter | `backend/geosim/ingestion/adapters/seismic.py` — `SeismicSegyAdapter` |
| Microseismic QuakeML ingestion adapter | `backend/geosim/ingestion/adapters/microseismic.py` — `MicroseismicQuakeMlAdapter` |

See also the [glossary](../glossary.md) for every term, the
[electromagnetic (TEM/AEM & MT)](electromagnetic.md) page for the fluid-sensitive
counterpart, [boreholes](boreholes.md) for direct velocity/temperature logs, and
[rock physics & favorability](../rock-physics.md) for how velocity and microseismic become
fracture-density and permeability evidence.
