# Electromagnetic — TEM/AEM & MT

!!! abstract "What you'll learn / why it matters"
    This page covers the **electromagnetic (EM)** family of surveys: methods that map the
    Earth's **electrical conductivity** without ever sticking electrodes in the ground.
    Instead they exploit a single fact of physics — a *changing magnetic field induces
    electric currents in a conductor*, and the way those currents grow, decay, and respond
    tells you how conductive the rock is and how deep the conductive stuff sits. We will
    build up [TEM/AEM](#tem-and-aem-time-domain-electromagnetics) (you make the field
    yourself and watch it decay) and [magnetotellurics, or MT](#magnetotellurics-mt) (you
    use the field nature gives you for free). EM is the family that lets you *see deep* —
    MT in particular is **the** method for the deep brine/reservoir conductor that no other
    technique can reach — but it pays for that depth with blur. By the end you'll know why,
    and exactly which bytes show up in an [EDI](#the-edi-file-format) or `.xyz` file and
    what [primitive](../data-model.md) they become.

If you have read the [electrical (ERT & IP)](electrical.md) page, you already know the
target property — **resistivity** $\rho$ (in ohm-metres, $\Omega\cdot m$) and its
reciprocal **conductivity** $\sigma = 1/\rho$ (in siemens per metre, $S/m$). Hot, salty,
clay-rich rock conducts electricity well (low $\rho$); cold, dry, crystalline rock does
not (high $\rho$). The difference between *electrical* methods and *electromagnetic* ones
is **how you get current to flow**:

| | Electrical (ERT/IP) | Electromagnetic (TEM/AEM/MT) |
|---|---|---|
| How current is driven | **Galvanic** — wires + electrodes inject current directly | **Inductive** — a *changing magnetic field* induces current; no electrical contact |
| Coupling to ground | Needs good electrode contact | Works over dry/resistive ground, ice, from a helicopter |
| Best at seeing | **Resistive** structure, shallow, sharp | **Conductive** structure, deeper, smooth |
| Depth reach | Tens to a few hundred metres | Hundreds of metres (TEM) to tens of kilometres (MT) |

!!! note "Define: electromagnetic induction (the one idea behind this whole page)"
    **Electromagnetic induction** is Faraday's law: a *changing* magnetic field $B$ creates
    a loop of electric current (an *eddy current*) in any conductor nearby. The better the
    conductor, the bigger and longer-lived the eddy currents. Those eddy currents have their
    own magnetic field, which we measure. **The decay or frequency-response of that secondary
    field is a fingerprint of the ground's conductivity.** Think of it like a signal-processing
    problem: you inject (or nature injects) an impulse/spectrum, the Earth is a filter whose
    coefficients are the conductivity-vs-depth profile, and you record the response. Recovering
    the filter from the response is an [inverse problem](../inversion.md) — and, like JPEG
    decompression, you can never get back more detail than the physics let in.

---

## A programmer's mental model

Picture conductivity-vs-depth as a 1-D array, sampled along a vertical column under one
station:

```python
# the "truth" the EM survey is trying to recover, under one station
depth_m      = [0,   50,  120,  400,  900, 2000]   # metres below surface
resistivity  = [40,   8,    8,  120,  120,   3 ]   # ohm-metres (low = conductive)
#                ^cover ^clay cap     ^resistive host  ^deep reservoir brine
```

No EM method ever measures this array directly. Each method returns a **transform** of it:

- **TEM/AEM** returns *apparent conductivity vs depth* — but each depth value is a
  smeared, weighted average over everything shallower, not a crisp sample.
- **MT** returns *apparent resistivity & phase vs period* — and *period* is a proxy for
  depth: short period → shallow, long period → deep, again as a blurred average.

This is exactly **lossy compression**. The native file is the compressed bitstream; the
"truth" array above is the original image; the deeper you look the more the compression
artifacts (blur) dominate. Keep that picture; every equation below is just quantifying the
blur.

---

## TEM and AEM (Time-domain Electromagnetics)

!!! note "Define: TEM, AEM, FDEM"
    - **TEM** (Time-domain ElectroMagnetics), also written **TDEM**: you drive a strong
      current through a wire loop on the ground, then **switch it off abruptly**. The
      collapsing magnetic field induces eddy currents in the ground; you record the
      *decay* of the secondary magnetic field over time (microseconds to milliseconds).
    - **AEM** (Airborne ElectroMagnetics): the same idea flown under a helicopter or fixed
      wing — a transmitter loop and receiver coil are towed over the terrain, mapping huge
      areas fast. The platform changes; the physics is identical.
    - **FDEM** (Frequency-domain EM): instead of switching off and watching decay, you
      transmit a continuous sine wave and measure the response *per frequency*. TEM and
      FDEM are Fourier transforms of each other — the same information, time vs frequency.

### The "smoke ring" — how a TEM sounding senses depth

When the transmitter current is cut, the induced eddy currents do not stay put. They form
a ring of current that **diffuses downward and outward** through the ground over time — the
classic **"smoke ring."** (It is called diffusion, not propagation, because in a conductor
EM energy spreads like heat or ink in water, *not* like a wave: there is no sharp wavefront,
just spreading and decaying.)

The depth of that smoke ring at elapsed time $t$ — i.e. how deep the survey is "feeling" at
that moment of the decay — is the **depth of investigation (DOI)**:

$$
\text{DOI} \;\propto\; \sqrt{\dfrac{t\,\rho}{\mu_0}}
$$

Symbol by symbol:

| Symbol | Meaning | Units |
|---|---|---|
| $t$ | elapsed time since transmitter switch-off ("decay gate") | seconds (s) |
| $\rho$ | ground resistivity (here a near-surface reference value) | $\Omega\cdot m$ |
| $\mu_0$ | magnetic permeability of free space, a constant $=4\pi\times10^{-7}\,\text{H/m}$ | H/m |
| DOI | depth the smoke ring has reached | metres (m) |

The intuition is the most important takeaway: **later in the decay = deeper you see.** Each
"decay gate" (a time sample of the decay curve) is one depth slice. Early gates (10 µs)
report the shallow ground; late gates (10 ms) report deep ground. And the smoke ring diffuses
*faster* through resistive ground (large $\rho$) — so over a resistor you reach a given depth
sooner. This is why a single TEM station, recording the whole decay curve, yields a whole
**conductivity-depth sounding**.

!!! example "How the synthetic generator implements the smoke ring"
    The T0 ("degrade-the-truth") forward model in
    `backend/geosim/synthgen/forward/em_mt.py` (`TDEMForward`) does exactly this. For each
    of 20 logarithmically-spaced decay gates from $10^{-5}$ to $10^{-2}$ s it computes the
    diffusion depth $d=\sqrt{2t\rho/\mu_0}$, then reports the **depth-weighted average
    conductivity** over $[0, d]$ — a triangular weight $w = \mathrm{clip}(1 - \text{depth}/d,\,0,\,1)$
    that down-weights the deeper, less-resolved part. That weighted average *is* the blur:
    the clay cap and the deep reservoir get smeared into one number per gate. Then 3–8 %
    Gaussian noise plus a late-time floor are added, because real late-time gates drown in
    noise (the secondary field has decayed to nothing).

### What TEM/AEM can and can't see

- **Can:** map a conductive layer (clay cap, saline aquifer, conductive ore) over a wide
  area, quickly, from the air, without ground contact. Excellent shallow-to-mid resolution.
- **Can't:** see very deep (the signal decays into noise after a few hundred metres to ~1 km,
  depending on conductivity), and it inherently *blurs* in depth (the weighted-average DOI).
  It is also "conductor-hungry": a thin conductor dominates the response and can hide what's
  beneath it — the **shielding** problem.

### The `.xyz` sounding file

TEM/AEM data arrives as plain-text **soundings** — one block of (time, depth, apparent
conductivity) rows per station. The synthetic generator writes `tem_soundings.xyz`:

```text title="tem_soundings.xyz (annotated)"
STATION X Y TIME_S DEPTH_M APP_COND_S_per_m   # (1) header: column roles
0 1250.00 1750.00 1.000000e-05 22.36 2.512000e-02   # (2) station 0, earliest gate (10 µs) → shallow
0 1250.00 1750.00 1.467799e-05 27.08 2.498000e-02   # (3) next gate, slightly deeper
...                                                  #     ...18 more gates for station 0...
0 1250.00 1750.00 1.000000e-02 707.10 8.090000e-04  # (4) last gate (10 ms) → deepest, lowest cond
1 1750.00 1750.00 1.000000e-05 22.36 1.998000e-02   # (5) station 1 begins; X/Y change
```

1. The one-line header names the columns. The ingestion adapter is tolerant: it matches
   header tokens against aliases (`easting`/`east`→`x`, `sigma_a`/`cond`→`app_cond`, etc.),
   so files from other vendors still parse.
2. `STATION` is an integer id; `X`/`Y` are the **plan position** of the station in metres
   (the [Engineering Frame](../spatial-framework.md), X-East/Y-North).
3. `TIME_S` is the decay gate in seconds; `DEPTH_M` is the smoke-ring DOI for that gate;
   `APP_COND_S_per_m` is the apparent conductivity ($S/m$) — the weighted average above.
4. Notice conductivity *drops* at the deep gate: the deep host here is resistive, so the
   late-time average is low-conductivity. That late gate is also the noisiest in real data.
5. A blank-line break and a change in `X`/`Y` start a new station's column.

### What it becomes after [ingestion](../ingestion.md)

The adapter `backend/geosim/ingestion/adapters/em.py` (`EmXyzAdapter`, `method="em"`,
`submethod="tdem"`) turns each row into **one observation record** at
$(x, y, \text{depth\_below\_surface})$, all collected into a single
[`Observation`](../data-model.md) of `geometryKind = "soundings"` carrying the canonical
`conductivity` value ($S/m$). The vertical column rides as native samples; the data model's
frozen geometry vocabulary classifies it as `soundings` (not `points`). Later, an
[inversion](../inversion.md) step *stitches* the per-station 1-D columns into a 3-D
`PropertyModel` (a conductivity `volume`) by interpolating laterally between stations and
**masking below each sounding's DOI** so the platform never silently extrapolates into
depths the data couldn't reach (this is the standard 1-D→3-D pattern; see
[uncertainty](../uncertainty.md)).

---

## Magnetotellurics (MT)

!!! note "Define: magnetotellurics"
    **Magnetotellurics (MT)** is a *passive* EM method: it uses the **natural,
    naturally-occurring electromagnetic fields** of the Earth as its source. There is no
    transmitter at all. You just plant electrodes and magnetometers and record the
    naturally varying electric field $E$ and magnetic field $H$ for hours to days.

Where do the natural fields come from? Two sources, conveniently spanning a huge frequency
range:

- **Long periods (slow variations, ~1 s to thousands of s):** the solar wind buffeting the
  Earth's magnetosphere — geomagnetic storms and sub-storms.
- **Short periods (fast variations, ~0.001 s to ~1 s):** worldwide lightning, whose energy
  rings around the planet in the *Schumann resonances*.

Together they illuminate the Earth with a broadband EM "noise" source for free. To a
programmer: nature is running a continuous broadband sweep, and MT is a *transfer-function
estimation* — measure input ($H$) and output ($E$) and divide.

### The impedance tensor — MT's core measurement

For each frequency, MT forms the **impedance** $Z$: the ratio of the horizontal electric
field to the horizontal magnetic field. In full generality this is a 2×2 complex **tensor**
relating the two horizontal $E$ components to the two horizontal $H$ components:

$$
\begin{bmatrix} E_x \\ E_y \end{bmatrix} =
\begin{bmatrix} Z_{xx} & Z_{xy} \\ Z_{yx} & Z_{yy} \end{bmatrix}
\begin{bmatrix} H_x \\ H_y \end{bmatrix}
$$

From one off-diagonal component (commonly $Z_{xy}$) you derive the two numbers everyone
actually plots:

$$
\rho_a \;=\; \frac{|Z|^2}{\omega\,\mu_0}, \qquad \phi \;=\; \arg Z
$$

| Symbol | Meaning |
|---|---|
| $Z$ | complex impedance $E/H$ at angular frequency $\omega$ ($\Omega$, but conventionally scaled) |
| $\rho_a$ | **apparent resistivity** ($\Omega\cdot m$) — the resistivity of a uniform earth that would give this $Z$ |
| $\phi$ | **phase** (degrees) — the lag between $E$ and $H$; a uniform earth gives exactly $45°$ |
| $\omega$ | angular frequency $=2\pi/T$ (rad/s), with $T$ the **period** |
| $\mu_0$ | permeability of free space, $4\pi\times10^{-7}$ H/m |

The phase is a beautifully simple diagnostic: $\phi > 45°$ means resistivity is *decreasing*
with depth (you're entering a conductor); $\phi < 45°$ means it's *increasing* (entering a
resistor); $\phi = 45°$ means uniform. It is essentially the *derivative* of the apparent-
resistivity curve.

### Skin depth — why period maps to depth

The whole power of MT is one equation. EM fields diffusing into a conductor are attenuated;
the depth at which the field has fallen to $1/e$ (about 37 %) of its surface value is the
**skin depth** $\delta$. For MT's diffusive (quasi-static) regime:

$$
\boxed{\;\delta \;=\; 503\,\sqrt{\rho\,T}\;}\quad\text{(metres)}
$$

| Symbol | Meaning | Units |
|---|---|---|
| $\delta$ | skin depth — the depth MT effectively "sees" at this period | metres (m) |
| $503$ | a constant that bundles $\sqrt{1/(\pi\mu_0)}$ for $\mu=\mu_0$ | $\sqrt{\Omega^{-1}\cdot m^{-1}\cdot s^{-1}}$ |
| $\rho$ | ground resistivity | $\Omega\cdot m$ |
| $T$ | the **period** of the recorded EM variation $=1/\text{frequency}$ | seconds (s) |

This is the magic: **period $T$ is a dial for depth.** Long periods (slow variations)
diffuse deeper before attenuating, so they see deep; short periods see shallow. A worked
table for a 100 $\Omega\cdot m$ host:

| Period $T$ | $\delta = 503\sqrt{\rho T}$ | What it senses |
|---|---|---|
| 0.001 s | $\approx 159$ m | shallow cover, clay cap |
| 0.1 s | $\approx 1{,}590$ m | reservoir level |
| 10 s | $\approx 15{,}900$ m (~16 km) | mid-crust |
| 1000 s | $\approx 159{,}000$ m (~159 km) | upper mantle |

A standard MT survey records ~30 periods spanning $0.001$ s to $1000$ s — so a single
station produces a sounding from the near-surface to the deep crust. **MT is the only common
exploration method that reaches kilometres deep.** That is why it is *the* tool for the deep
geothermal reservoir conductor (hot saline brine = very low resistivity).

!!! warning "MT is deep but smooth — the resolution tax"
    Because each period averages resistivity over the *entire* depth interval $[0, \delta]$
    (a diffusion average, not a sharp sample), MT's depth resolution degrades with depth: by
    a few km, a thin layer is just a gentle bump in the curve. In signal terms, MT is a
    strong **low-pass filter in depth**. It tells you *there is a deep conductor*, not its
    exact top, thickness, or sharpness. Always carry this as [uncertainty](../uncertainty.md).

### The ERT-vs-MT depth split (a fusion test, on purpose)

This is the headline interplay between this page and the
[electrical (ERT)](electrical.md) page, and it is *physically required*, not a bug:

| Target | ERT sees it… | MT sees it… |
|---|---|---|
| **Shallow clay cap** (conductor, ~100–500 m) | **Sharply** — ERT's strength is shallow, high-resolution structure | **Smoothly** — it's at MT's shortest periods, blurred |
| **Deep reservoir conductor** (brine, km-scale) | **Not at all** — it's below ERT's depth of investigation | **This is MT's domain** — only MT reaches it |

So a fused model will show ERT and MT *agreeing* on the clay cap (one sharp, one fuzzy) and
*disagreeing at depth* — ERT goes silent (no data) while MT lights up the deep conductor.
That disagreement is the correct answer, and the synthetic generator builds scenarios to
reproduce it (doc 05 §4.2). It is a built-in test that [fusion](../fusion.md) is honouring
each method's true depth window rather than blindly averaging.

### The EDI file format

MT data is exchanged in **EDI** (Electrical Data Interchange, an SEG/EMAP standard). An EDI
file is a flat text file of `>BLOCK` directives. The synthetic generator's `write_edi`
(in `em_mt.py`) emits one EDI per station with exactly the blocks the ingestion adapter
parses:

```text title="ST000.edi (annotated)"
>HEAD                                    # (1) header block
  DATAID=ST000                           #     station/site name
  ACQBY=geosim.synthgen                  #     who acquired it
  FILEBY=geosim.synthgen
  EMPTY=1.0E+32                          #     the "no data" sentinel value

>=DEFINEMEAS                             # (2) measurement-setup block
  REFLOC=1250.00,1750.00                 #     site plan position (X,Y) in metres

>=MTSECT                                 # (3) the MT data section
  NFREQ=30                               #     number of frequencies in this site

>FREQ ROT=ZROT // 30                     # (4) the FREQUENCY axis (Hz), high → low
  1.000000E+03  6.812921E+02  4.641589E+02  3.162278E+02  2.154435E+02
  ... (30 values, 5 per line, decreasing) ...
  1.000000E-03

>RHOXY ROT=ZROT // 30                    # (5) APPARENT RESISTIVITY (ohm·m), xy component
  4.210000E+01  4.050000E+01  3.880000E+01  3.510000E+01  2.900000E+01
  ... (one value per frequency, same order as >FREQ) ...
  3.120000E+00

>PHSXY ROT=ZROT // 30                    # (6) PHASE (degrees), xy component
  4.480000E+01  4.510000E+01  4.620000E+01  5.010000E+01  5.530000E+01
  ... (one value per frequency) ...
  3.890000E+01

>END                                     # (7) end of file
```

1. **`>HEAD`** — site identity and the `EMPTY` sentinel that flags missing samples.
2. **`>=DEFINEMEAS` / `REFLOC`** — where the site is. The adapter reads `REFLOC=x,y` to
   place the station in the [Engineering Frame](../spatial-framework.md).
3. **`>=MTSECT` / `NFREQ`** — declares the data section and how many frequencies follow.
4. **`>FREQ`** — the frequency axis in Hz, written high-to-low (EDI convention; note
   $\text{frequency} = 1/T$, so high frequency = short period = shallow).
5. **`>RHOXY`** — apparent resistivity $\rho_a$ in $\Omega\cdot m$, one value per frequency.
   `XY` denotes the off-diagonal impedance component $Z_{xy}$ used.
6. **`>PHSXY`** — phase $\phi$ in **degrees**. (The data model stores phase in milliradians;
   the [ingestion normalizer](../ingestion.md) does the conversion — the adapter keeps it
   native.)
7. Real EDI files have far more blocks (`>ZXYR`/`>ZXYI` raw impedance, `>TXR`/`>TXI`
   tipper, error blocks `>RHOXY.VAR`). The generator emits the minimal `>FREQ`/`>RHOXY`/
   `>PHSXY` triad, and the adapter is built to ignore blocks it doesn't recognise.

### What it becomes after [ingestion](../ingestion.md)

`backend/geosim/ingestion/adapters/mt.py` (`MtEdiAdapter`, `method="mt"`) parses each EDI
into **one [`Observation`](../data-model.md) of `geometryKind = "tensor"`** at the site
$(x, y, 0)$, splitting the two curves into canonical keys: `resistivity` (apparent $\rho$,
$\Omega\cdot m$) and `phase_mrad` (the phase, ingested as degrees and canonicalized to
milliradians by the normalizer). The frequency axis rides in the observation's metadata.
Each period is one record, so the value columns stay aligned and the station's bounding box
lands on its location. As with TEM, a later [inversion](../inversion.md) stitches the
per-site soundings into a resistivity `volume`, DOI-masked.

If an EDI is missing `>RHOXY` the parse **fails loudly** (you can't have an MT sounding with
no apparent resistivity); a missing `>PHSXY` degrades gracefully with a structured warning;
a missing `REFLOC` places the site at the origin and warns. This is the project's
"never silently wrong" stance (doc 03 §6).

---

## The rigorous (T1) forward — getting the physics exactly right

The smoke-ring weighted-average (TEM) and skin-depth box-average (the T0 `MTForward`) are
*approximations* good enough to exercise the pipeline. For MT — one of the three methods the
project builds rigorously first (doc 05 §6) — there is a **physically exact** forward:
`MTRigorousForward` / `layered_mt_impedance` in `em_mt.py`.

It treats the truth resistivity column under each station as a stack of layers and computes
the true surface plane-wave impedance by the **Wait/Cagniard recursion** — the analytic
plane-wave ($k_x\to 0$) limit of [empymod](../inversion.md)'s layered-earth solver. The
machinery, briefly:

- per-layer wavenumber $k_i = \sqrt{i\,\omega\,\mu_0/\rho_i}$ (so the per-layer skin depth
  is exactly $1/\mathrm{Re}(k_i) = 503\sqrt{\rho_i T}$ — the same equation, now exact);
- intrinsic impedance $Z_i = i\,\omega\,\mu_0/k_i$;
- an upward recursion from the basement half-space,
  $Z = Z_n\,(1-\Gamma E)/(1+\Gamma E)$ with reflection coefficient
  $\Gamma = (Z_n - Z)/(Z_n + Z)$ and $E = e^{-2 k_n h_n}$ ($h_n$ = layer thickness).

The complex $Z$ then gives $\rho_a = |Z|^2/(\omega\mu_0)$ and $\phi = \arg Z$ *correctly*,
so the period→depth mapping is honoured exactly: short periods resolve the shallow clay cap
(a conductor → low $\rho_a$, phase $>45°$), the mid band the resistive host, and long periods
diffuse down to the deep reservoir conductor — the same ERT-vs-MT depth split, now from first
principles. Crucially, it emits the **same EDI files**, so ingestion is unchanged: only the
numbers inside `>RHOXY`/`>PHSXY` get more truthful. The module even cross-checks itself
against empymod's own layered kernel for a uniform half-space ($\rho_a = \rho$ exactly).

!!! tip "Why this matters for the course"
    The T0/T1 split is the project's honesty mechanism in miniature: the *file format and
    data model never change*, only the fidelity of the physics that produced the numbers.
    That's the same discipline you'd want in any data pipeline — swap the model, keep the
    contract. See [forward modeling & inversion](../inversion.md) for the full story, and
    [the synthetic data generator](../synthetic-data.md) for how T0/T1 are selected.

---

## Key takeaways

- **One idea:** a changing magnetic field induces eddy currents whose decay (TEM) or
  frequency response (MT) is a fingerprint of the ground's **conductivity** — no electrodes
  needed (inductive, not galvanic).
- **TEM/AEM** = you switch off a transmitter and watch the **smoke ring** diffuse downward;
  later decay gates see deeper, $\text{DOI}\propto\sqrt{t\rho/\mu_0}$. Output: a
  conductivity-depth sounding per station, blurred by a depth-weighted average. Native file:
  `.xyz`. Primitive: `Observation(soundings)` of `conductivity`.
- **MT** uses **natural** EM fields and the **skin depth** $\delta = 503\sqrt{\rho T}$ turns
  **period into depth** — long periods see kilometres down. MT is **the deep-conductor
  method** (deep brine/reservoir) but smooth/low-resolution. Native file: **EDI**
  (`>FREQ`/`>RHOXY`/`>PHSXY`). Primitive: `Observation(tensor)` of `resistivity` +
  `phase_mrad`.
- **ERT-vs-MT depth split:** ERT images the shallow clay cap sharply; only MT reaches the
  deep conductor. Their disagreement at depth is *physically correct* and a deliberate
  [fusion](../fusion.md) test.
- Both stitch 1-D soundings into a 3-D `volume` later, always **DOI-masked** so the platform
  never extrapolates past what the data could resolve.
- The project ships a **rigorous (T1) MT forward** (exact Wait/Cagniard layered impedance,
  anchored to `empymod`) that emits the *same* EDI files — fidelity changes, contract doesn't.

## Where this lives in the code

| Concern | Path |
|---|---|
| TEM/AEM forward (smoke-ring soundings) | `backend/geosim/synthgen/forward/em_mt.py` — `TDEMForward` |
| MT forward (T0 skin-depth average) | `backend/geosim/synthgen/forward/em_mt.py` — `MTForward`, `write_edi` |
| MT forward (T1 exact layered impedance) | `backend/geosim/synthgen/forward/em_mt.py` — `MTRigorousForward`, `layered_mt_impedance` |
| EM `.xyz` ingestion adapter | `backend/geosim/ingestion/adapters/em.py` — `EmXyzAdapter` |
| MT EDI ingestion adapter | `backend/geosim/ingestion/adapters/mt.py` — `MtEdiAdapter` |
| 1-D→3-D sounding stitching | [ingestion](../ingestion.md) / [inversion](../inversion.md) (doc 03 §4, doc 10) |

See also the [glossary](../glossary.md) for every term, the
[electrical (ERT & IP)](electrical.md) page for the galvanic cousin and the depth-split
counterpart, and [rock physics & favorability](../rock-physics.md) for how resistivity
becomes temperature, fluid, and clay-alteration evidence.
