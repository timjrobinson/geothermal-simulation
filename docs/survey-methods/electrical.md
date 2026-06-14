# Electrical — ERT & IP

> **What you'll learn / why it matters.** Electrical methods inject current into the ground
> and watch how it flows. They are the platform's **shallow, sharp** sensors — and for
> geothermal they are gold, because the thing they see best, the conductive **clay cap**, is
> exactly the seal that traps a geothermal reservoir. This page builds DC resistivity (ERT)
> from Ohm's law, explains the **pseudosection** and why its values are "apparent" rather
> than true, defines the electrode **arrays** (dipole-dipole, Wenner), then covers **induced
> polarization (IP)** and its three different "keys" (time-domain ms, frequency-domain mV/V,
> and phase mrad). It ends with the honest tradeoff — ERT is sharp but **loses depth fast**
> ($\mathrm{DOI} \approx 0.15$–$0.2 \times$ array length) — and the real `.stg` file the
> platform ingests. Read [how to read these pages](index.md) first for *DOI*, *resolution
> kernel*, and *forward vs inverse*.

## The physics: push current, measure voltage, infer resistance

Rock conducts electricity, but badly and unevenly. **Electrical resistivity** $\rho$
(rho, units **ohm-metres, $\Omega\cdot m$**) measures *how strongly a material opposes
electric current* — the inverse of **conductivity** $\sigma$ ($\sigma = 1/\rho$). The whole
method rests on a fact that makes electricity a fantastic geothermal probe: **resistivity
drops dramatically in the presence of hot, salty water and clay.**

- Dry, solid, cold crystalline rock: **very resistive** (hundreds to thousands of
  $\Omega\cdot m$).
- Porous rock saturated with **saline fluid**: **conductive** (ions carry the current).
- **Clay** (especially the clay produced by hydrothermal alteration): **very conductive**
  (clay surfaces hold mobile ions).

So a low-resistivity blob underground means *fluid, or clay, or both* — and in a geothermal
system the cap above a reservoir is a clay-rich, conductive layer. Mapping resistivity maps
the cap.

### Archie's law — why resistivity tracks fluid

For clay-free, fluid-filled rock, the empirical **Archie's law** links bulk resistivity to
porosity and fluid:

$$
\rho_{\text{rock}} \;=\; a\,\phi^{-m}\,S_w^{-n}\,\rho_w
$$

- $\rho_{\text{rock}}$ — the bulk rock resistivity you measure ($\Omega\cdot m$).
- $\rho_w$ — the **resistivity of the pore fluid** (low for hot, salty brine).
- $\phi$ — **porosity** (fraction of pore space, 0–1).
- $S_w$ — **water saturation** (fraction of pores filled with water, 0–1).
- $a$ — a tortuosity constant (≈ 0.6–1).
- $m$ — the **cementation exponent** (≈ 1.3–2.5; how connected the pores are).
- $n$ — the **saturation exponent** (≈ 2).

The takeaway: **more porosity, more water, and saltier/hotter water all push resistivity
down.** This is the bridge from a geophysical number ($\rho$) to the geothermal targets
(fluid, permeability) — and it's exactly the kind of relationship the platform's
[rock physics](../rock-physics.md) layer exploits. (Archie breaks down when clay is
present — clay conducts through its own surface chemistry, not just the pore fluid — which is
precisely why you also run **IP**, below, to tell clay and brine apart.)

### How ERT measures it: Ohm's law with four electrodes

You can't stick a resistance meter into bedrock. Instead, **electrical resistivity
tomography (ERT)** — also just "DC resistivity" — uses **four electrodes** in the ground:

1. Push a known **current** $I$ into the ground through two **current electrodes**, usually
   labeled **A** and **B**.
2. Measure the resulting **voltage difference** $\Delta V$ between two **potential
   electrodes**, **M** and **N**.
3. Apply Ohm's law, corrected for the 3-D geometry, to get an **apparent resistivity**:

$$
\rho_a \;=\; K\,\frac{\Delta V}{I}
$$

where $K$ is a **geometric factor** that depends only on where the four electrodes sit. This
quadruple (A, B, M, N) is called a **quadrupole**. Move the electrodes around, repeat
hundreds of times at different spacings, and you build up a picture.

!!! abstract "The CT-scan / inverse-problem analogy"
    Each reading is a single line-integral-like probe through the ground — like one X-ray
    angle in a CT scan. No single reading is an image; the image emerges only after you
    combine many readings and **invert** them. ERT is, quite literally, electrical
    tomography.

## Why "apparent" resistivity, and what a pseudosection is

This is the concept students trip on, so slow down here.

The $\rho_a$ you compute from each reading is the resistivity the ground *would* have **if it
were uniform**. The real ground is not uniform — so $\rho_a$ is a **blend** of the true
resistivities of all the rock the current passed through, weighted by a **sensitivity
kernel** (a [resolution kernel](index.md#resolution-kernel)). It is *not* the true
resistivity at any one point. Hence **apparent** resistivity.

When you plot every reading's $\rho_a$ at a made-up location — horizontally under the
electrode-array midpoint, vertically at a **pseudodepth** proportional to the electrode
separation — you get a **pseudosection**: a 2-D image that *looks* like a depth slice but is
really a plot of measurements against a proxy for depth.

!!! warning "A pseudosection is raw data dressed up as a picture"
    The pseudosection is **not** a model of the earth. Its vertical axis is "pseudodepth," a
    geometric stand-in, not true depth; its values are apparent, not true, resistivity. To
    get *true* resistivity at *true* depth you must run the **inverse problem** (an ERT
    inversion). The pseudosection is the input to that inversion, not its output. In the
    data model the pseudosection is an `Observation` (`geometryKind: "profile2d"`); the
    inverted result is a separate `PropertyModel`.

You can watch a pseudosection being *forward-modeled* in
`backend/geosim/synthgen/forward/electrical.py` (`build_pseudosection`): for each
dipole-dipole quadrupole it places a depth-decaying Gaussian "banana" sensitivity kernel
under the array midpoint at pseudodepth $\approx n\cdot a/2$, takes the (log-)weighted
average of the true resistivity column, and zeroes out sensitivity below the DOI. That code
*is* the apparent-resistivity-and-pseudodepth machinery, in NumPy.

## Electrode arrays (define: dipole-dipole, Wenner)

An **array** is the geometric *pattern* in which you arrange and step the four electrodes.
The pattern controls the tradeoff between **depth reach**, **lateral resolution**, and
**signal strength**. The two you'll meet most:

| Array | Layout (along the line) | Strengths | Weaknesses |
|---|---|---|---|
| **Wenner** | four equally spaced electrodes `A — M — N — B` at spacing $a$; step the whole set | strong signal (high $\Delta V$), robust in noise, good vertical resolution | poor lateral resolution; slower; shallower reach for a given layout |
| **Dipole-dipole** | a close current pair `A B`, then a close potential pair `M N` placed $n$ dipole-lengths away | excellent **lateral** resolution, sees vertical structure (faults, dikes) well, efficient with multichannel gear | weaker signal at large $n$ (noisier deep), needs care |

The synthetic forward and the AGI `.stg` writer use **dipole-dipole** (see the
`"Type: dipole-dipole"` header and the `array` metadata in
`electrical.py`), because its lateral sharpness suits mapping a clay cap and faults. The
**pseudodepth** for dipole-dipole grows with the separation $n$: deeper readings come from
moving the potential pair farther from the current pair, which is also why they get weaker
and noisier with depth.

## The honest tradeoff: ERT is shallow & sharp, and loses depth FAST

This is ERT's defining limitation. **Depth of investigation scales with the size of your
electrode layout:**

$$
\mathrm{DOI} \;\approx\; 0.15\text{–}0.2 \times L_{\text{array}}
$$

where $L_{\text{array}}$ is the total length of the electrode spread. To see 200 m deep you
need roughly a **1–1.3 km** electrode line. The synthetic forward hard-codes
$\mathrm{DOI} = 0.18 \times \text{array length}$ and applies a sigmoid cutoff that **kills
sensitivity below the DOI** — beyond it, the readings carry no real information and an
inversion there is just inventing.

| ERT is good at | ERT is blind to |
|---|---|
| **Shallow, sharp** resolution (the clay cap, near-surface faults) | **Depth** — DOI is only ~15–20 % of array length; deep targets are invisible |
| Mapping low-resistivity **fluid/clay** zones | Telling **clay from brine** apart (both are conductive) — that's IP's job |
| High **lateral** resolution (dipole-dipole) | Signal drops sharply with depth (noisy at large $n$) |
| Cheap, ground-based, fast to deploy | The deep reservoir conductor (below DOI) — that's where [MT](electromagnetic.md) takes over |

!!! note "Why ERT and MT are partners"
    ERT is sharp but shallow; **magnetotellurics (MT)** is smooth but reaches kilometres
    deep. In a geothermal program ERT maps the *top* of the conductive clay cap in detail
    while MT images the *whole* cap down to the reservoir. See
    [electromagnetic methods](electromagnetic.md).

---

## IP — induced polarization: the same gear, an extra question

**Induced polarization (IP)** is acquired with the *same four-electrode setup as ERT* — often
in the same survey, the same `.stg` file family — but it measures something different and
complementary: how much the ground **temporarily stores charge**, like a leaky capacitor.

When you inject current and then **shut it off**, certain materials don't let the voltage
collapse instantly — they hold a residual, decaying voltage. This **chargeability** arises
from two main sources:

- **Disseminated metallic/sulphide grains** (electrode polarization at grain surfaces).
- **Clay minerals** (membrane polarization in the pore network).

That makes IP the perfect partner to ERT for the geothermal **clay-vs-brine ambiguity**:
both a clay cap and a briny reservoir look conductive (low $\rho$) to ERT, but **clay is
strongly chargeable and brine is not.** High conductivity *plus* high chargeability points to
**clay**; high conductivity *without* chargeability points to **fluid**. IP resolves what ERT
cannot.

### The three keys of IP (define each)

IP is reported in three different ways depending on whether you work in the **time domain**
or the **frequency domain**. The platform keeps these as **separate property keys** so units
and colormaps never get confused (see `_LABEL_MAP` in
`backend/geosim/ingestion/adapters/_stg.py`):

| Key (property) | Domain | Unit | What it is |
|---|---|---|---|
| **`chargeability_time_ms`** | time-domain | **ms** | the residual voltage **decay time** — integrate the decaying voltage after current shutoff over a time window |
| **`chargeability_mv_v`** | time-domain | **mV/V** | apparent chargeability — residual voltage as a fraction of the primary voltage (millivolts of decay per volt of signal) |
| **`phase_mrad`** | frequency-domain | **mrad** | the **phase lag** between an AC current and the measured voltage (chargeable ground delays the voltage; the lag, in milliradians, measures it) |

!!! tip "Time domain vs frequency domain — the signals analogy"
    This is the same duality a CS person knows from DSP. **Time-domain IP** is like an
    *impulse response*: hit the ground with current, cut it, and watch the voltage **decay**
    (measured in ms or mV/V). **Frequency-domain IP** is like a *transfer function*: drive
    the ground with AC at various frequencies and measure the **phase lag** (mrad) of the
    response. They describe the same physical property — chargeability — from the two
    classic viewpoints.

The synthetic forward (`IPForward` in `electrical.py`) generates an apparent-chargeability
pseudosection in **mV/V**, with noise that **worsens with depth** (a realistic detail — deep
IP readings are notoriously noisy). It shares the exact same sensitivity-kernel machinery as
ERT, just averaging chargeability **linearly** instead of resistivity logarithmically.

## DOI & resolution (IP)

IP shares ERT's geometry, so it shares ERT's **DOI ($\approx 0.15$–$0.2 \times$ array
length)** and its shallow-and-sharp character. IP data is generally **noisier** than the
resistivity acquired alongside it, and that noise gets worse at depth — so IP's *effective*
depth reach is usually a bit shallower than the co-located ERT.

## Native file format (annotated): the AGI `.stg`

Both ERT and IP are ingested from an **AGI SuperSting `.stg`** text file — the format written
by AGI's widely used resistivity/IP instruments. The synthetic forward writes a faithful
`.stg` (`_write_stg` in `electrical.py`) and the same file round-trips back through
ingestion. The two methods produce the *same column layout*; only the declared **value**
quantity differs.

### ERT pseudosection — `ert_lineAA.stg`

```text
AGI SuperSting Synthetic Pseudosection (geosim.synthgen)   # (1) free-text banner
Type: dipole-dipole                                        # (2) the electrode ARRAY used
records: 132  value: apparent_resistivity_ohm_m            # (3) count + the measured QUANTITY
Idx,A_x,A_y,B_x,B_y,M_x,M_y,N_x,N_y,pseudodepth,value      # (4) the column header
1,0.00,0.00,25.00,0.00,50.00,0.00,75.00,0.00,18.75,210.4   # (5) one quadrupole reading
2,0.00,0.00,25.00,0.00,75.00,0.00,100.00,0.00,31.25,185.7
3,25.00,0.00,50.00,0.00,75.00,0.00,100.00,0.00,18.75,166.2
```

1. A human-readable banner (instrument/origin). Adapters ignore it for data but may sniff it.
2. **`Type:`** — the electrode **array** (here dipole-dipole). Recorded into the
   observation's `meta.array`.
3. **`value:`** — the crucial line: it declares *what quantity the `value` column holds*.
   `apparent_resistivity_ohm_m` → the adapter maps this to the `resistivity` property with
   source unit $\Omega\cdot m$. (This single label is how one parser serves both ERT and IP —
   see `value_property_and_unit` in `_stg.py`.)
4. The column header. Each record carries the **full quadrupole geometry**, not just a point:
   - `Idx` — record index.
   - `A_x,A_y / B_x,B_y` — the two **current** electrodes (metres).
   - `M_x,M_y / N_x,N_y` — the two **potential** electrodes (metres).
   - `pseudodepth` — the geometric proxy depth for this reading (metres below surface).
   - `value` — **the measurement**: apparent resistivity in $\Omega\cdot m$.
5. One row = one four-electrode reading. The adapter computes the **midpoint** of the four
   electrodes as the plotting X/Y, places it at `pseudodepth` below surface, and keeps the
   raw electrode positions in `meta.electrodes`.

### IP pseudosection — `ip_lineAA.stg`

Byte-for-byte the same layout; only the declared quantity changes:

```text
records: 132  value: apparent_chargeability_mv_v           # (1) value label = an IP quantity
Idx,A_x,A_y,B_x,B_y,M_x,M_y,N_x,N_y,pseudodepth,value
1,0.00,0.00,25.00,0.00,50.00,0.00,75.00,0.00,18.75,12.3    # (2) value is now mV/V, not ohm-m
```

1. The `value:` label now reads `apparent_chargeability_mv_v`, so the IP adapter maps it to
   the **`chargeability_mv_v`** property (unit mV/V). A `chargeability_time` label would map
   to `chargeability_time_ms` (ms), and `phase` to `phase_mrad` (mrad) — the three keys.
2. Identical geometry columns; the measurement is now chargeability.

!!! note "How the platform tells ERT and IP `.stg` files apart"
    Both adapters `sniff()` the same `.stg` columns, but they disambiguate on the `value:`
    label: a file declaring *resistivity* scores high for the ERT adapter and low for IP, and
    vice-versa for *chargeability*/*phase*. The highest score wins — see
    `backend/geosim/ingestion/adapters/ert.py` and `ip.py`.

## The normalized primitives ERT & IP become

Ingestion turns each `.stg` into **one `Observation`** with
`geometryKind: "profile2d"` — the immutable raw pseudosection. Coordinates are the quadrupole
midpoints, placed at `pseudodepth` below surface (`z_convention: "depth_below_surface"`); the
raw electrode positions and array type live in `meta`.

- **ERT `.stg`** → `Observation(profile2d, resistivity)` (apparent ρ, unit $\Omega\cdot m$).
- **IP `.stg`** → `Observation(profile2d, chargeability_mv_v)` (or `chargeability_time_ms` /
  `phase_mrad` depending on the label).

To get *true* resistivity/chargeability at *true* depth you run the **inverse problem**. The
result is a `PropertyModel` whose support is the native **`section`** (`support.kind:
"section"`) — a 2-D "curtain" hanging in 3-D along the survey line's polyline, *not* forced
into the voxel grid until you resample it to the fused grid (a non-destructive choice — see
[fusion](../fusion.md)). For a 3-D ERT/IP survey it becomes a `volume` instead.

```text
ert_lineAA.stg ──▶ Observation(profile2d, resistivity)              # raw apparent ρ, immutable
ip_lineAA.stg  ──▶ Observation(profile2d, chargeability_mv_v)       # raw apparent chargeability
   (later, via inversion) ──▶ PropertyModel(section | volume, resistivity / chargeability)
```

## Where ERT & IP are strong for geothermal

ERT and IP are the **shallow workhorses** for the part of a geothermal system you can reach
from the surface with electrodes:

- **ERT** maps the **conductive clay cap** sharply and cheaply — the seal that defines a
  reservoir's top — and the near-surface faults that may feed it.
- **IP** disambiguates the conductor: **clay (chargeable) vs brine (not)**, sharpening the
  alteration picture.
- Together they hand off to **MT** for the deep cap and reservoir (below ERT's DOI), and they
  are calibrated against **[well logs](boreholes.md)** that measure resistivity directly in
  the borehole. The cross-method story — conductive + chargeable + (later) hot + permeable —
  is exactly what [fusion](../fusion.md) and [rock physics](../rock-physics.md) assemble.

## Key takeaways

- ERT injects current through electrodes **A, B** and measures voltage at **M, N**, giving
  **apparent resistivity** $\rho_a = K\,\Delta V/I$ — a sensitivity-weighted blend, *not*
  true resistivity at a point.
- **Archie's law** ($\rho \propto \phi^{-m} S_w^{-n}\rho_w$) is why low resistivity means
  **fluid/porosity** — the bridge to geothermal targets; **clay** also conducts, breaking
  Archie and motivating IP.
- A **pseudosection** plots apparent values against **pseudodepth** — it is raw data
  (`Observation`, `profile2d`), not an earth model; the model comes from **inversion**.
- **Arrays** trade depth/resolution/signal: **Wenner** = strong signal, less lateral detail;
  **dipole-dipole** = sharp lateral detail (the platform's choice).
- ERT is **shallow & sharp** but **loses depth fast**:
  $\mathrm{DOI} \approx 0.15$–$0.2 \times$ array length — to see 200 m down you need a ~1 km
  line.
- **IP** uses the same gear to measure **chargeability** in three keys — time-domain **ms**,
  time-domain **mV/V**, frequency-domain phase **mrad** — and resolves the geothermal
  **clay-vs-brine** ambiguity that ERT alone cannot.
- Both ingest from an **AGI `.stg`** into `Observation(profile2d)`; an inversion yields a
  **`section`** (or `volume`) `PropertyModel`.

## Where this lives in the code

- Forward models (earth → data): `backend/geosim/synthgen/forward/electrical.py` —
  `build_pseudosection` (the sensitivity-kernel + pseudodepth + DOI logic), `ERTForward`
  (apparent resistivity), `IPForward` (apparent chargeability), and `_write_stg` (the AGI
  `.stg` writer).
- Ingestion adapters (native files → primitives):
  `backend/geosim/ingestion/adapters/ert.py` (ERT `.stg` → `profile2d` resistivity),
  `ip.py` (IP `.stg` → `profile2d` chargeability/phase), and the shared byte parser
  `_stg.py` (`STG_COLUMNS`, `parse_stg`, `value_property_and_unit`, `_LABEL_MAP`).
- Design references: `design/OVERVIEW.md` §3 (ERT/IP rows), `design/03-ingestion-adapters.md`
  §2 (ert/ip rows, the `profile2d` → `section` placement rule in §3d), and design doc 05 §4
  (the forward-model + DOI contract).
