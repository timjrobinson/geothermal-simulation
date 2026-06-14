# Geothermal energy, for programmers

> **What you'll learn / why it matters.** This is the crucial background page. If you can write code but have
> never thought about rocks, start here. You'll learn what geothermal energy *is*, why the Earth is hot, what the
> **geothermal gradient** is and why some places are hot at shallow depth, the single headline idea of the whole
> field — the **geothermal play** (heat **and** fluid **and** permeability, all in the same place) — the
> difference between **hydrothermal** systems and **EGS**, the conceptual model of a fault-controlled upflow with
> a clay cap that the simulator's flagship scenario actually models, *why* we can't just look and must infer the
> subsurface from physics, and why drilling is where the money and risk concentrate. Every term is defined the
> first time it appears and linked to the [glossary](glossary.md).

## What geothermal energy is, and why we want it

**Geothermal energy** is heat stored in the Earth's interior. The plan is simple to state: drill into hot rock,
get the heat to the surface (usually by circulating water that comes back as steam or hot water), and run a power
plant or heat a building with it.

Why bother, when it means drilling kilometres of rock?

- It is **carbon-free** and runs **24/7** — unlike solar and wind, it is not weather-dependent. In power-grid
  terms it is *baseload*: always on.
- The resource is **enormous** — the heat under our feet dwarfs all fossil reserves; the problem has always been
  *accessing it economically*, not *quantity*.
- It has a **small surface footprint** per unit of energy.

The catch, and the reason this software exists: **you cannot see the resource.** It is kilometres down, hidden by
opaque rock. Drilling a single well can cost millions of dollars, and a well drilled into the wrong spot finds no
heat, or hot rock but no water, or water but no cracks for it to flow through — and produces nothing. So before
you drill, you must *infer* what is down there from indirect physical measurements made at the surface and in a
few existing boreholes. That inference, fused across many measurement types, is the product these docs describe.

## Why the Earth is hot: heat flow and the geothermal gradient

The interior of the Earth is hot — thousands of degrees at the core — from two sources: leftover heat from the
planet's formation, and ongoing **radioactive decay** in the crust and mantle. That heat continuously **flows**
outward toward the cold surface. The rate of that flow is called **heat flow** (measured in milliwatts per square
metre, mW/m²); it is, in effect, the planet leaking thermal energy.

Because heat flows from hot (deep) to cold (surface), temperature *increases* with depth. The rate of that
increase is the **geothermal gradient**:

$$
\text{geothermal gradient} \;=\; \frac{\Delta T}{\Delta z}\qquad \text{units: }^{\circ}\text{C/km}
$$

where $\Delta T$ is the temperature change over a depth interval $\Delta z$. **°C/km** simply means "degrees
Celsius hotter per kilometre you go down." A CS analogy: if depth is the x-axis and temperature the y-axis, the
geothermal gradient is the *slope* of the temperature-vs-depth line.

- The **normal / average** continental gradient is about **25–30 °C/km**. So at 3 km depth you might be only
  ~75–90 °C above the surface temperature — warm, but not great for power. To reach the ~200 °C useful for
  electricity at a normal gradient you'd have to drill *very* deep, which is expensive.
- Some regions have a **much higher gradient** because heat flow is elevated there. The
  **Basin and Range** province of the western United States (Nevada, Utah) is the classic example: crustal
  stretching ("extensional tectonics") thins the crust and lets heat up, giving gradients well above normal. The
  simulator's flagship synthetic earth uses a **~45 °C/km** gradient to model exactly this setting. At that
  gradient you can reach useful temperatures at *drillable* depths — which is why these regions are the prime
  hunting grounds.

!!! note "Why a higher gradient changes the economics, not just the physics"
    Drilling cost rises steeply (faster than linearly) with depth. A region where 200 °C sits at 2–3 km instead
    of 6–7 km is the difference between a viable project and an uneconomic one. The geothermal gradient is, in a
    sense, the *exchange rate between heat and drilling dollars*.

A subtlety worth knowing: a high gradient near the surface can come from heat being *carried upward by moving
water* (advection), not just conducted through rock. A plume of hot water rising along a fault locally bends the
temperature-vs-depth curve sharply upward. That is precisely the kind of target we hunt for — and it is why the
*shape* of the temperature field, not just its average slope, matters.

## The headline idea: a geothermal play needs THREE things, co-located

This is the single most important concept in the whole field. **A productive geothermal resource — a
"play"[^play] — requires three independent things to occur in the *same place*:**

[^play]: **Play** — borrowed from petroleum geology: a conceptual model of a working subsurface resource system,
    i.e. the combination of conditions that makes a deposit economically producible.

1. **Heat** — the rock and fluid must actually be hot enough (≈ 150–250 °C for electricity).
2. **Fluid** — there must be water (or steam) present to *carry* that heat to a well. Hot dry rock is useless to
   a conventional plant; you need a working fluid in the rock's pore space.
3. **Permeability** — the rock must have connected cracks and pores so the fluid can *flow* fast enough to
   sustain production. Hot water trapped in impermeable rock can't be pumped out economically.

!!! abstract "The whole game, in one line"
    > Geothermal exploration is a search for **heat ∧ fluid ∧ permeability**, all in the same volume of rock.

The conjunction (`∧`, logical AND) is the point, and it is why fusion is non-negotiable. Each survey method is
good evidence for *one or two* of these and largely blind to the others:

- A **temperature** measurement in a well tells you about *heat* — but only at that one well.
- An **electrical/electromagnetic** survey finds *low resistivity*, which can mean hot, salty water in porous
  rock — evidence for *fluid* (and indirectly heat) — but it cannot tell you if the rock has *permeability*.
- **Seismic** imaging finds the *structure* (layers, faults) that *might* host permeability — but it is nearly
  blind to temperature and fluid.
- **Microseismic** (tiny earthquakes) can light up *active, permeable* fractures — evidence for *permeability* —
  but says little about heat.

No single method tells you where all three coincide. The platform's job is to line them all up in 3-D and show
you the volume where the evidence for heat **and** fluid **and** permeability *overlaps*. That overlap is a
drilling target. The [rock physics & favorability](rock-physics.md) page turns this AND into actual math; note
that the system's *default* favorability combination is a **fuzzy AND** — *non-compensatory*, meaning strong
evidence for heat cannot paper over the *absence* of permeability. That choice exists precisely because the three
ingredients are a conjunction, not a sum.

## Defining the rock-and-fluid vocabulary

Before going further, here are the terms the rest of the docs assume. Each is also in the
[glossary](glossary.md).

| Term | Plain-English definition | Why it matters here |
|---|---|---|
| **Reservoir** | The body of hot, permeable, fluid-filled rock you actually produce from. | The "target volume." Finding its extent is the goal. |
| **Porosity** ($\phi$) | The fraction of a rock's volume that is empty pore space (0–1, or %). A sponge has high porosity. | Holds the fluid. More pore space ⇒ more fluid ⇒ (often) lower resistivity and lower seismic velocity. |
| **Permeability** | How easily fluid can *flow through* the connected pore space. | The hardest ingredient to detect remotely; high porosity does **not** guarantee high permeability (the pores must connect). |
| **Brine** | Hot, salty groundwater (high dissolved-salt content, "TDS" = total dissolved solids). | Saltwater conducts electricity far better than fresh water — this is *why* electrical/EM methods can sense geothermal fluid. |
| **Permeable fracture** | A crack or fault that fluid flows along. | In many geothermal systems the permeability *is* the fracture network, not the rock matrix. |
| **Intrusion** | A body of once-molten rock (magma) that pushed into older rock and solidified. | A young intrusion can be the *heat source* powering a system. |
| **Basement** | The deep, old, hard crystalline rock (e.g. granite) beneath the younger sedimentary/volcanic layers. | Often hot but low-permeability — the EGS frontier (below). |
| **Alteration** | Chemical change of rock by hot fluid over time (minerals dissolve and new ones grow). | A *fingerprint* of past/present fluid flow. It changes the rock's physical properties in diagnostic ways (next). |
| **Clay cap** | A shallow layer of clay minerals formed by alteration, sitting *above* a reservoir. | Clay is electrically conductive and nearly impermeable — it both *seals* the reservoir and shows up as a strong, shallow conductor that surveys can find. |

### Why alteration is a gift to geophysics

When hot, often acidic fluid percolates through rock for a long time, it chemically transforms it. Two effects
matter enormously because they are *detectable from the surface*:

- It produces **clay minerals**, which are very electrically **conductive** (low resistivity). A clay-rich zone
  lights up like a beacon to electrical and electromagnetic surveys. The shallow clay layer above a reservoir —
  the **clay cap** — is one of the most reliable indirect signs of a geothermal system.
- It **destroys magnetite**, the iron-oxide mineral that makes rock magnetic. So altered rock has *low magnetic
  susceptibility*, producing a **magnetic low** over the system — a second, independent fingerprint.

This is the crux of why we measure physics rather than temperature everywhere: we cannot put a thermometer in
every cubic metre of rock, but the *side effects* of a hot fluid system (conductive clay, destroyed magnetism,
changed density and seismic velocity) ripple outward into properties we *can* sense at the surface. Fusion is the
art of reading the temperature/fluid/permeability story back out of those side effects.

## Hydrothermal systems vs EGS

There are two broad ways to have (or create) a producible geothermal reservoir.

=== "Hydrothermal (conventional)"

    A **hydrothermal system** is a *natural* convection cell: heat (often from an intrusion or just high regional
    heat flow) drives groundwater to circulate. Water descends, gets heated at depth, becomes buoyant, and rises
    along permeable pathways — typically **faults**. Where it rises, it heats and alters the shallow rock,
    building a clay cap, and creates a hot, fluid-filled, permeable reservoir.

    All three ingredients (heat, fluid, permeability) occur **naturally and together**. The exploration challenge
    is purely to *find* the reservoir and its margins before drilling. This is what the flagship
    `great-basin-v1` synthetic earth models, and what most of these docs assume.

=== "EGS (Enhanced Geothermal Systems)"

    Many places have **heat** but lack natural **fluid pathways / permeability** — for example hot, dry
    crystalline **basement** granite. **EGS** *engineers* the missing ingredient: you drill into hot rock and
    **stimulate** it (inject fluid under pressure to open and connect fractures), creating an artificial
    reservoir, then circulate water through the new fracture network between an injection well and a production
    well.

    This is the frontier the industry is pushing hard right now. Two names worth knowing:

    - **FORGE** (Frontier Observatory for Research in Geothermal Energy) — a U.S. Department of Energy field
      laboratory near **Milford, Utah**, dedicated to EGS research in hot granite.
    - **Fervo Energy** — a company that has demonstrated commercial-scale EGS using horizontal drilling and
      fracturing techniques borrowed from the shale industry.

    EGS makes the **4-D / monitoring** problem central: during stimulation you watch the fracture network grow in
    real time via **microseismic** (a cloud of tiny earthquakes mapping where rock is cracking) and **InSAR**
    (satellite radar measuring millimetre ground deformation as fluid is injected). The simulator's
    `egs-granite-v1` scenario models exactly this story. See
    [seismic & microseismic](survey-methods/seismic.md) and [InSAR](survey-methods/insar.md).

## The conceptual model the simulator actually builds

The flagship synthetic earth, **`great-basin-v1`** (a Basin-and-Range hydrothermal play, modelled on the
Nevada/Utah setting), encodes the textbook conceptual model. Knowing this model makes every later page concrete —
it is the "test fixture" the whole pipeline is validated against.

```
        SURFACE  (alluvium-filled valley, ~1600 m elevation)
   ────────────────────────────────────────────────────────────
   ░░░░░░░░░░░  alluvium  (young, soft, porous valley fill)
   ▒▒▒▒▒▒▒▒▒  CLAY CAP  ← conductive, impermeable seal (alteration)   ◄ shallow conductor
   ▒▒▒▒▒▒  volcanics  ▒▒▒▒▒▒
   ▓▓▓▓▓▓▓▓▓  carbonate  ▓▓▓▓▓     ╱ range-front
   ████████  RESERVOIR  ████████  ╱  NORMAL FAULT   ◄ heat + fluid + permeability
   ██  (hot, altered, fractured, ╱  (60° dip, ~700 m
   ██   saline → conductive)    ╱    throw, the conduit)
   ████████████████████████████╱
   ▆▆▆▆▆▆▆▆▆  basement granite  ▆▆▆▆▆▆▆▆▆
                ▲ hot fluid rises along the fault from depth
```

The pieces, and the survey signature each produces:

- A **range-front normal fault** dipping ~60° with ~700 m of throw (vertical offset). It is both the master
  structure that drops the valley down *and* the **conduit** the hot fluid rises along.
- A **fault-controlled hydrothermal upflow**: hot brine (~220 °C) rises along the fault, heating, altering, and
  fracturing the rock around it. This creates the **reservoir** — hot, fractured (permeable), and saline (so
  electrically conductive).
- A **clay cap** above the reservoir: a shallow, conductive, impermeable smile that seals it.
- The resulting **joint, multi-method signature** — the reason this is the flagship test:
  | Evidence | Method that sees it | Pointing at |
  |---|---|---|
  | Shallow + deep **conductor** (low resistivity) | electromagnetic (MT/EM), electrical (ERT) | fluid + clay alteration |
  | **Magnetic low** over the system | magnetics | alteration (destroyed magnetite) |
  | Basin + fault **structure** | gravity, seismic | the architecture hosting permeability |
  | A **hot well** | borehole temperature log | heat, directly, at one point |
  | A **microseismic** cloud on the fault | passive seismic | active permeable fractures |

No one method sees the whole story; the platform's job is to overlay them so the heat ∧ fluid ∧ permeability
overlap pops out. The exact properties and forward models are in [the synthetic data generator](synthetic-data.md).

## Why we can't just look — inference from physics

Here is the situation a programmer should internalize: **the subsurface is the ultimate opaque, lossy,
under-sampled data source.**

- You **cannot observe it directly** except at the few points where a well has been drilled — and wells are
  millions of dollars each, so you have a handful at most.
- Every surface survey is an **indirect, lossy, low-pass-filtered, noisy** measurement of a 3-D field. Deep
  features are smeared out; methods average over large volumes; signal decays with depth.
- Turning measurements back into an earth model is an **inverse problem**: ill-posed and **non-unique** — many
  different earths could produce the same surface data (just as many programs can produce the same output). You
  cannot recover the truth uniquely; you can only constrain it.

The escape from non-uniqueness is *more, independent constraints*: each additional survey method removes some of
the ambiguity the others leave. A model that fits gravity *and* MT *and* seismic *and* a well is far more
constrained than any one alone. **That is the entire argument for a fusion platform** — and it is laid out in
detail on [the core problem](core-problem.md) page.

## Why drilling is where the risk lives

Exploration surveys are *cheap* relative to a well. A geothermal **well can cost on the order of millions of
dollars** and take weeks to drill, and the geothermal failure modes are brutal because of the three-ingredient
problem:

- Drill into rock that is **hot but dry** (no fluid) → no production.
- Drill into rock that is **hot and wet but tight** (no permeability) → fluid won't flow → no production.
- Miss the reservoir **laterally or in depth** because the model placed it wrong → a dry hole.

So the economic logic is: **spend on surveys and modelling to de-risk the drill location**, because the cost of
the surveys is dwarfed by the cost of a failed well. Every rung of the [fusion ladder](overview.md#the-progressive-fusion-ladder)
exists to shrink that drilling risk — culminating in the [well-planning](well-planning.md) tools that let you
place and predict a trajectory *in the model* before committing real steel to the ground.

## Key takeaways

- **Geothermal energy** is heat from inside the Earth; it is carbon-free, always-on baseload — but invisible and
  expensive to reach, which is why we *infer* the subsurface from physics.
- The Earth gets hotter with depth at the **geothermal gradient** (°C/km): ~25–30 °C/km is normal; high-heat-flow
  provinces like the **Basin and Range** reach ~45 °C/km, putting useful heat at drillable depth.
- A **geothermal play** needs **heat ∧ fluid ∧ permeability**, all in the *same place*. This conjunction is the
  central idea and the reason fusion is mandatory.
- **Alteration** by hot fluid is a gift: it creates conductive **clay** (visible to electrical/EM) and **destroys
  magnetism** (a magnetic low) — diagnostic fingerprints of a system.
- **Hydrothermal** systems have all three ingredients naturally; **EGS** (FORGE, Fervo) *engineers* the missing
  permeability into hot dry rock, making 4-D monitoring central.
- The flagship synthetic earth models a **fault-controlled upflow with a clay cap**, producing the textbook joint
  signature (conductor + magnetic low + structure + hot well).
- **Drilling is the dominant cost and risk**; the whole platform exists to de-risk where to drill.

## Where this lives in the code

- The conceptual model is encoded by the synthetic generator: `backend/geosim/synthgen/` — see `scene.py` (the
  scene spec), `compiler.py` (lithology + state → property volumes), `rockphysics.py` (the rock-physics rules
  that turn geology into resistivity/density/velocity/etc.), `resistivity.py`, and `truth.py` (the retained
  ground-truth bundle). The flagship scenario is `great-basin-v1`.
- The "heat ∧ fluid ∧ permeability" conjunction becomes runnable math in the favorability engine:
  `backend/geosim/synthgen/rockphysics.py` (synthetic side) and `frontend/src/lib/favorability.ts` (the
  fuzzy-AND default combination in the viewer).
