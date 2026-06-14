# The Geothermal Underground Simulator

> **Who this is for.** You can program. You may have never heard the words *magnetotellurics*
> or *dogleg severity*. By the end of these docs you will understand both — and exactly how a
> piece of software takes a dozen wildly different ways of measuring the Earth and fuses them
> into a single, queryable, viewable 3-D model you can drill a geothermal well into.
>
> No geoscience background is assumed. Every geophysical term is defined the first time it
> appears, and there is a [Glossary](glossary.md) for everything.

## What is this thing?

Imagine you want to find **heat underground** — hot rock and hot water, kilometres down, that
you could drill into to make carbon-free electricity. You cannot see through rock. So instead
you measure the Earth *indirectly*, from the surface and from boreholes, using physics:

- pull a sensor across the ground and measure tiny changes in **gravity** (denser rock pulls harder),
- inject electrical current and measure how the ground **resists** it (hot salty water conducts),
- thump the ground and time the **echoes** (sound reflects off rock layers),
- watch the ground **swell or sink** from a satellite,
- lower instruments down a **borehole** and read temperature and rock properties directly,

…and a dozen more. Each method "sees" something different, at a different depth, resolution,
and reliability. **No single method tells you where to drill.** The whole game is *combining*
them.

This software is the machine that does the combining. It:

1. **Ingests** raw files from every survey method, in their real industry formats.
2. **Reconciles** them into one coordinate system, one set of units, one data model.
3. **Fuses** them onto a shared 3-D grid so you can compare and cross-plot them cell-by-cell.
4. **Transforms** the geophysics (resistivity, velocity, density…) into the things a geothermal
   engineer actually cares about (temperature, fluid, permeability) via **rock physics**.
5. **Visualizes** all of it in a browser as a 3-D earth model you can slice, fly through, and
   drill virtual wells into — always carrying **uncertainty** so you never trust a number more
   than the data deserves.

It ships with a **synthetic earth** (a fully known fake planet) so every step can be validated
against ground truth.

## A 30-second mental model

```
 raw survey files            normalized "earth model"           interpretation
 (SEG-Y, LAS, EDI, …)   →    Observations / Property Models  →  fused grid → favorability
 every method, every            (one frame, one set              "drill here" + a planned
 format, every unit              of units, provenance)            well + its predicted log
```

Three primitive data types carry *everything*:

| Primitive | Plain-English meaning | Example |
|---|---|---|
| **Observation** | what was measured, where (raw, immutable) | 400 gravity readings at GPS points |
| **Property Model** | a continuous 3-D field of one physical property | a resistivity cube from an inversion |
| **Geological Feature** | a discrete shape someone interpreted | a fault surface, a well path |

Everything lives in one **Engineering Frame** (a local X-East / Y-North / Z-Up metre grid), and
every artifact carries **provenance** (where did this number come from?) and **uncertainty**
(how much should I trust it?).

## How to read these docs

The site is ordered as a course. If you read top-to-bottom you'll never hit a term you haven't met.

1. **The Big Picture** — [what this is](overview.md), a [geothermal primer](geothermal-primer.md)
   (start here if "geothermal" is fuzzy), and [why fusing surveys is hard](core-problem.md).
2. **Foundations** — the [coordinate/depth/units](spatial-framework.md) machinery and the
   [data model](data-model.md) every method maps onto.
3. **The Survey Methods** — one page per family of measurement: the physics, what it can and
   can't see, the **real file format with annotated examples**, and the normalized output.
   This is the heart of the docs. Start with [how to read these pages](survey-methods/index.md).
4. **Merging the Data** — [ingestion](ingestion.md), [fusion](fusion.md),
   [rock physics & favorability](rock-physics.md), and [uncertainty](uncertainty.md).
5. **Using the Model** — the [3-D viewer](visualization.md), [well planning](well-planning.md),
   and [forward modeling & inversion](inversion.md).
6. **Reference** — the [synthetic data generator](synthetic-data.md), the
   [codebase architecture](architecture.md), and the [glossary](glossary.md).

## The headline idea, if you remember nothing else

> Geothermal exploration is a search for three things that must occur **in the same place**:
> **heat**, **fluid**, and **permeability** (cracks for the fluid to flow through). Each survey
> method is good evidence for *one* of these and blind to the others. The platform's job is to
> line them all up in 3-D and show you where all three coincide — that spot is a drilling target.

Ready? Start with [**What this simulator is →**](overview.md)
