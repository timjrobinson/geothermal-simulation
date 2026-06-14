# Geology maps & geochemistry

> **What you'll learn / why it matters.** Before any geophysics, a geologist walks the
> ground with a hammer and a notebook and draws a **map**: where rock unit A meets rock
> unit B (a *contact*), where the ground is broken (a *fault*), what kind of rock is
> where. That map is **human interpretation** turned into geometry — the structural
> skeleton that every other dataset hangs on. Separately, a geochemist collects **water
> and gas samples** from hot springs and wells and runs them through a lab; the chemistry
> of that water is a remarkably good **thermometer for the reservoir kilometres below**,
> because the water "remembers" how hot it was down there. This page covers how a
> geologist's drawing becomes machine-readable **Features**, how chemistry becomes point
> **Observations**, the vector and tabular formats that carry them, and the "geothermometer"
> equations that turn a water analysis into a reservoir temperature.

---

## 1. Two different kinds of evidence

This page is two methods that share one trait — they're **surface, human-centric** data,
not instrument cubes:

- **Geology maps** are *vector geometry plus interpretation*: lines and polygons drawn by
  a person, each tagged with what it means. They are the structural framework.
- **Geochemistry** is *point samples plus lab numbers*: a handful of locations, each with a
  table of measured concentrations. They are sparse but chemically rich.

Neither produces a 3D volume on its own. Both anchor and constrain the volumes the
geophysical methods produce.

---

## 2. Surface geology — interpretation as geometry

A geological map records three kinds of object:

- **Contacts** — the boundary line where one rock **unit** meets another (e.g. where
  alluvium gives way to volcanics). The geological equivalent of a class boundary.
- **Faults** — fractures where rock has slipped. Critically, faults are often the
  **plumbing** of a geothermal system: they're the cracks (the *permeability*) that let
  hot water rise. In our flagship scenario a **range-front normal fault** is both the
  master structure and the fluid conduit (doc 05 §7.1).
- **Units** — areas of one rock type, drawn as polygons.

> **Contact** — the surface where two different rock units meet.
>
> **Fault** — a fracture in the rock across which the two sides have moved relative to
> each other. Classified by *how* they moved: **normal** (extension, one side drops),
> **reverse** (compression, one side rides up), **strike-slip** (sideways).
>
> **Unit** (or *lithology*) — a body of rock of one type/age (sandstone, granite, …).

### Why geology maps are "2.5D"

A map is drawn in **plan view** — a flat $(x, y)$ projection, like looking straight down.
The lines and polygons have **no inherent depth**. But the real ground isn't flat: a
contact mapped at $(x, y)$ actually sits on hilly **topography** at some elevation $z$. So
the platform **drapes** the 2D geometry onto the surface elevation model to give it a $z$:

$$
z(x, y) = \text{surfaceModel}(x, y)
$$

This is **2.5D** — genuine 2D geometry given a single elevation value per point by hanging
it on the terrain (think of a [texture](../glossary.md) stretched over a 3D mesh: the
texture is 2D, the mesh gives it the third dimension). It is *not* full 3D: a fault drawn
on the map is just its **surface trace**, a line where the fault plane intersects the
ground. The fault actually dips down into the earth as a *plane*, but that 3D extent isn't
in the map — it has to be modelled later.

!!! note "From a geologist's drawing to Features — the pipeline"
    The geologist's interpretation arrives as vector geometry (lines/polygons in a CRS).
    Ingestion (1) reprojects it to the [Engineering Frame](../spatial-framework.md), (2)
    drapes it onto the `surfaceModel` to assign $z$, and (3) classifies each object into a
    typed **`GeologicalFeature`**. Contacts and faults become line/surface features; rock
    units become either surface features or, *if* a 3D geological model exists, watertight
    **unit solids**. A flat map alone can only give surface features — turning units into
    solid 3D bodies needs a geomodel (GemPy), which is a downstream step the map adapter
    flags rather than fabricates (doc 03 §5).

### Interpretive uncertainty

A map line is a **human opinion**, not a measurement. Where a contact is exposed in a road
cut it's certain; where it's buried under soil it's a dashed "inferred" guess. The
platform carries an **interpretive** uncertainty tag on map features
(`backend/geosim/synthgen/forward/surface.py`, `GeologyMapForward` writes
`"uncertainty": "interpretive"`), which the model treats as a **qualitative** confidence
tier (low/medium/high), *not* a number — see [uncertainty](../uncertainty.md). You should
never multiply an interpretive line into a precise-looking probability.

---

## 3. Geochemistry — water as a thermometer

When you can't drill, you can still sample what comes *out* of the ground: hot-spring
water, fumarole gas, well fluids. The dissolved chemistry is a window onto the reservoir.

### Geothermometers — the key idea

Deep in the reservoir, hot water sits in **chemical equilibrium** with the surrounding
minerals: at a given temperature, certain minerals dissolve to a specific, predictable
concentration. As that water rises quickly to the surface, the chemistry **freezes in** —
the reactions are too slow to re-equilibrate on the way up. So the surface water still
carries the chemical "signature" of its last deep, hot equilibrium. A **geothermometer**
is an empirical equation that reads that signature back out as a **reservoir temperature**.

!!! example "Analogy: reconstructing a value from a frozen cache"
    The water is a **cache** populated at reservoir temperature. The trip to the surface
    is too fast to invalidate the cache, so the surface sample still holds the deep value.
    The geothermometer is the function that decodes the cached chemistry back into the
    temperature that produced it — inferring a hidden parameter from a preserved
    observable, the same inverse-problem shape as the geophysical methods.

The classic **silica (quartz) geothermometer** uses dissolved silica $\text{SiO}_2$
(in mg/kg), which tracks quartz solubility:

$$
T\ (^\circ\text{C}) = \frac{1309}{5.19 - \log_{10}(\text{SiO}_2)} - 273.15
$$

- $\text{SiO}_2$ — dissolved silica concentration (mg/kg).
- $T$ — estimated last-equilibrium (reservoir) temperature in °C; the $-273.15$ converts
  the kelvin-based fit to °C.

The widely used **Na–K geothermometer** uses the ratio of sodium to potassium, which
re-equilibrates slowly and so reflects deep temperatures:

$$
T\ (^\circ\text{C}) = \frac{1217}{\log_{10}(\text{Na}/\text{K}) + 1.483} - 273.15
$$

- $\text{Na}/\text{K}$ — the molar ratio of sodium to potassium concentrations.

Different geothermometers (silica, Na–K, Na–K–Ca, gas ratios) re-equilibrate at different
rates and depths, so geochemists run several and look for agreement — disagreement itself
is information (mixing with shallow groundwater, slow cooling). These temperature
estimates feed straight into the heat side of [favorability](../rock-physics.md), and they
are **cheap** compared to drilling.

---

## 4. The native file formats (annotated)

### 4.1 Vector geology — Shapefile / GeoJSON / GeoPackage

Geological maps arrive as **vector GIS data**. The three common containers all hold the
same idea — geometries + attribute tables — in different packaging:

| Format | What it is | Notes |
|---|---|---|
| **Shapefile** | the legacy GIS format | actually a *bundle* of files (`.shp`+`.shx`+`.dbf`+`.prj`); the `.prj` holds the CRS |
| **GeoJSON** | JSON-based, single text file | human-readable; what our synthetic generator emits |
| **GeoPackage** | a SQLite database (`.gpkg`) | one file, many layers; modern default |

GeoJSON is the easiest to read. Here is an annotated map matching what `GeologyMapForward`
writes (faults as line traces, contacts as points):

```json
{
  "type": "FeatureCollection",
  "name": "great-basin-v1-geology",
  "features": [
    {
      "type": "Feature",
      "geometry": {                                  // a fault's SURFACE TRACE (a line)
        "type": "LineString",
        "coordinates": [[-6000, -3000], [6000, 1000]]  // Engineering metres (East, North)
      },
      "properties": {
        "kind": "fault",
        "id": "range-front",
        "faultKind": "normal",                       // normal/reverse/strikeslip
        "uncertainty": "interpretive"                // human opinion → qualitative tier
      }
    },
    {
      "type": "Feature",
      "geometry": {                                  // a CONTACT between two units (a point here)
        "type": "Point",
        "coordinates": [125.0, 340.0]
      },
      "properties": {
        "kind": "contact",
        "unitA": "alluvium",                         // unit on one side
        "unitB": "volcanics",                        // unit on the other
        "uncertainty": "interpretive"
      }
    }
  ]
}
```

Each `Feature` is one geometry (`LineString`, `Point`, or `Polygon`) plus a `properties`
table. The coordinates are in the source CRS in a real file (and reprojected on ingest);
in synthetic/local mode they are already Engineering metres (read with `geopandas`/`fiona`,
doc 03 §2).

### 4.2 Geochemistry — CSV / LIMS assay table

Lab results (a **LIMS** export — Laboratory Information Management System) are tabular: one
row per sample, columns for location and each measured species:

```csv
sample_id,x,y,elev,type,T_emer_C,SiO2,Na,K,Ca,Cl,pH
SP-01,820.0,310.5,1605.0,spring,68.4,210.0,640.0,42.0,18.0,310.0,7.1
SP-02,-440.0,1120.0,1640.0,spring,54.2,145.0,520.0,30.0,22.0,260.0,7.4
WL-03,0.0,0.0,1600.0,well,98.0,340.0,810.0,61.0,12.0,420.0,6.8
# x,y,elev : sample location, Engineering metres (point Observation coords)
# type     : spring | well | fumarole
# T_emer_C : temperature measured AT the surface (emergence) — NOT the reservoir T
# SiO2,Na,K,Ca,Cl : dissolved species concentrations (mg/kg) → feed the geothermometers
# pH       : acidity
```

The surface temperature `T_emer_C` is what you measure with a thermometer at the spring;
the **reservoir** temperature is *computed* from `SiO2` and `Na`/`K` via the §3 equations —
that computed value is the prize (parsed with `pandas`/`openpyxl`, doc 03 §2).

---

## 5. What it becomes in the model

After [ingestion](../ingestion.md), the two methods land as different primitives (see
[the data model](../data-model.md)):

| Native | Normalized primitive | Detail |
|---|---|---|
| Fault trace (line) | **`GeologicalFeature`** `featureKind: "fault"` | `FaultDetail` (faultType, dip, throw) |
| Contact (line) | **`GeologicalFeature`** `featureKind: "surface"` | a draped interpreted surface |
| Rock unit (polygon) | **`GeologicalFeature`** `featureKind: "surface"` → (with a geomodel) `"unitSolid"` | watertight solid only if a 3D model exists |
| Geochem sample (row) | **`ObservationSet`** `geometryKind: "points"` | one point per sample, assays as columns |

So a geologist's map becomes a set of typed **Features** (faults, surfaces, and — only if
a geomodel is built — unit solids), each carrying its interpretive-uncertainty tag and
draped onto topography for its $z$. Geochemistry becomes **point Observations**: a small
table where each row is a sample at an $(x, y, z)$ location with its measured species as
columns (often placed at the surface, or at a well's MD if the sample came from a well, doc
03 §2). These Features supply the structural skeleton that constrains
[fusion](../fusion.md), and the geothermometer temperatures feed the heat term of
[favorability](../rock-physics.md).

!!! tip "Why keep faults as Features and not bake them into the grid"
    A fault is a **discrete shape someone interpreted**, with editable geometry and an
    audit trail — exactly the [Feature](../data-model.md) primitive's job. Baking it into
    a voxel grid would destroy the interpretation and freeze a resampling choice.
    Features stay sharp and re-editable; the fused grid samples *from* them when needed.

---

## Key takeaways

- **Geology maps** are human **interpretation as geometry** — contacts, faults, units —
  drawn in plan view and **draped onto topography** to become **2.5D** (real 2D geometry +
  one elevation per point, *not* full 3D).
- A map fault is only its **surface trace**; its dipping plane and rock **unit solids**
  require a downstream **geomodel** the map adapter flags rather than invents.
- Map features carry an **interpretive** (qualitative) uncertainty tag — treat as
  low/med/high confidence, never a false-precision number.
- **Geochemistry** turns sparse **water/gas samples** into reservoir temperatures via
  **geothermometers** (silica, Na–K): the water "freezes in" its deep equilibrium
  chemistry, and an empirical equation decodes it back to temperature — cheap evidence for
  the **heat** side of favorability.
- Native formats: **Shapefile / GeoJSON / GeoPackage** (vector) and **CSV / LIMS** assays
  (tabular).
- Normalized form: **surface / fault / unitSolid `Feature`s** for the map, **point
  `Observation`s** for the chemistry.

## Where this lives in the code

- Synthetic geology forward (contacts + fault traces from the lithology field → GeoJSON):
  `backend/geosim/synthgen/forward/surface.py` (`GeologyMapForward`).
- Synthetic geochem/heat-flow points (temperature & sample points → CSV):
  `backend/geosim/synthgen/forward/borehole.py` (`HeatFlowForward`).
- Ingestion contract for `geology` (geopandas/fiona) and `geochem` (pandas/openpyxl):
  doc `03-ingestion-adapters.md` §2 rows for `geology` / `geochem`.
- Coordinate / drape handling: `backend/geosim/spatial/frame.py`,
  `backend/geosim/spatial/vertical.py`.
