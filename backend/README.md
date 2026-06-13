# geosim — backend

FastAPI + Python geoscience backend for the Geothermal Underground Simulator.

## Packages

| Package | Doc | Responsibility |
|---|---|---|
| `geosim.spatial` | 01 | Engineering Frame, CRS/datum transforms (pyproj), units registry (pint), property-type registry, depth/MD/TVD + minimum-curvature |
| `geosim.datamodel` | 02 | Observation / PropertyModel / Feature / FusedEarthModel schemas, provenance, IDs |
| `geosim.storage` | 04 | Zarr v3 writer/reader, COG, raw store, content-addressing |
| `geosim.catalog` | 04 | SQLAlchemy models + spatial index (Postgres/PostGIS, SQLite fallback) |
| `geosim.jobs` | 04 | RQ + Redis job runner, job contract |
| `geosim.plugins` | 08 | PluginRegistry + 6 extension points, discovery, manifest validation |
| `geosim.ingestion` | 03 | Ingestion adapters + normalization pipeline |
| `geosim.fusion` | 07 | Resampling, cross-plot/stats, rock-physics transforms, favorability, uncertainty |
| `geosim.synthgen` | 05 | Synthetic ground-truth earth + per-method forward models |
| `geosim.inversion` | 10 | (later) InversionEngine plugins |
| `geosim.api` | 04 | FastAPI app, endpoints, capabilities |

## Setup

```bash
uv sync --extra dev          # core + dev tooling
uv run pytest                # run tests
uv run ruff check .          # lint

# optional heavy deps
uv sync --extra ingest --extra fusion --extra postgres
```
