# Geothermal Underground Simulator

A browser-based 3D underground simulator for geothermal drilling that fuses every
subsurface survey method (gravity, magnetics, ERT/IP, EM/TEM, MT, seismic, InSAR,
well logs, heat-flow, geochemistry) into a single georeferenced 3D earth model and
renders it in the browser.

See [`design/`](design/) for the full architecture. `design/OVERVIEW.md` is the map;
`design/ROADMAP.md` is the milestone plan (M0–M9); `design/DECISIONS.md` is the
authoritative decision log.

## Repository layout

```
backend/    FastAPI + Python geoscience stack (spatial frame, data model, storage,
            catalog, jobs, plugins, fusion, synthetic generator, inversion)
frontend/   React + TypeScript + Vite + Three.js (react-three-fiber) 3D viewer
design/     Architecture & design docs (source of truth)
```

## Tech stack

- **Frontend:** React + TypeScript + Vite, Zustand state, Three.js via react-three-fiber,
  WebGL2 ray-marching (WebGPU progressive enhancement), Observable Plot / D3 panels.
- **Backend:** Python + FastAPI; `pyproj`, `pint`, `xarray`, `zarr`, `numpy`, `rasterio`,
  `verde`, `segyio`, `lasio`, `ObsPy`; later `SimPEG`/`PyGIMLi` (inversion), `GemPy` (geomodel).
- **Storage:** PostgreSQL + PostGIS catalog (SQLite + SpatiaLite portable fallback);
  Zarr v3 (3D/4D volumes), COG (2D), glTF/VTK (meshes), GeoJSON, LAZ/3D-Tiles (points).
- **Jobs:** RQ + Redis.

## Engineering Frame (doc 01) — the one invariant

Everything internal lives in a single **Engineering Frame**: local right-handed ENU
(X=East, Y=North, Z=Up), metres, floating origin. Georeferencing is an optional rigid
transform to a real CRS. Bulk arrays are *always* stored in Engineering coordinates, so
local and georeferenced data share one code path and re-anchoring never reprocesses arrays.

## Development

```bash
# backend
cd backend && uv sync && uv run pytest

# frontend
cd frontend && npm install && npm run dev

# infra (Postgres + PostGIS + Redis) — when Docker is available
docker compose up -d
```

Local testing without Docker uses the SQLite + SpatiaLite catalog fallback (doc 04 §2.1).
