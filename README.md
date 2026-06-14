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
            catalog, jobs, plugins, ingestion, fusion, synthetic generator,
            planning, inversion, geomodel)
frontend/   React + TypeScript + Vite + Three.js (react-three-fiber) 3D viewer
docs/       MkDocs Material documentation site (teaches geothermal engineering
            from scratch for programmers) + the flashcard deck
study/      Study tooling: AI exam server + flashcard generator (use local `claude -p`)
design/     Architecture & design docs (source of truth)
Makefile    All dev/run/test/docs/study commands — run `make help`
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

## Getting started

Everything is driven by the **Makefile** — run `make help` for the full list.

```bash
make setup            # backend venv (uv) + deps, and frontend npm deps
make run              # prints the two-terminal recipe below

# the app (two terminals):
make demo             # 1) seeded API + sample data on :8000
make run-frontend     # 2) Vite 3D viewer on :5173  → open the forwarded port
```

- The default install uses the **SQLite + inline-jobs** fallback, so it runs with no
  Docker. For the Postgres + PostGIS + Redis stack, `make infra-up` then point the
  backend at it.
- Heavy solvers (rigorous forward models, GemPy, SimPEG/PyGIMLi) are optional:
  `make install-backend-full`. Without them the core platform runs fine and their tests
  auto-skip.

### Tests, lint, build

```bash
make test             # backend test suite (pytest)
make test-frontend    # frontend unit tests
make test-all         # both
make lint             # ruff (backend)
make typecheck        # tsc (frontend)
make build-frontend   # production frontend build
make check            # full CI gate: lint + typecheck + tests + build
```

## Documentation

A full **MkDocs Material** site teaches the whole project — and geothermal engineering
from first principles — for readers who know programming but not geoscience (24 pages:
the survey methods and their data formats, the fusion pipeline, rock physics, the viewer,
well planning, inversion, a glossary, and more).

```bash
make install-docs     # one-time: mkdocs-material
make docs             # live-reload docs site on :8001
make docs-build       # strict static build to ./site
```

## Study tools

An AI-powered study layer over the docs, using the **local Claude model** (`claude -p`,
no API key — it reuses your Claude Code auth):

- **Per-page exams** — a "Generate Exam" button at the bottom of every docs page writes
  questions from that page, you answer in free text, and it grades you (score + feedback
  + ideal answers). Weak answers can be turned into flashcards in one click.
- **Spaced-repetition flashcards** — an ~888-card deck (`docs/flashcards/deck.json`,
  generated from the docs) with an Anki-style SM-2 scheduler; self-grade 0–5 and the
  cards you find hard resurface more often. Studying needs no server (it's a static deck).
- **Progress dashboard** — flashcard mastery, cards-mastered-over-time, review activity,
  and per-topic exam scores. All progress is stored in your browser (`localStorage`).

```bash
make study            # build docs + serve them WITH the exam/flashcard API on :8002
make study-api        # run only the exam API (pair with `make docs` on :8001)
make flashcards       # (re)generate the flashcard deck from the docs via `claude -p`
```

Open the `:8002` port → every page has *Generate Exam*; the **Study** section has the
**Flashcards** and **Progress dashboard** pages.
