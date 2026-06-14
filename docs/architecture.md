# Codebase architecture

> **What you'll learn / why it matters.** Every other page in these docs explains a *concept* — coordinate frames, the data model, a survey method, fusion. This page is the map from those concepts to the **actual code**: where each idea lives, how a request flows end-to-end, how the plugin system lets you add a whole new survey method without touching core, and how to run the thing. If you're a developer about to open the repo, start here. We'll assume you know software architecture but not the geoscience — terms are linked to the [glossary](glossary.md).

The system is a **monorepo** with two halves: a **Python + FastAPI backend** (`backend/geosim/…`) that owns all the geoscience, and a **React + TypeScript + Three.js frontend** (`frontend/src/…`) that owns the 3-D viewer. They talk over a plain REST + WebSocket API. The guiding architectural principle, end to end, is **method-agnosticism**: neither the core backend nor the frontend hard-codes the list of survey methods or physical properties. That list comes from a **plugin registry** and is shipped to the client at runtime via one `/api/capabilities` document. Adding a survey method is "write one plugin package," not "edit core."

---

## 1. The monorepo at a glance

```text
simulation/
├── backend/
│   ├── geosim/            # the Python package — all backend logic (below)
│   ├── scripts/           # dev_server.py (the seeded demo server) etc.
│   ├── tests/             # pytest suite
│   └── pyproject.toml     # deps + optional extras (ingest, fusion, forward-t1, …)
├── frontend/
│   ├── src/               # React + TS + Three.js (scene/ lib/ ui/ + store.ts)
│   └── package.json       # Vite app
├── design/                # the design docs (sources of truth: 01–10, OVERVIEW, DECISIONS)
├── docs/                  # this MkDocs site
├── Makefile               # the developer command reference (see §6)
└── docker-compose.yml     # OPTIONAL Postgres+PostGIS + Redis (defaults need neither)
```

### 1.1 The backend packages (`backend/geosim/`)

Each package maps onto a design doc and a chapter of these docs. Core packages depend *inward*; the plugin system is the only place core depends on extensions, and even then only through abstract interfaces.

| Package | Responsibility | Concept page |
|---|---|---|
| `spatial/` | CRS, datums, the **Engineering Frame**, units (`pint`) registry, the property-type registry | [Coordinates, depth & units](spatial-framework.md) |
| `catalog/` | the metadata DB — SQLAlchemy models, Alembic migrations, PostGIS spatial columns; projects, datasets, jobs, provenance | [The data model](data-model.md) |
| `storage/` | the **bulk** array stores — Zarr volumes, COG/GeoTIFF rasters, glTF meshes, multiresolution pyramids, the raw-file store | [The data model](data-model.md) |
| `ingestion/` | the per-method format adapters + normalization pipeline + gridding | [Ingestion](ingestion.md) |
| `synthgen/` | the synthetic data generator — ground-truth earth + forward models | [The synthetic data generator](synthetic-data.md) |
| `fusion/` | resampling onto the fused grid, cross-plotting/statistics, rock-physics transforms, favorability, calibration | [Fusion](fusion.md) · [Rock physics](rock-physics.md) |
| `planning/` | drilling targets, well trajectories, intersection, predicted logs, drillability/risk, export (WITSML/CSV) | [Drilling & well planning](well-planning.md) |
| `inversion/` | the inversion-engine interface, mock engine, cooperative/joint inversion harness (heavy solvers are optional/later) | [Forward modeling & inversion](inversion.md) |
| `geomodel/` | implicit geological model builder (GemPy, optional) + writer | [Fusion](fusion.md) |
| `plugins/` | the **one registry + six extension points** — the extensibility spine (this page §3) | this page |
| `jobs/` | async job runners — inline (no-service) and RQ+Redis tiers | this page §4 |
| `api/` | the FastAPI app + routers; the REST/WebSocket surface, including `/api/capabilities` | this page §5 |

### 1.2 The frontend (`frontend/src/`)

The frontend is a Vite single-page app. It has three concerns, mirrored in three directories, plus a Zustand store:

| Directory | Responsibility |
|---|---|
| `scene/` | the Three.js / react-three-fiber 3-D scene — `Scene.tsx` plus one component per renderable kind: `VolumeLayer` / `StreamingVolumeLayer` (GPU ray-marched volumes), `SliceLayer`, `ClipBox`, `FeatureLayer`, `WellLayer`, `PointCloudLayer` (microseismic), `RasterLayer`, `TerrainLayer`, `PickTargetLayer`. |
| `lib/` | the non-React engine room — `api.ts` (typed REST client), brick streaming + decoding (`bricks.ts`, `brick.worker.ts`, `brickDecode.ts`, `brickPool.ts`), shaders, colormaps, transfer functions, LOD, cross-plot/favorability/wells/planning math, time handling. |
| `ui/` | the React panels — `DiscoveryPanel` (load a project), `ControlPanel`, `AnalysisPanel`, `CrossPlot`, `Histogram`, `CorrelationHeatmap`, `FavorabilityPanel`, `TransferFnEditor`, `MembershipCurveEditor`, `LogTrackPanel`, `PredictedLogTracks`, `PlanningPanel`, `RiskReadout`, `TimeSlider`, `ScenarioTable`. |
| `store.ts` | the single Zustand store — loaded project, layers, selection, time, transfer functions, etc. |

The stack is React + TypeScript + **Three.js** (via `@react-three/fiber` and `@react-three/drei`); volume rendering is GPU ray-marching of a `Data3DTexture` with per-property transfer functions. Details are in [the 3D viewer](visualization.md).

---

## 2. The architectural layers

The OVERVIEW describes the system as a stack of layers, top (browser) to bottom (spatial foundations):

```text
┌──────────────────────────────────────────────────────────────┐
│  CLIENT (browser) — React + TS + Three.js (react-three-fiber) │  frontend/src/
│  3D scene · layer manager · volume render · slices · sections  │
│  well paths · microseismic · time slider · cross-plot panels   │
└──────────────▲───────────────────────────────┬────────────────┘
               │  REST / WebSocket (tiles, slices, queries, jobs)
┌──────────────┴───────────────────────────────▼────────────────┐
│  API / SERVING — FastAPI                                       │  geosim/api/
├────────────────────────────────────────────────────────────────┤
│  PROCESSING — resample→fused grid · rock-physics transforms ·  │  geosim/fusion/
│  derived properties · [pluggable] inversion · geomodel         │  geosim/inversion/ geomodel/
├────────────────────────────────────────────────────────────────┤
│  DOMAIN / MODEL — observations · property models · features ·  │  geosim/catalog/ (schema)
│  fused grid · time                                             │
├────────────────────────────────────────────────────────────────┤
│  INGESTION — per-method format adapters (plugins)             │  geosim/ingestion/
├────────────────────────────────────────────────────────────────┤
│  STORAGE — catalog DB + chunked array store + raw store       │  geosim/catalog/ geosim/storage/
├────────────────────────────────────────────────────────────────┤
│  SPATIAL FRAMEWORK — CRS, datums, units registry, provenance  │  geosim/spatial/
└────────────────────────────────────────────────────────────────┘
        SYNTHETIC DATA GENERATOR (feeds Ingestion)  ──  geosim/synthgen/
```

Crucially, the **plugin/extensibility framework spans several layers at once** (ingestion, property types, transforms, forward models, renderers, inversion engines). That cross-cutting framework is the next section.

---

## 3. The plugin architecture (the extensibility spine)

This is the single most important architectural idea, and it's the one most worth understanding before reading any code. The requirement (from the OVERVIEW) is that this is a **research/R&D platform**: someone must be able to add a brand-new survey method without editing core ingestion, fusion, storage, or viewer code. The answer is **one registry + six extension points**.

### 3.1 One registry, six extension points

Every pluggable thing is a **Contribution**: a typed object implementing one of a small, fixed set of **Extension Point** interfaces, registered under a string key in a single `PluginRegistry` singleton.

| # | Extension point | Interface (in `geosim/plugins/contracts.py`) | What it teaches the system |
|---|---|---|---|
| a | **Ingestion adapter** | `IngestionAdapter` | how to parse a native file format → normalized primitives |
| b | **Property type** | `PropertyType` (declarative data, not code) | that a new physical quantity exists, with its unit, colormap, scaling, display range |
| c | **Rock-physics transform** | `Transform` | how to derive new property fields from existing ones |
| d | **Forward model** | `ForwardModel` | how to simulate what a method *would* measure over a synthetic earth |
| e | **Renderer / transfer function** | `RendererSpec` (declarative; frontend implements) | how to draw a property/primitive in the 3-D viewer |
| f | **Inversion engine** | `InversionEngine` | how to invert observations into a property model (later phase) |

!!! note "Canonical keys, not invented ones"
    Plugins don't *mint* method or property strings. They register against the **canonical registries**: the `(method, submethod)` pairs (`backend/geosim/plugins/methods.py` — e.g. `gravity`, `mt`, `ert`, `seismic` with submethods like `reflection`/`refraction`) and the canonical property-type keys. A plugin *may* register a brand-new property type, but method/submethod values come from the canonical set — no rogue variants like `"seismic_reflection"`. Load-time validation quarantines violators.

### 3.2 Method bundles — the unit you actually create

The six points aren't independent features bolted on separately. A **survey method** (gravity, MT, ERT…) is the natural unit that bundles several of them at once. A **method bundle** is one Python package that declares a manifest and registers a coherent set: an adapter + property type(s) + a default transfer function + optionally a forward model + a transform — all for one canonical `(method, submethod)` pair.

```python
# the entire wiring of a brand-new method "SP" (spontaneous potential), illustrative
from geosim.plugins import register, PropertyType, RendererSpec, manifest
from .adapter import SPAdapter
from .forward import SPForwardModel

manifest("plugin.json")                       # load + validate the manifest

register.property_type(PropertyType(
    key="self_potential", canonical_unit="mV",
    default_colormap="RdBu", default_scaling="linear",
    display_range=(-200, 200),
))
register.adapter(SPAdapter)                    # how to ingest SP files (doc 03)
register.forward_model(SPForwardModel)         # optional — only if you want synthetic SP
register.renderer(RendererSpec(
    key="volume.raymarch", applies_to=["self_potential"],
    default_transfer_function=DIVERGING_TF,
))
```

After this, **with zero edits to core**: the synthetic generator can produce SP data, ingestion auto-routes SP files to `SPAdapter`, the fused grid accepts an SP volume with correct units/colour, and the viewer offers an SP layer with a sensible default transfer function. That is the R&D payoff. (See [§7 below](#7-how-to-add-a-new-survey-method-the-rd-payoff).)

### 3.3 Discovery: two channels, one registry

```python
# the stable registry surface (backend/geosim/plugins/registry.py, paraphrased)
class PluginRegistry:
    def adapter_for_format(self, fmt: str) -> IngestionAdapter | None: ...
    def property_type(self, key: str) -> PropertyType: ...
    def transforms(self) -> list[Transform]: ...
    def forward_model(self, method: str) -> ForwardModel | None: ...
    def inversion_engines(self) -> list[InversionEngine]: ...
    def renderer_specs(self) -> list[RendererSpec]: ...
    def capabilities(self) -> CapabilitiesDocument:    # the /api/capabilities payload
        ...
```

Two discovery channels converge on this one registry:

1. **First-party / built-in plugins** ship in the repo and register via **decorators** (`@register.adapter`, `register.property_type(...)`) — zero packaging ceremony.
2. **Third-party plugins** are installed Python distributions that advertise themselves via `importlib.metadata` **entry points** under the group `geosim.plugins`. At startup core enumerates the group and imports each module, which runs the same decorators.

Core code (ingestion service, fusion engine, serving layer) only ever talks to the `PluginRegistry` interface — it never imports a concrete plugin. That's what lets core evolve independently. The single stable surface a plugin may import is the `geosim.plugins` package (the six Protocols, the dataclasses, `register`/`manifest`); everything else in core is private.

### 3.4 Two orthogonal axes: trust vs process isolation

A subtle but important point: **security** and **process placement** are *separate* axes, and conflating them causes confusion.

- **Trust (security) — one global decision.** The tool is **local-first, single-user**, so plugins are **trusted code** at the app's own trust level, run **in-process by default**, no sandboxing. This matches the `pip` norm (SimPEG, lasio, ObsPy are all in-process). The future hosted/multi-user mode is the only thing that changes this — and the seam for it is isolated.
- **Process isolation (engineering) — per contribution.** Every contribution declares an `executionMode`:

| `executionMode` | For | Where it runs |
|---|---|---|
| `in_process` *(default)* | lightweight adapters/transforms/property types/forward models | the FastAPI process; arrays passed by reference (no serialization cost) |
| `worker_process` | heavy CPU jobs that shouldn't block the API | a separate Python worker (the RQ/Redis tier) |
| `container` | engines with conflicting/heavy native deps | a dedicated container image |
| `remote_worker` | heavy/conflicting engines (e.g. SimPEG/PyGIMLi MT/EM inversion) | a remote (possibly GPU) worker |

This is a **dependency/CPU** decision, *not* a security one — a `container` engine is still trusted; it just lives apart so its heavy deps don't poison the API venv. This is exactly why the heavy [inversion](inversion.md) solvers are *optional* extras and run out-of-process. In `backend/pyproject.toml` these appear as opt-in extras: `forward-t1` (harmonica/empymod/pykonal), `geomodel` (GemPy), `inversion`, `postgres` — the default install runs the whole core platform without any of them.

---

## 4. The job system

Long-running work (synthetic builds, fusion resampling, inversions) runs as **jobs**, not inline in the request. There is one `JobRunner` contract — `enqueue(kind, params, fn) -> job_id` plus a job-state model — with two interchangeable executors (`backend/geosim/jobs/`):

| Runner | When | Behaviour |
|---|---|---|
| `InlineJobRunner` *(default)* | the no-service local tier | runs the job **synchronously**, in-process — so the whole app works with no Redis. |
| `RQJobRunner` | the async tier | hands the job to an **RQ + Redis** worker (`make worker`); Redis is never required at import/test time. |

A job function receives a `ProgressReporter` (`report(...)` / `cancelled`) and pushes progress over a channel that a WebSocket endpoint streams to the browser. The catalog has a `jobs` table; the API exposes `GET /jobs/{jid}`, `POST /jobs/{jid}:cancel`, and `WS /jobs/{jid}/progress`.

---

## 5. The API surface & an end-to-end request

The FastAPI app is built by `create_app(settings)` in `backend/geosim/api/app.py`, which wires three foundation services behind dependency injection so the stack runs with **no Docker/Redis/Postgres**:

- **catalog** — a SQLAlchemy session factory (default: SQLite in-memory),
- **storage** — a `storage_root` whose per-project bulk-store tree (`arrays/ grids/ meshes/ vectors/ points/ raw/ cache/`) is materialized on project create,
- **jobs** — a `JobRunner` (default `InlineJobRunner`).

The inversion (SimPEG/discretize) and geomodel (GemPy) routers are imported **lazily** because they depend on heavy optional extras — the core must run on the default install without them.

### 5.1 Key endpoints

| Area | Endpoint(s) | Purpose |
|---|---|---|
| **Capabilities** | `GET /api/capabilities` | the single backend→frontend contract (§5.2) |
| **Projects** | `POST /projects`, `GET /projects`, `GET/PATCH/DELETE /projects/{pid}` | project CRUD; create materializes the dir + catalog rows + `SpatialFrame` |
| **Artifacts** | `GET /projects/{pid}/artifacts`, `…/features`, `…/time-extent` | list what a project holds; 4-D time extent |
| **Property models** | `GET /{pm_id}`, `…/volume`, `…/volume/meta`, `…/zarr/{path}`, `POST /{pm_id}/slice` | stream a volume (incl. raw Zarr bricks) and cut slices |
| **Fusion** | `POST /fused`, `…/{grid_id}/resample`, `…/sample`, `…/crossplot`, `…/cluster`, `…/transform`, `…/favorability`, `…/calibrate` | build the fused grid; resample layers onto it; cross-plot/cluster; run rock-physics transforms; compute favorability |
| **Features** | `GET /features/{fid}`, `…/geometry`, `…/points` | faults, horizons, microseismic clouds |
| **Wells / planning** | `POST /projects/{pid}/targets`, `…/wells`, `GET /wells/{wid}/trajectory|positions|export`, `POST /wells/{wid}/predict|solve` | drilling targets, well paths, predicted logs, export |
| **Inversion** | `GET /inversion-engines`, `POST /property-models:invert` | list engines; enqueue an inversion job (202 Accepted) |
| **Jobs** | `GET /jobs/{jid}`, `POST /jobs/{jid}:cancel`, `WS /jobs/{jid}/progress` | poll/cancel/stream long jobs |
| **Transforms** | `GET /transforms` | list registered rock-physics transforms |

### 5.2 `/api/capabilities` — the contract that makes the client method-agnostic

On startup the React app fetches **one** document derived straight from `PluginRegistry.capabilities()`. It lists the property types (key, unit, colormap, scaling, display range), the methods (formats, what they produce, whether they have a forward model), the renderers, the transforms, and the loaded plugins. The layer manager, colour-mapping UI, transfer-function editor, and method picker are all driven by this document — so adding a backend method *automatically* lights up the right UI with **no frontend edit**.

The client ships a **fixed catalog** of renderer implementations keyed by the same `renderer.key` (e.g. `volume.raymarch`, `wellpath.tube`); `/api/capabilities` only *selects among* the renderers the client already bundles, with a graceful fallback for unknown keys. No third-party JavaScript is loaded.

### 5.3 A request, end to end

```text
1. Build data     : make scenario        → synthgen writes scenarios/<id>/measured/ + truth/
2. Seed a project : dev_server.py         → ingestion adapters parse measured/ → primitives
                                            → catalog rows + Zarr/COG/glTF in storage/
3. Browser opens  : GET /api/capabilities → client learns the methods/properties/renderers
4. Load a layer   : GET /{pm_id}/volume   → streams multiresolution Zarr bricks
5. Render         : VolumeLayer ray-marches the Data3DTexture with the property's transfer fn
6. Fuse           : POST /fused, …/resample, …/favorability → derived volumes (as jobs)
7. Plan a well    : POST /projects/{pid}/targets, …/wells; GET …/predict → trajectory + log
```

Provenance is stamped automatically on every artifact (which plugin + version + contribution produced it), so every number is traceable — see [the data model](data-model.md) and [uncertainty](uncertainty.md).

---

## 6. How to run it

Everything is driven by the **Makefile** (run `make` or `make help` for the full list). The defaults select the **no-service embedded tier** — SQLite in-memory + temp storage + inline jobs — so you need neither Docker, Redis, nor Postgres to develop.

```bash
make setup            # create the backend venv (uv) + install core/dev/ingest/fusion deps;
                      # install frontend npm deps

# --- the two-terminal dev loop ---
make demo             # Terminal 1: PERSISTENT seeded API on :8000 (real data the viewer loads)
make run-frontend     # Terminal 2: Vite viewer on http://localhost:5173
                      #   then open the viewer; load the seeded project via the discovery panel

# --- synthetic data ---
make scenarios                          # list available scenarios
make scenario SCENARIO=great-basin-v1   # build the flagship into scenarios/<id>/

# --- quality gate ---
make test-all         # backend pytest + frontend unit tests
make check            # full CI gate: lint + typecheck + tests + frontend build

# --- optional heavier tiers ---
make install-backend-full   # heavy solvers: T1 forwards, GemPy, SimPEG/PyGIMLi, Postgres
make infra-up               # Postgres+PostGIS + Redis via docker compose (optional)
make worker                 # an RQ background-job worker (needs Redis)
make migrate                # apply Alembic catalog migrations
make docs                   # serve THIS docs site with live reload on :8001
```

`make run-backend` is the lighter ephemeral variant (in-memory SQLite, no seed data) — good for hitting `/api` and a mock UI. `make demo` is the one that gives the viewer real, persistent data.

---

## 7. How to add a new survey method (the R&D payoff)

This is the whole point of the architecture. To add a method "SP" (spontaneous potential), you create **one package** — no edits to ingestion, fusion, storage, or viewer code:

```text
backend/plugins/sp/            (or a pip-installable geosim_sp/ for third-party)
├── plugin.json                # the manifest: id, version, api_version, method, provides{}, execution_modes{}
├── __init__.py                # runs the @register decorators (see §3.2)
├── adapter.py                 # SPAdapter(IngestionAdapter)   → parse SP files (doc 03 / ingestion.md)
├── forward.py                 # SPForwardModel(ForwardModel)  → simulate SP over the synthetic earth (optional)
└── transfer.py                # the default RendererSpec / transfer function (doc 06 / visualization.md)
```

At load time the registry validates the manifest schema, API-version compatibility, interface conformance, that `(method, submethod)` is canonical, and property-type integrity (unit exists in the `pint` registry, no key clashes). A bad plugin is **quarantined** (logged, excluded) but never crashes the app; its status shows in the plugin-health view. A plugin may register *fewer* contributions — an ingest-only adapter for a real format with no forward model is a first-class `single-contribution` bundle.

Because everything routes through the one registry and the `/api/capabilities` contract, the method is then live everywhere: the synthetic generator can produce it, ingestion routes its files, the fused grid carries its property with correct units/colour, and the viewer offers it as a layer — all without a single core edit.

---

## Key takeaways

- It's a **monorepo**: a Python/FastAPI backend (`backend/geosim/*` — all geoscience) and a React/TS/Three.js frontend (`frontend/src/{scene,lib,ui}` + `store.ts`), talking over REST + WebSocket.
- The architecture is **method-agnostic** end to end: the survey-method and property lists are *not* hard-coded; they come from a **plugin registry** and reach the client through one **`/api/capabilities`** document.
- **One registry + six extension points** (adapter, property type, transform, forward model, renderer, inversion engine). A new method = **one plugin package + a manifest, no core changes** — this is the R&D payoff.
- **Trust and process-isolation are separate axes**: plugins are trusted and run `in_process` by default; heavy/conflicting engines opt into `worker_process`/`container`/`remote_worker` for dependency/CPU reasons, which is why inversion/T1 solvers are optional extras.
- Long work runs as **jobs** (`InlineJobRunner` default, `RQJobRunner` for the Redis tier) with WebSocket progress.
- The whole stack runs with **no Docker/Redis/Postgres** by default (SQLite in-memory + temp storage + inline jobs); `make demo` + `make run-frontend` is the dev loop.

## Where this lives in the code

| Concern | Path |
|---|---|
| FastAPI app factory + core endpoints | `backend/geosim/api/app.py` |
| API routers (fusion, features, planning, property models, inversion, geomodel) | `backend/geosim/api/*.py` |
| Plugin registry + six contracts + manifest + canonical methods | `backend/geosim/plugins/{registry,contracts,manifest,methods,register}.py` |
| Job runners (inline + RQ) | `backend/geosim/jobs/runner.py` |
| Storage layout (bulk stores) | `backend/geosim/storage/layout.py` |
| Catalog DB models + migrations | `backend/geosim/catalog/` |
| Seeded dev server | `backend/scripts/dev_server.py` |
| Frontend entry + scene + store | `frontend/src/{main.tsx,App.tsx,store.ts}`, `frontend/src/scene/Scene.tsx` |
| Typed REST client | `frontend/src/lib/api.ts` |
| Developer commands | `Makefile` |
