# 08 — Plugin Architecture & Extensibility Framework

> Parent: `OVERVIEW.md` §4 (plugin framework) and §10 row 8. This doc defines the
> *single* extensibility mechanism that ties the whole stack together: how new
> survey methods, property types, ingestion adapters, rock-physics transforms,
> forward models, renderers/transfer-functions, and (later) inversion engines
> register and compose **without core changes**.
>
> This is the R&D-platform requirement from the OVERVIEW made concrete. Sibling
> docs (03 ingestion, 05 synthetic, 06 viewer, 07 fusion, 10 inversion) own the
> *internals* of each extension type; this doc owns the *registration contract*
> they all conform to. Where a sibling must conform, it is flagged
> **[sibling-doc binds here]**.

---

## 1. The central idea: one registry, six extension points

Every pluggable thing in the system is a **Contribution**: a typed object that
implements one of a small, fixed set of **Extension Point** interfaces and is
registered under a string key. There is exactly **one registration mechanism**
(§3); the six extension points differ only in the interface they satisfy.

| # | Extension point | Interface (backend) | Owning doc | Phase |
|---|---|---|---|---|
| a | **Ingestion adapter** | `IngestionAdapter` | 03 | 1+ |
| b | **Property type** | `PropertyType` (declarative) | 01 §5, 02 | 0+ |
| c | **Rock-physics transform** | `Transform` | 07 | 3+ |
| d | **Forward model** | `ForwardModel` | 05 | 1+ |
| e | **Renderer / transfer function** | `RendererSpec` (declarative, frontend) | 06 | 1+ |
| f | **Inversion engine** | `InversionEngine` | 10 | 6 (later) |

**Key insight:** these six are not independent features bolted on separately —
a *survey method* (gravity, MT, ERT…) is the natural unit that bundles several
of them at once (§5). So the framework is built around two layers:

1. **Contributions** — the six atomic extension points above.
2. **Method bundles** — a cohesive package that registers a coherent set of
   contributions (an adapter + property type(s) + a default transfer function +
   optionally a forward model + transform) as one installable unit.

A new survey method is therefore: write one plugin package, declare a manifest,
register its contributions. No edit to core ingestion, fusion, storage, or viewer
code.

---

## 2. Trust model & execution model (decide this first)

The OVERVIEW scopes this as **local-first, single-user** (§Context, §5). That
fact dominates every plugin-security decision:

| Property | Decision | Rationale |
|---|---|---|
| **Trust boundary** | Plugins are **trusted code**, same trust level as the app itself. | Single user installs them deliberately on their own machine, exactly like `pip install`. There is no untrusted multi-tenant input. |
| **Execution** | **In-process**, in the FastAPI Python process (or its job workers). | Geoscience plugins need `numpy`/`xarray`/`segyio` arrays by reference; serializing volumes across a sandbox boundary would dominate runtime. |
| **Isolation** | **None by default.** Plugins can import anything and touch the filesystem. | Matches the `pip`/entry-point ecosystem norm (SimPEG, lasio, ObsPy are all in-process). |
| **Distribution** | Plugins are ordinary **Python packages** (built-in ones ship in-repo; third-party ones `pip install`). | Reuses Python packaging; no bespoke plugin format. |

> **This is a deliberate, documented trust choice, not an oversight.** It is the
> correct default for a local single-user R&D tool. It is also explicitly the
> thing that must change before any **hosted/multi-user** mode (OVERVIEW §Context
> "designed so it can grow"). The seam for that future is isolated to §11.

**What we *do* enforce even on trusted plugins** (cheap, catches bugs not
attacks): manifest schema validation, interface conformance checks, version
compatibility, and capability declaration — all at load time (§8). These make
plugins *predictable*, not *contained*.

---

## 3. The uniform registration mechanism

### 3.1 Discovery: entry points + in-repo registry, both feeding one registry

We use **two discovery channels that converge on the same registry**, chosen to
get the best of decorators (zero-config for first-party code) and entry points
(clean third-party install):

1. **First-party / built-in plugins** (everything that ships in the repo —
   gravity, MT, ERT, the synthetic forward models) live under
   `backend/plugins/<name>/` and are discovered by **importing the package**,
   where a **decorator** registers each contribution. Zero packaging ceremony
   during core development.

2. **Third-party plugins** are installed Python distributions that advertise
   themselves via **`importlib.metadata` entry points** under the group
   `geosim.plugins`. At startup the core enumerates that group and imports each
   advertised module (which runs the same decorators).

Both paths end at one **`PluginRegistry`** singleton. Decorator vs entry-point is
purely *how the module gets imported*; the registration call is identical.

```python
# backend/plugins/gravity/__init__.py  (a first-party method bundle)
from geosim.plugins import register, PropertyType, IngestionAdapter

@register.property_type
PROP_DENSITY = PropertyType(
    key="density",
    canonical_unit="kg/m3",        # must match doc 01 §5 registry
    default_colormap="viridis",
    default_scaling="linear",
    display_range=(1800, 3200),
)

@register.adapter
class GravityCSVAdapter(IngestionAdapter):
    method = "gravity"
    formats = ["csv", "grd", "netcdf"]
    def parse(self, raw, ctx): ...      # → NormalizedBundle  [doc 03 owns this]
```

```toml
# third-party plugin's pyproject.toml — the entry-point channel
[project.entry-points."geosim.plugins"]
my_seismic = "geosim_seismic_plus:plugin"   # module exposing a manifest + register() calls
```

> **Why not config-driven discovery (an explicit list in a config file)?**
> Rejected as the *primary* mechanism: it makes adding a method a two-step edit
> (write code *and* edit central config), which contradicts "no core changes."
> Config is retained only as an **override layer** — a project/user setting can
> *disable* a discovered plugin or *pin* a version, but never has to *enable* one.

### 3.2 The registry API (the stable core surface)

```python
class PluginRegistry:
    def adapters(self) -> dict[str, IngestionAdapter]
    def adapter_for_format(self, fmt: str) -> IngestionAdapter | None
    def property_type(self, key: str) -> PropertyType
    def transforms(self) -> list[Transform]
    def forward_model(self, method: str) -> ForwardModel | None
    def inversion_engines(self) -> list[InversionEngine]
    def renderer_specs(self) -> list[RendererSpec]      # serialized to frontend
    def manifest(self, plugin_id: str) -> PluginManifest
    def capabilities(self) -> CapabilitiesDocument      # the /capabilities payload (§7)
```

Core code (ingestion service, fusion engine, serving layer) only ever talks to
**this interface** — it never imports a concrete plugin. That is what lets core
evolve independently (§9).

---

## 4. The six extension-point contracts (at the seam, not the internals)

Each interface is defined here only to the depth the *registration* needs; the
behavioural contract is owned by the sibling doc.

### (a) Ingestion adapter — **[doc 03 binds here]**
```python
class IngestionAdapter(Protocol):
    method: str                 # "gravity", "mt", ...
    formats: list[str]          # native format keys it claims (OVERVIEW §3 table)
    def sniff(self, raw: RawFile) -> float          # 0..1 confidence it can parse this
    def parse(self, raw: RawFile, ctx: IngestContext) -> NormalizedBundle
# NormalizedBundle = { observations[], property_models[], features[], crs, units, provenance }
```
Doc 03 owns parsing rules, the per-method format table, and normalization. This
doc only fixes the signature and that it returns the OVERVIEW §3 normalized
primitive, tagged with provenance (§6).

### (b) Property type — declarative, **[doc 01 §5 + doc 02 bind here]**
The one extension point that is pure data, not code. It is the registry that doc
01 §5 calls "a property type registry (feeds doc 02 & 08)."
```python
PropertyType(
    key: str,                  # "resistivity", "density", "chargeability"
    canonical_unit: str,       # must exist in doc 01 pint registry
    default_colormap: str,
    default_scaling: "linear" | "log",
    display_range: tuple[float, float],
    description: str = "",
)
```
Registering a property type is what teaches the *whole stack* (units, storage
metadata, colour mapping, viewer defaults) how to handle a new physical quantity
— declared once, per OVERVIEW §1 spec.

### (c) Rock-physics transform — **[doc 07 binds here]**
```python
class Transform(Protocol):
    key: str
    inputs:  list[str]         # property-type keys it consumes
    outputs: list[str]         # property-type keys it produces (often new ones)
    def apply(self, fields: dict[str, Field], params: dict) -> dict[str, Field]
```
Doc 07 owns the maths, uncertainty propagation, and the fused-grid resampling
the transform runs on. A transform *may* register a new output property type
(e.g. `geothermal_favorability`) as part of its bundle.

### (d) Forward model — **[doc 05 binds here]**
```python
class ForwardModel(Protocol):
    method: str
    def simulate(self, earth: GroundTruthEarth, geom: AcquisitionGeometry,
                 noise: NoiseSpec) -> RawFile     # emits a native-format file (OVERVIEW §8)
```
Doc 05 owns the ground-truth earth spec and the physics. The plugin contract is
just: given the synthetic earth, emit a file the *same method's adapter* can
ingest — closing the OVERVIEW §8 round-trip.

### (e) Renderer / transfer function — declarative, **[doc 06 binds here]**
Backend-registered as a **serializable spec** (so the frontend can discover it
via `/capabilities`); the *implementation* is frontend code (§7.2).
```python
RendererSpec(
    key: str,                  # "volume.raymarch", "wellpath.tube", "microseismic.cloud"
    applies_to: list[str],     # property keys or primitive kinds it renders
    default_transfer_function: TransferFunction,   # opacity/colour ramp, isovalue defaults
    ui_panel: str | None = None,                   # optional custom React panel id
)
```
Doc 06 owns the Three.js scene graph and shaders. This doc fixes that a renderer
is *declared* on the backend and *resolved* to a React component on the client.

### (f) Inversion engine — **[doc 10 binds here, later]**
```python
class InversionEngine(Protocol):
    key: str                   # "simpeg.dc", "pygimli.ert", "simpeg.joint"
    methods: list[str]         # survey methods it can invert
    def invert(self, observations: list[Observation], mesh: Mesh,
               config: dict, job: JobHandle) -> PropertyModel
```
Phase 6. Listed now so the registry shape doesn't change later: inversion is
*just another contribution type*, run as a background job (OVERVIEW §5).

---

## 5. The Method Bundle — how one plugin packages a whole survey method

A survey method is the cohesive unit. A **method bundle** is one Python package
that declares a **manifest** and registers its contributions together. This is
the artifact a contributor actually creates to "add a new method."

### 5.1 Manifest

```jsonc
// plugin manifest — declared in code or as plugin.json; validated at load (§8)
PluginManifest {
  "id": "geosim.method.mt",          // globally unique, reverse-DNS-ish
  "name": "Magnetotellurics",
  "version": "1.2.0",                 // semver of THIS plugin
  "api_version": "1.x",              // core plugin-API contract it targets (§9)
  "kind": "method-bundle",            // or "single-contribution"
  "provides": {
    "adapters":        ["mt.edi", "mt.modem"],
    "property_types":  ["resistivity"],         // may reuse an existing one
    "transforms":      [],
    "forward_models":  ["mt"],
    "renderers":       ["volume.raymarch"],     // may reuse a core renderer
    "inversion_engines": []
  },
  "requires_property_types": ["resistivity"],   // capability negotiation (§7.3)
  "python_requires": ">=3.11",
  "dependencies": ["mtpy>=2.0"]                 // declared; installed via pip
}
```

### 5.2 Skeleton: adding a new method (worked example — "spontaneous potential")

Everything needed to add a brand-new method "SP" lives in one package:

```
backend/plugins/sp/                  # (or a pip-installable geosim_sp/ for 3rd-party)
├── plugin.json                      # the manifest above, for "sp"
├── __init__.py                      # runs the @register decorators
├── adapter.py                       # SPAdapter(IngestionAdapter)   → doc 03
├── forward.py                       # SPForwardModel(ForwardModel)  → doc 05  (optional)
└── transfer.py                      # default RendererSpec/transfer  → doc 06
```

```python
# backend/plugins/sp/__init__.py — the entire wiring of a new method
from geosim.plugins import register, PropertyType, RendererSpec, manifest
from .adapter import SPAdapter
from .forward import SPForwardModel

manifest("plugin.json")              # load + validate manifest

register.property_type(PropertyType(
    key="self_potential", canonical_unit="mV",
    default_colormap="RdBu", default_scaling="linear",
    display_range=(-200, 200),
))
register.adapter(SPAdapter)
register.forward_model(SPForwardModel)        # optional — only if synthetic gen wanted
register.renderer(RendererSpec(
    key="volume.raymarch", applies_to=["self_potential"],
    default_transfer_function=DIVERGING_TF,
))
```

After this, with **zero edits to core**: the synthetic generator can produce SP
data, ingestion auto-routes SP files to `SPAdapter`, the fused grid accepts an
SP volume with correct units/colour, and the viewer offers an SP layer with a
sensible default transfer function. That is the R&D requirement satisfied.

A plugin may register **fewer** contributions (e.g. an ingest-only adapter for a
real-world format with no forward model) — `kind: "single-contribution"` bundles
are first-class.

---

## 6. Provenance: which plugin/version produced an artifact — **[doc 02 binds here]**

Every artifact the system stores (observation, property model, derived volume,
synthetic file) records the contribution and plugin version that produced it, so
results are reproducible and auditable (OVERVIEW §2 "provenance links").

```jsonc
ProvenanceRecord {           // attached to every artifact; schema owned by doc 02
  "produced_by": {
    "plugin_id": "geosim.method.mt",
    "plugin_version": "1.2.0",
    "api_version": "1.x",
    "contribution": "adapter:mt.edi" | "transform:res_temp" | "forward:mt" | "inversion:simpeg.dc"
  },
  "inputs": ["<artifact-id>", ...],     // for transforms/inversions: lineage
  "params": { ... },                    // the exact config used
  "source_crs": "...", "source_unit": "..."   // ties into doc 01 §7
}
```

The registry stamps `produced_by` automatically when a contribution runs, so a
plugin author cannot forget it. Doc 02 owns the on-disk/catalog schema; this doc
fixes that the **plugin id + version + contribution key** are part of it.

---

## 7. Frontend extensibility

The frontend must learn what the backend can do **at runtime** — it cannot
hard-code the property/method list, or adding a method would mean a frontend
edit (violating "no core changes").

### 7.1 The `/capabilities` endpoint (backend → client)

On startup the React app fetches a single capabilities document derived from
`PluginRegistry.capabilities()`:

```jsonc
GET /api/capabilities  →
{
  "api_version": "1.x",
  "property_types": [
    { "key": "resistivity", "unit": "ohm.m", "colormap": "turbo",
      "scaling": "log", "display_range": [1, 10000] },
    { "key": "density", "unit": "kg/m3", "colormap": "viridis",
      "scaling": "linear", "display_range": [1800, 3200] }
  ],
  "methods": [
    { "id": "mt", "name": "Magnetotellurics", "formats": ["edi","modem"],
      "produces": ["resistivity"], "has_forward_model": true }
  ],
  "renderers": [
    { "key": "volume.raymarch", "applies_to": ["resistivity","density"],
      "default_transfer_function": { ... }, "ui_panel": null }
  ],
  "transforms": [
    { "key": "res_temp", "inputs": ["resistivity"], "outputs": ["temperature_likelihood"] }
  ],
  "plugins": [ { "id": "geosim.method.mt", "version": "1.2.0" } ]
}
```

This is the **single contract** that makes the frontend method-agnostic: the
layer manager, colour-mapping UI, transfer-function editor, and method picker are
all driven by this document. Property types declared once on the backend (doc 01
§5) flow straight to the UI — units, default colormap, log/linear, range — with
no client-side duplication.

### 7.2 Client-side renderer/panel registry

Renderers and custom panels have a *declarative* half (the `RendererSpec` from
`/capabilities`) and an *implementation* half (React/Three.js code). The client
keeps a parallel registry keyed by the same `renderer.key`:

```typescript
// frontend: a client renderer registry mirrors the backend renderer keys
registerRenderer("volume.raymarch", RayMarchVolume);     // built-in (doc 06)
registerRenderer("wellpath.tube",   WellPathTube);
registerPanel("crossplot",          CrossPlotPanel);

// resolution at runtime: capabilities.renderers drives which to mount
function rendererFor(spec: RendererSpec): React.FC {
  return clientRenderers[spec.key] ?? FallbackRenderer;   // graceful unknown-key fallback
}
```

A **third-party frontend plugin** (rarer; most methods reuse `volume.raymarch`)
ships as an ES module that calls `registerRenderer`/`registerPanel` and is loaded
when its `ui_panel`/renderer key appears in capabilities. If a backend renderer
key has no client implementation, the client uses `FallbackRenderer` and surfaces
a warning — capability negotiation (§7.3) rather than a crash.

### 7.3 Capability negotiation

- Backend advertises `api_version`; client checks compatibility (§9) and warns on
  mismatch.
- Client renders only the renderers/panels it actually implements; unknown keys
  degrade gracefully.
- A method that `requires_property_types` not present is disabled with an
  explanatory message rather than failing silently.

---

## 8. Validation at load time

Even for trusted plugins, the registry validates on load — to catch *bugs*, fail
fast, and keep the system predictable. A failing plugin is **quarantined**
(logged, excluded from the registry) but does not crash the app.

| Check | What it verifies | On failure |
|---|---|---|
| **Manifest schema** | Manifest parses against the `PluginManifest` JSON-Schema | quarantine + error |
| **API-version compat** | `manifest.api_version` satisfies core's supported range (§9) | quarantine + error |
| **Interface conformance** | Each contribution implements its Protocol (signatures, required attrs) | quarantine + error |
| **Property-type integrity** | `canonical_unit` exists in the doc 01 `pint` registry; no key/unit clash with an existing property type | quarantine + error |
| **Key uniqueness** | No two adapters claim the same format with equal `sniff` confidence; no duplicate contribution keys | warn / deterministic tie-break |
| **Dependency presence** | Declared deps importable | quarantine + actionable message |

Validation results are exposed at `/api/plugins` (status per plugin) so the UI
can show a plugin-health panel. A `geosim plugins validate` CLI runs the same
checks offline for plugin authors.

---

## 9. Versioning & a stable plugin API surface

Two independent version axes:

1. **Plugin version** (`manifest.version`) — semver of the individual plugin;
   recorded in provenance (§6) so an artifact is tied to the exact code.
2. **Plugin API version** (`manifest.api_version`) — semver of the *core
   contract* (the interfaces in §4 + the registry API in §3.2 + manifest schema).
   This is the stability promise that lets core evolve without breaking plugins.

**Stable surface (the only thing plugins may import):** the `geosim.plugins`
package — the six Protocols, `PropertyType`/`RendererSpec`/`Transform` dataclasses,
the `register`/`manifest` helpers, and the `NormalizedBundle`/`Field`/`RawFile`
DTOs. Everything else in core (`geosim.core.*`, services, storage internals) is
**private** and may change freely.

```python
# everything a plugin is allowed to depend on lives behind one import
from geosim.plugins import (
    register, manifest,
    IngestionAdapter, Transform, ForwardModel, InversionEngine,
    PropertyType, RendererSpec, TransferFunction,
    NormalizedBundle, Field, RawFile, IngestContext, JobHandle,
)
```

**Compatibility policy:** core declares a supported `api_version` *range*; a
plugin targeting `1.x` runs on any core `1.*`. Breaking changes bump to `2.x`,
and core may run a compatibility shim for the previous major for one release.
Additive changes (new optional manifest fields, new extension-point methods with
defaults) stay within a major version.

---

## 10. End-to-end: registration → use (sequence)

```
startup ──▶ enumerate entry-points (geosim.plugins) + import backend/plugins/*
        ──▶ each module runs @register / manifest()  ──▶ PluginRegistry
        ──▶ validate every contribution (§8)         ──▶ quarantine failures
                                                       │
run time:                                              ▼
  ingest file   ──▶ registry.adapter_for_format()  ──▶ adapter.parse() ──▶ stamp provenance (§6)
  synth data    ──▶ registry.forward_model(method) ──▶ simulate() ──▶ file ──▶ same adapter
  fusion L3     ──▶ registry.transforms()          ──▶ transform.apply() ──▶ derived volume + provenance
  viewer load   ──▶ GET /api/capabilities          ──▶ client mounts renderers by key
  inversion(6)  ──▶ registry.inversion_engines()   ──▶ engine.invert() as job
```

---

## 11. The hosted/multi-user seam (future, isolated here)

When the tool grows beyond local single-user (OVERVIEW §Context), the trust model
of §2 must change — and *only* §2 changes, because everything routes through the
registry:

- **Plugin signing / allow-list** — install only vetted/signed plugins.
- **Out-of-process execution** — run untrusted contributions in a subprocess or
  container with a serialized array boundary (the `Field`/`RawFile` DTOs are
  already the only thing crossing the boundary, by design of §9).
- **Resource limits & quotas** per plugin job.

No interface in §4, no manifest field, and no provenance/capabilities contract
needs to change to add this — that is the payoff of routing all extensibility
through one registry.

---

## Decisions locked in

1. **One registry, six extension points** (adapter, property type, transform,
   forward model, renderer, inversion engine). A new survey method = one plugin
   package + manifest; **no core changes**.
2. **In-process, trusted execution** for the local-first single-user tool.
   Plugins are ordinary Python packages at the app's own trust level. The
   hosted/multi-user sandbox is a future change isolated to §2/§11.
3. **Dual discovery → one registry:** first-party plugins use **decorators**
   (zero ceremony); third-party plugins use **`importlib.metadata` entry points**
   (group `geosim.plugins`). Config is an *override/disable* layer only, never the
   primary enable path.
4. **Method Bundle** is the cohesive unit: one package bundles adapter +
   property type(s) + default transfer function + optional forward model +
   optional transform, declared in one **manifest** (validated at load).
5. **`/api/capabilities`** is the single backend→frontend contract; the React app
   is method-agnostic and driven entirely by it. Renderers are declared on the
   backend and resolved to React components by key on the client, with graceful
   fallback for unknown keys.
6. **Two version axes:** plugin version (in provenance, per artifact) and plugin
   **API version** (the stability promise). Plugins may import **only**
   `geosim.plugins`; all other core is private and free to evolve.
7. **Load-time validation** (manifest schema, interface conformance, property-type
   integrity, API-version compat) quarantines bad plugins without crashing core.
8. **Provenance stamps plugin id + version + contribution key** on every artifact
   automatically (binds doc 02).

### Cross-doc bindings (siblings must conform to these seams)
- **Doc 03** — `IngestionAdapter.parse → NormalizedBundle`; `sniff` for format routing.
- **Doc 01 §5 / Doc 02** — the `PropertyType` declarative registry; canonical units come from the doc 01 `pint` registry.
- **Doc 07** — `Transform` (inputs/outputs as property-type keys); may register new output property types.
- **Doc 05** — `ForwardModel.simulate` must emit a file the *same method's* adapter ingests (closes the §8 round-trip).
- **Doc 06** — backend `RendererSpec` ↔ client renderer registry, matched by `key`.
- **Doc 10** — `InversionEngine` as a registered contribution run as a job (Phase 6).

---

## Open questions for you

1. **Backend plugin discovery mechanism — confirm the decorator + entry-point
   hybrid.** *Why it matters:* it sets the developer ergonomics for every new
   method and whether third-party plugins are first-class. Options: (a) **hybrid
   — decorators for first-party, entry-points for third-party** *(recommended
   default; best ergonomics + clean install)*; (b) entry-points only (uniform but
   heavier for in-repo dev); (c) config-driven manifest list (explicit but
   contradicts "no core changes").

2. **In-process vs sandboxed execution & trust model.** *Why it matters:* it
   fixes the security posture and how hard the future hosted mode is. Options:
   (a) **in-process, trusted, no isolation** *(recommended default for local
   single-user; matches the SimPEG/lasio ecosystem and avoids array-serialization
   cost)*; (b) in-process but with a capability allow-list/signing from day one
   (more ceremony, little benefit while single-user); (c) out-of-process sandbox
   now (premature; large perf + complexity cost).

3. **How much frontend code may a plugin ship?** *Why it matters:* decides
   whether the frontend stays fully method-agnostic (capabilities-driven only) or
   becomes a true plugin host that loads third-party JS. Options: (a) **backend
   declares renderers/transfer-functions; client implements a fixed catalog and
   reuses them by key — no third-party JS for now** *(recommended default;
   simplest, covers ~all methods via `volume.raymarch` + standard panels)*;
   (b) allow dynamically-loaded ES-module frontend plugins (maximally extensible,
   but adds a JS plugin loader + trust questions on the client).
