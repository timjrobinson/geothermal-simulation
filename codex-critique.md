# Codex Critique: Geothermal Simulator Design Docs

This is a strong architecture concept, but the docs currently read like a set of partially reconciled drafts rather than an implementation contract. The biggest issue is not that individual sections are weak; it is that many sections contain old decisions, open questions, and incompatible schema/API assumptions that directly contradict the later decision log. If implementation started from these docs as-is, different developers would build different systems.

The core design should be kept, but it needs a hard consolidation pass. Delete stale prose. Collapse duplicate contracts. Make the data model, storage layout, API, plugin contract, and phasing agree in one direction.

## Highest-Risk Problems

### 1. The docs are not a single source of truth

`DECISIONS.md` says it is authoritative, but several detailed docs still contain the overridden design as active prose. The revision banners help, but they do not fix the problem because the body still says the opposite.

Examples:

- `04-storage-and-serving.md` says at the top that PostgreSQL + PostGIS and RQ + Redis are the decisions, but section 1/2 still presents SQLite + SpatiaLite as the default, and section 9.4 still says FastAPI `BackgroundTasks` are the default.
- `05-synthetic-data-generator.md` says in the revision banner that rigorous T1 physics should be first for MT/gravity/seismic, but section 6, decisions locked in, and open questions all still recommend T0 plausible for all methods first.
- `07-fusion-and-rock-physics.md` says the full rock-physics table and weighted+fuzzy favorability are decided, but its open questions still ask which transform set and favorability method to choose.
- `09-drilling-well-planning.md` says WITSML trajectory export is in scope, but the export priority table marks WITSML as P1 and the open questions still ask whether it matters.
- `03-ingestion-adapters.md` still says partial failure threshold is `>X%`, while `DECISIONS.md` says `>10%`.
- `03-ingestion-adapters.md` still references `geosim.adapters` entry points, while `08-plugin-architecture.md` standardizes on `geosim.plugins`.

Recommendation: delete the overridden paragraphs, not just annotate them. Keep historical rationale in `DECISIONS.md` only. In the design docs, every "Open questions for you" section that is already resolved should be removed or renamed "Historical decisions already resolved".

### 2. Scope is dangerously overstuffed for a greenfield MVP

The design says "integration & visualization platform" and "MVP consumes already-processed/inverted models", but the docs keep pulling in heavy scientific and operational commitments:

- rigorous MT, gravity, and seismic forward models;
- full rock-physics library including Waxman-Smits, dual-water, permeability proxies, fracture density, fuzzy favorability, uncertainty propagation;
- WebGL2 plus WebGPU abstraction;
- Postgres + Redis from day one;
- plugin registry, entry points, method bundles, capabilities endpoint;
- Zarr v3 sharding, OME-Zarr multiscales, browser Blosc/zstd decode;
- WITSML export;
- future remote/GPU inversion workers.

This is too much to treat as one build. The roadmap notices this and recommends a lean integration spine first, but the decisions and detailed docs undercut that recommendation.

Recommendation: split "platform spine" from "science depth" explicitly:

- Spine: local project, one synthetic property volume, one real Zarr layout, one ingest path, one viewer, one slice endpoint, one provenance path.
- Science packs: synthetic forward models, rock physics, inversion, well planning, and advanced uncertainty as separately versioned milestones.

If this is not done, the project will likely spend months building framework and infrastructure before proving the core claim: many subsurface datasets, one 3D earth.

### 3. Data model and storage schema do not actually match

`02-data-model.md` defines `Dataset` as a rich envelope with `extent`, `spatialFrameId`, `originCrs`, `provenanceId`, `version`, `tags`, `createdBy`, and an inline typed `payload`. `04-storage-and-serving.md` then creates normalized SQL tables where `datasets` lacks many of those fields, and where typed objects live in `property_models`, `observations`, and `features`.

This can be fine, but the docs do not state the mapping cleanly enough. Right now it is unclear which contract wins when the JSON schema and SQL schema differ.

Specific gaps:

- `datasets` in doc 04 lacks `extent`, `spatial_frame_id`, `origin_crs`, `provenance_id`, `version`, `tags`, and `created_by`.
- Doc 02 says every dataset has exactly one `Provenance`; doc 04's provenance table targets any artifact, with no required FK from `datasets`.
- Doc 02 treats `FusedEarthModel` as a payload kind; doc 04 says fused volumes get both a `datasets.kind='fusedModel'` row and a `property_models` row. That may be correct, but it is not explicitly modeled.
- Doc 02 says a project may hold multiple fused models with layers; doc 04 has no table for fused-model layer membership or `FusedLayer` metadata.

Recommendation: add a "Logical-to-Physical Mapping" table to doc 04 and make it exhaustive. Every required field in doc 02 must be either a column, a JSONB path, or explicitly derived. If doc 04 is the implementation schema, doc 02 should stop implying a JSONB-document-per-record catalog.

### 4. "Self-contained project directory" conflicts with PostgreSQL from the start

The docs repeatedly say a project is a copyable/zippable directory containing `catalog.sqlite`. But the decision log says PostgreSQL + PostGIS from the start.

Those are different product behaviors:

- SQLite project: copy one folder and open it elsewhere.
- PostgreSQL project: copy arrays/raw/cache plus `pg_dump`, then restore into a running database.

The docs try to keep both by saying "project-as-directory plus pg_dump", but that is not the same local-first UX. It also complicates backups, project sharing, tests, and onboarding.

Recommendation: pick one default UX and make the other a secondary mode. If PostgreSQL is truly the default, delete `catalog.sqlite` from the authoritative project layout and define export/import as a first-class operation. If local-first portability matters more, reconsider using PostgreSQL before the hosted phase.

### 5. The Zarr layout is contradictory

Doc 02 says a property-model Zarr group looks like:

```text
<datasetId>.zarr/
  resistivity/
  resistivity_sigma/
  _pyramid/1/2/...
```

Doc 04 says:

```text
pm_<id>.zarr/
  0/
  1/
  <property>_sigma/
```

Those are incompatible object paths. The viewer, API, and storage writer cannot all follow both.

Other Zarr issues:

- Doc 02 says "Zarr coordinate arrays per CF conventions OR implied by origin+spacing"; this is too loose for a browser reader. Pick one minimum required convention.
- Doc 04 assumes browser Blosc(zstd) decode for Zarr v3. That is a risky dependency and should be verified early. If JS Zarr v3 + Blosc support is not mature enough, this becomes a hidden platform blocker.
- "Chunk == GPU brick == Zarr chunk path" is good, but the layout must be frozen before frontend work begins.

Recommendation: create one authoritative Zarr group spec with exact paths for value arrays, sigma arrays, DOI/mask arrays, pyramids, attrs, coordinate metadata, chunk keys, and compression. Everything else should link to it.

### 6. Support geometry vocabulary is still inconsistent

Doc 02 says `PropertyModel.support.kind` is only `volume | grid2d | mesh`. Doc 03 still says 2D profiles/sections are stored with `support = section`, and well logs with `support = well_path`.

This is more than naming. ERT, seismic 2D, and vertical curtain sections are neither ordinary 2D rasters nor full 3D volumes. If they are represented as `mesh`, the docs need to say exactly how: vertices, cells, value location, line geometry, pseudo-depth/elevation convention, and resampling behavior.

Other geometry mismatches:

- `Grid2DSupport` cannot cleanly represent a vertical ERT curtain unless `zLevel` can be an arbitrary embedded surface/curtain, which it currently cannot.
- InSAR time series are described as COG/GeoTIFF time-series in storage, but doc 02's time axis implies array dimensions. A folder of COGs, a multi-band COG, and a Zarr array are different contracts.
- `ObservationSet.geometryKind="profile2d"` and `PropertyModel.support.kind="mesh"` need an explicit conversion path for inverted 2D sections.

Recommendation: either add `SectionSupport` as a first-class support type or mandate `MeshSupport` for all 2D-in-3D sections with a precise schema. Do not leave "section" as an old name in ingestion.

### 7. Method names are not normalized

The method taxonomy differs across docs:

- Overview has "seismic reflection/refraction" in one row.
- Doc 02 `MethodKey` has only `"seismic"`.
- Doc 03 uses `seismic_reflection` and `seismic_refraction`.
- Doc 10 uses "DC resistivity", "ERT", "EM", "AEM", "TDEM/FDEM" with different groupings.

This will break plugin registration, routing, dataset filtering, and capabilities.

Recommendation: create one canonical `MethodKey` registry in doc 08 or doc 02, and require every doc to use it. If reflection and refraction are separate adapters, they need separate keys everywhere. If they are subtypes under `seismic`, define `method="seismic"` plus `submethod`.

### 8. Provenance is overpromising reversibility

The docs are right to log every CRS/unit/datum transform. But they sometimes imply that conversions are reversible and auditable in a broad sense.

Be careful:

- Unit conversion and CRS transformation can be reversible enough if parameters are recorded.
- Vertical datum conversion may depend on geoid grids and library versions.
- Gridding, interpolation, downsampling, clipping, filtering, inversion, and synthetic forward modeling are not reversible.
- Storing only params is not enough for bit reproduction unless dependency versions and algorithm versions are pinned.

Recommendation: split provenance into `reversibleTransforms` and `irreversibleDerivations`. Do not use "reversible" as a blanket claim. For irreversible steps, require enough metadata for repeatability, not inversion.

### 9. The local-to-georeferenced promotion story is too clean

The Engineering Frame plus optional geo-anchor is a good design. But the docs overstate "georeference later with zero reprocessing" as if this always makes the dataset physically meaningful.

It works for assigning a rigid transform to local coordinates. It does not solve:

- local datasets authored against a synthetic or flat surface then anchored to a real DEM;
- surveys with unknown scale, rotation, vertical datum, or coordinate handedness;
- basin-scale projects where projection distortion is not a pure rigid transform;
- depth-below-surface data when the surface model changes after anchoring.

Recommendation: distinguish "assigning an anchor" from "validating georeferencing". Add a georeferencing quality/status field: `unknown`, `assumed_local`, `anchored`, `validated`, `survey_controlled`. Do not let the UI imply that local synthetic data becomes real just by setting an anchor.

## Science and Interpretation Issues

### 10. The rock-physics layer risks false precision

The transform library is ambitious, but many proposed transforms are non-unique and site-dependent. `resistivity -> temperature` through Archie + Arps is especially risky unless porosity, salinity, clay content, saturation, and alteration are independently constrained.

In the synthetic world, those fields exist because the generator authored them. In real projects, they are often unknown, inferred, or circularly derived from the same geophysical data. That means the platform could produce a polished temperature or favorability volume that looks more certain than the evidence supports.

Recommendation:

- Rename early transforms from deterministic outputs to likelihood/proxy outputs unless calibrated by wells.
- Require every transform to declare assumptions and calibration status.
- Put well-log calibration at the center of the transform workflow, not as a secondary cross-plot feature.
- Make default favorability output carry an "evidence overlap" and "assumption burden" indicator.

### 11. Weighted favorability can encode bad geothermal logic

Weighted linear favorability is transparent, but it is compensatory. A high temperature score can offset absent permeability or absent fluid. That contradicts the geothermal requirement that heat, fluid, and permeability co-exist.

The docs acknowledge this and add fuzzy logic, but the decisions still say "weighted-linear default" in places.

Recommendation: default the geothermal play score to fuzzy conjunction for required evidence, with weighted linear only as an exploratory mode. If weighted linear remains default, the UI must make missing required evidence visually obvious.

### 12. Uncertainty model is useful but too narrow

Per-cell sigma is a practical rendering/fusion layer, but it is not sufficient to represent inversion uncertainty, spatial correlation, resolution, or model non-uniqueness. The docs know this, but some downstream math treats sigma as if cells and properties are independent.

Risks:

- Delta-method propagation assumes local linearity and mostly independent inputs.
- Favorability confidence may look quantitative even when dominated by unknown parameters.
- A smooth gravity/MT inversion can have low per-cell variance but poor resolving power.

Recommendation: keep sigma, but add a required `uncertaintyQuality` / `uncertaintyTier` and a separate `resolutionSupport` concept. Make confidence displays qualitative when inputs are proxy-level.

### 13. Synthetic generator is trying to be both demo engine and physics benchmark

The synthetic generator has two incompatible jobs:

- provide cheap, broad, native-looking data for an early end-to-end demo;
- provide rigorous physics for validating fusion and inversion.

The docs currently oscillate between those jobs. T1 MT/seismic can easily become a research project by itself. Native-format generation for every method also risks wasting time on file writers before the ingestion/viewer pipeline is proven.

Recommendation:

- Keep `unit-cube-v1` and a simple `great-basin-lite` as the first demo sources.
- Emit a minimal set of formats first: CSV, LAS, GeoTIFF/COG, simple EDI, simple SEG-Y only if needed.
- Treat T1 rigorous MT/seismic as validation packs after the platform can already ingest and render T0.
- Be explicit that T0 synthetic data is not suitable for validating inversion physics.

### 14. The "one geology -> all properties" idea is good, but the docs need a calibration path

Synthetic truth derives all properties from lithology/state, which is correct. Real projects will not have truth fields for alteration, fracture density, salinity, or porosity.

Recommendation: add a real-data calibration workflow:

1. Ingest well logs and lab/core/geochemistry.
2. Estimate site-specific rock-physics parameters.
3. Run transforms with calibrated parameter distributions.
4. Mark uncalibrated transforms as proxy outputs.

Without this, the synthetic workflow and real workflow will diverge sharply.

## Architecture and Implementation Issues

### 15. Plugin trust model conflicts with inversion dependency isolation

Doc 08 says plugins are trusted, in-process, no isolation by default. Doc 10 says inversion engines should run in isolated environments/subprocesses because SimPEG/PyGIMLi/GemPy dependencies are heavy and conflicting.

Both are reasonable for different contribution types. They should not be forced into one execution model.

Recommendation: add `executionMode` to plugin contributions:

- `in_process` for lightweight trusted adapters/transforms;
- `worker_process` for heavy CPU jobs;
- `container` or `remote_worker` for inversion/MT/EM later.

Then doc 08 and doc 10 agree instead of contradicting each other.

### 16. Frontend plugin story is half-enabled and half-forbidden

`DECISIONS.md` says no third-party JS for now: backend declares renderers/colormaps and the client has a fixed renderer catalog. Doc 08 still describes dynamically loaded ES modules for third-party frontend plugins.

Recommendation: delete or clearly defer dynamic frontend plugins. For now, `/api/capabilities` should only select from renderers already shipped in the frontend.

### 17. Browser volume rendering plan is plausible but underspecified

The WebGL2 brick-pool/page-table plan is advanced. It can work, but it is not trivial:

- WebGL2 lacks compute shaders.
- Texture size limits vary widely.
- Trilinear filtering across brick boundaries needs ghost voxels or seam handling.
- A 3D atlas plus page table needs careful coordinate math and fallback when bricks are missing.
- Browser-side Zarr v3 + Blosc/zstd decode must be proven early.

Recommendation: make the first viewer milestone simpler: one resident 3D texture at a modest size, then add brick streaming. Do not make virtual texturing part of the first proof unless you want rendering infrastructure to dominate the build.

### 18. API sketches are inconsistent

Examples:

- Doc 04 defines `POST /property-models/{id}/slice`; doc 06 sometimes says `GET /slice`.
- Doc 04 `SampleRequest` supports points or a straight line, but doc 09 needs batched sampling along a curved well path/deviation survey.
- Doc 07 defines `/fused/{gridId}/sample`, while doc 04 defines `/projects/{pid}/sample`.
- Doc 10 defines `POST /jobs/inversion`, while doc 04 job API uses resource-specific endpoints returning job ids.

Recommendation: create one API contract doc or OpenAPI stub and make the design docs descriptive, not independently authoritative. At minimum, centralize sample, slice, job, and artifact URLs in doc 04.

### 19. Observation errors are required later but not actually modeled now

Doc 10 says inversion data weights come from per-datum observation errors in doc 02. Doc 02's `ObservationSet` has column specs and methodData, but no required convention for per-observation uncertainty/error columns.

Recommendation: add a standard observation-error convention now:

- `value_sigma` columns for scalar observations;
- covariance/error model references for tensors/traces;
- units and property type for error columns;
- a rule for default noise floors when absent.

Otherwise inversion will bolt on a parallel uncertainty schema later.

### 20. Categorical/lithology fields violate the current "one property per model" simplification

`DECISIONS.md` says categorical/lithology fields are `PropertyModel` with a class-probability axis. Doc 02's core array convention is `(t,z,y,x)` and one property per model. A class axis is not defined in the dimension convention.

Recommendation: explicitly define categorical array shapes:

- hard labels: `(z,y,x)` integer with a category table;
- probabilities: `(class,z,y,x)` or `(t,class,z,y,x)`;
- uncertainty: entropy/confidence or class probabilities, not a separate sigma with the same semantics as continuous properties.

### 21. Temperature as canonical `degC` needs care

Using Celsius for display is fine. Using Celsius as a canonical internal unit can create subtle errors with `pint` because absolute temperatures and temperature deltas are different quantities.

Recommendation: either store thermodynamic temperature internally as kelvin and display Celsius, or define strict rules: `degC` for absolute temperatures, `delta_degC`/`K` for gradients and uncertainty. Temperature sigma should not be plain `degC` if the unit library treats it as absolute.

### 22. Chargeability is overloaded

Doc 01 lists chargeability as `mV/V (or ms)`. Those are not interchangeable; frequency-domain IP and time-domain IP can use different chargeability definitions.

Recommendation: split property types or add measurement subtype:

- `chargeability_mv_v`
- `chargeability_time_ms`
- possibly `phase_mrad`

Do not let one canonical unit hide method differences.

### 23. Derived volumes and property models are conflated in storage

Doc 04 says a fused model "gets a `property_models` row (it is a property model in catalog terms)." But doc 02 defines `FusedEarthModel` as a container grid with layers, not just one property.

Recommendation: keep three concepts separate:

- `PropertyModel`: one physical/derived property on a support.
- `FusedEarthModel`: a shared grid/container.
- `FusedLayer`: one source property resampled onto that grid.

A favorability volume is a `PropertyModel`. A fused grid with many layers is not itself just a `PropertyModel`.

### 24. The storage docs overuse "content-addressed" without defining reference counting

Several sections say unchanged arrays are shared by sha256 and old versions are cheap. But Zarr groups are directories with many chunk objects and metadata. It is unclear whether content addressing is per file, per chunk, per array, or per artifact manifest.

Recommendation: define content addressing precisely:

- raw files: whole-file sha256;
- Zarr chunks: immutable object ETags maybe content hashes;
- artifact versions: manifest hash over metadata plus chunk references;
- garbage collection: mark-and-sweep from catalog references.

Otherwise "keep all versions cheaply" may be false.

## Drilling and Planning Issues

### 25. The "crude drillability flag" is decided but not specified

The decision log says core includes a crude drillability flag. Doc 09 mostly specifies DLS constraints, but not a real drillability heuristic beyond DLS exceedance.

Recommendation: either define the crude flag concretely or remove the claim. A minimal version could combine max DLS, build rate, total MD/TVD ratio, hole inclination, and lithology hardness proxy. Keep it explicitly non-engineering-grade.

### 26. Fault proximity is treated as risk without context

Fault proximity can be good for permeability and bad for drilling/losses/seismicity. The current risk formula always increases risk near faults, while favorability may also increase near faults. That is not wrong, but it needs clearer separation:

- productivity opportunity;
- drilling hazard;
- induced-seismicity hazard;
- uncertainty from structural interpretation.

Recommendation: do not collapse all fault proximity into one risk scalar. Show separate channels and let the composite score be configured by use case.

### 27. WITSML export is under-scoped

WITSML trajectory export is non-trivial: schema version, coordinate reference systems, measured depth datum, units, station types, metadata, and validation all matter.

Recommendation: if WITSML is genuinely in scope, add a small conformance target: exact WITSML version, minimum required fields, validation tool/library, and one round-trip test. If not, leave it as P1/P2 and stop calling it part of the supported set.

## Things That Should Be Deleted or Moved

Delete from implementation-facing docs:

- Resolved open-question sections in docs 03, 05, 07, 08, 09, 10.
- Old SQLite-default and BackgroundTasks-default prose in doc 04.
- Old T0-first decisions in doc 05 if T1-first is truly decided, or the T1-first revision if the roadmap wins.
- `support = section` / `support = well_path` wording in doc 03 unless those become real support kinds.
- `geosim.adapters` entry-point examples in doc 03.
- Dynamic third-party frontend ES module loading in doc 08 unless it is actually in scope.
- Duplicate API endpoint sketches outside doc 04, or mark them explicitly non-authoritative.

Move to appendix or later-phase notes:

- WebGPU compute path details.
- Sparse octree future path.
- Full joint inversion details.
- Full WITSML/Compass export details.
- Full rock-physics table beyond the first calibrated transforms.
- Rigorous MT/seismic synthetic forward modeling.

## What I Would Keep

The design has several good foundations:

- Engineering Frame as the internal coordinate system.
- Observations vs PropertyModels vs GeologicalFeatures.
- Raw files immutable with provenance.
- Native-resolution originals preserved; fused grid is derived.
- Property-type registry for units/display/interpolation defaults.
- Synthetic generator as an end-to-end validation source.
- Browser viewer as the core product experience.
- Inversion output re-entering as ordinary `PropertyModel`.

Those are the spine. The problem is the amount of unresolved and duplicated machinery wrapped around them.

## Recommended Consolidation Pass

1. Freeze canonical vocabularies: `MethodKey`, `PropertyTypeKey`, `support.kind`, `geometryKind`, `featureKind`, job states, artifact kinds.
2. Make doc 02 the logical schema only and doc 04 the physical schema only. Add exact field mapping between them.
3. Pick one Zarr layout and delete the other.
4. Pick one async/database default and delete stale alternatives from the main path.
5. Reduce the first milestone to one property volume, one storage layout, one API path, one viewer path.
6. Reclassify ambitious science features as plugins/milestones, not MVP requirements.
7. Add calibration status and assumption tracking to rock-physics outputs before they are shown as targetable volumes.
8. Turn all remaining "flags to other docs" into actual resolved contracts or tracked TODOs.

## Bottom Line

The architecture is promising, but the docs currently describe several overlapping systems: a local-first desktop-style project, a Postgres-backed app, a browser volume-rendering engine, a synthetic physics lab, a plugin platform, a rock-physics workbench, an inversion orchestration system, and a drilling planner. Those can eventually be one product, but they cannot all be the MVP contract.

The most important deletion is stale resolved prose. The most important design correction is to make the data/storage/API contracts exact. The most important scope correction is to build the integration spine before the rigorous science stack.
