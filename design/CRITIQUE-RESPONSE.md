# Response to the Codex Critique

Tracks every numbered point in `codex-critique.md`: what was **fixed**, what was
**kept as-is** (with reasoning), and where the fix lives. The critique was strong and
largely correct — its core thesis (stale prose, inexact contracts, spine-before-science)
drove this pass.

## Fixed

| # | Issue | Fix & location |
|---|---|---|
| 1 | Stale prose contradicting decisions | Deleted/converted overridden body prose & resolved "Open questions" across docs 03,04,05,06,07,08,09,10. `>X%`→`>10%` (03); `geosim.adapters`→`geosim.plugins` (03). |
| 3 | Doc 02 ↔ doc 04 schema mismatch | Added a **Logical→Physical mapping table** + missing `datasets` columns (extent, spatial_frame_id, origin_crs, submethod, provenance_id NOT NULL, version, tags, created_by) in doc 04 §2. |
| 4 | "Copyable folder" vs Postgres | Removed `catalog.sqlite` from the authoritative layout; added a first-class **Export/Import** bundle (array stores + `pg_dump`) in doc 04 §3.1. |
| 5 | Contradictory Zarr layout | Froze **one authoritative spec** in doc 02 §10.2 (`<datasetId>.zarr/<property>/<level>/c/...`); doc 04/06 now reference it. Pinned coordinate convention (origin+spacing required). Flagged browser-Blosc decode as an M0/M1 spike. |
| 6 | `support=section`/`well_path` | Added **`SectionSupport`** (doc 02 §4) for ERT/2D-seismic curtains; fixed doc 03 vocabulary; clarified InSAR time-series = Zarr `[t,…]`. |
| 7 | Method names not normalized | Canonical **`MethodKey` + `submethod`** registry in doc 02 §2; docs 03/08/10 aligned (seismic→reflection/refraction; em→tdem/fdem/aem; ert/ip→dc_resistivity/ip_*). |
| 8 | Provenance over-promising reversibility | Scoped reversibility: `Transform` (`exact`/`with_pinned_deps`) vs `Step` derivations (repeatable-not-reversible) in doc 02 §7 + doc 01 §7. |
| 9 | Local→georef "too clean" | Added **`georefStatus`** (unknown→survey_controlled) in doc 01 §2; clarified "zero reprocessing" = bytes, not physical validity. |
| 10/14 | Rock-physics false precision / calibration | Transforms now declare `assumptions` + `calibrationStatus`; **uncalibrated outputs are labelled likelihood/proxy**; added a central **well-log calibration workflow** (doc 07 §4.8). |
| 11 | Compensatory favorability | **Fuzzy-conjunction is now the default play score** (heat ∧ fluid ∧ permeability); weighted-linear demoted to exploratory. Added evidence-overlap + assumption-burden indicators. *(Confirmed by user 2026-06-14.)* |
| 12 | Uncertainty too narrow | Added `UncertaintySpec.tier` (quantitative/proxy/qualitative/unknown) + `independence` flag (doc 02 §6); doc 07 shows qualitative confidence for proxy inputs and routes resolving-power through DOI not σ. |
| 15 | Plugin trust vs inversion isolation | Added **`executionMode`** (in_process/worker_process/container/remote_worker) in doc 08; doc 10 references it — the two now agree (trust axis ≠ isolation axis). |
| 16 | Frontend plugin half-enabled | Deleted dynamic third-party ES-module loading; fixed client renderer catalog via `/api/capabilities` (doc 08 §7.2). |
| 17 | Viewer over-ambitious first cut | Added **renderer staging** (doc 06 §1.3): M1 = single resident 3D texture; brick-pool/virtual-texturing = M2+. |
| 18 | Inconsistent API sketches | Doc 04 declared the **authoritative API**; fixed `GET /slice`→`POST .../slice` (06), added **path/polyline sampling** for curved wells (04 §9.3 + 09), inversion uses doc 04 job endpoints (10). |
| 19 | Observation errors not modeled | Added per-observation **sigma columns + errorModel + default noise floors** (doc 02 §3); doc 10 consumes them. |
| 20 | Categorical fields vs (t,z,y,x) | Defined categorical array shapes — hard labels `(z,y,x)` int + category table, or probabilities `(class,z,y,x)` (doc 02 §10.2). |
| 21 | Temperature canonical °C (pint bug) | **Canonical temperature = kelvin**; gradients K/km; σ in K; °C is display-only (doc 01 §5). |
| 22 | Chargeability overloaded | Split into `chargeability_time_ms` / `chargeability_mv_v` / `phase_mrad` (doc 01 §5, doc 02 §1). |
| 23 | Derived vs fused conflated | Separate `fused_models` (container) + `fused_layers` tables; favorability is an ordinary PropertyModel (doc 04 §2/§7). |
| 24 | Content-addressing undefined | Defined raw=whole-file sha256, chunks=immutable, version=manifest hash, GC=mark-and-sweep (doc 04 §8.1). |
| 25 | Drillability flag unspecified | Concretely specified the non-engineering-grade pass/warn check (DLS, build/turn rate, MD/TVD, inclination, lithology-hardness proxy) — doc 09 §4.6. |
| 26 | Fault proximity collapsed to one scalar | Split into productivity / drilling-hazard / seismicity / structural-uncertainty channels (doc 09 §7.2). |
| 27 | WITSML under-scoped | Added a conformance target: WITSML 2.0, minimum fields, validation lib, round-trip test (doc 09 §9.1). |

## Kept as-is (deliberate — these are your decisions, not defects)

The critique repeatedly urged *cutting* scope. Where that scope was a decision you made,
we **kept the decision and sequenced it** instead of deleting it:

- **#2 / #13 — "too much for an MVP" (rigorous MT/gravity/seismic, full rock-physics table, Postgres+Redis, WITSML).**
  These are your explicit choices. We did **not** revert them. Instead the mechanism the
  critique actually wants is already in `ROADMAP.md`: the **lean T0 spine ships first
  (M0–M2)**, and the heavy science is **later milestones** (M3 rigorous physics, M5 full
  rock-physics, M9 inversion) that run beside the critical path. We added the explicit
  caveat that **T0 synthetic data is not valid for inversion validation** (doc 05). If you
  later want to *defer* any of these, that's a one-line roadmap change — but it's your call,
  not the critique's.
- **#11 default favorability** — we changed the *default* to fuzzy-conjunction (a scientific
  correctness improvement within your "ship both" decision), but **both methods still ship**.
  **Confirmed by the user (2026-06-14).**

## Confirmed

1. **Favorability default = fuzzy-conjunction** (heat ∧ fluid ∧ permeability), with
   weighted-linear as an exploratory mode — **confirmed 2026-06-14**. You chose "ship both,
   defer Bayesian"; the *default within that* wasn't pinned. The critique (#11) made the
   geothermal case for fuzzy-AND (linear lets high temperature mask absent permeability),
   and you signed off.

## Not done (intentionally deferred, low value now)

- Sparse-octree storage (#5 sub-point) — dense pyramids ship first; revisit only if very
  large/sparse volumes demand it (noted in doc 04).
- Full WITSML/Compass/named-tool export beyond the 2.0 conformance target (#27) — waits for
  a real downstream tool in the loop.
