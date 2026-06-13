"""M7 exit criteria — drilling target & well planning, backend half (doc-ROADMAP M7).

The doc-ROADMAP M7 gate (design/ROADMAP.md §"M7 — Drilling target & well planning"):

    Exit: plan a well to the synthetic hot zone, see the predicted temperature/lithology/risk
    log along its path, and export a WITSML trajectory.

The *interactive* half (target-drag re-solve + predicted-log-along-the-tube render) is the
FRONTEND check (see ``blockers`` in the structured result). This test proves the **backend
half** of that gate end-to-end, exactly per doc 09 (§3 target, §4 trajectory + min-curvature,
§5–§6 predicted log + geothermal outputs, §7 risk, §9.1 WITSML round-trip):

1. **One earth, fuse + transforms** — compile the flagship ``great-basin-v1`` scene (doc 05
   §7.1: a Basin-&-Range hydrothermal play with a fault-controlled ~220 °C upflow) at a
   deliberately **coarse** truth grid (~19×15×15 cells) so the whole build + fuse + transform +
   plan + predict + export runs in ~1 s. The fault-controlled upflow — hot AND porous AND
   fractured — is preserved. The co-located truth temperature (canonical kelvin) / resistivity /
   P-velocity stand in for the inverted survey models the fusion engine consumes (doc 07 §0);
   the microseismic event cloud (tracking the true fracture density) bins onto the fused grid.
   :func:`build_fused_model` + :func:`resample_to_fused` co-register them, then the shipped
   rock-physics transforms derive **porosity** (Vp→φ) and a **fracture-density** index
   (microseismic KDE); a fuzzy-AND :func:`compute_favorability` (hot ∧ porous ∧ fractured, all
   *required*) yields the ``[0,1]`` **favorability** volume + its paired confidence (doc 07
   §4.6). The favorability volume is resampled back onto the fused grid so the planner can sample
   it along the path.

2. **Pick a target at the synthetic hot zone** (doc 09 §3) — the favorability hot-spot cell is
   the target bullseye; :func:`enrich_target` stamps the points-mode fused sample (temperature
   °C from canonical K, favorability, σ/confidence) onto it, tied to the fused model id for
   stale detection (§3.3). The enriched temperature lands on the known ~220 °C upflow.

3. **Solve a deviated well to it** (doc 09 §4) — a **build-hold-land** :func:`solve_survey`
   emits the deviation survey from an offset wellhead to the target; the planned well IS a
   deviation survey identical to an ingested well (§4.1). It **lands inside tolerance**, the
   per-interval **DLS stays within the metric °/30 m ceiling**, inclination is within max, and
   the crude :func:`drillability_flag` returns an advisory ``ok|warn`` (never ``fail``, §4.6).

4. **Predict the log along the curved path** (doc 09 §5–§7) — :func:`predict_log` densifies the
   survey ON the min-curvature arc, batch-samples the fused layers (value + σ) at the curved
   vertices, and assembles the predicted temperature/lithology/risk log with the geothermal
   summary. **THE M7 EXIT assertions:** the **BHT at TD is near the hot-zone temperature**
   (~220 °C) with a σ band and is the hottest point along the path; the well is **in window**
   over a real pay fraction; **fracture intersections** are counted; the per-station **risk** is
   the transparent weighted blend of its driver terms (a glass box, §7.4).

5. **Export + round-trip a WITSML trajectory** (doc 09 §9.1) — :func:`export_witsml_trajectory`
   emits a WITSML 2.0 (and 1.4.1.1 legacy) ``trajectory``; it validates structurally and
   **round-trips**: export → re-import → each station's **(MD, inc, azi)** and derived
   **(TVD, N, E)** match within tolerance (MD/TVD/N/E ≤ 0.01 m, inc/azi ≤ 0.01°), and the MD
   datum (KB elevation) survives. (The Energistics XSDs + ``xmlschema`` are not installed here,
   so the writer validates structurally and flags ``schema_validated is False`` — doc 09 §9.1
   "validate structurally without the XSD and note it".)

All I/O is to ``tmp_path`` with in-memory SQLite (doc 04 §2.1 fallback) — no Docker / Postgres /
Redis, coarse grids throughout.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

# Importing the rock-physics package self-registers the §4.2 transform library + its property
# types (porosity / microseismic / fracture_density) — must precede any REGISTRY use on them.
import geosim.fusion.rockphys  # noqa: F401
from geosim.catalog import (
    Dataset,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    SpatialFrameRow,
    create_all,
    make_engine,
    new_id,
    session_factory,
)
from geosim.fusion import (
    Evidence,
    FavorabilitySpec,
    TransferFn,
    build_fused_model,
    compute_favorability,
    resample_to_fused,
    run_transform,
)
from geosim.fusion.rockphys import MicroseismicDensity, VelocityToPorosity
from geosim.fusion.rockphys.fracture import events_to_count_volume
from geosim.planning import (
    DesignSpec,
    DrillTarget,
    PlannedWell,
    RiskWeights,
    TargetTolerance,
    TrajectoryConstraints,
    drillability_flag,
    enrich_target,
    predict_log,
    solve_survey,
)
from geosim.planning.export import (
    export_witsml_trajectory,
    parse_witsml_trajectory,
    validate_witsml_trajectory,
)
from geosim.spatial import min_curvature_positions
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)
from geosim.synthgen import compile_scene
from geosim.synthgen.scenarios import get_scenario

# Round-trip tolerances (doc 09 §9.1).
TOL_LEN_M = 0.01
TOL_ANG_DEG = 0.01

# The known ~220 °C upflow band (doc 05 §7.1 ``temp_peak``) → kelvin, with generous slack so
# the assertion keys on "near the hot zone" not an exact synthetic number.
HOT_ZONE_C = 220.0
HOT_ZONE_FLOOR_C = 190.0  # the BHT must clear this — squarely in the upflow, not background

# The well's MD datum / kelly-bushing elevation (Engineering metres, ≈ scene surface).
KB_ELEV_M = 1700.0


# ─────────────────────────────── coarse great-basin earth ───────────────────────────────


def _coarse_great_basin():
    """Compile ``great-basin-v1`` at a coarse truth grid (same geology, ~19×15×15 cells).

    The shipped flagship truth grid is millions of cells; we only need the co-located property
    + state volumes, so we replace the fine spacings with coarse ones (doc 05 §2 allows the
    truth-grid spacing to be chosen) to keep this gate ~1 s while preserving the fault-controlled
    hydrothermal upflow's multi-method signature (hot ∧ porous ∧ fractured).
    """
    spec = get_scenario("great-basin-v1").scene
    coarse_frame = replace(spec.frame, dx=800.0, dy=800.0, dz=400.0)
    return compile_scene(replace(spec, frame=coarse_frame))


# ─────────────────────────────────── native-model I/O ───────────────────────────────────


def _bbox(origin, spacing, shape) -> dict:
    oz, oy, ox = origin
    dz, dy, dx = spacing
    nz, ny, nx = shape
    return {
        "xmin": ox, "xmax": ox + dx * (nx - 1),
        "ymin": oy, "ymax": oy + dy * (ny - 1),
        "zmin": oz, "zmax": oz + dz * (nz - 1),
    }


def _write_native_pm(session, layout, pid, *, prop, values, origin, spacing, unit, method):
    """Persist a co-located native :class:`PropertyModel` (Zarr + catalog rows)."""
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    write_property_model(zarr_path, prop, values.astype(np.float32), grid=grid, overwrite=True)

    bbox_json = json.dumps(_bbox(origin, spacing, values.shape))
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method=method, kind="propertyModel",
        status="ready", extent_json=bbox_json, spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="m7@test",
    ))
    session.flush()
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop,
        canonical_unit=unit, support="volume", store_uri=str(zarr_path),
        shape_json=json.dumps(list(values.shape)),
        spacing_json=json.dumps(list(spacing)), origin_json=json.dumps(list(origin)),
        bbox_json=bbox_json,
    ))
    session.commit()
    return session.get(PropertyModel, pm_id)


def _microseismic_count_volume(earth) -> np.ndarray:
    """A binned microseismic event cloud co-located with the true fractured upflow (doc 07 §4.2).

    Density tracks the truth fracture-density state field (more active fracturing ⇒ more events);
    deterministic RNG so the gate is reproducible.
    """
    sub = ~earth.above_surface
    frac = np.where(sub, earth.state.fracture_density, 0.0)
    nz, ny, nx = earth.shape
    oz, oy, ox = earth.origin
    dz, dy, dx = earth.spacing

    weights = frac.reshape(-1)
    weights = weights / weights.sum()
    rng = np.random.default_rng(0)
    n_events = 600
    cell_idx = rng.choice(weights.size, size=n_events, p=weights)
    iz, iy, ix = np.unravel_index(cell_idx, (nz, ny, nx))
    jitter = rng.uniform(-0.4, 0.4, size=(n_events, 3))
    xs = ox + (ix + jitter[:, 0]) * dx
    ys = oy + (iy + jitter[:, 1]) * dy
    zs = oz + (iz + jitter[:, 2]) * dz
    return np.column_stack([xs, ys, zs])


# ─────────────────────────────────────── fixture ───────────────────────────────────────


@pytest.fixture(scope="module")
def m7_planned(tmp_path_factory):
    """The full M7 backend pipeline: earth → fuse+transforms → favorability → target → well → log.

    Yields a dict with the session, fused model, the picked target, the resolved planned well,
    the predicted log, and the hot-zone bookkeeping the assertions key on.
    """
    earth = _coarse_great_basin()
    sub = ~earth.above_surface
    origin, spacing, shape = earth.origin, earth.spacing, earth.shape

    storage_root = tmp_path_factory.mktemp("store")
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1 fallback)
    create_all(engine)
    Session = session_factory(engine)
    session = Session()

    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    bbox = _bbox(origin, spacing, shape)
    session.add(Project(id=pid, name="m7-great-basin", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({k: bbox[k] for k in ("xmin", "xmax", "ymin", "ymax")}),
        depth_range_json=json.dumps({"zmin": bbox["zmin"], "zmax": bbox["zmax"]}),
        frame_json=json.dumps({"mode": "local"}),
    ))
    session.commit()

    # 1) Native co-located models. Temperature is the TRUE upflow field (canonical kelvin) so the
    # predicted-log BHT reads the real hot zone; resistivity / Vp / microseismic feed the
    # rock-physics + favorability chain.
    temp_true = earth.property_volume("temperature")  # canonical kelvin
    resistivity = earth.property_volume("resistivity")
    velocity_p = earth.property_volume("velocity_p")
    micro_events = _microseismic_count_volume(earth)

    temp_pm = _write_native_pm(session, layout, pid, prop="temperature", values=temp_true,
                               origin=origin, spacing=spacing, unit="kelvin", method="welllog")
    res_pm = _write_native_pm(session, layout, pid, prop="resistivity", values=resistivity,
                              origin=origin, spacing=spacing, unit="ohm*m", method="mt")
    vel_pm = _write_native_pm(session, layout, pid, prop="velocity_p", values=velocity_p,
                              origin=origin, spacing=spacing, unit="m/s", method="seismic")

    # 2) Fuse + resample everything onto a shared support (doc 07 §1–§2).
    fem, grid = build_fused_model(
        session, layout, pid,
        source_property_model_ids=[temp_pm.id, res_pm.id, vel_pm.id],
        spacing=spacing, name="m7-fused",
    )
    counts = events_to_count_volume(micro_events, grid)
    mic_pm = _write_native_pm(session, layout, pid, prop="microseismic", values=counts,
                              origin=origin, spacing=spacing, unit="dimensionless",
                              method="microseismic")
    for pm in (temp_pm, res_pm, vel_pm, mic_pm):
        resample_to_fused(session, fem, pm.id)
    session.refresh(fem)

    # 3) Rock-physics transforms → porosity (Vp→φ) + fracture-density index (microseismic KDE).
    poro = run_transform(
        session, layout, fem, VelocityToPorosity(),
        params={"v_matrix_m_s": 6000.0, "v_fluid_m_s": 1500.0, "model": "wyllie"},
        storage_root=storage_root,
    )
    frac = run_transform(
        session, layout, fem, MicroseismicDensity(),
        params={"bandwidth_cells": 1.0}, storage_root=storage_root,
    )

    # 4) Favorability = hot (TRUE temperature) ∧ porous ∧ fractured, all required (doc 07 §4.6).
    spec = FavorabilitySpec(
        evidence=[
            Evidence(source=temp_pm.id, target="temperature",
                     transfer=TransferFn("ramp", lo=440.0, hi=490.0), weight=0.4, role="required"),
            Evidence(source=poro.model_id, target="porosity",
                     transfer=TransferFn("ramp", lo=0.025, hi=0.07), weight=0.3, role="required"),
            Evidence(source=frac.model_id, target="fracture_density",
                     transfer=TransferFn("ramp", lo=0.15, hi=0.6), weight=0.3, role="required"),
        ],
        method="fuzzy",
    )
    fav_result = compute_favorability(session, layout, fem, spec, storage_root=storage_root)
    # Resample the favorability volume back onto the fused grid so the planner can sample it
    # along the path (doc 09 §5.1) and enrich the target with it (§3.2).
    resample_to_fused(session, fem, fav_result.model_id, storage_root=storage_root)
    session.refresh(fem)

    # 5) Pick the target at the favorability hot-spot (doc 09 §3.2 click-on-isosurface).
    fav = open_property_model(
        session.get(PropertyModel, fav_result.model_id).store_uri
    ).read_level("favorability", 0)
    finite = np.isfinite(fav) & sub
    masked = np.where(finite, fav, -np.inf)
    tiz, tiy, tix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    oz, oy, ox = origin
    dz, dy, dx = spacing
    target_xyz = (ox + tix * dx, oy + tiy * dy, oz + tiz * dz)
    true_target_c = float(earth.property_volume("temperature")[tiz, tiy, tix]) - 273.15

    target = DrillTarget(
        id=new_id(IdKind.FEATURE), name="great-basin upflow", project_id=pid, kind="point",
        location=target_xyz, tolerance=TargetTolerance(radius_m=120.0, tvd_window_m=60.0),
        desired_temperature_c=200.0, min_temperature_c=150.0, kb_elev_m=KB_ELEV_M,
    )
    enrich_target(session, fem, target, storage_root=storage_root)

    # 6) Solve a build-hold-land well from an offset wellhead to the target (doc 09 §4.4).
    wellhead = (target_xyz[0] - 1200.0, target_xyz[1] - 800.0)
    constraints = TrajectoryConstraints(max_dls_deg30m=4.0, max_inc_deg=80.0)
    design = DesignSpec(method="build-hold-land", target=target_xyz, kop_md_m=2000.0,
                        build_rate_deg30m=2.5)
    solve = solve_survey(design, wellhead, KB_ELEV_M, constraints)
    well = PlannedWell(
        id=new_id(IdKind.FEATURE), name="GB-W01", project_id=pid, wellhead=wellhead,
        kb_elev_m=KB_ELEV_M, deviation_survey=solve.survey, design=design,
        constraints=constraints,
    )

    # 7) Predict the log along the curved path (doc 09 §5–§7).
    log = predict_log(session, fem, well, md_step_m=50.0, target=target,
                      risk_weights=RiskWeights(), storage_root=storage_root)

    bundle = {
        "session": session, "fem": fem, "fem_id": fem.id,
        "target": target, "target_xyz": target_xyz, "true_target_c": true_target_c,
        "well": well, "solve": solve, "wellhead": wellhead, "constraints": constraints,
        "log": log,
    }
    yield bundle
    session.close()


# ─────────────────────── 1+2: hot-zone target enrichment (doc 09 §3) ───────────────────────


def test_target_lands_on_the_synthetic_hot_zone(m7_planned):
    """The favorability hot-spot target is enriched with the known ~220 °C upflow (§3.2/§3.3).

    The picked bullseye sits in the fault-controlled hydrothermal upflow; the points-mode fused
    enrichment stamps the temperature (°C, from canonical K), favorability and σ/confidence on,
    tied to the fused model id for stale detection (§3.3).
    """
    target, fem_id = m7_planned["target"], m7_planned["fem_id"]
    true_c = m7_planned["true_target_c"]
    e = target.enrichment
    assert e is not None
    assert e.model_version == fem_id  # tied to the fused model for stale detection (§3.3)

    # The enriched temperature is the known hot zone — near the synthetic upflow temperature.
    assert e.temperature_c is not None and e.temperature_c.value is not None
    assert e.temperature_c.value == pytest.approx(true_c, abs=2.0)
    assert e.temperature_c.value >= HOT_ZONE_FLOOR_C
    assert e.temperature_c.value == pytest.approx(HOT_ZONE_C, abs=15.0)

    # The hot-spot really is favorable (hot ∧ porous ∧ fractured all co-located).
    assert e.favorability is not None and e.favorability.value is not None
    assert e.favorability.value > 0.6

    # depthTVD derived from z + KB (§3.3): KB elevation minus the canonical Engineering z.
    assert e.depth_tvd_m == pytest.approx(KB_ELEV_M - target.location[2])

    # Stale detection (§3.3): a different model version reads as stale.
    assert target.is_stale("some-other-fem-id")
    assert not target.is_stale(fem_id)


# ───────────────────────────── 3: solve a deviated well (doc 09 §4) ─────────────────────────────


def test_well_lands_in_tolerance_with_dls_in_range(m7_planned):
    """The build-hold-land well lands inside tolerance with DLS within the °/30 m ceiling (§4.4).

    A planned well IS a deviation survey identical to an ingested well (§4.1); the solver lands
    the curved trajectory inside the target tolerance, keeping the per-interval DLS under the
    metric ceiling and inclination within max — the geometry that gates export (§4.4).
    """
    solve = m7_planned["solve"]
    target = m7_planned["target"]
    constraints = m7_planned["constraints"]

    # Lands inside the target tolerance radius (§4.4 validation).
    assert solve.landing_error_m is not None
    assert solve.landing_error_m < target.tolerance.radius_m
    # DLS within the metric °/30 m ceiling; inclination within max — neither exceeded.
    assert not solve.dls_exceeded
    assert solve.max_dls_deg30m <= constraints.max_dls_deg30m + 1e-6
    assert not solve.inc_exceeded
    assert solve.max_inc_deg <= constraints.max_inc_deg + 1e-6
    # The build arc actually deviates (a real deviated well, not a vertical degenerate).
    assert solve.max_inc_deg > 10.0


def test_positions_reuse_the_shared_min_curvature_integrator(m7_planned):
    """The well's positions ARE the shared min-curvature integrator (doc 09 §4.3 reuse).

    Per doc 09 §4.3 the survey→position integrator is the SAME backend routine an ingested
    well uses — :func:`geosim.spatial.min_curvature_positions` — not a planner-private copy.
    """
    well = m7_planned["well"]
    pos = well.positions()
    direct = min_curvature_positions(
        well.deviation_survey, well.wellhead, kb_elev=well.kb_elev_m
    )
    assert np.allclose(pos.enu, direct.enu)
    assert np.allclose(pos.tvd, direct.tvd)
    # The track descends from the KB datum (TVD grows; Engineering z decreases).
    assert pos.tvd[-1] > pos.tvd[0]
    assert pos.enu[-1, 2] < pos.enu[0, 2]


def test_drillability_flag_is_advisory(m7_planned):
    """The crude drillability flag returns an advisory ``ok|warn`` — never ``fail`` (§4.6)."""
    well, solve = m7_planned["well"], m7_planned["solve"]
    flag = drillability_flag(solve.survey, well.positions())
    assert flag.verdict in ("ok", "warn")  # advisory only (§4.6)
    names = {c.name for c in flag.checks}
    assert names == {"dls", "buildRate", "turnRate", "mdTvdRatio", "maxInc", "hardness"}
    # A gentle 2.5°/30 m build to a deep target should be drillable.
    assert flag.verdict == "ok"


# ─────────────── 4: predicted log + geothermal outputs + risk (doc 09 §5–§7) ───────────────


def test_predicted_log_has_temperature_lithology_risk_curves(m7_planned):
    """The predicted log carries the temperature / risk curves along the curved path (doc 09 §5).

    One station per ~MD step, each with Engineering XYZ on the min-curvature arc, a temperature
    sample, and a per-station risk — the curves the viewer colour-maps along the well tube.
    """
    log = m7_planned["log"]
    assert len(log.stations) > 50  # a dense predicted log along the path
    s0, s_td = log.stations[0], log.stations[-1]
    # Temperature curve present at every station (the headline geothermal log).
    assert "temperatureC" in s0.values and "temperatureC" in s_td.values
    # MD is monotone from the datum and TVD grows down the path.
    md = [s.md for s in log.stations]
    assert md == sorted(md)
    assert log.stations[-1].tvd > log.stations[0].tvd
    # Each station carries a risk scalar in [0,1] (the risk track, §7.4).
    assert all(0.0 <= s.risk <= 1.0 for s in log.stations)


def test_bht_is_near_the_hot_zone_temperature(m7_planned):
    """THE M7 EXIT (geothermal): predicted BHT lands near the synthetic hot-zone temp (§6).

    The well is landed in the ~220 °C upflow, so the predicted bottom-hole temperature (the
    temperature at TD) is near the known hot-zone temperature with a σ band, is the hottest
    point along the path, and clears the geothermal floor. The in-window pay fraction is a real
    number and productive fracture intersections are counted (§6).
    """
    log = m7_planned["log"]
    true_c = m7_planned["true_target_c"]
    summary = log.summary

    # BHT near the hot zone (~220 °C) with a σ band (doc 09 §6 "BHT … ± σ").
    assert summary.bht_c is not None
    assert summary.bht_c == pytest.approx(true_c, abs=5.0)
    assert summary.bht_c == pytest.approx(HOT_ZONE_C, abs=15.0)
    assert summary.bht_c >= HOT_ZONE_FLOOR_C
    assert summary.bht_sigma_c is not None and summary.bht_sigma_c > 0.0

    # The hottest point along the path is at/near TD (the landed reservoir is the hot target).
    assert summary.max_temp_c is not None
    assert summary.max_temp_c >= summary.bht_c - 1e-6
    assert summary.max_temp_c == pytest.approx(summary.bht_c, abs=5.0)

    # A real pay fraction in window (above the min-temperature floor + favorability) (§6).
    assert summary.in_window_fraction > 0.05
    # Productive fracture intersections counted along the fractured upflow (§6 EGS proxy).
    assert summary.productive_fracture_intersections >= 1


def test_risk_is_a_transparent_weighted_blend(m7_planned):
    """Risk is a glass-box weighted blend of its driver terms (doc 09 §7.4).

    Each station's risk equals the (capped) sum of its weighted driver terms — temperature
    confidence, hazard, DLS exceedance, structural uncertainty — always shown alongside the
    number, and the aggregate mean ≤ peak risk is reported.
    """
    log = m7_planned["log"]
    for s in log.stations:
        assert set(s.risk_drivers) == {
            "tempConfidence", "hazard", "dlsExceedance", "structuralUncertainty"
        }
        assert s.risk == pytest.approx(min(1.0, sum(s.risk_drivers.values())), abs=1e-9)
    assert 0.0 <= log.summary.mean_risk <= log.summary.peak_risk <= 1.0


# ─────────────────────── 5: WITSML export + round-trip (doc 09 §9.1) ───────────────────────


def _assert_witsml_round_trips(well, version):
    """THE M7 EXIT (export): export → validate → re-import → compare within tolerance (§9.1)."""
    xml = export_witsml_trajectory(well, version=version, model_version="m7-fused")

    # Structural validation only — the Energistics XSDs / xmlschema are unavailable here (§9.1).
    vr = validate_witsml_trajectory(xml)
    assert vr.well_formed
    assert vr.structural_ok, vr.errors
    assert vr.schema_validated is False
    assert "structural validation only" in vr.note

    parsed = parse_witsml_trajectory(xml)
    assert parsed.version == version
    arr = parsed.as_arrays()
    survey = np.asarray(well.deviation_survey)
    pos = well.positions()
    wx, wy = float(well.wellhead[0]), float(well.wellhead[1])

    # (MD, inc, azi) + derived (TVD, N, E) survive the round trip within tolerance (§9.1).
    assert np.max(np.abs(arr["md"] - survey[:, 0])) <= TOL_LEN_M
    assert np.max(np.abs(arr["inc"] - survey[:, 1])) <= TOL_ANG_DEG
    assert np.max(np.abs(arr["azi"] - survey[:, 2])) <= TOL_ANG_DEG
    assert np.max(np.abs(arr["tvd"] - pos.tvd)) <= TOL_LEN_M
    # dispNs/dispEw are displacements from the wellhead → add it back to compare to ENU N/E.
    assert np.max(np.abs((arr["dispNs"] + wy) - pos.enu[:, 1])) <= TOL_LEN_M
    assert np.max(np.abs((arr["dispEw"] + wx) - pos.enu[:, 0])) <= TOL_LEN_M

    # The MD datum (KB elevation + kind) survives the round trip (§9.1).
    assert parsed.md_datum_elev_m == pytest.approx(well.kb_elev_m, abs=TOL_LEN_M)
    assert parsed.md_datum_kind == "KB"
    return parsed


def test_witsml_20_trajectory_exports_and_round_trips(m7_planned):
    """THE M7 EXIT: a WITSML 2.0 trajectory exports and round-trips within tolerance (§9.1)."""
    well = m7_planned["well"]
    parsed = _assert_witsml_round_trips(well, "2.0")
    assert len(parsed.stations) == np.asarray(well.deviation_survey).shape[0]
    # The model version rides the trajectory metadata (service company, §9.1).
    assert "m7-fused" in export_witsml_trajectory(well, version="2.0", model_version="m7-fused")


def test_witsml_141_legacy_trajectory_round_trips(m7_planned):
    """The WITSML 1.4.1.1 legacy alt round-trips too (doc 09 §9.1)."""
    well = m7_planned["well"]
    _assert_witsml_round_trips(well, "1.4.1.1")
