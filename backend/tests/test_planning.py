"""Tests for the well-planning core (doc 09).

Small/coarse grids, local temp dirs, SQLite in-memory — no Docker/Postgres/Redis.

On a small fused grid with a planted **hot + favorable + fractured** zone (a FORGE-style
target volume) this exercises the doc-09 contract end to end:

- **DrillTarget enrichment** (§3): a target picked in the hot zone gets the right
  temperature (°C, from canonical K) / favorability / confidence stamped on, tied to the
  fused model's id for stale detection.
- **Design solvers** (§4.4): a build-hold-land well lands inside tolerance, with DLS within
  the ceiling and a sane drillability flag; vertical/S-curve solvers run.
- **Minimum curvature** (§4.3): positions REUSE :func:`geosim.spatial.min_curvature_positions`
  and the densified path follows the curved arc, not the chord.
- **Predicted log** (§5–§6): values + σ + confidence at curved-path stations; BHT at TD
  matches the hot zone; the in-window fraction is high; fracture intersections counted.
- **Risk** (§7.4): a transparent weighted blend with a per-station driver breakdown whose
  weighted terms sum to the score.
- The **API** surface (§10) runs: POST targets / wells, solve, positions, predict.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api.app import Settings, create_app
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
from geosim.fusion import build_fused_model, resample_to_fused
from geosim.planning import (
    DesignSpec,
    DrillTarget,
    PlannedWell,
    RiskWeights,
    TargetTolerance,
    TrajectoryConstraints,
    densify_survey,
    drillability_flag,
    enrich_target,
    predict_log,
    solve_survey,
    well_positions,
)
from geosim.spatial import min_curvature_positions
from geosim.storage import GridSpec, ensure_project_layout, write_property_model

# A coarse grid spanning x,y ∈ [0,1000], z ∈ [-2000, 0] (Engineering metres, Z-up).
SHAPE = (21, 11, 11)  # (nz, ny, nx)
ORIGIN = (-2000.0, 0.0, 0.0)  # (z0, y0, x0)
SPACING = (100.0, 100.0, 100.0)  # (dz, dy, dx)

# Hot zone centre (Engineering x, y, z): deep, offset east — a build-hold-land target.
HOT_XYZ = (600.0, 500.0, -1500.0)


# ───────────────────────────────── fixtures ─────────────────────────────────


@pytest.fixture
def env(tmp_path):
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    storage_root = tmp_path
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    session.add(Project(id=pid, name="planning-test", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000}),
        depth_range_json=json.dumps({"zmin": -2000, "zmax": 0}),
        frame_json=json.dumps({"mode": "local"}),
    ))
    session.commit()
    yield session, layout, storage_root, pid
    session.close()


def _axis_coords():
    oz, oy, ox = ORIGIN
    dz, dy, dx = SPACING
    nz, ny, nx = SHAPE
    z = oz + dz * np.arange(nz)
    y = oy + dy * np.arange(ny)
    x = ox + dx * np.arange(nx)
    return z, y, x


def _gaussian_blob(centre, sigma=350.0):
    """A 3-D Gaussian bump in [0,1] centred at ``centre`` (Engineering x,y,z)."""
    z, y, x = _axis_coords()
    gz, gy, gx = np.meshgrid(z, y, x, indexing="ij")
    cx, cy, cz = centre
    r2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma**2))


def _planted_fields():
    """Plant a co-located hot + favorable + fractured zone at ``HOT_XYZ``.

    Temperature ramps with depth (a geothermal gradient) PLUS a hot bump at the zone, so the
    deepest/target station is the hottest. Favorability + fracture density peak at the zone.
    """
    z, _y, _x = _axis_coords()
    blob = _gaussian_blob(HOT_XYZ)
    # Background gradient: ~30 °C/km from a 15 °C surface → kelvin.
    depth = -z  # +down depth below datum
    grad_c = 15.0 + 0.030 * depth[:, None, None]  # °C, broadcast over (z,1,1)
    grad_c = np.broadcast_to(grad_c, SHAPE).copy()
    temp_c = grad_c + 120.0 * blob  # hot bump (+120 °C at the zone)
    temp_k = temp_c + 273.15  # canonical kelvin

    fav = 0.05 + 0.9 * blob  # favorability peaks ~0.95 at the zone
    frac = 0.05 + 0.9 * blob  # fracture density peaks at the zone
    return temp_k, fav, frac


def _make_native_pm(session, layout, pid, *, prop, values, unit):
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=ORIGIN, spacing=SPACING, cell_ref="center")
    write_property_model(zarr_path, prop, values, grid=grid, overwrite=True)
    nz, ny, nx = values.shape
    oz, oy, ox = ORIGIN
    dz, dy, dx = SPACING
    bbox = json.dumps({
        "xmin": ox, "xmax": ox + dx * (nx - 1), "ymin": oy, "ymax": oy + dy * (ny - 1),
        "zmin": oz, "zmax": oz + dz * (nz - 1),
    })
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method="synthetic",
        kind="propertyModel", status="ready", extent_json=bbox, spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop, canonical_unit=unit,
        support="volume", store_uri=str(zarr_path), shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(SPACING)), origin_json=json.dumps(list(ORIGIN)),
        bbox_json=bbox,
    ))
    session.commit()
    return pm_id


def _build_fused(session, layout, pid):
    temp_k, fav, frac = _planted_fields()
    pms = [
        _make_native_pm(session, layout, pid, prop="temperature", values=temp_k, unit="kelvin"),
        _make_native_pm(session, layout, pid, prop="favorability", values=fav,
                        unit="dimensionless"),
        _make_native_pm(session, layout, pid, prop="fracture_density", values=frac,
                        unit="dimensionless"),
    ]
    fem, _grid = build_fused_model(
        session, layout, pid, source_property_model_ids=pms, spacing=SPACING, name="fused-plan",
    )
    for pm_id in pms:
        resample_to_fused(session, fem, pm_id, method="trilinear", interp_space="linear")
    session.refresh(fem)
    return fem


# ───────────────────────────────── target ─────────────────────────────────


def test_target_enrichment_picks_up_hot_zone(env):
    session, layout, storage_root, pid = env
    fem = _build_fused(session, layout, pid)
    tgt = DrillTarget(
        id=new_id(IdKind.FEATURE), name="hot-zone-A", project_id=pid, kind="point",
        location=HOT_XYZ, tolerance=TargetTolerance(50.0, 25.0),
        desired_temperature_c=180.0, min_temperature_c=150.0, kb_elev_m=0.0,
    )
    enrich_target(session, fem, tgt, storage_root=storage_root)
    e = tgt.enrichment
    assert e is not None
    assert e.model_version == fem.id  # tied to the fused model for stale detection (§3.3)
    # Hot bump (+120 °C) on top of ~60 °C gradient at 1500 m ≈ 180 °C.
    assert e.temperature_c.value > 150.0
    assert e.favorability.value > 0.6
    assert e.favorability.confidence is not None
    # depthTVD derived from z + KB (§3.3).
    assert e.depth_tvd_m == pytest.approx(1500.0)
    # Stale detection: a different model version reads as stale.
    assert tgt.is_stale("some-other-fem-id")
    assert not tgt.is_stale(fem.id)


# ───────────────────────────────── solvers ─────────────────────────────────


def test_build_hold_land_lands_in_tolerance(env):
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    design = DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0,
                        build_rate_deg30m=3.0)
    res = solve_survey(design, (0.0, 0.0), 0.0, cons)
    # Lands inside the 50 m tolerance window (§4.4).
    assert res.landing_error_m < 50.0
    # DLS within the ceiling, inclination within max (§4.4 validation).
    assert not res.dls_exceeded
    assert res.max_dls_deg30m <= 5.0 + 1e-6
    assert not res.inc_exceeded
    # The build arc actually deviates (not a vertical degenerate).
    assert res.max_inc_deg > 10.0


def test_vertical_and_s_curve_solvers_run(env):
    cons = TrajectoryConstraints(max_dls_deg30m=6.0, max_inc_deg=92.0)
    rv = solve_survey(DesignSpec(method="vertical", target=(0.0, 0.0, -1000.0)),
                      (0.0, 0.0), 0.0, cons)
    assert rv.positions.tvd[-1] == pytest.approx(1000.0)
    assert float(np.max(rv.survey[:, 1])) == pytest.approx(0.0)  # stays vertical

    rs = solve_survey(
        DesignSpec(method="S-curve", target=(400.0, 0.0, -1500.0), kop_md_m=400.0,
                   build_rate_deg30m=3.0, drop_rate_deg30m=3.0, hold_inc_deg=35.0),
        (0.0, 0.0), 0.0, cons,
    )
    assert rs.max_inc_deg == pytest.approx(35.0, abs=1.0)  # builds to the hold inclination
    assert rs.survey[-1, 1] == pytest.approx(0.0, abs=1e-6)  # drops back to vertical
    assert not rs.dls_exceeded


def test_positions_reuse_shared_integrator(env):
    """The planner's positions are EXACTLY the shared min-curvature integrator (§4.3 reuse)."""
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    res = solve_survey(DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0),
                       (0.0, 0.0), 0.0, cons)
    direct = min_curvature_positions(res.survey, (0.0, 0.0), kb_elev=0.0)
    mine = well_positions(res.survey, (0.0, 0.0), 0.0)
    assert np.allclose(mine.enu, direct.enu)
    assert np.allclose(mine.tvd, direct.tvd)


def test_densify_follows_the_curved_arc(env):
    """Densified stations lie on the min-curvature arc, not the chord (§4.3/§5.1)."""
    survey = np.array([[0, 0, 0], [600, 0, 90], [1200, 60, 90]], dtype=float)
    dense = densify_survey(survey, 30.0)
    # The densified vertices reproduce the original stations' positions exactly.
    pos_dense = min_curvature_positions(dense, (0.0, 0.0), kb_elev=0.0)
    pos_orig = min_curvature_positions(survey, (0.0, 0.0), kb_elev=0.0)
    # The last vertex of each matches.
    assert np.allclose(pos_dense.enu[-1], pos_orig.enu[-1], atol=1.0)
    # A curved build has more horizontal reach than a straight chord between endpoints.
    chord = pos_orig.enu[-1] - pos_orig.enu[0]
    # The densified path length exceeds the straight chord (it curves).
    seg = np.linalg.norm(np.diff(pos_dense.enu, axis=0), axis=1).sum()
    assert seg > np.linalg.norm(chord)


# ───────────────────────────────── drillability ─────────────────────────────────


def test_drillability_flag_sane(env):
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    res = solve_survey(DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0),
                       (0.0, 0.0), 0.0, cons)
    flag = drillability_flag(res.survey, res.positions)
    assert flag.verdict in ("ok", "warn")  # never "fail" (§4.6)
    names = {c.name for c in flag.checks}
    assert names == {"dls", "buildRate", "turnRate", "mdTvdRatio", "maxInc", "hardness"}
    # A gentle 3°/30 m build to a deep target should be drillable (ok).
    assert flag.verdict == "ok"

    # A deliberately savage dogleg trips the DLS check to warn (still not fail).
    savage = np.array([[0, 0, 0], [60, 0, 0], [90, 80, 0]], dtype=float)
    pos = min_curvature_positions(savage, (0.0, 0.0), kb_elev=0.0)
    bad = drillability_flag(savage, pos)
    assert bad.verdict == "warn"
    dls_check = next(c for c in bad.checks if c.name == "dls")
    assert dls_check.verdict == "warn"
    assert dls_check.md_interval_m is not None


# ───────────────────────────────── predicted log + risk ─────────────────────────────────


def test_predicted_log_bht_and_in_window(env):
    session, layout, storage_root, pid = env
    fem = _build_fused(session, layout, pid)
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    res = solve_survey(DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0),
                       (0.0, 0.0), 0.0, cons)
    well = PlannedWell(
        id=new_id(IdKind.FEATURE), name="W-01", project_id=pid,
        wellhead=(0.0, 0.0), kb_elev_m=0.0, deviation_survey=res.survey, constraints=cons,
    )
    target = DrillTarget(
        id=new_id(IdKind.FEATURE), name="hot", project_id=pid, kind="point",
        location=HOT_XYZ, tolerance=TargetTolerance(120.0, 60.0),
        min_temperature_c=150.0, kb_elev_m=0.0,
    )
    log = predict_log(session, fem, well, md_step_m=10.0, target=target, storage_root=storage_root)

    # One station per ~10 m of MD, each with Engineering XYZ on the curved path (§5.2).
    assert len(log.stations) > 50
    s0 = log.stations[0]
    assert "temperatureC" in s0.values
    # BHT at TD matches the planted hot zone (~180 °C) with a σ band (§6).
    bht = log.summary.bht_c
    assert bht is not None and bht > 150.0
    assert log.summary.bht_sigma_c is not None and log.summary.bht_sigma_c > 0.0
    # Max temperature is at/near TD (the hottest point) (§6).
    assert log.summary.max_temp_c >= bht - 1e-6
    # The target sits in a hot+favorable zone, so the in-window fraction near TD is non-trivial.
    assert log.summary.in_window_fraction > 0.05
    # Fracture intersections counted where fracture density crosses the threshold (§6).
    assert log.summary.productive_fracture_intersections >= 1


def test_risk_is_transparent_weighted_blend_with_drivers(env):
    session, layout, storage_root, pid = env
    fem = _build_fused(session, layout, pid)
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    res = solve_survey(DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0),
                       (0.0, 0.0), 0.0, cons)
    well = PlannedWell(
        id=new_id(IdKind.FEATURE), name="W-01", project_id=pid,
        wellhead=(0.0, 0.0), kb_elev_m=0.0, deviation_survey=res.survey, constraints=cons,
    )
    weights = RiskWeights()  # default drilling-feasibility weights (0.40/0.30/0.10/0.20)
    log = predict_log(session, fem, well, md_step_m=10.0, risk_weights=weights,
                      storage_root=storage_root)

    for s in log.stations:
        # Risk is in [0,1] and equals the sum of its weighted driver terms (glass box, §7.4).
        assert 0.0 <= s.risk <= 1.0
        assert set(s.risk_drivers) == {
            "tempConfidence", "hazard", "dlsExceedance", "structuralUncertainty"
        }
        assert s.risk == pytest.approx(min(1.0, sum(s.risk_drivers.values())), abs=1e-9)
    # Aggregate mean/peak risk are reported (§7.4).
    assert 0.0 <= log.summary.mean_risk <= log.summary.peak_risk <= 1.0


def test_risk_weights_are_tunable(env):
    """Re-weighting changes the score transparently (§7.4 use-case-configurable)."""
    session, layout, storage_root, pid = env
    fem = _build_fused(session, layout, pid)
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    res = solve_survey(DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0),
                       (0.0, 0.0), 0.0, cons)
    well = PlannedWell(
        id=new_id(IdKind.FEATURE), name="W-01", project_id=pid,
        wellhead=(0.0, 0.0), kb_elev_m=0.0, deviation_survey=res.survey, constraints=cons,
    )
    base = predict_log(session, fem, well, md_step_m=20.0, storage_root=storage_root)
    # Crank the structural-uncertainty weight to 1.0 — a different, still-transparent score.
    heavy = predict_log(
        session, fem, well, md_step_m=20.0, storage_root=storage_root,
        risk_weights=RiskWeights(temp_confidence=0.0, hazard=0.0, dls_exceedance=0.0,
                                 structural_uncertainty=1.0),
    )
    assert base.summary.mean_risk != pytest.approx(heavy.summary.mean_risk, abs=1e-6)


# ───────────────────────────────── API surface (§10) ─────────────────────────────────


def test_api_targets_wells_solve_positions_predict(tmp_path):
    """The doc-09 §10 endpoints run end-to-end against a shared in-process app."""
    settings = Settings(storage_root=tmp_path / "store")
    app = create_app(settings)
    tc = TestClient(app)

    # Create a project.
    r = tc.post("/projects", json={"name": "p", "frame": {
        "mode": "local",
        "roi": {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000},
        "depth_range": {"zmin": -2000, "zmax": 0},
    }})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]

    # Build the fused grid using the app's own catalog engine + storage root.
    Session = session_factory(app.state.engine)
    session = Session()
    layout = ensure_project_layout(app.state.storage_root, pid)
    fem = _build_fused(session, layout, pid)
    fem_id = fem.id
    session.close()

    # POST target (enriched).
    rt = tc.post(f"/projects/{pid}/targets", json={
        "fused_model_id": fem_id, "name": "hot", "location": list(HOT_XYZ),
        "desired_temperature_c": 180.0, "min_temperature_c": 150.0, "kb_elev_m": 0.0,
    })
    assert rt.status_code == 201, rt.text
    tgt = rt.json()
    assert tgt["sampled"]["temperatureC"]["value"] > 150.0
    assert tgt["sampled"]["modelVersion"] == fem_id
    tgt_id = tgt["id"]

    # POST well via intent (a build-hold-land solver runs).
    rw = tc.post(f"/projects/{pid}/wells", json={
        "name": "W-01", "wellhead": [0.0, 0.0], "kb_elev_m": 0.0,
        "target_ids": [tgt_id],
        "design": {"method": "build-hold-land", "target": list(HOT_XYZ),
                   "kop_md_m": 500.0, "build_rate_deg30m": 3.0},
        "max_dls_deg30m": 5.0,
    })
    assert rw.status_code == 201, rw.text
    wj = rw.json()
    assert wj["solve"]["landingError_m"] < 50.0
    assert not wj["solve"]["dlsExceeded"]
    wid = wj["id"]

    # GET positions (+ drillability).
    rp = tc.get(f"/wells/{wid}/positions")
    assert rp.status_code == 200, rp.text
    pj = rp.json()
    assert len(pj["md"]) == len(pj["tvd"]) == len(pj["enu"])
    assert pj["drillability"]["verdict"] in ("ok", "warn")

    # POST solve (re-solve).
    rs = tc.post(f"/wells/{wid}/solve", json={
        "design": {"method": "build-hold-land", "target": list(HOT_XYZ), "kop_md_m": 600.0},
        "max_dls_deg30m": 5.0,
    })
    assert rs.status_code == 200, rs.text
    assert rs.json()["solve"]["landingError_m"] < 50.0

    # POST predict (predicted log + summary + risk + drillability).
    rpr = tc.post(f"/wells/{wid}/predict", json={
        "fused_model_id": fem_id, "md_step_m": 10.0, "target_id": tgt_id,
    })
    assert rpr.status_code == 200, rpr.text
    log = rpr.json()
    assert log["summary"]["bhtC"] > 150.0
    assert log["summary"]["inWindowFraction"] >= 0.0
    assert "riskDrivers" in log["stations"][0]
    assert log["drillability"]["verdict"] in ("ok", "warn")
