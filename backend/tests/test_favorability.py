"""Tests for the geothermal favorability derived volume (doc 07 §4.6).

Small/coarse grids, local temp dirs, SQLite in-memory — no Docker/Postgres/Redis.

On a small fused grid with a *planted* heat + fluid + permeability anomaly (all three
co-located in one corner) plus a **dry-hot** corner (heat only, no fluid/perm), this checks
the headline non-compensatory property of the shipped default:

- **fuzzy-conjunction (default)** scores HIGH only where all required evidence co-occurs; the
  dry-hot cell scores LOW (an absent required layer pulls the cell toward 0);
- **weighted-linear (exploratory)** scores the dry-hot cell HIGHER than fuzzy does (the
  compensatory blind spot), and its missing-required guard flags/excludes cells lacking a
  required layer;
- the **evidence-overlap** and **assumption-burden** honesty diagnostics are produced (overlap
  = fraction of required layers covering the cell; burden = fraction of evidence whose source
  is an uncalibrated/proxy transform);
- the membership curves (ramp / sigmoid / gaussian-band) map raw → [0,1];
- the ``POST /fused/{gridId}/favorability`` endpoint runs (sync) and rejects deferred Bayesian.
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
from geosim.fusion import (
    Evidence,
    FavorabilitySpec,
    TransferFn,
    build_fused_model,
    compute_favorability,
    fused_grid_from_row,
    membership,
    resample_to_fused,
)
from geosim.spatial import Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)

SHAPE = (4, 4, 4)
ORIGIN = (-200.0, 0.0, 0.0)
SPACING = (50.0, 50.0, 50.0)


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
    frame = SpatialFrame(roi=Aabb(0, 200, 0, 200), depth_range=DepthRange(-200, 0))
    session.add(Project(id=pid, name="favorability-test", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    yield session, layout, storage_root, pid
    session.close()


def _aabb(xmin, xmax, ymin, ymax, zmin, zmax) -> str:
    return json.dumps(
        {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}
    )


def _make_native_pm(session, layout, pid, *, prop, values, unit, method="synthetic",
                    derivation=None):
    """Write a native PropertyModel; optionally stamp a derivation block (proxy/uncal source)."""
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=ORIGIN, spacing=SPACING, cell_ref="center")
    write_property_model(zarr_path, prop, values, grid=grid, overwrite=True)

    nz, ny, nx = values.shape
    oz, oy, ox = ORIGIN
    dz, dy, dx = SPACING
    bbox = _aabb(ox, ox + dx * (nx - 1), oy, oy + dy * (ny - 1), oz, oz + dz * (nz - 1))
    params_json = json.dumps({"derivation": derivation}) if derivation else None
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic",
                           params_json=params_json))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method=method, kind="propertyModel",
        status="ready", extent_json=bbox, spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    row = PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop, canonical_unit=unit,
        support="volume", store_uri=str(zarr_path), shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(SPACING)), origin_json=json.dumps(list(ORIGIN)),
        bbox_json=bbox,
    )
    session.add(row)
    session.commit()
    return row


def _planted_fields():
    """Plant a co-located heat+fluid+perm anomaly + a dry-hot decoy.

    - the [0,0,0] cell ("hot+wet+permeable") has HIGH temperature, fluid, and fracture density;
    - the [-1,-1,-1] cell ("dry-hot") has HIGH temperature but LOW fluid and LOW fracture
      density — the compensatory trap.
    Everywhere else is a low background.
    """
    temp = np.full(SHAPE, 320.0)        # background ~47 °C
    fluid = np.full(SHAPE, 0.1)
    perm = np.full(SHAPE, 0.05)

    temp[0, 0, 0] = 520.0               # hot
    fluid[0, 0, 0] = 0.9                # wet
    perm[0, 0, 0] = 0.9                 # permeable

    temp[-1, -1, -1] = 540.0            # dry-hot decoy: very hot…
    fluid[-1, -1, -1] = 0.05            # …but no fluid…
    perm[-1, -1, -1] = 0.02             # …and no path
    return temp, fluid, perm


def _build_fused_with_evidence(session, layout, pid, *, perm_uncalibrated=False):
    """Build a fused grid with temperature, water_saturation, fracture_density resampled in."""
    temp, fluid, perm = _planted_fields()
    temp_pm = _make_native_pm(session, layout, pid, prop="temperature", values=temp,
                              unit="kelvin")
    fluid_pm = _make_native_pm(session, layout, pid, prop="water_saturation", values=fluid,
                               unit="dimensionless")
    # The permeability proxy may be flagged as an uncalibrated/proxy transform output so it
    # drives the assumption-burden diagnostic (doc 07 §4.6).
    perm_deriv = (
        {"kind": "transform", "calibrationStatus": "uncalibrated", "tier": "proxy"}
        if perm_uncalibrated else None
    )
    perm_pm = _make_native_pm(session, layout, pid, prop="fracture_density", values=perm,
                              unit="dimensionless", derivation=perm_deriv)

    fem, _grid = build_fused_model(
        session, layout, pid,
        source_property_model_ids=[temp_pm.id, fluid_pm.id, perm_pm.id],
        spacing=SPACING, name="fused-fav",
    )
    for pm in (temp_pm, fluid_pm, perm_pm):
        resample_to_fused(session, fem, pm.id, method="trilinear", interp_space="linear")
    session.refresh(fem)
    return fem, temp_pm, fluid_pm, perm_pm


def _evidence(temp_id, fluid_id, perm_id):
    """Heat AND fluid AND permeability, all required (the classic geothermal conjunction)."""
    return [
        Evidence(source=temp_id, target="temperature",
                 transfer=TransferFn("ramp", lo=400.0, hi=520.0), weight=0.4, role="required"),
        Evidence(source=fluid_id, target="water_saturation",
                 transfer=TransferFn("ramp", lo=0.2, hi=0.8), weight=0.3, role="required"),
        Evidence(source=perm_id, target="fracture_density",
                 transfer=TransferFn("ramp", lo=0.1, hi=0.8), weight=0.3, role="required"),
    ]


# ───────────────────────────── membership curves ─────────────────────────────


def test_membership_curves_map_to_unit_interval():
    x = np.array([100.0, 200.0, 300.0, np.nan])
    ramp = membership(x, TransferFn("ramp", lo=150.0, hi=250.0))
    np.testing.assert_allclose(ramp[:3], [0.0, 0.5, 1.0])
    assert np.isnan(ramp[3])  # NaN coverage stays NaN — no invented evidence

    desc = membership(x, TransferFn("ramp", lo=250.0, hi=150.0))  # descending
    np.testing.assert_allclose(desc[:3], [1.0, 0.5, 0.0])

    sig = membership(np.array([0.0, 10.0, 20.0]), TransferFn("sigmoid", center=10.0, k=1.0))
    assert sig[0] < 0.5 < sig[2] and abs(sig[1] - 0.5) < 1e-9

    band = membership(np.array([5.0, 10.0, 15.0]), TransferFn("gaussian-band", center=10.0,
                                                              width=2.0))
    assert band[1] == pytest.approx(1.0) and band[0] < band[1] and band[2] < band[1]


# ─────────────────── fuzzy is non-compensatory; weighted is not ───────────────────


def test_fuzzy_high_only_where_all_required_cooccur(env):
    """Fuzzy-AND scores HIGH only where heat AND fluid AND perm co-occur (doc 07 §4.6)."""
    session, layout, root, pid = env
    fem, t, f, p = _build_fused_with_evidence(session, layout, pid)

    spec = FavorabilitySpec(evidence=_evidence(t.id, f.id, p.id), method="fuzzy")
    result = compute_favorability(session, layout, fem, spec, storage_root=root)

    reader = open_property_model(session.get(PropertyModel, result.model_id).store_uri)
    fav = reader.read_level("favorability", 0)
    assert fav.shape == fused_grid_from_row(fem).shape

    hot_wet_perm = fav[0, 0, 0]
    dry_hot = fav[-1, -1, -1]
    background = fav[1, 1, 1]

    # the co-located play scores high; the dry-hot decoy scores LOW (non-compensatory);
    # the dry-hot cell is no better than the cold background despite soaring temperature.
    assert hot_wet_perm > 0.7
    assert dry_hot < 0.2
    assert dry_hot < hot_wet_perm
    assert dry_hot <= background + 0.05


def test_weighted_scores_dry_hot_higher_than_fuzzy(env):
    """Weighted-linear is compensatory: the dry-hot cell scores HIGHER than under fuzzy."""
    session, layout, root, pid = env
    fem, t, f, p = _build_fused_with_evidence(session, layout, pid)
    ev = _evidence(t.id, f.id, p.id)

    fuzzy = compute_favorability(
        session, layout, fem,
        FavorabilitySpec(evidence=ev, method="fuzzy"), storage_root=root,
    )
    # weighted with neutral missing-policy so the dry-hot cell stays scored (not NaN'd).
    weighted = compute_favorability(
        session, layout, fem,
        FavorabilitySpec(evidence=ev, method="weighted", missing_policy="neutral"),
        storage_root=root,
    )

    f_reader = open_property_model(session.get(PropertyModel, fuzzy.model_id).store_uri)
    w_reader = open_property_model(session.get(PropertyModel, weighted.model_id).store_uri)
    f_fav = f_reader.read_level("favorability", 0)
    w_fav = w_reader.read_level("favorability", 0)

    # the compensatory blind spot: weighted lets the soaring temperature inflate the dry-hot
    # cell well above fuzzy's near-zero score.
    assert w_fav[-1, -1, -1] > f_fav[-1, -1, -1] + 0.2
    # at the genuine play both agree it is favorable.
    assert w_fav[0, 0, 0] > 0.6 and f_fav[0, 0, 0] > 0.6


def test_weighted_missing_required_guard_excludes_uncovered_cells(env):
    """A cell missing a required layer is flagged + excluded, not averaged (doc 07 §4.6)."""
    session, layout, root, pid = env
    # Plant a NaN region in the permeability layer → those cells lack a required layer.
    temp, fluid, perm = _planted_fields()
    perm[0, :, :] = np.nan  # whole top slab has no permeability coverage
    t = _make_native_pm(session, layout, pid, prop="temperature", values=temp, unit="kelvin")
    f = _make_native_pm(session, layout, pid, prop="water_saturation", values=fluid,
                        unit="dimensionless")
    p = _make_native_pm(session, layout, pid, prop="fracture_density", values=perm,
                        unit="dimensionless")
    fem, _g = build_fused_model(session, layout, pid,
                                source_property_model_ids=[t.id, f.id, p.id],
                                spacing=SPACING, name="fused-guard")
    for pm in (t, f, p):
        resample_to_fused(session, fem, pm.id, method="trilinear", interp_space="linear")
    session.refresh(fem)

    ev = _evidence(t.id, f.id, p.id)
    weighted = compute_favorability(
        session, layout, fem,
        FavorabilitySpec(evidence=ev, method="weighted", missing_policy="nodata"),
        storage_root=root,
    )
    reader = open_property_model(session.get(PropertyModel, weighted.model_id).store_uri)
    fav = reader.read_level("favorability", 0)
    overlap = open_property_model(
        session.get(PropertyModel, weighted.overlap_model_id).store_uri
    ).read_level("evidence_overlap", 0)

    # cells with no permeability (top slab) are flagged: excluded from the favorability output…
    assert np.isnan(fav[0]).all()
    assert weighted.n_missing_required > 0
    # …and their overlap is recorded < 1 where it is scored elsewhere (the honesty diagnostic).
    finite_overlap = overlap[np.isfinite(overlap)]
    assert finite_overlap.size > 0 and finite_overlap.max() == pytest.approx(1.0)


# ─────────────────────────── honesty diagnostics ───────────────────────────


def test_overlap_and_assumption_burden_diagnostics_produced(env):
    """Evidence-overlap + assumption-burden companion volumes are written (doc 07 §4.6)."""
    session, layout, root, pid = env
    fem, t, f, p = _build_fused_with_evidence(session, layout, pid, perm_uncalibrated=True)

    spec = FavorabilitySpec(evidence=_evidence(t.id, f.id, p.id), method="fuzzy")
    result = compute_favorability(session, layout, fem, spec, storage_root=root)

    # all four volumes catalogued and readable.
    assert result.confidence_model_id and result.overlap_model_id and result.burden_model_id
    overlap = open_property_model(
        session.get(PropertyModel, result.overlap_model_id).store_uri
    ).read_level("evidence_overlap", 0)
    burden = open_property_model(
        session.get(PropertyModel, result.burden_model_id).store_uri
    ).read_level("assumption_burden", 0)
    conf = open_property_model(
        session.get(PropertyModel, result.confidence_model_id).store_uri
    ).read_level("confidence", 0)

    # all three required layers cover the whole grid here ⇒ overlap == 1 everywhere scored.
    finite = np.isfinite(overlap)
    np.testing.assert_allclose(overlap[finite], 1.0)
    # the permeability evidence is an uncalibrated/proxy source → 1 of 3 contributing ⇒ ~1/3.
    np.testing.assert_allclose(burden[np.isfinite(burden)], 1.0 / 3.0, atol=1e-6)
    # confidence = overlap*(1-burden) = 1*(1-1/3) = 2/3 here.
    np.testing.assert_allclose(conf[np.isfinite(conf)], 2.0 / 3.0, atol=1e-6)

    # the derivation block records kind=favorability + the evidence + proxy flags (doc 07 §4.3).
    pm = session.get(PropertyModel, result.model_id)
    prov = session.get(Provenance, session.get(Dataset, pm.dataset_id).provenance_id)
    deriv = json.loads(prov.params_json)["derivation"]
    assert deriv["kind"] == "favorability" and deriv["method"] == "fuzzy"
    proxies = [e["proxy"] for e in deriv["evidence"]]
    assert proxies.count(True) == 1  # only the permeability source is proxy


# ─────────────────────────────── REST endpoint ───────────────────────────────


@pytest.fixture
def client_env(tmp_path):
    storage_root = tmp_path / "store"
    storage_root.mkdir()
    app = create_app(Settings(storage_root=str(storage_root)))
    client = TestClient(app)
    Session = app.state.session_factory
    session = Session()

    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    frame = SpatialFrame(roi=Aabb(0, 200, 0, 200), depth_range=DepthRange(-200, 0))
    session.add(Project(id=pid, name="rest-fav", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({"xmin": 0, "xmax": 200, "ymin": 0, "ymax": 200}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()
    fem, t, f, p = _build_fused_with_evidence(session, layout, pid)
    ids = (fem.id, t.id, f.id, p.id)
    session.close()
    yield client, pid, ids


def _evidence_payload(t_id, f_id, p_id):
    return [
        {"source": t_id, "target": "temperature",
         "transferFn": {"type": "ramp", "lo": 400.0, "hi": 520.0}, "weight": 0.4,
         "role": "required"},
        {"source": f_id, "target": "water_saturation",
         "transferFn": {"type": "ramp", "lo": 0.2, "hi": 0.8}, "weight": 0.3,
         "role": "required"},
        {"source": p_id, "target": "fracture_density",
         "transferFn": {"type": "ramp", "lo": 0.1, "hi": 0.8}, "weight": 0.3,
         "role": "required"},
    ]


def test_endpoint_favorability_sync(client_env):
    client, pid, (fem_id, t_id, f_id, p_id) = client_env
    r = client.post(f"/fused/{fem_id}/favorability", json={
        "project_id": pid, "method": "fuzzy",
        "evidence": _evidence_payload(t_id, f_id, p_id),
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "sync"
    assert body["output_property"] == "favorability"
    assert body["model_id"] and body["confidence_model_id"]
    assert body["overlap_model_id"] and body["burden_model_id"]
    assert body["n_required"] == 3


def test_endpoint_favorability_bayesian_deferred_is_400(client_env):
    client, pid, (fem_id, t_id, f_id, p_id) = client_env
    r = client.post(f"/fused/{fem_id}/favorability", json={
        "project_id": pid, "method": "bayesian",
        "evidence": _evidence_payload(t_id, f_id, p_id),
    })
    assert r.status_code == 400, r.text
    assert "deferred" in r.json()["detail"].lower()
