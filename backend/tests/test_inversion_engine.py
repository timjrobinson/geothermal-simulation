"""Tests for the inversion engine framework (doc 10 §2–§4, §7).

Engine-agnostic harness only — the trivial in-framework :class:`MockLinearEngine` (a
linear toy) stands in for SimPEG/PyGIMLi so the pipeline is proven without a heavy solver
(TINY meshes, FEW iterations). Covered:

- :func:`build_tensor_domain` — core / padding / active-cells correct (doc 10 §4).
- the mock engine runs through the harness → an ordinary PropertyModel + MANDATORY
  uncertainty + InversionProvenance + a fused resample (doc 10 §0, §2.3, §4.4, §7).
- params validation rejects bad input PRE-enqueue (doc 10 §3).
- progress (φ_d / φ_m) + cooperative cancel work (doc 10 §3).
- the engine registers + serves over ``POST /property-models:invert`` returning a job id.
"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api.app import Settings, create_app
from geosim.catalog import (
    Dataset,
    IdKind,
    Observation,
    Project,
    PropertyModel,
    Provenance,
    SpatialFrameRow,
    create_all,
    make_engine,
    new_id,
    session_factory,
)
from geosim.inversion import (
    CoreRegion,
    InversionResult,
    PaddingSpec,
    ParamValidationError,
    build_tensor_domain,
    default_uncertainty,
    run_inversion,
    validate_params,
)
from geosim.inversion.engine import register_inversion_engine
from geosim.inversion.mock import MOCK_SPEC, MockLinearEngine
from geosim.jobs import Cancelled, InlineJobRunner, JobState, ProgressChannel, ProgressReporter
from geosim.plugins import get_registry
from geosim.storage import ensure_project_layout, open_property_model

# ───────────────────────────────── fixtures ─────────────────────────────────


@pytest.fixture
def env(tmp_path):
    """In-memory catalog + temp storage + a seeded project (flat:0 surface) + 1 observation."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="inv-test", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400}),
        depth_range_json=json.dumps({"zmin": -400, "zmax": 0}),
        surface_model="flat:0",
        frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()
    obs_id = _seed_observation(session, pid)
    yield session, layout, tmp_path, pid, obs_id
    session.close()


def _seed_observation(session, pid) -> str:
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    obs_id = new_id(IdKind.OBSERVATION)
    box = {"xmin": 200, "xmax": 200, "ymin": 200, "ymax": 200, "zmin": -200, "zmax": -200}
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="mt-obs", method="mt", kind="obs", status="ready",
        extent_json=json.dumps(box), spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="point",
        primary_property="resistivity",
        values_json=json.dumps({
            "coords": [[-200.0, 200.0, 200.0]],
            "values": {"resistivity": [10.0]},
            "sigma": {"resistivity": [1.0]},
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


def _core() -> CoreRegion:
    return CoreRegion(origin=(-200.0, 0.0, 0.0), extent=(200.0, 200.0, 200.0),
                      cell_size=(40.0, 40.0, 40.0))


# ───────────────────────────── domain: mesh / padding / active cells ─────────────────────────────


def test_tensor_domain_core_padding_active_cells():
    """Core block size, expanding padding, and topography active mask (doc 10 §4)."""
    core = _core()
    assert core.n_core() == (5, 5, 5)  # 200/40

    # No padding, no surface → all cells active, mesh == core.
    plain = build_tensor_domain(core)
    assert plain.mesh.shape_cells == (5, 5, 5)
    assert plain.n_cells == 125
    assert plain.n_active == 125

    # 2 padding cells each side → 5 + 4 = 9 per axis; expanding (factor 1.3).
    dom = build_tensor_domain(core, padding=PaddingSpec(n_pad=2, factor=1.3), surface_z=0.0)
    assert dom.mesh.shape_cells == (9, 9, 9)
    # Padding cells grow geometrically away from the core.
    hx = dom.mesh.h[0]
    assert hx[0] > hx[1]  # on the negative side the OUTERMOST pad cell is the largest
    assert np.isclose(hx[2], 40.0)  # core cells are uniform 40 m

    # Active mask: every core cell sits below z=0 (deepest core top is -40 m), so all core
    # cells are active; air pad cells above the surface are inactive (doc 10 §4.3).
    assert dom.n_active < dom.n_cells
    sz, sy, sx = dom.core_slices()
    assert (sz, sy, sx) == (slice(2, 7), slice(2, 7), slice(2, 7))


def test_extract_core_and_grid_alignment():
    """``extract_core`` returns the (nz,ny,nx) core; core_grid is cell-centred (doc 10 §4.4)."""
    core = _core()
    dom = build_tensor_domain(core, padding=PaddingSpec(n_pad=2))
    cube = dom.extract_core(np.arange(dom.n_cells, dtype=float))
    assert cube.shape == (5, 5, 5)
    (oz, oy, ox), (dz, dy, dx) = dom.core_grid()
    # Cell-centre origin = min corner + half a cell.
    assert np.isclose(oz, -200.0 + 20.0)
    assert (dz, dy, dx) == (40.0, 40.0, 40.0)


# ───────────────────────────── params validation (doc 10 §3) ─────────────────────────────


def test_validate_params_applies_defaults_and_passes():
    out = validate_params({"target_value": 5.0}, dict(MOCK_SPEC.params_schema))
    assert out["target_value"] == 5.0
    assert out["background_value"] == 100.0  # default applied
    assert out["max_iterations"] == 4


def test_validate_params_rejects_missing_required():
    with pytest.raises(ParamValidationError, match="target_value"):
        validate_params({}, dict(MOCK_SPEC.params_schema))


def test_validate_params_rejects_bad_type_enum_and_bounds():
    schema = dict(MOCK_SPEC.params_schema)
    with pytest.raises(ParamValidationError, match="number"):
        validate_params({"target_value": "hot"}, schema)
    with pytest.raises(ParamValidationError, match="minimum"):
        validate_params({"target_value": -1.0}, schema)
    with pytest.raises(ParamValidationError, match="maximum"):
        validate_params({"target_value": 1.0, "max_iterations": 999}, schema)
    with pytest.raises(ParamValidationError, match="unknown"):
        validate_params({"target_value": 1.0, "nope": 1}, schema)


# ───────────────────────────── default uncertainty (doc 10 §2.3) ─────────────────────────────


def test_default_uncertainty_is_depth_inflated_and_finite():
    vals = np.full((4, 2, 2), 100.0, dtype=np.float32)
    sigma = default_uncertainty(vals, "resistivity")
    assert sigma.shape == vals.shape
    assert np.all(np.isfinite(sigma))
    # Z-up: index 0 deepest → larger σ than the shallow top (DOI proxy).
    assert sigma[0].mean() > sigma[-1].mean()


def test_inversion_result_requires_uncertainty():
    with pytest.raises(ValueError, match="MANDATORY"):
        InversionResult(values=np.zeros((2, 2, 2)), sigma=None)
    with pytest.raises(ValueError, match="match"):
        InversionResult(values=np.zeros((2, 2, 2)), sigma=np.zeros((3, 3, 3)))


# ─────────────────────── harness: full pipeline (doc 10 §0, §4.4, §7) ───────────────────────


def test_mock_engine_runs_through_harness(env):
    """Mock engine → PropertyModel + uncertainty + provenance + fused resample (doc 10 §0)."""
    session, layout, storage_root, pid, obs_id = env
    state = JobState(id="job_x", kind="invert")
    reporter = ProgressReporter(state, ProgressChannel(), __import__("threading").Event())

    dom = build_tensor_domain(_core(), padding=PaddingSpec(n_pad=2, factor=1.3), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, MockLinearEngine(),
        domain=dom, observation_ids=[obs_id], params={"target_value": 10.0},
        reporter=reporter,
    )

    # The recovered model is an ORDINARY PropertyModel (doc 10 §0).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm is not None and pm.property == "resistivity" and pm.support == "volume"
    assert pm.uncertainty_uri == "resistivity_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    assert reader.properties == ["resistivity"]
    vol = reader.read_level("resistivity", 0)
    assert vol.shape == (5, 5, 5)  # the CORE block only — padding never leaks (doc 10 §4)

    # MANDATORY uncertainty present + finite (doc 10 §2.3).
    assert reader.has_sigma("resistivity")
    sigma = reader.read_sigma_level("resistivity", 0)
    assert sigma.shape == vol.shape and np.all(np.isfinite(sigma)) and np.all(sigma > 0)

    # InversionProvenance: process tag + engine fingerprint + observation input edge (doc 10 §7).
    prov = session.get(Provenance, res.provenance_id)
    assert prov.process == "invert:mock.linear"
    params = json.loads(prov.params_json)
    assert params["engineId"] == "mock.linear"
    assert params["observationIds"] == [obs_id]
    assert params["mesh"]["type"] == "tensor"
    assert params["mesh"]["core"]["nCore"] == [5, 5, 5]
    assert params["iterations"] == 4
    assert {(i.input_kind, i.input_id) for i in prov.inputs} == {("observation", obs_id)}

    # Fused resample of the recovered core (doc 10 §4.4).
    assert res.fused_model_id is not None and res.fused_layer_id is not None

    # Recovery is in the right ballpark: the anomaly centre pulls toward the target value.
    assert float(np.nanmin(vol)) < 100.0  # below background somewhere (toward target=10)


def test_harness_validates_params_before_running(env):
    """Bad params raise before any catalog write (doc 10 §3)."""
    session, layout, storage_root, pid, obs_id = env
    dom = build_tensor_domain(_core())
    n_before = session.query(PropertyModel).count()
    with pytest.raises(ParamValidationError):
        run_inversion(
            session, layout, pid, MockLinearEngine(),
            domain=dom, observation_ids=[obs_id], params={},  # missing target_value
        )
    assert session.query(PropertyModel).count() == n_before


# ───────────────────────────── progress + cancel (doc 10 §3) ─────────────────────────────


def test_progress_reports_phi_metrics(env):
    """The harness threads φ_d / φ_m progress events through the reporter (doc 10 §3)."""
    session, layout, storage_root, pid, obs_id = env
    state = JobState(id="job_p", kind="invert")
    channel = ProgressChannel()
    reporter = ProgressReporter(state, channel, __import__("threading").Event())

    dom = build_tensor_domain(_core(), padding=PaddingSpec(n_pad=1))
    run_inversion(
        session, layout, pid, MockLinearEngine(),
        domain=dom, observation_ids=[obs_id],
        params={"target_value": 10.0, "max_iterations": 3}, reporter=reporter,
    )
    msgs = [e.message or "" for e in channel.history]
    assert any("phi_d" in m for m in msgs)  # iteration metrics surfaced
    assert channel.history[-1].progress == 1.0  # reaches completion


def test_cooperative_cancel(env):
    """A cancel request aborts the run cooperatively → Cancelled (doc 10 §3)."""
    session, layout, storage_root, pid, obs_id = env
    cancel_event = __import__("threading").Event()
    cancel_event.set()  # pre-cancel: the engine sees it on the first iteration
    state = JobState(id="job_c", kind="invert")
    reporter = ProgressReporter(state, ProgressChannel(), cancel_event)

    dom = build_tensor_domain(_core())
    with pytest.raises(Cancelled):
        run_inversion(
            session, layout, pid, MockLinearEngine(),
            domain=dom, observation_ids=[obs_id],
            params={"target_value": 10.0}, reporter=reporter,
        )


def test_cancel_via_inline_runner_marks_job_cancelled(env):
    """Through the JobRunner state machine a pre-cancelled job ends 'cancelled' (doc 04 §9.4)."""
    session, layout, storage_root, pid, obs_id = env
    runner = InlineJobRunner()
    dom = build_tensor_domain(_core())

    def _job(params, reporter):
        reporter._cancel_event.set()  # request cancel before the engine iterates
        return run_inversion(
            session, layout, pid, MockLinearEngine(),
            domain=dom, observation_ids=[obs_id], params=params, reporter=reporter,
        ).to_payload()

    job_id = runner.enqueue("invert", {"target_value": 10.0}, _job, project_id=pid)
    assert runner.get(job_id).status.value == "cancelled"


# ───────────────────────────── API surface (doc 10 §3, doc 08 §4f) ─────────────────────────────


@pytest.fixture
def registered_engine():
    """Register the mock engine on the process registry for the API tests; clean up after."""
    reg = get_registry()
    engine = register_inversion_engine(MockLinearEngine())
    yield engine
    reg._inversion_engines.pop("mock.linear", None)  # keep the singleton clean between tests


def test_api_lists_engine_and_inverts(tmp_path, registered_engine):
    """End-to-end over the API: engine palette + POST :invert → job → recovered PropertyModel."""
    app = create_app(Settings(storage_root=tmp_path, job_runner=InlineJobRunner()))
    client = TestClient(app)

    # project + frame
    proj = client.post("/projects", json={
        "name": "inv-api",
        "frame": {
            "mode": "local",
            "roi": {"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400},
            "depth_range": {"zmin": -400, "zmax": 0},
            "surface_model": "flat:0",
        },
    }).json()
    pid = proj["id"]

    # seed an observation directly through the app's session factory.
    session = app.state.session_factory()
    obs_id = _seed_observation(session, pid)
    session.close()

    # engine palette (doc 10 §2)
    engines = client.get("/inversion-engines").json()
    assert any(e["id"] == "mock.linear" and e["output_property"] == "resistivity"
               for e in engines)

    body = {
        "project_id": pid,
        "engine_id": "mock.linear",
        "observation_ids": [obs_id],
        "core": {"origin": [-200, 0, 0], "extent": [200, 200, 200], "cell_size": [40, 40, 40]},
        "params": {"target_value": 10.0},
        "n_pad": 2,
    }
    r = client.post("/property-models:invert", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    # Inline runner → job already terminal; reuse GET /jobs/{id} (doc 04 §9.2).
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "succeeded"
    pm_id = job["result"]["propertyModelId"]

    meta = client.get(f"/property-models/{pm_id}").json()
    assert meta["property"] == "resistivity"
    assert meta["hasSigma"] is True
    assert meta["shape"] == [5, 5, 5]


def test_api_rejects_bad_params_before_enqueue(tmp_path, registered_engine):
    """A param schema violation 400s and creates no job (doc 10 §3)."""
    app = create_app(Settings(storage_root=tmp_path, job_runner=InlineJobRunner()))
    client = TestClient(app)
    proj = client.post("/projects", json={
        "name": "inv-bad",
        "frame": {
            "mode": "local",
            "roi": {"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400},
            "depth_range": {"zmin": -400, "zmax": 0},
        },
    }).json()
    pid = proj["id"]
    session = app.state.session_factory()
    obs_id = _seed_observation(session, pid)
    session.close()

    r = client.post("/property-models:invert", json={
        "project_id": pid,
        "engine_id": "mock.linear",
        "observation_ids": [obs_id],
        "core": {"origin": [-200, 0, 0], "extent": [200, 200, 200], "cell_size": [40, 40, 40]},
        "params": {},  # missing required target_value
    })
    assert r.status_code == 400
    assert "target_value" in r.text


def test_api_unknown_engine_404(tmp_path):
    app = create_app(Settings(storage_root=tmp_path, job_runner=InlineJobRunner()))
    client = TestClient(app)
    proj = client.post("/projects", json={
        "name": "inv-404",
        "frame": {
            "mode": "local",
            "roi": {"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400},
            "depth_range": {"zmin": -400, "zmax": 0},
        },
    }).json()
    r = client.post("/property-models:invert", json={
        "project_id": proj["id"],
        "engine_id": "does.not.exist",
        "observation_ids": ["obs_x"],
        "core": {"origin": [-200, 0, 0], "extent": [200, 200, 200], "cell_size": [40, 40, 40]},
        "params": {},
    })
    assert r.status_code == 404
