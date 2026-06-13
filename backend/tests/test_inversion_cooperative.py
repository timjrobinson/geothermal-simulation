"""Tests for cooperative (sequential) inversion — doc 10 §6 stage 5b.

Cooperative coupling is an ORCHESTRATION of ordinary §3 single-method jobs (a tiny DAG),
NOT a joint solver (5c stays roadmap, doc 10 §6). These tests stay TINY (mock engines,
small cores, few iterations) and prove the pipeline, not production accuracy:

- stage A runs, then A's recovered CORE model is threaded into stage B as a reference /
  structure weight via ``ctx.reference_model`` (doc 10 §6 5b);
- the DAG completes and each stage is an ordinary persisted PropertyModel (doc 10 §0);
- the 5b coupling is recorded in stage B's InversionProvenance (doc 10 §7);
- the API ``POST /property-models:cooperative-invert`` launches the DAG → a parent job id.
"""

import json
import threading

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
from geosim.inversion import CoreRegion
from geosim.inversion.cooperative import (
    COUPLING_STAGE,
    CooperativeStage,
    ReferenceModel,
    cooperative_invert,
)
from geosim.inversion.engine import (
    InversionContext,
    InversionEngineSpec,
    InversionResult,
    register_inversion_engine,
)
from geosim.inversion.mock import MockLinearEngine
from geosim.jobs import InlineJobRunner, JobState, ProgressChannel, ProgressReporter
from geosim.plugins import get_registry
from geosim.storage import ensure_project_layout, open_property_model

# ───────────────────────────────── fixtures ─────────────────────────────────


@pytest.fixture
def env(tmp_path):
    """In-memory catalog + temp storage + a seeded project (flat:0) + 1 observation."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="coop-test", storage_root=str(tmp_path)))
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
    return CoreRegion(origin=(-200.0, 0.0, 0.0), extent=(160.0, 160.0, 160.0),
                      cell_size=(40.0, 40.0, 40.0))


def _reporter(job_id="job_coop") -> ProgressReporter:
    return ProgressReporter(JobState(id=job_id, kind="invert:cooperative"),
                            ProgressChannel(), threading.Event())


# A tiny engine that RECORDS the reference threaded onto its context (doc 10 §6 5b). It
# proves stage A's recovered model reaches stage B and uses it as a starting model.
class _RefAwareEngine:
    """A mock engine that reads ``ctx.reference_model`` and starts from it (doc 10 §6)."""

    seen_reference: ReferenceModel | None = None

    spec = InversionEngineSpec(
        id="mock.refaware",
        kind="mock",
        library="mock",
        methods=["ert"],
        output_property="resistivity",
        coupling="standalone",
        params_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def run(self, ctx: InversionContext) -> InversionResult:
        ref = getattr(ctx, "reference_model", None)
        self.__class__.seen_reference = ref
        nz, ny, nx = ctx.domain.core.n_core()
        if ref is not None:
            # Start from the partner model (a reference / starting model, doc 10 §6 5b).
            start = np.asarray(ref.values, dtype=float)
        else:
            start = np.full((nz, ny, nx), 50.0, dtype=float)
        ctx.progress(0.5, "refaware", iteration=1, phi_d=0.5, phi_m=0.1)
        recovered = start * 1.01  # a trivial "refinement" of the reference
        sigma = 0.1 * np.abs(recovered) + 1.0
        return InversionResult(
            values=recovered.astype(np.float32),
            sigma=sigma.astype(np.float32),
            iterations=1, final_phi_d=0.5, final_phi_m=0.1,
            metrics={"engine": "mock.refaware"},
        )


# ───────────────────────────── ReferenceModel structure weight ─────────────────────────────


def test_structure_weight_normalised_and_peaks_at_gradients():
    """The structure weight is a normalised gradient-magnitude field (doc 10 §6)."""
    vals = np.zeros((4, 4, 4), dtype=float)
    vals[:, :, 2:] = 100.0  # a sharp x-boundary at index 2
    ref = ReferenceModel(values=vals, property="density", engine_id="a",
                         source_property_model_id="pm_a")
    w = ref.structure_weight()
    assert w.shape == vals.shape
    assert float(w.max()) == pytest.approx(1.0)  # normalised to [0, 1]
    assert float(w.min()) >= 0.0
    # The weight is highest near the boundary (a smooth region is ~0).
    assert w[0, 0, 0] == pytest.approx(0.0)


# ───────────────────────────── the cooperative DAG (doc 10 §6 5b) ─────────────────────────────


def test_cooperative_dag_threads_a_into_b(env):
    """A (mock) → B (ref-aware): A's recovered model reaches B as a reference (doc 10 §6 5b)."""
    session, layout, storage_root, pid, obs_id = env
    _RefAwareEngine.seen_reference = None
    stage_a = CooperativeStage(
        name="A", engine=MockLinearEngine(), observation_ids=[obs_id], core=_core(),
        params={"target_value": 10.0}, n_pad=1, surface_z=0.0,
    )
    stage_b = CooperativeStage(
        name="B", engine=_RefAwareEngine(), observation_ids=[obs_id], core=_core(),
        params={}, depends_on="A", n_pad=1, surface_z=0.0,
    )

    result = cooperative_invert(
        session, layout, pid, [stage_a, stage_b],
        reporter=_reporter(), storage_root=storage_root,
    )

    # The DAG completed in order, both stages persisted (doc 10 §0).
    assert result.order == ["A", "B"]
    assert set(result.stages) == {"A", "B"}

    # Stage A's recovered CORE model was passed into stage B as a reference (doc 10 §6 5b).
    seen = _RefAwareEngine.seen_reference
    assert seen is not None
    assert seen.engine_id == "mock.linear"
    assert seen.property == "resistivity"
    assert seen.source_property_model_id == result.stages["A"].property_model_id
    # B started from A's model: B ≈ A * 1.01.
    a_reader = open_property_model(layout.zarr_path(result.stages["A"].property_model_id))
    a_vol = a_reader.read_level("resistivity", 0)
    assert np.allclose(seen.values, a_vol)


def test_cooperative_b_is_ordinary_property_model_with_uncertainty(env):
    """Stage B's result is an ordinary PropertyModel + MANDATORY uncertainty (doc 10 §0, §2.3)."""
    session, layout, storage_root, pid, obs_id = env
    stages = [
        CooperativeStage(name="A", engine=MockLinearEngine(), observation_ids=[obs_id],
                         core=_core(), params={"target_value": 10.0}, surface_z=0.0),
        CooperativeStage(name="B", engine=_RefAwareEngine(), observation_ids=[obs_id],
                         core=_core(), params={}, depends_on="A", surface_z=0.0),
    ]
    result = cooperative_invert(session, layout, pid, stages, storage_root=storage_root)

    b = result.stages["B"]
    pm = session.get(PropertyModel, b.property_model_id)
    assert pm is not None and pm.property == "resistivity" and pm.support == "volume"
    assert pm.uncertainty_uri == "resistivity_sigma"

    reader = open_property_model(layout.zarr_path(b.property_model_id))
    assert reader.has_sigma("resistivity")
    sigma = reader.read_sigma_level("resistivity", 0)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)


def test_cooperative_provenance_records_5b_coupling(env):
    """Stage B's InversionProvenance records the 5b coupling + partner (doc 10 §7)."""
    session, layout, storage_root, pid, obs_id = env
    stages = [
        CooperativeStage(name="A", engine=MockLinearEngine(), observation_ids=[obs_id],
                         core=_core(), params={"target_value": 10.0}, surface_z=0.0),
        CooperativeStage(name="B", engine=_RefAwareEngine(), observation_ids=[obs_id],
                         core=_core(), params={}, depends_on="A", surface_z=0.0),
    ]
    result = cooperative_invert(session, layout, pid, stages, storage_root=storage_root)

    # The orchestrator records the handoff.
    assert len(result.couplings) == 1
    c = result.couplings[0]
    assert c["stage"] == COUPLING_STAGE == "5b"
    assert c["child"] == "B" and c["partner"] == "A"
    assert c["partnerEngine"] == "mock.linear"

    # Stage B's provenance row carries the coupling in metrics (doc 10 §7).
    prov = session.get(Provenance, result.stages["B"].provenance_id)
    assert prov.process == "invert:mock.refaware"
    params = json.loads(prov.params_json)
    coupling = params["metrics"]["coupling"]
    assert coupling["stage"] == "5b"
    assert coupling["strategy"] == "cooperative"
    assert coupling["partnerEngine"] == "mock.linear"
    assert coupling["partnerProperty"] == "resistivity"
    assert coupling["partnerPropertyModelId"] == result.stages["A"].property_model_id

    # Stage A is a plain standalone run — no coupling recorded (it's the root).
    prov_a = session.get(Provenance, result.stages["A"].provenance_id)
    assert "coupling" not in json.loads(prov_a.params_json)["metrics"]


def test_cooperative_rejects_forward_dependency(env):
    """A stage depending on a not-yet-run stage is rejected (doc 10 §6 ordered DAG)."""
    session, layout, storage_root, pid, obs_id = env
    stages = [
        CooperativeStage(name="B", engine=_RefAwareEngine(), observation_ids=[obs_id],
                         core=_core(), params={}, depends_on="A", surface_z=0.0),
    ]
    with pytest.raises(ValueError, match="depends on"):
        cooperative_invert(session, layout, pid, stages, storage_root=storage_root)


# ───────────────────────────── API surface (doc 10 §6, doc 08 §4f) ─────────────────────────────


@pytest.fixture
def registered_engines():
    """Register the mock + ref-aware engines for the API tests; clean up after."""
    reg = get_registry()
    register_inversion_engine(MockLinearEngine())
    register_inversion_engine(_RefAwareEngine())
    yield
    reg._inversion_engines.pop("mock.linear", None)
    reg._inversion_engines.pop("mock.refaware", None)


def test_api_cooperative_launches_dag(tmp_path, registered_engines):
    """POST :cooperative-invert → parent job id; the DAG runs both stages (doc 10 §6)."""
    app = create_app(Settings(storage_root=tmp_path, job_runner=InlineJobRunner()))
    client = TestClient(app)
    proj = client.post("/projects", json={
        "name": "coop-api",
        "frame": {
            "mode": "local",
            "roi": {"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400},
            "depth_range": {"zmin": -400, "zmax": 0},
            "surface_model": "flat:0",
        },
    }).json()
    pid = proj["id"]
    session = app.state.session_factory()
    obs_id = _seed_observation(session, pid)
    session.close()

    core = {"origin": [-200, 0, 0], "extent": [160, 160, 160], "cell_size": [40, 40, 40]}
    body = {
        "project_id": pid,
        "stages": [
            {"name": "A", "engine_id": "mock.linear", "observation_ids": [obs_id],
             "core": core, "params": {"target_value": 10.0}, "n_pad": 1},
            {"name": "B", "engine_id": "mock.refaware", "observation_ids": [obs_id],
             "core": core, "params": {}, "depends_on": "A", "n_pad": 1},
        ],
    }
    r = client.post("/property-models:cooperative-invert", json=body)
    assert r.status_code == 202, r.text
    payload = r.json()
    assert payload["stages"] == ["A", "B"]
    job_id = payload["job_id"]

    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "succeeded", job
    result = job["result"]
    assert result["strategy"] == "cooperative" and result["stage"] == "5b"
    assert result["order"] == ["A", "B"]
    assert len(result["couplings"]) == 1
    assert result["couplings"][0]["partner"] == "A"

    # The final (B) model is a real, serveable PropertyModel (doc 10 §0).
    pm_id = result["final"]["propertyModelId"]
    meta = client.get(f"/property-models/{pm_id}").json()
    assert meta["property"] == "resistivity"
    assert meta["hasSigma"] is True


def test_api_cooperative_rejects_bad_dependency_before_enqueue(tmp_path, registered_engines):
    """A forward dependency 400s and creates no job (doc 10 §3, §6)."""
    app = create_app(Settings(storage_root=tmp_path, job_runner=InlineJobRunner()))
    client = TestClient(app)
    proj = client.post("/projects", json={
        "name": "coop-bad",
        "frame": {"mode": "local",
                  "roi": {"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400},
                  "depth_range": {"zmin": -400, "zmax": 0}},
    }).json()
    pid = proj["id"]
    session = app.state.session_factory()
    obs_id = _seed_observation(session, pid)
    session.close()

    core = {"origin": [-200, 0, 0], "extent": [160, 160, 160], "cell_size": [40, 40, 40]}
    r = client.post("/property-models:cooperative-invert", json={
        "project_id": pid,
        "stages": [
            {"name": "B", "engine_id": "mock.refaware", "observation_ids": [obs_id],
             "core": core, "params": {}, "depends_on": "A"},
        ],
    })
    assert r.status_code == 400
    assert "depends on" in r.text
