"""Tests for the PyGIMLi ERT inversion engine (doc 10 §8, §9).

A TINY end-to-end ERT round-trip: PyGIMLi forward-simulates a single-anomaly (conductive
block in a resistive halfspace) dipole-dipole pseudosection, the engine inverts it back, and
we assert the recovered resistivity recovers the anomaly's sign + location within ERT's
(loose) resolution and that the harness turns it into an ordinary PropertyModel + a MANDATORY
coverage-derived uncertainty + InversionProvenance (doc 10 §0, §2.3, §7).

Everything is kept FAST (a short electrode line, a coarse PyGIMLi auto-mesh, few iterations,
loose tolerance) — this proves the PIPELINE + rough recovery, not production accuracy
(per the component brief).
"""

import json
import threading

import numpy as np
import pytest

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
from geosim.inversion import CoreRegion, PaddingSpec, build_tensor_domain, run_inversion
from geosim.inversion.engines.ert_pygimli import ERT_PYGIMLI_SPEC, PygimliERTInversion
from geosim.jobs import JobState, ProgressChannel, ProgressReporter
from geosim.storage import ensure_project_layout, open_property_model

pytestmark = pytest.mark.filterwarnings("ignore")


# ───────────────────────────── synthetic ERT survey (PyGIMLi forward) ─────────────────────────────


# Survey geometry (Engineering metres, W-E line at y=0, flat surface z=0). Kept short to stay
# fast: 12 electrodes × 10 m = a 110 m line, dipole-dipole.
N_ELEC = 12
SPACING = 10.0
BACKGROUND_RES = 100.0  # ohm·m halfspace
ANOMALY_RES = 10.0  # ohm·m conductive block
ANOMALY_X = 55.0  # block centre along the line
ANOMALY_DEPTH = 15.0  # block centre depth (m below surface)
ANOMALY_R = 8.0  # block radius (m)


def _simulate_pseudosection(seed: int = 7):
    """PyGIMLi forward sim → (electrode XY, ABMN quadrupoles, apparent ρ) for one block.

    A conductive circular block in a resistive halfspace, sampled by a dipole-dipole line.
    Returns plain NumPy so the platform Observation can be built without any PyGIMLi types
    (mirroring how a real ingested ERT pseudosection arrives, doc 03 §2).
    """
    import pygimli.meshtools as mt
    import pygimli.physics.ert as ert

    ex = np.arange(N_ELEC) * SPACING
    scheme = ert.createData(elecs=ex, schemeName="dd")

    world = mt.createWorld(
        start=[-30, 0], end=[float(ex[-1]) + 30, -60], worldMarker=True
    )
    block = mt.createCircle(pos=[ANOMALY_X, -ANOMALY_DEPTH], radius=ANOMALY_R, marker=2)
    mesh = mt.createMesh(world + block, quality=32, area=8.0)
    rhomap = [[1, BACKGROUND_RES], [2, ANOMALY_RES]]

    data = ert.simulate(
        mesh, scheme=scheme, res=rhomap, noiseLevel=0.02, noiseAbs=1e-6, seed=seed
    )

    # sensor XY (line at y=0) + ABMN sensor indices + apparent resistivity.
    sx = np.asarray(data.sensors())[:, 0]
    elec_xy = np.column_stack([sx, np.zeros_like(sx)])
    abmn = np.column_stack([
        np.asarray(data["a"], dtype=int),
        np.asarray(data["b"], dtype=int),
        np.asarray(data["m"], dtype=int),
        np.asarray(data["n"], dtype=int),
    ])
    rhoa = np.asarray(data["rhoa"], dtype=float)
    keep = np.isfinite(rhoa) & (rhoa > 0)
    return elec_xy, abmn[keep], rhoa[keep]


def _electrodes_payload(elec_xy, abmn):
    """Build the ``meta.electrodes`` A/B/M/N XY lists the ERT Observation carries (doc 03 §2)."""
    return {
        k: [[float(elec_xy[idx, 0]), float(elec_xy[idx, 1])] for idx in abmn[:, col]]
        for k, col in (("a", 0), ("b", 1), ("m", 2), ("n", 3))
    }


# ───────────────────────────── env fixture ─────────────────────────────


@pytest.fixture(scope="module")
def survey():
    """Forward-simulate the pseudosection once for the module (the slow part)."""
    return _simulate_pseudosection()


@pytest.fixture
def env(tmp_path, survey):
    """In-memory catalog + temp storage + a project (flat:0 surface) + the ERT Observation."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="ert-inv", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": -30, "xmax": 140, "ymin": -20, "ymax": 20}),
        depth_range_json=json.dumps({"zmin": -60, "zmax": 0}),
        surface_model="flat:0",
        frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()
    obs_id = _seed_ert_observation(session, pid, survey)
    yield session, layout, tmp_path, pid, obs_id
    session.close()


def _seed_ert_observation(session, pid, survey) -> str:
    """Seed one ``profile2d`` ERT Observation from the simulated pseudosection (doc 03 §2)."""
    elec_xy, abmn, rhoa = survey
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    obs_id = new_id(IdKind.OBSERVATION)
    midx = 0.25 * (
        elec_xy[abmn[:, 0], 0] + elec_xy[abmn[:, 1], 0]
        + elec_xy[abmn[:, 2], 0] + elec_xy[abmn[:, 3], 0]
    )
    box = {
        "xmin": float(elec_xy[:, 0].min()), "xmax": float(elec_xy[:, 0].max()),
        "ymin": 0.0, "ymax": 0.0, "zmin": -50.0, "zmax": 0.0,
    }
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:ert-stg-v1"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="ert-line", method="ert", kind="obs", status="ready",
        extent_json=json.dumps(box), spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    coords = np.column_stack([
        np.full(midx.shape, -10.0), np.zeros_like(midx), midx,
    ]).tolist()  # (z, y, x) Engineering — only meta.electrodes is used by the engine
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="profile2d",
        primary_property="resistivity",
        values_json=json.dumps({
            "coords": coords,
            "values": {"resistivity": [float(v) for v in rhoa]},
        }),
        meta_json=json.dumps({
            "array": "dipole-dipole",
            "electrodes": _electrodes_payload(elec_xy, abmn),
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


def _core() -> CoreRegion:
    """A coarse CORE block straddling the survey line + the anomaly (Z-up, (z,y,x))."""
    # x: 0..110 (the line), y: thin slab around y=0, z: -50..0 (surface). 10 m core cells.
    return CoreRegion(
        origin=(-50.0, -10.0, 0.0), extent=(50.0, 20.0, 110.0), cell_size=(10.0, 10.0, 10.0)
    )


# ───────────────────────────── spec / registration ─────────────────────────────


def test_spec_is_pygimli_ert_resistivity():
    spec = ERT_PYGIMLI_SPEC
    assert spec.id == "pygimli.ert"
    assert spec.kind == "ert"
    assert spec.library == "pygimli"
    assert "ert" in spec.methods
    assert spec.output_property == "resistivity"
    assert spec.compute == "worker_process"  # heavy native solve (doc 08 §2.1, doc 10 §8)


def test_engine_self_registered_on_import():
    """Importing the module registers ``pygimli.ert`` on the process registry (doc 08 §4f)."""
    from geosim.plugins import get_registry

    ids = {e.spec.id for e in get_registry().inversion_engines()}
    assert "pygimli.ert" in ids


# ───────────────────────────── full ERT round-trip (doc 10 §0, §8) ─────────────────────────────


def test_ert_inversion_recovers_conductive_anomaly(env):
    """Forward-simulated conductive block → invert → recovered ρ + uncertainty + provenance."""
    session, layout, storage_root, pid, obs_id = env
    state = JobState(id="job_ert", kind="invert")
    reporter = ProgressReporter(state, ProgressChannel(), threading.Event())

    dom = build_tensor_domain(_core(), padding=PaddingSpec(n_pad=1, factor=1.3), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, PygimliERTInversion(),
        domain=dom, observation_ids=[obs_id],
        params={"lam": 20.0, "max_iterations": 4, "para_dx": 0.3},
        reporter=reporter,
    )

    # Output is an ORDINARY resistivity PropertyModel (doc 10 §0).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm is not None
    assert pm.property == "resistivity" and pm.support == "volume"
    assert pm.canonical_unit == "ohm*m"  # registry-driven unit
    assert pm.uncertainty_uri == "resistivity_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    nz, ny, nx = dom.core.n_core()
    vol = reader.read_level("resistivity", 0)
    assert vol.shape == (nz, ny, nx)  # CORE block only — PyGIMLi mesh never leaks (doc 10 §4)

    # MANDATORY uncertainty present, finite, positive (doc 10 §2.3).
    assert reader.has_sigma("resistivity")
    sigma = reader.read_sigma_level("resistivity", 0)
    assert sigma.shape == vol.shape
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)

    # ── recovery: sign + location within ERT's (loose) resolution ──
    # Recovered resistivity is finite + physical everywhere.
    assert np.all(np.isfinite(vol)) and np.all(vol > 0)

    # The conductive block is genuinely recovered: the recovered minimum sits well below the
    # 100 ohm·m background (toward the 10 ohm·m anomaly), proving the sign of the anomaly.
    assert float(np.min(vol)) < 0.6 * BACKGROUND_RES

    # ...and it is LOCATED near the true block: take the y-slab centre column, find the most
    # conductive core cell, and assert it is in the right ballpark in (x, depth).
    (oz, oy, ox), (dz, dy, dx) = dom.core_grid()
    jy = ny // 2  # centre y-slab (the plane of the line)
    slab = vol[:, jy, :]  # (z, x)
    kz, kx = np.unravel_index(int(np.argmin(slab)), slab.shape)
    x_hit = ox + dx * kx
    z_hit = oz + dz * kz
    # loose: within ~25 m along the line and ~25 m in depth of the true block centre.
    assert abs(x_hit - ANOMALY_X) <= 25.0, f"x {x_hit} vs {ANOMALY_X}"
    assert abs((-z_hit) - ANOMALY_DEPTH) <= 25.0, f"depth {-z_hit} vs {ANOMALY_DEPTH}"

    # Convergence record + InversionProvenance fingerprint (doc 10 §3, §7).
    assert res.iterations >= 1
    prov = session.get(Provenance, res.provenance_id)
    assert prov.process == "invert:pygimli.ert"
    params = json.loads(prov.params_json)
    assert params["engineId"] == "pygimli.ert"
    assert params["engineLibrary"] == "pygimli"
    assert params["observationIds"] == [obs_id]
    assert params["mesh"]["type"] == "tensor"
    assert params["metrics"]["nMeasurements"] >= 10
    assert params["metrics"]["paraDomainCells"] >= 1
    assert {(i.input_kind, i.input_id) for i in prov.inputs} == {("observation", obs_id)}

    # Fused resample of the recovered core (doc 10 §4.4).
    assert res.fused_model_id is not None and res.fused_layer_id is not None


def test_ert_uncertainty_tracks_coverage(env):
    """Coverage-derived σ is largest where PyGIMLi resolves least (deep cells; doc 10 §2.3)."""
    session, layout, storage_root, pid, obs_id = env
    dom = build_tensor_domain(_core(), padding=PaddingSpec(n_pad=1), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, PygimliERTInversion(),
        domain=dom, observation_ids=[obs_id],
        params={"lam": 20.0, "max_iterations": 3},
        resample_fused=False,
    )
    reader = open_property_model(layout.zarr_path(res.property_model_id))
    vol = reader.read_level("resistivity", 0)
    sigma = reader.read_sigma_level("resistivity", 0)

    # Relative σ deepest (Z-up index 0) exceeds relative σ shallowest — ERT loses resolution
    # with depth, so the coverage-weighted σ inflates there (doc 10 §2.3).
    rel = sigma / np.maximum(vol, 1e-6)
    assert rel[0].mean() > rel[-1].mean()


def test_ert_inversion_rejects_empty_observations(env):
    """An ERT inversion with no electrode quadrupoles is a hard error (doc 03 §2)."""
    session, layout, storage_root, pid, obs_id = env
    # Replace the obs payload with one that has no electrode geometry.
    empty_id = new_id(IdKind.OBSERVATION)
    obs = session.get(Observation, obs_id)
    session.add(Observation(
        id=empty_id, dataset_id=obs.dataset_id, project_id=pid, geometry_kind="profile2d",
        primary_property="resistivity",
        values_json=json.dumps({"coords": [], "values": {"resistivity": []}}),
        meta_json=json.dumps({"array": "dipole-dipole"}),
        bbox_json=obs.bbox_json,
    ))
    session.commit()
    dom = build_tensor_domain(_core())
    with pytest.raises(ValueError, match="no ERT measurements"):
        run_inversion(
            session, layout, pid, PygimliERTInversion(),
            domain=dom, observation_ids=[empty_id], params={}, resample_fused=False,
        )
