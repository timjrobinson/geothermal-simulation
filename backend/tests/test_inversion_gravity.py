"""Tests for the SimPEG linear gravity inversion engine (doc 10 §8, §9).

The doc 10 §9 "plumbing-proof" engine: forward-simulate a compact dense block with SimPEG,
invert it back through the engine + harness, and assert the recovered density anomaly is
(a) the right SIGN (positive over a dense block), (b) LOCATED correctly *laterally* (the
peak sits near the true block in x/y — gravity has no intrinsic depth resolution, so depth
is only loosely constrained, doc 10 §8), and (c) a real PropertyModel + MANDATORY
uncertainty + InversionProvenance (doc 10 §0, §2.3, §7).

Everything is kept TINY + FAST (a 10×10×8 mesh, few iterations, loose tolerance) — this
proves the pipeline + rough recovery, NOT production accuracy (doc 10 §9).
"""

import json

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
from geosim.inversion import (
    CoreRegion,
    PaddingSpec,
    build_tensor_domain,
    run_inversion,
)
from geosim.inversion.engines.gravity_simpeg import (
    GRAVITY_SIMPEG_SPEC,
    SimpegGravityInversion,
)
from geosim.plugins import get_registry
from geosim.storage import ensure_project_layout, open_property_model

# True dense block (Engineering metres, Z-up): centred at (x, y) = (200, 200), z = -100 m.
_BLOCK_XY = (200.0, 200.0)
_BLOCK_Z = -100.0
_BLOCK_HALF = 60.0
_BLOCK_DRHO = 600.0  # +600 kg/m³ excess density
_BACKGROUND = 2670.0


def _core() -> CoreRegion:
    """A small 10×10×8 core over a 400×400 m ROI, 200 m deep (doc 10 §9 TINY mesh)."""
    return CoreRegion(
        origin=(-200.0, 0.0, 0.0),  # (z0, y0, x0)
        extent=(200.0, 400.0, 400.0),  # (dz, dy, dx) → (8, 10, 10) cells
        cell_size=(25.0, 40.0, 40.0),
    )


def _forward_block_stations() -> tuple[np.ndarray, np.ndarray]:
    """SimPEG-forward a compact dense block → station coords (z,y,x) + Bouguer mGal.

    Builds the SAME kind of ``TensorMesh`` the inversion will use, puts a +Δρ block in it,
    and simulates ``gz`` on a station grid just above the surface. Returns coords in the
    platform Engineering ``(z, y, x)`` order + the anomaly so the test can persist them as
    an Observation exactly like ingestion would (doc 02 §10.2).
    """
    from simpeg import maps
    from simpeg.potential_fields import gravity

    domain = build_tensor_domain(_core(), padding=PaddingSpec(n_pad=1, factor=1.3), surface_z=0.0)
    mesh = domain.mesh
    active = domain.active_cells
    cc = mesh.cell_centers  # (x, y, z)

    true_full = np.zeros(mesh.n_cells)
    bx, by = _BLOCK_XY
    block = (
        (np.abs(cc[:, 0] - bx) <= _BLOCK_HALF)
        & (np.abs(cc[:, 1] - by) <= _BLOCK_HALF)
        & (np.abs(cc[:, 2] - _BLOCK_Z) <= _BLOCK_HALF)
    )
    true_full[block] = _BLOCK_DRHO
    true_active = true_full[active]

    # Station grid just above the flat surface (z = +25 m), covering the ROI.
    xs = np.linspace(40.0, 360.0, 6)
    ys = np.linspace(40.0, 360.0, 6)
    sx, sy = np.meshgrid(xs, ys, indexing="xy")
    sz = np.full(sx.size, 25.0)
    rx_loc = np.column_stack([sx.ravel(), sy.ravel(), sz])  # SimPEG (x, y, z)

    rx = gravity.receivers.Point(rx_loc, components="gz")
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey, mesh=mesh, rhoMap=maps.IdentityMap(nP=int(active.sum())),
        active_cells=active, store_sensitivities="ram",
    )
    dpred = sim.dpred(true_active)

    # Re-order station coords to platform Engineering (z, y, x) Z-up (doc 02 §10.2).
    coords_zyx = np.column_stack([rx_loc[:, 2], rx_loc[:, 1], rx_loc[:, 0]])
    return coords_zyx, np.asarray(dpred, dtype=float)


@pytest.fixture
def env(tmp_path):
    """In-memory catalog + temp storage + project (flat:0) + a forward-simulated gravity obs."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="grav-inv", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400}),
        depth_range_json=json.dumps({"zmin": -200, "zmax": 0}),
        surface_model="flat:0",
        frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()

    coords_zyx, anomaly = _forward_block_stations()
    obs_id = _seed_gravity_observation(session, pid, coords_zyx, anomaly)
    yield session, layout, tmp_path, pid, obs_id
    session.close()


def _seed_gravity_observation(session, pid, coords_zyx, anomaly) -> str:
    """Persist the forward-simulated stations as a ``gravity`` points Observation."""
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    obs_id = new_id(IdKind.OBSERVATION)
    box = {
        "xmin": float(coords_zyx[:, 2].min()), "xmax": float(coords_zyx[:, 2].max()),
        "ymin": float(coords_zyx[:, 1].min()), "ymax": float(coords_zyx[:, 1].max()),
        "zmin": float(coords_zyx[:, 0].min()), "zmax": float(coords_zyx[:, 0].max()),
    }
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:synthetic-gravity"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="grav-obs", method="gravity", kind="obs",
        status="ready", extent_json=json.dumps(box), spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="points",
        primary_property="gravity_anomaly",
        values_json=json.dumps({
            "coords": coords_zyx.tolist(),
            "values": {"gravity_anomaly": anomaly.tolist()},
            "sigma": {},
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


# ──────────────────── spec + registration (doc 10 §2, doc 08 §4f) ────────────────────


def test_engine_spec_and_registration():
    """The engine advertises a gravity→density spec + self-registers (doc 10 §2, §8)."""
    spec = GRAVITY_SIMPEG_SPEC
    assert spec.id == "simpeg.gravity"
    assert spec.kind == "gravity"
    assert spec.library == "simpeg"
    assert spec.output_property == "density"
    assert "gravity" in spec.methods
    assert spec.compute == "worker_process"  # heavy → off the request thread (doc 08 §2.1)
    assert spec.mesh_types == ("tensor",)

    reg = get_registry()
    assert "simpeg.gravity" in reg._inversion_engines


# ──────────────────── full forward→invert pipeline (doc 10 §8, §9) ────────────────────


def test_gravity_inversion_recovers_block(env):
    """Forward-simulate a dense block, invert it back → located density anomaly (doc 10 §8)."""
    session, layout, storage_root, pid, obs_id = env

    domain = build_tensor_domain(_core(), padding=PaddingSpec(n_pad=1, factor=1.3), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, SimpegGravityInversion(),
        domain=domain, observation_ids=[obs_id],
        params={
            "background_density": _BACKGROUND,
            "max_iterations": 6,
            "beta0_ratio": 1.0,
        },
        resample_fused=True,
    )

    # The recovered model is an ORDINARY density PropertyModel (doc 10 §0).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm is not None
    assert pm.property == "density"
    assert pm.support == "volume"
    assert pm.canonical_unit == "kg/m**3"
    assert pm.uncertainty_uri == "density_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    assert reader.properties == ["density"]
    vol = reader.read_level("density", 0)  # (nz, ny, nx) absolute density
    assert vol.shape == (8, 10, 10)  # CORE only — padding never leaks (doc 10 §4)
    assert np.all(np.isfinite(vol))

    # MANDATORY uncertainty present + positive + finite (doc 10 §2.3).
    assert reader.has_sigma("density")
    sigma = reader.read_sigma_level("density", 0)
    assert sigma.shape == vol.shape
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)

    # The recovered anomaly Δρ = density − background.
    d_rho = vol - _BACKGROUND
    # (a) right SIGN: a dense block ⇒ a POSITIVE anomaly somewhere (doc 10 §8).
    assert float(d_rho.max()) > 0.0
    assert float(d_rho.max()) > abs(float(d_rho.min()))  # net excess mass is positive

    # (b) located LATERALLY near the true block (gravity has no depth resolution → only
    #     x/y are well-constrained, doc 10 §8). Find the peak cell's (x, y) and assert it
    #     sits within ~one core cell of the true block centre (loose tolerance).
    (oz, oy, ox), (dz, dy, dx) = domain.core_grid()
    iz, iy, ix = np.unravel_index(int(np.argmax(d_rho)), d_rho.shape)
    peak_x = ox + dx * ix
    peak_y = oy + dy * iy
    assert abs(peak_x - _BLOCK_XY[0]) <= 1.5 * dx
    assert abs(peak_y - _BLOCK_XY[1]) <= 1.5 * dy

    # The lateral mass centroid (positive cells, density-weighted) also lands near the block.
    pos = np.clip(d_rho, 0.0, None)
    if pos.sum() > 0:
        xc = ox + dx * np.arange(d_rho.shape[2])
        yc = oy + dy * np.arange(d_rho.shape[1])
        wx = pos.sum(axis=(0, 1))
        wy = pos.sum(axis=(0, 2))
        cx = float((xc * wx).sum() / wx.sum())
        cy = float((yc * wy).sum() / wy.sum())
        assert abs(cx - _BLOCK_XY[0]) <= 2.0 * dx
        assert abs(cy - _BLOCK_XY[1]) <= 2.0 * dy

    # Convergence diagnostics recorded (doc 10 §3, §7).
    assert res.iterations >= 1
    assert res.final_phi_d is not None and res.final_phi_d >= 0.0
    assert res.final_phi_m is not None and res.final_phi_m >= 0.0

    # InversionProvenance: engine fingerprint + observation edge + gravity diagnostics (doc 10 §7).
    prov = session.get(Provenance, res.provenance_id)
    assert prov.process == "invert:simpeg.gravity"
    params = json.loads(prov.params_json)
    assert params["engineId"] == "simpeg.gravity"
    assert params["engineLibrary"] == "simpeg"
    assert params["engineKind"] == "gravity"
    assert params["observationIds"] == [obs_id]
    assert params["mesh"]["type"] == "tensor"
    assert params["metrics"]["nStations"] == 36
    assert {(i.input_kind, i.input_id) for i in prov.inputs} == {("observation", obs_id)}

    # Fused resample of the recovered core (doc 10 §4.4).
    assert res.fused_model_id is not None
    assert res.fused_layer_id is not None


def test_gravity_inversion_requires_stations(env):
    """An observation set with no gravity_anomaly values is rejected (doc 10 §8)."""
    session, layout, storage_root, pid, _obs_id = env

    # Seed an observation carrying a NON-gravity property only.
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    empty_id = new_id(IdKind.OBSERVATION)
    box = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="no-grav", method="gravity", kind="obs",
        status="ready", extent_json=json.dumps(box), spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(Observation(
        id=empty_id, dataset_id=ds_id, project_id=pid, geometry_kind="points",
        primary_property="gravity_anomaly",
        values_json=json.dumps({"coords": [[0.0, 0.0, 0.0]], "values": {}, "sigma": {}}),
        bbox_json=json.dumps(box),
    ))
    session.commit()

    domain = build_tensor_domain(_core(), surface_z=0.0)
    with pytest.raises(ValueError, match="no gravity stations"):
        run_inversion(
            session, layout, pid, SimpegGravityInversion(),
            domain=domain, observation_ids=[empty_id],
            params={"max_iterations": 2}, resample_fused=False,
        )
