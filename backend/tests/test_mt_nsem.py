"""Tests for the SimPEG 3-D NSEM magnetotelluric inversion engine (doc 10 §8, §9).

The doc 10 §9 "geophysically-proper resistivity model": forward-simulate a compact
*conductive block* in a resistive half-space with SimPEG NSEM, persist the per-site
apparent-resistivity + phase soundings as platform MT Observations, invert them back
through the engine + harness, and assert the recovered resistivity (a) is a real
``resistivity`` PropertyModel + MANDATORY uncertainty + InversionProvenance, (b) recovers a
CONDUCTOR (the recovered ρ drops below the background somewhere), and (c) LOCALISES the
block — the most-conductive cell sits near the true block centre, within 3-D MT smoothness
limits (doc 10 §8).

Everything is kept TINY + FAST (an 8×8×8 padded mesh ⇒ a 4×4×4 core, 9 sites, 3 periods,
few Gauss-Newton iterations). NSEM uses a sparse DIRECT solver (``pymatsolver`` LU here), so
each forward solve is a complex Maxwell factorisation per period — the test is sized so the
whole forward+invert finishes in a few minutes on the CPU (NO GPU — this 3-D sparse PDE
solve does not benefit from one, doc 10 §8). This proves the pipeline + rough recovery, NOT
production accuracy; the real FORGE 113-site run is documented as an offline heavy job, not
run here.
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
from geosim.inversion.engines.mt_nsem import (
    _AIR_SIGMA,
    MT_NSEM_SPEC,
    SimpegMTInversion,
)
from geosim.plugins import get_registry
from geosim.storage import ensure_project_layout, open_property_model

# True conductive block (Engineering metres, Z-up): centred at (x, y) = (0, 0), z = -800 m.
_BG_RHO = 100.0  # background half-space resistivity (Ω·m)
_BLOCK_RHO = 5.0  # conductive block resistivity (Ω·m)
_BLOCK_XY = (0.0, 0.0)
_BLOCK_Z = -800.0
_PERIODS = np.array([0.1, 1.0, 10.0])  # s (3 periods → shallow→deep sampling)
_ORIENTS = ("xy", "yx")


def _core() -> CoreRegion:
    """A small 4×4×4 core over a 1600 m cube ROI centred under the site grid (doc 10 §9)."""
    return CoreRegion(
        origin=(-1600.0, -800.0, -800.0),  # (z0, y0, x0)
        extent=(1600.0, 1600.0, 1600.0),  # (dz, dy, dx) → (4, 4, 4) cells
        cell_size=(400.0, 400.0, 400.0),
    )


def _padding() -> PaddingSpec:
    """Padding so the NSEM mesh boundary sits far from the target (doc 10 §4.2)."""
    return PaddingSpec(n_pad=2, factor=1.5)


def _site_grid() -> np.ndarray:
    """A 3×3 MT station grid on the flat surface (z = 0), SimPEG ``(x, y, z)`` order."""
    xs = np.linspace(-600.0, 600.0, 3)
    sx, sy = np.meshgrid(xs, xs, indexing="xy")
    return np.column_stack([sx.ravel(), sy.ravel(), np.zeros(sx.size)])


def _forward_block_soundings() -> list[dict]:
    """NSEM-forward a conductive block → per-site apparent-ρ + phase soundings (doc 05 §4).

    Builds the SAME ``TensorMesh`` the inversion will use, puts a conductive block in a
    resistive half-space, forward-simulates ``Z_xy``/``Z_yx`` apparent resistivity + phase
    at a 3×3 site grid across 3 periods, and unpacks the prediction back into per-site
    sounding curves via the survey receiver slices. Returns a list of site dicts the test
    persists as MT ``tensor`` Observations exactly like ingestion would (doc 03 §2).
    """
    import pymatsolver
    from simpeg import maps
    from simpeg.electromagnetics import natural_source as nsem

    domain = build_tensor_domain(_core(), padding=_padding(), surface_z=0.0)
    mesh = domain.mesh
    active = domain.active_cells

    bg_sigma = 1.0 / _BG_RHO
    sigma_primary = np.full(mesh.n_cells, bg_sigma)
    sigma_primary[~active] = _AIR_SIGMA
    act_map = maps.InjectActiveCells(
        mesh=mesh, active_cells=active, value_inactive=np.log(_AIR_SIGMA)
    )
    sigma_map = maps.ExpMap(mesh) * act_map

    sites_xyz = _site_grid()

    # Survey: per-site app-ρ + phase receivers for each orientation; one source per period.
    # Track (site_index, orientation, component, receiver) so we can unpack dpred by slice.
    rx_meta: list[tuple[int, str, str, object]] = []
    per_period_rx: dict[float, list] = {}
    for period in _PERIODS:
        rx_list = []
        for si, loc in enumerate(sites_xyz):
            for orient in _ORIENTS:
                rho_rx = nsem.receivers.Impedance(
                    loc.reshape(1, 3), orientation=orient, component="apparent_resistivity"
                )
                ph_rx = nsem.receivers.Impedance(
                    loc.reshape(1, 3), orientation=orient, component="phase"
                )
                rx_list.append(rho_rx)
                rx_list.append(ph_rx)
                rx_meta.append((si, orient, "rho", rho_rx))
                rx_meta.append((si, orient, "phase", ph_rx))
        per_period_rx[float(period)] = rx_list

    source_list = [
        nsem.sources.PlanewaveXYPrimary(per_period_rx[float(p)], 1.0 / float(p))
        for p in _PERIODS
    ]
    survey = nsem.survey.Survey(source_list)

    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey, sigmaPrimary=sigma_primary, sigmaMap=sigma_map
    )
    sim.solver = pymatsolver.SolverLU

    # True model: a conductive block at the core centre.
    m_true = np.full(int(active.sum()), np.log(bg_sigma))
    cc = mesh.cell_centers[active]
    bx, by = _BLOCK_XY
    block = (
        (np.abs(cc[:, 0] - bx) <= 400.0)
        & (np.abs(cc[:, 1] - by) <= 400.0)
        & (cc[:, 2] <= _BLOCK_Z + 400.0)
        & (cc[:, 2] >= _BLOCK_Z - 400.0)
    )
    m_true[block] = np.log(1.0 / _BLOCK_RHO)
    dpred = np.asarray(sim.dpred(m_true), dtype=float)

    # Unpack dpred into per-site sounding curves (resistivity + phase) keyed by period.
    slices = survey.get_all_slices()
    n_sites = sites_xyz.shape[0]
    # site -> {"rho": {period: val}, "phase": {period: val}} aggregated across orientations.
    rho_curves: list[dict[float, float]] = [{} for _ in range(n_sites)]
    phase_curves: list[dict[float, float]] = [{} for _ in range(n_sites)]
    for src, period in zip(source_list, _PERIODS, strict=True):
        for si, _orient, comp, rx in rx_meta:
            if rx not in src.receiver_list:
                continue
            val = float(dpred[slices[src, rx]][0])
            # Average the two off-diagonal orientations (xy/yx ≈ equal for this 1-D-ish block).
            target = rho_curves if comp == "rho" else phase_curves
            key = float(period)
            target[si][key] = (target[si].get(key, val) + val) / 2.0 if key in target[si] else val

    out: list[dict] = []
    for si in range(n_sites):
        x, y, _z = sites_xyz[si]
        periods_sorted = sorted(rho_curves[si].keys())
        freq = [1.0 / p for p in periods_sorted]
        rho = [rho_curves[si][p] for p in periods_sorted]
        # Engineering phase is stored in milliradians (doc 03 §2 normaliser); convert deg→mrad.
        phase_mrad = [np.radians(phase_curves[si][p]) * 1000.0 for p in periods_sorted]
        out.append({
            "x": float(x), "y": float(y),
            "freq": freq, "rho": rho, "phase_mrad": phase_mrad,
        })
    return out


@pytest.fixture(scope="module")
def soundings() -> list[dict]:
    """Forward-simulate ONCE (the NSEM forward is the slow part) and share across tests."""
    return _forward_block_soundings()


@pytest.fixture
def env(tmp_path, soundings):
    """In-memory catalog + temp storage + project (flat:0) + forward-simulated MT obs."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="mt-inv", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": -800, "xmax": 800, "ymin": -800, "ymax": 800}),
        depth_range_json=json.dumps({"zmin": -1600, "zmax": 0}),
        surface_model="flat:0",
        frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()

    obs_ids = [_seed_mt_observation(session, pid, s) for s in soundings]
    yield session, layout, tmp_path, pid, obs_ids
    session.close()


def _seed_mt_observation(session, pid, sounding: dict) -> str:
    """Persist one site sounding as an MT ``tensor`` Observation (doc 03 §2 / doc 05 §4).

    Coords are Engineering ``(z, y, x)`` Z-up (doc 02 §10.2) — one row per period sample at
    the site. ``values.resistivity`` = apparent ρ (Ω·m), ``values.phase_mrad`` = phase
    (mrad), the frequency axis rides in ``meta.frequency_hz`` (Hz), exactly as the MT EDI
    adapter emits + the engine reads.
    """
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    obs_id = new_id(IdKind.OBSERVATION)
    x, y = sounding["x"], sounding["y"]
    n = len(sounding["freq"])
    coords = [[0.0, float(y), float(x)] for _ in range(n)]  # (z, y, x), all at the site
    box = {
        "xmin": float(x), "xmax": float(x), "ymin": float(y), "ymax": float(y),
        "zmin": 0.0, "zmax": 0.0,
    }
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:synthetic-mt"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="mt-obs", method="mt", kind="obs",
        status="ready", extent_json=json.dumps(box), spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="tensor",
        primary_property="resistivity",
        values_json=json.dumps({
            "coords": coords,
            "values": {
                "resistivity": list(sounding["rho"]),
                "phase_mrad": list(sounding["phase_mrad"]),
            },
            "sigma": {},
        }),
        bbox_json=json.dumps(box),
        meta_json=json.dumps({"frequency_hz": list(sounding["freq"]), "component": "xy"}),
    ))
    session.commit()
    return obs_id


# ──────────────────── spec + registration (doc 10 §2, doc 08 §4f) ────────────────────


def test_engine_spec_and_registration():
    """The engine advertises an mt→resistivity spec + self-registers (doc 10 §2, §8)."""
    spec = MT_NSEM_SPEC
    assert spec.id == "simpeg.mt.nsem"
    assert spec.kind == "mt"
    assert spec.library == "simpeg"
    assert spec.output_property == "resistivity"
    assert "mt" in spec.methods
    assert spec.compute == "worker_process"  # heavy sparse PDE solve (doc 08 §2.1)
    assert spec.mesh_types == ("tensor",)

    reg = get_registry()
    assert "simpeg.mt.nsem" in reg._inversion_engines


def test_collect_sites_parses_soundings(soundings):
    """The site collector unpacks app-ρ/phase curves + the frequency axis (doc 05 §4)."""
    # Build engine-agnostic obs dicts like the harness load_observations would.
    obs = [{
        "coords": [[0.0, s["y"], s["x"]]] * len(s["freq"]),
        "values": {"resistivity": s["rho"], "phase_mrad": s["phase_mrad"]},
        "meta": {"frequency_hz": s["freq"]},
    } for s in soundings]
    sites = SimpegMTInversion._collect_sites(obs)
    assert len(sites) == len(soundings)
    s0 = sites[0]
    assert s0["periods"].size == len(soundings[0]["freq"])
    assert np.all(s0["rho"] > 0)
    assert s0["phase"] is not None
    # SimPEG (x, y, z) location is recovered from the (z, y, x) coords.
    assert s0["loc"][:2] == (soundings[0]["x"], soundings[0]["y"])


def test_collect_sites_requires_mt_data():
    """Observations with no resistivity sounding yield no sites (doc 05 §4)."""
    assert SimpegMTInversion._collect_sites([
        {"coords": [[0.0, 0.0, 0.0]], "values": {"density": [1.0]}, "meta": {}},
    ]) == []


# ──────────────────── full forward→invert pipeline (doc 10 §8, §9) ────────────────────


def test_mt_inversion_recovers_conductor(env):
    """Forward-simulate a conductive block, invert it back → localised conductor (doc 10 §8)."""
    session, layout, storage_root, pid, obs_ids = env

    domain = build_tensor_domain(_core(), padding=_padding(), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, SimpegMTInversion(),
        domain=domain, observation_ids=obs_ids,
        params={
            "background_resistivity": _BG_RHO,
            "max_iterations": 4,
            "beta0_ratio": 1.0,
            "rho_min": 1.0,
            "rho_max": 1000.0,
        },
        resample_fused=True,
    )

    # The recovered model is an ORDINARY resistivity PropertyModel (doc 10 §0).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm is not None
    assert pm.property == "resistivity"
    assert pm.support == "volume"
    assert pm.canonical_unit == "ohm*m"
    assert pm.uncertainty_uri == "resistivity_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    assert reader.properties == ["resistivity"]
    vol = reader.read_level("resistivity", 0)  # (nz, ny, nx) absolute resistivity (Ω·m)
    assert vol.shape == (4, 4, 4)  # CORE only — padding never leaks (doc 10 §4)
    assert np.all(np.isfinite(vol)) and np.all(vol > 0)

    # MANDATORY uncertainty present + positive + finite (doc 10 §2.3).
    assert reader.has_sigma("resistivity")
    sigma = reader.read_sigma_level("resistivity", 0)
    assert sigma.shape == vol.shape
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)

    # (a) recovers a CONDUCTOR: the recovered resistivity drops below the background
    #     somewhere (a conductive block ⇒ a low-ρ anomaly, doc 10 §8).
    assert float(vol.min()) < _BG_RHO
    # the conductive anomaly is the dominant departure (more conductive than resistive).
    log_dev = np.log10(vol / _BG_RHO)
    assert abs(float(log_dev.min())) > abs(float(log_dev.max()))

    # (b) LOCALISES laterally near the true block: the most-conductive cell's (x, y) sits
    #     within ~one core cell of the true block centre (3-D MT smoothness — loose, doc 10 §8).
    (oz, oy, ox), (dz, dy, dx) = domain.core_grid()
    iz, iy, ix = np.unravel_index(int(np.argmin(vol)), vol.shape)
    peak_x = ox + dx * ix
    peak_y = oy + dy * iy
    assert abs(peak_x - _BLOCK_XY[0]) <= 1.5 * dx
    assert abs(peak_y - _BLOCK_XY[1]) <= 1.5 * dy

    # the conductive mass centroid (conductance-weighted) also lands near the block laterally.
    cond = np.clip(1.0 / vol - 1.0 / _BG_RHO, 0.0, None)  # excess conductance
    if cond.sum() > 0:
        xc = ox + dx * np.arange(vol.shape[2])
        yc = oy + dy * np.arange(vol.shape[1])
        wx = cond.sum(axis=(0, 1))
        wy = cond.sum(axis=(0, 2))
        cx = float((xc * wx).sum() / wx.sum())
        cy = float((yc * wy).sum() / wy.sum())
        assert abs(cx - _BLOCK_XY[0]) <= 2.0 * dx
        assert abs(cy - _BLOCK_XY[1]) <= 2.0 * dy

    # Convergence diagnostics recorded (doc 10 §3, §7).
    assert res.iterations >= 1
    assert res.final_phi_d is not None and res.final_phi_d >= 0.0
    assert res.final_phi_m is not None and res.final_phi_m >= 0.0

    # InversionProvenance: engine fingerprint + observation edges + MT diagnostics (doc 10 §7).
    prov = session.get(Provenance, res.provenance_id)
    assert prov.process == "invert:simpeg.mt.nsem"
    pj = json.loads(prov.params_json)
    assert pj["engineId"] == "simpeg.mt.nsem"
    assert pj["engineLibrary"] == "simpeg"
    assert pj["engineKind"] == "mt"
    assert set(pj["observationIds"]) == set(obs_ids)
    assert pj["mesh"]["type"] == "tensor"
    assert pj["metrics"]["nSites"] == len(obs_ids)
    assert pj["metrics"]["gpuAccelerated"] is False  # sparse direct solve — CPU (doc 10 §8)
    assert "SolverLU" in pj["metrics"]["solver"] or "Pardiso" in pj["metrics"]["solver"]
    assert {(i.input_kind, i.input_id) for i in prov.inputs} == {
        ("observation", oid) for oid in obs_ids
    }

    # Fused resample of the recovered core (doc 10 §4.4).
    assert res.fused_model_id is not None
    assert res.fused_layer_id is not None


def test_mt_inversion_requires_soundings(env):
    """An observation set with no MT resistivity sounding is rejected (doc 10 §8)."""
    session, layout, storage_root, pid, _obs_ids = env

    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    empty_id = new_id(IdKind.OBSERVATION)
    box = {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "zmin": 0, "zmax": 0}
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="no-mt", method="mt", kind="obs",
        status="ready", extent_json=json.dumps(box), spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(Observation(
        id=empty_id, dataset_id=ds_id, project_id=pid, geometry_kind="tensor",
        primary_property="resistivity",
        values_json=json.dumps({"coords": [[0.0, 0.0, 0.0]], "values": {}, "sigma": {}}),
        bbox_json=json.dumps(box),
    ))
    session.commit()

    domain = build_tensor_domain(_core(), padding=_padding(), surface_z=0.0)
    with pytest.raises(ValueError, match="no MT soundings"):
        run_inversion(
            session, layout, pid, SimpegMTInversion(),
            domain=domain, observation_ids=[empty_id],
            params={"max_iterations": 1}, resample_fused=False,
        )
