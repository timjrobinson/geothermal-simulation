"""Real Utah FORGE gravity inversion validation (doc 10 §8, §9).

The synthetic ``test_inversion_gravity.py`` proves the forward→invert *pipeline*; this test
proves the SimPEG gravity engine recovers a **sensible density model from the REAL Utah
FORGE field survey** (``data/utah-forge/measured/gravity/Utah_FORGE_Gravity_Data.txt``,
the complete Bouguer anomaly ``gCBGA`` over ~3700 stations).

Flow (mirrors ``data/load_utah_forge.py`` / ``test_real_forge_load.py``):

1. Build the georeferenced FORGE project via :meth:`SpatialFrame.for_real_site`
   (``frame.json`` anchor) and **ingest the real native file** with the production
   :func:`geosim.ingestion.ingest_file` pipeline — this exercises the real gravity adapter +
   normalizer (lon/lat EPSG:4326 → Engineering metres, mGal canonicalised), proving the
   field format actually loads.
2. From the ingested Engineering-frame stations, cut a **tractable local window** around the
   FORGE site and remove the regional trend (a standard planar regional/residual separation
   — the absolute complete-Bouguer field is a regional ~-200 mGal signal; the density
   inversion fits the *residual* anomaly). The windowed residual is persisted as the
   ``gravity_anomaly`` Observation the engine inverts.
3. Build a COARSE :class:`ModelDomain` over the window (1 km lateral / 0.5 km vertical core
   cells, 3 km deep) so the linear inversion finishes in a few minutes on CPU (doc 10 §9),
   run :class:`SimpegGravityInversion` through the harness, and assert a **sensible recovered
   density-anomaly model**: finite values in a plausible range, spatially coherent, with the
   data misfit decreasing over the iterations (doc 10 §3, §8).

Skips gracefully if the real dataset is not present on disk. Kept CI-fast: a small window,
a coarse mesh, and few iterations (the heavy production run on a fine mesh is documented in
the engine, not run here — doc 10 §9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
FORGE = REPO / "data" / "utah-forge"
GRAVITY_FILE = FORGE / "measured" / "gravity" / "Utah_FORGE_Gravity_Data.txt"

_needs_forge = pytest.mark.skipif(
    not GRAVITY_FILE.exists(), reason="real Utah FORGE gravity dataset not present"
)

# Local inversion window around the FORGE anchor (Engineering metres, half-width).
_WINDOW_M = 6000.0
_BACKGROUND = 2670.0  # the survey's Bouguer reduction density (metadata: 2.67 g/cm³)


# ─────────────────────────────── project + ingest fixture ───────────────────────────────


@pytest.fixture(scope="module")
def forge_project(tmp_path_factory):
    """A georeferenced FORGE project with the REAL gravity file ingested (doc 03, doc 10)."""
    from geosim.api.frame_io import frame_row_kwargs
    from geosim.catalog.db import create_all, make_engine, session_factory
    from geosim.catalog.ids import IdKind, new_id
    from geosim.catalog.models import Project, SpatialFrameRow
    from geosim.ingestion import ingest_file
    from geosim.spatial import Aabb, DepthRange, SpatialFrame
    from geosim.storage import ensure_project_layout

    f = json.loads((FORGE / "frame.json").read_text())
    frame = SpatialFrame.for_real_site(
        lon=f["anchor_lonlat"][0],
        lat=f["anchor_lonlat"][1],
        surface_elev=f["surface_elev_m"],
        roi=Aabb(**f["roi"]),
        depth_range=DepthRange(**f["depth_range"]),
    )

    root = tmp_path_factory.mktemp("forge-gravity")
    engine = make_engine(f"sqlite:///{root / 'catalog.db'}")
    create_all(engine)
    Session = session_factory(engine)
    pid = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=pid, name="FORGE gravity", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(project_id=pid, **frame_row_kwargs(frame))
        s.add(proj)
        s.commit()
    layout = ensure_project_layout(root, pid)

    # Ingest the real native file through the production pipeline (doc 03 §7).
    with Session() as s:
        rep = ingest_file(
            s, root, pid, GRAVITY_FILE, method_hint="gravity", crs_hint="EPSG:4326"
        )
        s.commit()
    status = rep.status.value if hasattr(rep.status, "value") else str(rep.status)
    assert status in {"ok", "ok_with_warnings"}, f"FORGE gravity ingest failed: {rep.warnings}"
    assert rep.n_observations >= 1

    return root, pid, Session, layout, frame


def _ingested_gravity(session, project_id):
    """Return the ingested gravity Observation's Engineering ``(z, y, x)`` coords + anomaly."""
    from geosim.catalog.models import Observation

    row = (
        session.query(Observation)
        .filter_by(project_id=project_id, primary_property="gravity_anomaly")
        .first()
    )
    assert row is not None, "no ingested gravity observation found"
    payload = json.loads(row.values_json)
    coords = np.asarray(payload["coords"], dtype=float)  # (N, 3) Engineering (z, y, x)
    anomaly = np.asarray(payload["values"]["gravity_anomaly"], dtype=float)  # mGal
    return coords, anomaly


def _seed_residual_observation(session, project_id, coords_zyx, residual):
    """Persist a windowed/detrended residual gravity Observation for the inversion."""
    from geosim.catalog.ids import IdKind, new_id
    from geosim.catalog.models import Dataset, Observation, Provenance

    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    obs_id = new_id(IdKind.OBSERVATION)
    box = {
        "xmin": float(coords_zyx[:, 2].min()), "xmax": float(coords_zyx[:, 2].max()),
        "ymin": float(coords_zyx[:, 1].min()), "ymax": float(coords_zyx[:, 1].max()),
        "zmin": float(coords_zyx[:, 0].min()), "zmax": float(coords_zyx[:, 0].max()),
    }
    session.add(Provenance(
        id=prov_id, project_id=project_id, target_kind="obs", target_id=ds_id,
        process="derive:gravity-residual",
    ))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=project_id, name="forge-gravity-residual", method="gravity",
        kind="obs", status="ready", extent_json=json.dumps(box), spatial_frame_id=project_id,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="test",
    ))
    session.flush()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=project_id, geometry_kind="points",
        primary_property="gravity_anomaly",
        values_json=json.dumps({
            "coords": coords_zyx.tolist(),
            "values": {"gravity_anomaly": residual.tolist()},
            "sigma": {},
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


def _window_and_detrend(coords_zyx, anomaly):
    """Cut a ±``_WINDOW_M`` window around the FORGE anchor + remove the regional trend.

    Returns the windowed station coords (receivers lifted just above the local surface, so
    every station sits in the active air halfspace) + the planar-detrended residual anomaly.
    """
    z, y, x = coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]
    win = (np.abs(x) <= _WINDOW_M) & (np.abs(y) <= _WINDOW_M) & np.isfinite(anomaly)
    xw, yw, zw, gw = x[win], y[win], z[win], anomaly[win]

    # Planar regional/residual separation: fit g ≈ a + b·x + c·y, invert the residual.
    design = np.column_stack([np.ones_like(xw), xw, yw])
    coef, *_ = np.linalg.lstsq(design, gw, rcond=None)
    residual = gw - design @ coef

    # Lift receivers a metre above the highest local station so they are clear of the
    # active topography (the engine masks air cells at the surface elevation).
    z_rx = np.full_like(zw, float(zw.max()) + 1.0)
    coords_out = np.column_stack([z_rx, yw, xw])
    return coords_out, residual, float(zw.max())


# ─────────────────────────────── the real validation ───────────────────────────────


@_needs_forge
def test_real_forge_gravity_inversion(forge_project):
    """Invert the REAL FORGE Bouguer residual → a sensible coherent density model (doc 10 §8)."""
    from geosim.inversion import (
        CoreRegion,
        PaddingSpec,
        build_tensor_domain,
        run_inversion,
    )
    from geosim.inversion.engines.gravity_simpeg import SimpegGravityInversion
    from geosim.storage import open_property_model

    root, pid, Session, layout, _frame = forge_project

    with Session() as session:
        coords_all, anomaly_all = _ingested_gravity(session, pid)

        # The ingested absolute complete-Bouguer field is a strong regional signal (mGal).
        assert np.all(np.isfinite(anomaly_all[np.isfinite(anomaly_all)]))
        assert anomaly_all.size > 1000  # the real survey has thousands of stations

        coords_zyx, residual, surface_z = _window_and_detrend(coords_all, anomaly_all)
        n_stations = coords_zyx.shape[0]
        assert n_stations >= 200, f"too few windowed stations ({n_stations})"
        # The residual is a *local* anomaly: small, zero-mean, with real structure.
        assert np.all(np.isfinite(residual))
        assert abs(float(residual.mean())) < 1.0
        assert 0.5 < float(residual.std()) < 50.0

        obs_id = _seed_residual_observation(session, pid, coords_zyx, residual)

        # COARSE tractable core over the window: 1 km lateral, 0.5 km vertical, 3 km deep.
        core = CoreRegion(
            origin=(-3000.0, -_WINDOW_M, -_WINDOW_M),       # (z0, y0, x0)
            extent=(3000.0, 2 * _WINDOW_M, 2 * _WINDOW_M),  # (dz, dy, dx)
            cell_size=(500.0, 1000.0, 1000.0),              # → (6, 12, 12) core cells
        )
        domain = build_tensor_domain(
            core, padding=PaddingSpec(n_pad=2, factor=1.4), surface_z=surface_z + 1.0
        )
        # Sanity: the window survey fits inside the core footprint.
        assert domain.core.n_core() == (6, 12, 12)

        res = run_inversion(
            session, layout, pid, SimpegGravityInversion(),
            domain=domain, observation_ids=[obs_id],
            params={
                "background_density": _BACKGROUND,
                "max_iterations": 8,
                "beta0_ratio": 0.5,
                "cooling_factor": 2.0,
                "rho_min": -400.0,
                "rho_max": 400.0,
                "use_gpu": "auto",  # NumPy on this CPU container; CuPy on the user's 4090.
            },
            resample_fused=True,
        )

        # ── (1) a real density PropertyModel with MANDATORY finite uncertainty (doc 10 §0) ──
        reader = open_property_model(layout.zarr_path(res.property_model_id))
        assert reader.properties == ["density"]
        vol = reader.read_level("density", 0)  # (nz, ny, nx) absolute density (kg/m³)
        assert vol.shape == (6, 12, 12)
        assert np.all(np.isfinite(vol))

        assert reader.has_sigma("density")
        sigma = reader.read_sigma_level("density", 0)
        assert sigma.shape == vol.shape
        assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)

        # ── (2) sensible density ANOMALY: finite, plausible magnitude, non-trivial ──
        d_rho = vol - _BACKGROUND
        assert np.all(np.isfinite(d_rho))
        # A residual-gravity density model: bounded well inside the solver bounds, and not
        # collapsed to zero (the inversion actually moved mass).
        peak = float(np.abs(d_rho).max())
        assert 0.0 < peak <= 400.0, f"implausible density anomaly peak {peak}"
        assert float(d_rho.std()) > 1e-3
        # Both signs of anomaly are present (a real residual has highs AND lows).
        assert float(d_rho.max()) > 0.0 and float(d_rho.min()) < 0.0

        # ── (3) spatially COHERENT: smoothness regularisation ⇒ small lateral gradients
        #        relative to the model amplitude (no salt-and-pepper, doc 10 §8). ──
        lateral_grad = float(np.abs(np.diff(d_rho, axis=2)).mean())
        assert lateral_grad < 0.6 * peak, "recovered model is not spatially coherent"

        # ── (4) data misfit DECREASED over the run (doc 10 §3). ──
        #   The starting misfit is χ²(m=0) = Σ (residual / σ)² with the engine's noise floor
        #   σ = rel_noise·max|anomaly| (default rel_noise=0.02). A converging inversion must
        #   leave the final φ_d well below that starting misfit.
        assert res.iterations >= 2
        assert res.final_phi_d is not None and res.final_phi_d >= 0.0
        assert res.final_phi_m is not None and res.final_phi_m > 0.0
        floor = max(0.02 * float(np.max(np.abs(residual))), 1e-6)
        phi_d0 = float(np.sum((residual / floor) ** 2))
        assert res.final_phi_d < phi_d0, "data misfit did not decrease from the m=0 start"

        # ── provenance + the GPU/backend record (doc 10 §7, §8) ──
        from geosim.catalog.models import Provenance

        prov = session.get(Provenance, res.provenance_id)
        assert prov.process == "invert:simpeg.gravity"
        params = json.loads(prov.params_json)
        assert params["engineId"] == "simpeg.gravity"
        assert params["metrics"]["nStations"] == n_stations
        # On this CPU container the dense-G path runs on NumPy; the metric records it.
        assert params["metrics"]["computeBackend"] == "numpy"
        assert params["metrics"]["useGpu"] is False

        # Fused resample of the recovered core succeeded (doc 10 §4.4).
        assert res.fused_model_id is not None and res.fused_layer_id is not None


# ─────────────────── GPU dense-G acceleration: NumPy-path correctness ───────────────────
#
# The user runs the dense-G (G·m / Gᵀ·r) products on an RTX 4090 via CuPy; this container
# has no GPU, so we verify the *fallback* path is exactly correct: the GPU simulation
# subclass must produce byte-identical forward/adjoint products to stock SimPEG when the
# active geosim.compute backend is NumPy (doc 10 §8 "stay correct + identical on CPU").


def _tiny_gravity_sim(gravity, cls):
    """Build a tiny gravity Simulation3DIntegral of class ``cls`` for product checks."""
    from discretize import TensorMesh
    from simpeg import maps

    mesh = TensorMesh([[(40.0, 8)], [(40.0, 8)], [(25.0, 6)]])
    active = np.ones(mesh.n_cells, dtype=bool)
    rx = gravity.receivers.Point(
        np.array([[120.0, 120.0, 30.0], [200.0, 80.0, 30.0], [80.0, 240.0, 30.0]]),
        components="gz",
    )
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    return cls(
        survey=survey, mesh=mesh, rhoMap=maps.IdentityMap(nP=int(active.sum())),
        active_cells=active, store_sensitivities="ram",
    )


def test_resolve_use_gpu_modes():
    """``use_gpu`` resolution: auto/off → NumPy here, on → error when no GPU (doc 10 §8)."""
    from geosim import compute
    from geosim.inversion.engines.gravity_simpeg import _resolve_use_gpu

    assert _resolve_use_gpu("off") is False
    # On this CPU container compute.gpu_available() is False, so auto degrades to NumPy.
    assert _resolve_use_gpu("auto") is compute.gpu_available()
    if not compute.gpu_available():
        with pytest.raises(ValueError, match="no CUDA device"):
            _resolve_use_gpu("on")


def test_gpu_simulation_matches_stock_on_numpy():
    """The GPU dense-G subclass == stock SimPEG forward/adjoint on the NumPy backend."""
    from simpeg.potential_fields import gravity

    from geosim.inversion.engines.gravity_simpeg import _make_gpu_simulation

    stock = _tiny_gravity_sim(gravity, gravity.simulation.Simulation3DIntegral)
    gpu = _tiny_gravity_sim(gravity, _make_gpu_simulation(gravity))

    rng = np.random.default_rng(7)
    nP = stock.G.shape[1]
    nD = stock.G.shape[0]
    m = rng.random(nP)
    v = rng.random(nP)
    r = rng.random(nD)

    # forward (fields / dpred), J·v, and Jᵀ·r must all be byte-identical to stock SimPEG.
    assert np.array_equal(gpu.fields(m), stock.fields(m))
    assert np.array_equal(gpu.dpred(m), stock.dpred(m))
    assert np.array_equal(gpu.Jvec(m, v), stock.Jvec(m, v))
    assert np.array_equal(gpu.Jtvec(m, r), stock.Jtvec(m, r))

    # G is staged through geosim.compute; on a CPU box that lands on a host ndarray.
    assert isinstance(gpu._device_G(), np.ndarray)
