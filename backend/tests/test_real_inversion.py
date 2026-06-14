"""Real Utah FORGE end-to-end inversion verification (doc 10 §0, §8, §9).

A single consolidated VERIFY test over the **real** Utah FORGE field data
(``data/utah-forge/``) that proves the three production inversion engines turn real native
observations into sensible, co-registered :class:`~geosim.catalog.PropertyModel` volumes
through the engine-agnostic harness (``geosim.inversion.harness.run_inversion``):

1. **Gravity → density** (:class:`SimpegGravityInversion`). The real complete-Bouguer
   survey (``measured/gravity/Utah_FORGE_Gravity_Data.txt``, ~3700 stations) is ingested
   through the production :func:`geosim.ingestion.ingest_file` pipeline, windowed +
   planar-detrended around the FORGE anchor, and inverted on a COARSE tractable core into
   an absolute **density** volume (kg/m³) with a MANDATORY finite uncertainty.

2. **TEM-1D → conductivity** (:class:`Tem1DInversion`). A handful of real FORGE WalkTEM
   ``.usf`` soundings (``measured/em/FORGE_TEM_USF/*.usf``) are parsed with the production
   EM adapter, placed at their (locally-anchored) Engineering XY, and inverted (per-sounding
   1-D log-σ Occam, then laterally stitched) into a **conductivity** volume (S/m).

3. **MT-3D → resistivity** (:class:`SimpegMTInversion`). The full 3-D NSEM sparse solve on
   the real MT array is too heavy for CI (doc 10 §8 — the GPU does not help the sparse
   direct factorisation). Instead this asserts the engine **builds a valid SimPEG NSEM
   survey from the REAL EDI observations** (one plane-wave source per period, finite
   app-ρ/phase data in SimPEG's predicted receiver order) WITHOUT running the inversion —
   proving the real MT field data feeds the engine boundary correctly.

Both recovered models are checked to be finite, physically-plausible, co-registered in the
**FORGE Engineering Frame** (doc 01), and persisted with a full InversionProvenance lineage.
Everything is kept CI-fast (small windows, coarse meshes, few iterations + GPU→NumPy
fallback on this CPU container); the heavy production fine-mesh runs are documented in the
engines, not run here (doc 10 §9). Skips gracefully if the real dataset is absent.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

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
    new_id,
)
from geosim.catalog.db import create_all, make_engine, session_factory
from geosim.inversion import CoreRegion, PaddingSpec, build_tensor_domain, run_inversion
from geosim.inversion.engines.gravity_simpeg import SimpegGravityInversion
from geosim.inversion.engines.mt_nsem import MT_NSEM_SPEC, SimpegMTInversion
from geosim.inversion.engines.tem1d import Tem1DInversion
from geosim.storage import ensure_project_layout, open_property_model

pytestmark = pytest.mark.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
FORGE = REPO / "data" / "utah-forge"
FRAME_JSON = FORGE / "frame.json"
GRAVITY_FILE = FORGE / "measured" / "gravity" / "Utah_FORGE_Gravity_Data.txt"
TEM_DIR = FORGE / "measured" / "em" / "FORGE_TEM_USF"
EDI_DIR = FORGE / "measured" / "mt" / "edi"

_BACKGROUND_DENSITY = 2670.0  # the survey's Bouguer reduction density (2.67 g/cm³)
_WINDOW_M = 6000.0            # gravity inversion half-window around the FORGE anchor


# ───────────────────────────── shared FORGE frame helper ─────────────────────────────


def _forge_frame():
    """The real georeferenced FORGE :class:`SpatialFrame` from ``frame.json`` (doc 01)."""
    from geosim.spatial import Aabb, DepthRange, SpatialFrame

    f = json.loads(FRAME_JSON.read_text())
    return SpatialFrame.for_real_site(
        lon=f["anchor_lonlat"][0],
        lat=f["anchor_lonlat"][1],
        surface_elev=f["surface_elev_m"],
        roi=Aabb(**f["roi"]),
        depth_range=DepthRange(**f["depth_range"]),
    )


def _project(root):
    """A georeferenced FORGE project (catalog + layout) anchored to the real frame."""
    from geosim.api.frame_io import frame_row_kwargs

    frame = _forge_frame()
    engine = make_engine(f"sqlite:///{root / 'catalog.db'}")
    create_all(engine)
    Session = session_factory(engine)
    pid = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=pid, name="FORGE real inversion", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(project_id=pid, **frame_row_kwargs(frame))
        s.add(proj)
        s.commit()
    layout = ensure_project_layout(root, pid)
    return pid, Session, layout, frame


# ═══════════════════════════ 1) real gravity → density ═══════════════════════════


def _ingest_gravity(session, root, pid):
    """Ingest the real FORGE Bouguer file → ``(coords_zyx, anomaly_mGal)`` (doc 03 §7)."""
    from geosim.ingestion import ingest_file

    rep = ingest_file(
        session, root, pid, GRAVITY_FILE, method_hint="gravity", crs_hint="EPSG:4326"
    )
    session.commit()
    status = rep.status.value if hasattr(rep.status, "value") else str(rep.status)
    assert status in {"ok", "ok_with_warnings"}, f"gravity ingest failed: {rep.warnings}"
    assert rep.n_observations >= 1

    row = (
        session.query(Observation)
        .filter_by(project_id=pid, primary_property="gravity_anomaly")
        .first()
    )
    assert row is not None, "no ingested gravity observation found"
    payload = json.loads(row.values_json)
    coords = np.asarray(payload["coords"], dtype=float)  # (N, 3) Engineering (z, y, x)
    anomaly = np.asarray(payload["values"]["gravity_anomaly"], dtype=float)  # mGal
    return coords, anomaly


def _window_and_detrend(coords_zyx, anomaly):
    """±``_WINDOW_M`` window around the anchor + planar regional/residual separation."""
    z, y, x = coords_zyx[:, 0], coords_zyx[:, 1], coords_zyx[:, 2]
    win = (np.abs(x) <= _WINDOW_M) & (np.abs(y) <= _WINDOW_M) & np.isfinite(anomaly)
    xw, yw, zw, gw = x[win], y[win], z[win], anomaly[win]
    design = np.column_stack([np.ones_like(xw), xw, yw])
    coef, *_ = np.linalg.lstsq(design, gw, rcond=None)
    residual = gw - design @ coef
    # Lift receivers above the highest local station so they sit in the active air halfspace.
    z_rx = np.full_like(zw, float(zw.max()) + 1.0)
    coords_out = np.column_stack([z_rx, yw, xw])
    return coords_out, residual, float(zw.max())


def _seed_points_obs(session, pid, coords_zyx, prop, values, *, method, name):
    """Persist a point Observation carrying ``values`` under property ``prop``."""
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    obs_id = new_id(IdKind.OBSERVATION)
    box = {
        "xmin": float(coords_zyx[:, 2].min()), "xmax": float(coords_zyx[:, 2].max()),
        "ymin": float(coords_zyx[:, 1].min()), "ymax": float(coords_zyx[:, 1].max()),
        "zmin": float(coords_zyx[:, 0].min()), "zmax": float(coords_zyx[:, 0].max()),
    }
    session.add(Provenance(
        id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
        process=f"derive:{name}",
    ))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=name, method=method, kind="obs", status="ready",
        extent_json=json.dumps(box), spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="verify",
    ))
    session.flush()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="points",
        primary_property=prop,
        values_json=json.dumps({
            "coords": coords_zyx.tolist(), "values": {prop: values}, "sigma": {},
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


@pytest.mark.skipif(not GRAVITY_FILE.exists(), reason="real FORGE gravity not present")
def test_real_forge_gravity_to_density(tmp_path):
    """Invert the REAL FORGE Bouguer residual → a finite, coherent density volume (doc 10 §8)."""
    pid, Session, layout, frame = _project(tmp_path)
    with Session() as session:
        coords_all, anomaly_all = _ingest_gravity(session, tmp_path, pid)
        assert anomaly_all.size > 1000  # the real survey has thousands of stations

        coords_zyx, residual, surface_z = _window_and_detrend(coords_all, anomaly_all)
        n_stations = coords_zyx.shape[0]
        assert n_stations >= 200, f"too few windowed stations ({n_stations})"
        # A local residual: small, ~zero-mean, real structure.
        assert np.all(np.isfinite(residual))
        assert abs(float(residual.mean())) < 1.0
        assert 0.5 < float(residual.std()) < 50.0

        obs_id = _seed_points_obs(
            session, pid, coords_zyx, "gravity_anomaly", residual.tolist(),
            method="gravity", name="forge-gravity-residual",
        )

        # COARSE tractable core over the window → (6, 12, 12) cells, 3 km deep (doc 10 §9).
        core = CoreRegion(
            origin=(-3000.0, -_WINDOW_M, -_WINDOW_M),
            extent=(3000.0, 2 * _WINDOW_M, 2 * _WINDOW_M),
            cell_size=(500.0, 1000.0, 1000.0),
        )
        domain = build_tensor_domain(
            core, padding=PaddingSpec(n_pad=2, factor=1.4), surface_z=surface_z + 1.0
        )
        assert domain.core.n_core() == (6, 12, 12)

        res = run_inversion(
            session, layout, pid, SimpegGravityInversion(),
            domain=domain, observation_ids=[obs_id],
            params={
                "background_density": _BACKGROUND_DENSITY,
                "max_iterations": 8,
                "beta0_ratio": 0.5,
                "rho_min": -400.0,
                "rho_max": 400.0,
                "use_gpu": "auto",  # NumPy on this CPU container; CuPy on the user's 4090.
            },
            resample_fused=True,
        )

        # A real density PropertyModel with a MANDATORY finite uncertainty (doc 10 §0, §2.3).
        pm = session.get(PropertyModel, res.property_model_id)
        assert pm.property == "density" and pm.support == "volume"
        assert pm.canonical_unit == "kg/m**3"
        reader = open_property_model(layout.zarr_path(res.property_model_id))
        vol = reader.read_level("density", 0)  # (nz, ny, nx) absolute density
        assert vol.shape == (6, 12, 12)
        assert np.all(np.isfinite(vol))
        assert reader.has_sigma("density")
        sigma = reader.read_sigma_level("density", 0)
        assert sigma.shape == vol.shape
        assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)

        # A sensible density anomaly: finite, plausible magnitude, both signs, coherent.
        d_rho = vol - _BACKGROUND_DENSITY
        peak = float(np.abs(d_rho).max())
        assert 0.0 < peak <= 400.0, f"implausible density anomaly peak {peak}"
        assert float(d_rho.std()) > 1e-3
        assert float(d_rho.max()) > 0.0 and float(d_rho.min()) < 0.0
        lateral_grad = float(np.abs(np.diff(d_rho, axis=2)).mean())
        assert lateral_grad < 0.6 * peak, "recovered density is not spatially coherent"

        # Data misfit decreased from the m=0 start (doc 10 §3).
        assert res.iterations >= 2
        assert res.final_phi_d is not None and res.final_phi_d >= 0.0
        floor = max(0.02 * float(np.max(np.abs(residual))), 1e-6)
        phi_d0 = float(np.sum((residual / floor) ** 2))
        assert res.final_phi_d < phi_d0, "data misfit did not decrease from m=0"

        # Co-registered in the FORGE frame: provenance + the recovered model carry the
        # project's Engineering-frame mesh origin (doc 01, doc 10 §7).
        prov = session.get(Provenance, res.provenance_id)
        assert prov.process == "invert:simpeg.gravity"
        params = json.loads(prov.params_json)
        assert params["metrics"]["nStations"] == n_stations
        assert params["metrics"]["computeBackend"] == "numpy"  # CPU container
        origin = json.loads(pm.origin_json)  # (z, y, x) Engineering origin
        assert origin[1] == pytest.approx(-_WINDOW_M + 500.0)  # core min-corner + ½ cell
        assert origin[2] == pytest.approx(-_WINDOW_M + 500.0)
        # Fused resample (co-location with every other property, doc 10 §4.4) succeeded.
        assert res.fused_model_id is not None and res.fused_layer_id is not None


# ═══════════════════════════ 2) real TEM-1D → conductivity ═══════════════════════════


def _real_tem_obs(n):
    """Parse ``n`` real FORGE WalkTEM ``.usf`` soundings → obs dicts at local Engineering XY."""
    from geosim.ingestion.adapters.em import EmXyzAdapter
    from geosim.ingestion.base import RawSource

    files = sorted(glob.glob(os.path.join(str(TEM_DIR), "*.usf")))[:n]
    adapter = EmXyzAdapter()
    parsed = []
    for f in files:
        pr = adapter.parse(
            RawSource(data=Path(f).read_bytes(), filename=os.path.basename(f), crs_hint=None)
        )
        parsed.extend(pr.observations)
    # Common UTM anchor → small local Engineering footprint (flat-surface test domain).
    eastings = [float(o.coords[0][0]) for o in parsed]
    northings = [float(o.coords[0][1]) for o in parsed]
    ax, ay = min(eastings), min(northings)
    out = []
    for o in parsed:
        e, nrt, _elev = (float(v) for v in o.coords[0])
        out.append({
            "coords": [[0.0, nrt - ay, e - ax]],  # Engineering (z, y, x), flat surface z=0
            "meta": o.meta,
        })
    return out


def _seed_soundings(session, pid, obs_dicts, *, primary):
    """Seed parsed soundings as ``soundings`` Observations (transient rides in meta)."""
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    xs = [d["coords"][0][2] for d in obs_dicts]
    ys = [d["coords"][0][1] for d in obs_dicts]
    box = {"xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys),
           "zmin": -300.0, "zmax": 0.0}
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:em-usf-v1"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="forge-tem", method="em", kind="obs", status="ready",
        extent_json=json.dumps(box), spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="verify",
    ))
    session.flush()
    obs_ids = []
    for d in obs_dicts:
        oid = new_id(IdKind.OBSERVATION)
        x, y = d["coords"][0][2], d["coords"][0][1]
        session.add(Observation(
            id=oid, dataset_id=ds_id, project_id=pid, geometry_kind="soundings",
            primary_property=primary,
            values_json=json.dumps({"coords": d["coords"], "values": {}}),
            meta_json=json.dumps(d["meta"]),
            bbox_json=json.dumps(
                {"xmin": x, "xmax": x, "ymin": y, "ymax": y, "zmin": -300.0, "zmax": 0.0}
            ),
        ))
        obs_ids.append(oid)
    session.commit()
    return obs_ids


@pytest.mark.skipif(
    not glob.glob(os.path.join(str(TEM_DIR), "*.usf")),
    reason="real FORGE TEM .usf soundings not present",
)
def test_real_forge_tem_to_conductivity(tmp_path):
    """Invert a few REAL FORGE WalkTEM soundings → a physical conductivity volume (doc 10 §8)."""
    obs_dicts = _real_tem_obs(n=4)
    assert len(obs_dicts) >= 3

    # A small LOCAL flat-surface project (the soundings carry only a local Engineering XY).
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="forge-tem", storage_root=str(tmp_path)))
    xs = [d["coords"][0][2] for d in obs_dicts]
    ys = [d["coords"][0][1] for d in obs_dicts]
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": min(xs) - 50, "xmax": max(xs) + 50,
                             "ymin": min(ys) - 50, "ymax": max(ys) + 50}),
        depth_range_json=json.dumps({"zmin": -300, "zmax": 0}),
        surface_model="flat:0",
        frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()
    obs_ids = _seed_soundings(session, pid, obs_dicts, primary="conductivity")

    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    ext_x, ext_y = max(x1 - x0, 100.0), max(y1 - y0, 100.0)
    core = CoreRegion(
        origin=(-180.0, y0 - 25.0, x0 - 25.0),
        extent=(180.0, ext_y + 50.0, ext_x + 50.0),
        cell_size=(45.0, max(ext_y / 2.0, 50.0), max(ext_x / 2.0, 50.0)),
    )
    dom = build_tensor_domain(core, padding=PaddingSpec(n_pad=1), surface_z=0.0)

    res = run_inversion(
        session, layout, pid, Tem1DInversion(),
        domain=dom, observation_ids=obs_ids,
        params={
            "layer_depths": [20.0, 45.0, 80.0, 130.0, 200.0],
            "background_conductivity": 0.02,
            "smoothness": 1.0,
            "max_iterations": 40,
            "fit_gain": True,  # absorb the unknown WalkTEM absolute calibration
            "time_min": 1e-5,  # drop early ramp-affected gates
            "time_max": 3e-3,  # drop late noise-floor gates
        },
        resample_fused=False,
    )

    # A real conductivity PropertyModel + MANDATORY finite, positive uncertainty (doc 10 §0).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm.property == "conductivity" and pm.support == "volume"
    reader = open_property_model(layout.zarr_path(res.property_model_id))
    vol = reader.read_level("conductivity", 0)  # (nz, ny, nx) S/m
    sigma = reader.read_sigma_level("conductivity", 0)
    assert np.all(np.isfinite(vol)) and np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)
    # Physically-sensible conductivity: strictly positive, in a plausible ground range
    # (1e-4 .. 10 S/m, the engine bounds) and not collapsed to a single value.
    assert np.all(vol > 0)
    assert float(vol.min()) >= 1e-4 - 1e-9 and float(vol.max()) <= 10.0 + 1e-6
    assert float(vol.max()) / float(vol.min()) > 1.0 + 1e-6, "σ(z) carries no structure"

    # Convergence diagnostics + provenance lineage (doc 10 §3, §7).
    assert res.iterations >= 1
    assert res.final_phi_d is not None and res.final_phi_d >= 0.0
    prov = session.get(Provenance, res.provenance_id)
    assert prov.process == "invert:empymod.tem1d"
    params = json.loads(prov.params_json)
    assert params["metrics"]["nSoundings"] == len(obs_ids)
    assert set(params["observationIds"]) == set(obs_ids)
    session.close()


# ═══════════════════════════ 3) real MT-3D → NSEM survey build ═══════════════════════════
#
# The full 3-D NSEM inversion of the real MT array is too heavy for CI — it is a sparse
# DIRECT solve the GPU does not accelerate (doc 10 §8). We instead prove the engine BUILDS a
# valid SimPEG NSEM survey from the REAL EDI observations (reprojected into the FORGE frame),
# without running the inversion: one plane-wave source per period, finite app-ρ/phase data in
# the order SimPEG predicts. This is the real-data half of the boundary the heavy run depends
# on; the recovery itself is covered by the synthetic ``test_mt_nsem.py``.


def _real_mt_obs(n):
    """Parse ``n`` real FORGE EDI soundings → obs dicts reprojected into the FORGE frame."""
    from geosim.ingestion.adapters.mt import MtEdiAdapter
    from geosim.ingestion.base import RawSource

    frame = _forge_frame()
    files = sorted(glob.glob(os.path.join(str(EDI_DIR), "*.edi")))[:n]
    obs = []
    for f in files:
        pr = MtEdiAdapter().parse(
            RawSource(data=Path(f).read_bytes(), filename=os.path.basename(f), crs_hint=None)
        )
        o = pr.observations[0]
        lon, lat, elev = (float(v) for v in o.coords[0])
        # EDI sites are lon/lat (EPSG:4326) → reproject into Engineering (E, N, up), doc 01.
        eng = frame.to_engineering([[lon, lat, elev]], src_crs=pr.source.crs)[0]
        z, y, x = float(elev), float(eng[1]), float(eng[0])
        nper = int(np.asarray(o.values["resistivity"]).size)
        obs.append({
            "coords": [[z, y, x]] * nper,
            "values": {
                "resistivity": list(o.values["resistivity"]),
                "phase_mrad": list(o.values["phase_mrad"]),
            },
            "meta": {"frequency_hz": list(o.meta["frequency_hz"])},
        })
    return obs


@pytest.mark.skipif(
    not glob.glob(os.path.join(str(EDI_DIR), "*.edi")),
    reason="real FORGE MT .edi soundings not present",
)
def test_real_forge_mt_builds_nsem_survey():
    """The MT engine builds a valid NSEM survey from REAL EDI observations (doc 10 §8)."""
    from simpeg.electromagnetics import natural_source as nsem

    obs = _real_mt_obs(n=4)
    assert len(obs) >= 3

    engine = SimpegMTInversion()
    # 1) the real EDI curves unpack into per-site app-ρ/phase soundings on a period axis.
    sites = engine._collect_sites(obs)
    assert len(sites) == len(obs)
    for s in sites:
        assert s["periods"].size > 0
        assert np.all(np.isfinite(s["periods"])) and np.all(s["periods"] > 0)
        assert np.all(np.isfinite(s["rho"])) and np.all(s["rho"] > 0)
        assert s["phase"] is not None and np.all(np.isfinite(s["phase"]))
    # Reprojected into the FORGE Engineering frame → sites sit inside the ROI (±15 km).
    for s in sites:
        x, y, _z = s["loc"]
        assert abs(x) <= 15000.0 and abs(y) <= 15000.0

    # 2) the engine assembles a real SimPEG NSEM Survey: one PlanewaveXYPrimary per UNIQUE
    #    period, finite dobs/standard_deviation aligned to SimPEG's receiver order. No solve.
    params = {k: v.get("default") for k, v in MT_NSEM_SPEC.params_schema["properties"].items()}
    survey, dobs, dstd = engine._build_survey(nsem, sites, ["xy", "yx"], params)

    all_periods = np.unique(np.concatenate([s["periods"] for s in sites]))
    assert len(survey.source_list) == all_periods.size  # one plane-wave source per period
    for src in survey.source_list:
        assert src.frequency > 0
    assert dobs.size > 0 and dobs.size == dstd.size == survey.nD
    assert np.all(np.isfinite(dobs)) and np.all(np.isfinite(dstd)) and np.all(dstd > 0)
    # The data vector mixes app-ρ (Ω·m, positive) and phase (degrees) samples — all finite.
    assert float(dobs.max()) > 0.0
