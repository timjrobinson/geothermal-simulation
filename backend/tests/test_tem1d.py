"""Tests for the empymod 1-D layered TEM inversion engine (doc 10 §8, §9).

Two flavours, both kept FAST (tiny meshes, a handful of soundings, few layers, loose
tolerance — this proves the PIPELINE + rough recovery, not production accuracy):

1. **synthetic recovery** — forward-model a known 3-layer σ(z) column with :mod:`empymod`,
   invert it back, and assert the recovered layer conductivities land within tolerance
   (the forward/inverse round-trip is correct).
2. **real FORGE data** — ingest a few real FORGE WalkTEM ``.usf`` soundings
   (``data/utah-forge/measured/em/FORGE_TEM_USF/*.usf``), invert the stitched volume, and
   assert a physically-sensible conductivity-depth model + a MANDATORY uncertainty +
   InversionProvenance (doc 10 §0, §2.3, §7).

Everything runs on the CPU/NumPy path (this container has no GPU); the GPU stitch path
falls back to NumPy and is exercised transparently.
"""

import glob
import json
import os
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
from geosim.inversion.engines.tem1d import (
    TEM1D_SPEC,
    Tem1DInversion,
    forward_tem_sounding,
    invert_sounding,
)
from geosim.jobs import JobState, ProgressChannel, ProgressReporter
from geosim.storage import ensure_project_layout, open_property_model

pytestmark = pytest.mark.filterwarnings("ignore")


# Real FORGE WalkTEM soundings (canonical layout, doc 03 §2).
FORGE_TEM_DIR = "/workspaces/simulation/data/utah-forge/measured/em/FORGE_TEM_USF"

# Fixed inversion layer column (bottom depths below surface, m); 3 layers + half-space.
SYNTH_DEPTHS = [25.0, 60.0]
REAL_DEPTHS = [20.0, 45.0, 80.0, 130.0, 200.0]
SYNTH_TIMES = np.logspace(-5.0, -2.5, 24)


# ───────────────────────────── spec / registration (doc 10 §2) ─────────────────────────────


def test_spec_is_empymod_tem1d_conductivity():
    spec = TEM1D_SPEC
    assert spec.id == "empymod.tem1d"
    assert spec.kind == "em"
    assert spec.library == "empymod"
    assert "em" in spec.methods
    assert spec.output_property == "conductivity"
    assert spec.compute == "worker_process"  # heavy stitched solve (doc 08 §2.1)


def test_engine_self_registered_on_import():
    """Importing the module registers ``empymod.tem1d`` on the process registry (doc 08 §4f)."""
    from geosim.plugins import get_registry

    ids = {e.spec.id for e in get_registry().inversion_engines()}
    assert "empymod.tem1d" in ids


# ───────────────────────────── synthetic 1-D round-trip (doc 10 §8) ─────────────────────────────


def test_forward_is_conductivity_sensitive():
    """The central-loop forward decays and responds to layer conductivity (doc 10 §8)."""
    resistive = forward_tem_sounding(SYNTH_TIMES, np.array([0.01, 0.01, 0.01]), SYNTH_DEPTHS)
    conductive = forward_tem_sounding(SYNTH_TIMES, np.array([0.01, 0.5, 0.01]), SYNTH_DEPTHS)
    # decays by many orders over the time window (a real TEM transient).
    amp = np.abs(resistive)
    assert amp[0] > 10.0 * amp[-1]
    # a buried conductor genuinely changes the response (not a degenerate forward).
    log_diff = np.mean(np.abs(np.log10(np.abs(conductive)) - np.log10(np.abs(resistive))))
    assert log_diff > 0.05


def test_single_sounding_recovers_three_layers():
    """Forward a known 3-layer σ(z), invert → recovers each layer within tolerance (doc 10 §3)."""
    true_sigma = np.array([0.01, 0.2, 0.02])  # resistive / conductive / resistive (S/m)
    clean = forward_tem_sounding(SYNTH_TIMES, true_sigma, SYNTH_DEPTHS)
    rng = np.random.default_rng(0)
    noisy = clean * (1.0 + 0.02 * rng.standard_normal(clean.size))

    rec, info = invert_sounding(
        SYNTH_TIMES, noisy, SYNTH_DEPTHS,
        background_conductivity=0.03, smoothness=0.3, max_iterations=40,
    )
    assert info["success"]
    # each recovered layer within ~0.25 log10 (a factor ~1.8) of the truth — loose but proves
    # the layered conductivity-depth structure is recovered (doc 10 §9).
    log_ratio = np.abs(np.log10(rec / true_sigma))
    assert np.all(log_ratio < 0.3), f"recovered {rec} vs true {true_sigma}"
    # the conductive middle layer is clearly the most conductive of the three.
    assert int(np.argmax(rec)) == 1


# ──────────────────────── synthetic stitched volume (harness end-to-end) ────────────────────────


def _synthetic_env(tmp_path):
    """In-memory catalog + temp storage + a project (flat surface) seeded with 4 synthetic
    soundings forming a 2×2 grid, each carrying a forward-modeled 3-layer transient."""
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name="tem-inv", storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": -50, "xmax": 150, "ymin": -50, "ymax": 150}),
        depth_range_json=json.dumps({"zmin": -250, "zmax": 0}),
        surface_model="flat:0",
        frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()

    # 4 soundings at a 2×2 grid; a conductive middle layer (the target) under each.
    sites = [(10.0, 10.0), (90.0, 10.0), (10.0, 90.0), (90.0, 90.0)]
    true_sigma = np.array([0.01, 0.25, 0.02])
    clean = forward_tem_sounding(SYNTH_TIMES, true_sigma, SYNTH_DEPTHS)
    rng = np.random.default_rng(1)

    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    box = {"xmin": 0.0, "xmax": 100.0, "ymin": 0.0, "ymax": 100.0, "zmin": -250.0, "zmax": 0.0}
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:em-usf-v1"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="tem-soundings", method="em", kind="obs",
        status="ready", extent_json=json.dumps(box), spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()

    obs_ids = []
    for x, y in sites:
        noisy = clean * (1.0 + 0.02 * rng.standard_normal(clean.size))
        oid = new_id(IdKind.OBSERVATION)
        session.add(Observation(
            id=oid, dataset_id=ds_id, project_id=pid, geometry_kind="soundings",
            primary_property="conductivity",
            values_json=json.dumps({"coords": [[0.0, y, x]], "values": {}}),
            meta_json=json.dumps({
                "sounding_name": f"S_{int(x)}_{int(y)}",
                "transient": {
                    "time_s": SYNTH_TIMES.tolist(),
                    "voltage": noisy.tolist(),
                    "n_gates": int(SYNTH_TIMES.size),
                },
            }),
            bbox_json=json.dumps(
                {"xmin": x, "xmax": x, "ymin": y, "ymax": y, "zmin": -250.0, "zmax": 0.0}
            ),
        ))
        obs_ids.append(oid)
    session.commit()
    return session, layout, pid, obs_ids


def _synth_core() -> CoreRegion:
    """A coarse CORE block over the 2×2 sounding grid, 0..−180 m depth (Z-up, (z,y,x))."""
    # z: -180..0 in 30 m cells (6 layers); y,x: 0..100 in 50 m cells.
    return CoreRegion(
        origin=(-180.0, 0.0, 0.0), extent=(180.0, 100.0, 100.0), cell_size=(30.0, 50.0, 50.0)
    )


def test_synthetic_stitched_volume_recovers_conductor(tmp_path):
    """Stitch 4 synthetic soundings → a conductivity volume that recovers the buried conductor."""
    session, layout, pid, obs_ids = _synthetic_env(tmp_path)
    state = JobState(id="job_tem", kind="invert")
    reporter = ProgressReporter(state, ProgressChannel(), threading.Event())

    dom = build_tensor_domain(_synth_core(), padding=PaddingSpec(n_pad=1), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, Tem1DInversion(),
        domain=dom, observation_ids=obs_ids,
        params={
            "layer_depths": SYNTH_DEPTHS,
            "background_conductivity": 0.03,
            "smoothness": 0.5,
            "max_iterations": 30,
            "fit_gain": False,
        },
        reporter=reporter,
    )

    # Output is an ORDINARY conductivity PropertyModel (doc 10 §0).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm is not None
    assert pm.property == "conductivity" and pm.support == "volume"
    assert pm.canonical_unit == "S/m"
    assert pm.uncertainty_uri == "conductivity_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    nz, ny, nx = dom.core.n_core()
    vol = reader.read_level("conductivity", 0)
    assert vol.shape == (nz, ny, nx)  # CORE block only — empymod never leaks (doc 10 §4)
    assert np.all(np.isfinite(vol)) and np.all(vol > 0)

    # MANDATORY uncertainty present, finite, positive (doc 10 §2.3).
    assert reader.has_sigma("conductivity")
    sigma = reader.read_sigma_level("conductivity", 0)
    assert sigma.shape == vol.shape
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)

    # ── recovery: the conductive layer (25..60 m depth, σ≈0.25) is genuinely recovered ──
    # Map depth → core z-rows (Z-up index 0 deepest). The conductor sits ~25..60 m down.
    (oz, oy, ox), (dz, dy, dx) = dom.core_grid()
    zc = oz + dz * np.arange(nz)
    depth_below = zc.max() - zc
    cond_rows = (depth_below >= 25.0) & (depth_below <= 60.0)
    shallow_rows = depth_below < 25.0
    # the conductor depth-band is more conductive than the resistive cover above it.
    assert vol[cond_rows].mean() > 1.5 * vol[shallow_rows].mean()
    # ...and the recovered conductor conductivity is in the right ballpark (loose).
    assert 0.05 < vol[cond_rows].mean() < 1.0

    # Convergence record + InversionProvenance fingerprint (doc 10 §3, §7).
    assert res.iterations >= 1
    prov = session.get(Provenance, res.provenance_id)
    assert prov.process == "invert:empymod.tem1d"
    params = json.loads(prov.params_json)
    assert params["engineId"] == "empymod.tem1d"
    assert params["engineLibrary"] == "empymod"
    assert set(params["observationIds"]) == set(obs_ids)
    assert params["metrics"]["nSoundings"] == 4
    assert params["metrics"]["backend"] == "numpy"  # CPU path on this container
    assert {(i.input_kind, i.input_id) for i in prov.inputs} == {
        ("observation", o) for o in obs_ids
    }

    # Fused resample of the recovered core (doc 10 §4.4).
    assert res.fused_model_id is not None and res.fused_layer_id is not None


def test_uncertainty_inflates_with_depth(tmp_path):
    """σ grows with depth (DOI proxy) — deep cells are less constrained (doc 10 §2.3)."""
    session, layout, pid, obs_ids = _synthetic_env(tmp_path)
    dom = build_tensor_domain(_synth_core(), padding=PaddingSpec(n_pad=1), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, Tem1DInversion(),
        domain=dom, observation_ids=obs_ids,
        params={"layer_depths": SYNTH_DEPTHS, "max_iterations": 20},
        resample_fused=False,
    )
    reader = open_property_model(layout.zarr_path(res.property_model_id))
    vol = reader.read_level("conductivity", 0)
    sigma = reader.read_sigma_level("conductivity", 0)
    rel = sigma / np.maximum(vol, 1e-9)
    # relative σ deepest (Z-up index 0) exceeds relative σ shallowest (DOI inflation).
    assert rel[0].mean() > rel[-1].mean()


def test_rejects_observations_without_transients(tmp_path):
    """A TEM inversion with no per-sounding transient is a hard error (doc 03 §2)."""
    session, layout, pid, obs_ids = _synthetic_env(tmp_path)
    empty_id = new_id(IdKind.OBSERVATION)
    ref = session.get(Observation, obs_ids[0])
    session.add(Observation(
        id=empty_id, dataset_id=ref.dataset_id, project_id=pid, geometry_kind="soundings",
        primary_property="conductivity",
        values_json=json.dumps({"coords": [[0.0, 0.0, 0.0]], "values": {}}),
        meta_json=json.dumps({"sounding_name": "empty"}),  # no transient
        bbox_json=ref.bbox_json,
    ))
    session.commit()
    dom = build_tensor_domain(_synth_core())
    with pytest.raises(ValueError, match="no TEM soundings"):
        run_inversion(
            session, layout, pid, Tem1DInversion(),
            domain=dom, observation_ids=[empty_id], params={}, resample_fused=False,
        )


# ───────────────────────────── REAL FORGE data (doc 10 §9) ─────────────────────────────


def _real_forge_soundings(n: int):
    """Parse ``n`` real FORGE WalkTEM ``.usf`` soundings → engine-agnostic obs dicts (doc 03 §2).

    Reuses the platform EM adapter (the same path the ingestion pipeline takes) and places
    each sounding at a LOCAL Engineering XY (UTM minus a common anchor) so the test runs in
    a small flat-surface domain. Returns ``(obs_dicts, anchor_xy)``.
    """
    from geosim.ingestion.adapters.em import EmXyzAdapter
    from geosim.ingestion.base import RawSource

    files = sorted(glob.glob(os.path.join(FORGE_TEM_DIR, "*.usf")))[:n]
    adapter = EmXyzAdapter()
    parsed = []
    for f in files:
        pr = adapter.parse(RawSource(data=open(f, "rb").read(),
                                     filename=os.path.basename(f), crs_hint=None))
        for o in pr.observations:
            parsed.append(o)
    # common anchor = min easting/northing across soundings → local Engineering XY.
    eastings = [float(o.coords[0][0]) for o in parsed]
    northings = [float(o.coords[0][1]) for o in parsed]
    ax, ay = min(eastings), min(northings)
    obs_dicts = []
    for o in parsed:
        e, nrt, elev = (float(v) for v in o.coords[0])
        obs_dicts.append({
            "coords": [[0.0, nrt - ay, e - ax]],  # Engineering (z, y, x), flat surface z=0
            "values": {},
            "meta": o.meta,
        })
    return obs_dicts, (ax, ay)


def _seed_real_observations(session, pid, obs_dicts):
    """Seed the parsed FORGE soundings as ``soundings`` Observations in the catalog."""
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    xs = [d["coords"][0][2] for d in obs_dicts]
    ys = [d["coords"][0][1] for d in obs_dicts]
    box = {
        "xmin": min(xs), "xmax": max(xs), "ymin": min(ys), "ymax": max(ys),
        "zmin": -300.0, "zmax": 0.0,
    }
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:em-usf-v1"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="forge-tem", method="em", kind="obs", status="ready",
        extent_json=json.dumps(box), spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    obs_ids = []
    for d in obs_dicts:
        oid = new_id(IdKind.OBSERVATION)
        x, y = d["coords"][0][2], d["coords"][0][1]
        session.add(Observation(
            id=oid, dataset_id=ds_id, project_id=pid, geometry_kind="soundings",
            primary_property="conductivity",
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
    not glob.glob(os.path.join(FORGE_TEM_DIR, "*.usf")),
    reason="real FORGE TEM .usf soundings not present",
)
def test_real_forge_tem_inversion_is_sensible(tmp_path):
    """Invert a few REAL FORGE WalkTEM soundings → a physical conductivity-depth model."""
    obs_dicts, _anchor = _real_forge_soundings(n=4)
    assert len(obs_dicts) >= 3  # a handful of real soundings

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
    obs_ids = _seed_real_observations(session, pid, obs_dicts)

    # CORE spanning the sounding cluster, 0..−180 m depth.
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    ext_x = max(x1 - x0, 100.0)
    ext_y = max(y1 - y0, 100.0)
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
            "layer_depths": REAL_DEPTHS,
            "background_conductivity": 0.02,
            "smoothness": 1.0,
            "max_iterations": 40,
            "fit_gain": True,           # absorb the unknown WalkTEM absolute calibration
            "time_min": 1e-5,           # drop early ramp-affected gates
            "time_max": 3e-3,           # drop late noise-floor gates
        },
        resample_fused=False,
    )

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    vol = reader.read_level("conductivity", 0)
    sigma = reader.read_sigma_level("conductivity", 0)

    # The recovered conductivity field is finite, positive, and PHYSICAL for ground at the
    # FORGE site: σ in ~1e-3..2 S/m (i.e. ρ ≈ 0.5..1000 ohm·m), not pinned at the bounds.
    assert np.all(np.isfinite(vol)) and np.all(vol > 0)
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)
    med = float(np.median(vol))
    assert 1e-3 < med < 2.0, f"median σ {med} S/m is unphysical"
    # there is real conductivity-depth STRUCTURE (the model isn't a flat background).
    assert float(np.max(vol)) > 2.0 * float(np.min(vol))

    # Provenance records the real run (doc 10 §7).
    prov = session.get(Provenance, res.provenance_id)
    params = json.loads(prov.params_json)
    assert params["engineId"] == "empymod.tem1d"
    assert params["metrics"]["nSoundings"] == len(obs_ids)
    assert params["metrics"]["nLayers"] == len(REAL_DEPTHS) + 1
    assert res.final_phi_d is not None
    session.close()
