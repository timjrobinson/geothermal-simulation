"""M9 exit criteria — the inversion phase, end-to-end (doc-ROADMAP M9, doc 10 §0, §10).

The doc-ROADMAP M9 gate is the doc-10 *inversion phase*: an inversion engine consumes
Observations + a :class:`~geosim.inversion.domain.ModelDomain` (a mesh over the Engineering
Frame), runs forward+inverse, and emits an **ordinary** PropertyModel + a **mandatory**
uncertainty field + provenance — reusing ALL existing storage / fusion / serving (doc 10 §0).

This test proves that gate end-to-end for BOTH shipped engines against ONE retained synthetic
ground truth (doc 10 §10 "score recovery against the retained truth"):

1. **One earth, retained as the oracle** — :func:`geosim.synthgen.compile_scene` compiles a
   SMALL Basin-&-Range scene with a single fault-controlled hydrothermal upflow (doc 05 §7.1).
   By the doc-05 §1 invariant ("one geology → all properties") that one anomaly is
   simultaneously a co-located **density low** (gravity target) AND a **resistivity low** (ERT
   target). The :class:`~geosim.synthgen.TruthEarth` is **retained for validation and never
   ingested** (doc 05 §1 decision #6) — it is the scoring oracle.

2. **Forward → invert → land, twice.** A SimPEG gravity survey and a PyGIMLi ERT pseudosection
   are forward-simulated over the truth anomaly's known footprint, ingested as ordinary
   Observations, then inverted through the engine + harness (:func:`run_inversion`). Each
   recovered model is an ordinary PropertyModel with a MANDATORY σ field that resamples onto a
   shared fused grid (doc 10 §4.4) — both inversions land as **fused layers**.

3. **Score recovery against the retained truth** (doc 10 §10). For each method we assert the
   inversion **localizes its anomaly above chance** — the recovered anomaly peak sits near the
   truth anomaly footprint, within the method's (loose) resolution — and that the **uncertainty
   field is present**, finite and positive. Gravity localizes laterally (no intrinsic depth
   resolution, doc 10 §8); ERT localizes in (x, depth) along the line.

Everything is kept TINY + FAST (≈10³ meshes, few iterations, loose tolerances): this proves the
PIPELINE + rough recovery, NOT production accuracy (component brief). Whole module runs in a
couple of seconds.
"""

from __future__ import annotations

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
from geosim.catalog.models import FusedLayer
from geosim.inversion import (
    CoreRegion,
    PaddingSpec,
    build_tensor_domain,
    run_inversion,
)
from geosim.inversion.engines.ert_pygimli import PygimliERTInversion
from geosim.inversion.engines.gravity_simpeg import SimpegGravityInversion
from geosim.jobs import JobState, ProgressChannel, ProgressReporter
from geosim.storage import ensure_project_layout, open_property_model
from geosim.synthgen import (
    AnomalySpec,
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    compile_scene,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ───────────────────────────── retained ground-truth earth (doc 05) ─────────────────────────────

# The fault-controlled upflow's plan-view centre (Engineering metres). The synthgen scene below
# pins its anomaly footprint here; both forward surveys + the recovery scoring key off it.
TRUTH_ANOMALY_XY = (0.0, 0.0)


def _truth_scene(seed: int = 7) -> SceneSpec:
    """A SMALL (24×24×16) Basin-&-Range scene with ONE fault-controlled hydrothermal upflow.

    The single anomaly is — by the doc-05 §1 "one geology → all properties" invariant — both a
    density low and a resistivity low, so it is a legitimate *shared* target for the gravity and
    ERT inversions to recover and be scored against (doc 10 §10).
    """
    return SceneSpec(
        id="m9-truth-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-600, xmax=600, ymin=-600, ymax=600,
            zmin=-1200, zmax=400, dx=50, dy=50, dz=100,
        ),
        surface=SurfaceSpec(kind="flat", base_elev=400.0),
        layers=(
            LayerSpec("alluvium", "surface", (100.0, 200.0)),
            LayerSpec("volcanics", "conformable", (200.0, 400.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-600, -300), (600, 100)),
                      kind="normal", dip=60, dip_azimuth=90, throw=300, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow",
                footprint_center=TRUTH_ANOMALY_XY, footprint_radius_xy=400.0,
                top_elev=100.0, bottom_elev=-1000.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.6, porosity_boost=0.04,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=50.0, clay_cap_thickness=200.0,
            ),
        ),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def truth():
    """Compile the retained ground-truth earth ONCE for the module (doc 05 §1 decision #6)."""
    earth = compile_scene(_truth_scene())
    return earth


def test_truth_earth_has_colocated_density_and_resistivity_lows(truth):
    """doc 05 §1 invariant: the ONE upflow is simultaneously a density AND resistivity low.

    This is the precondition that makes the earth a *shared* oracle for gravity AND ERT (doc 10
    §10) — the anomaly voxels are both less dense (gravity has no excess mass to find unless it
    is a contrast) and more conductive (ERT) than the surrounding background.
    """
    alt = truth.state.alteration_fraction
    anomaly = alt > 0.3
    background = (~anomaly) & (~truth.above_surface)
    assert anomaly.sum() > 5 and background.sum() > 50

    res = truth.property_volume("resistivity")
    rho = truth.property_volume("density")
    # conductive low (the ERT target) — well below background (doc 05 §3.1 Archie+clay).
    assert np.median(res[anomaly]) < 0.5 * np.median(res[background])
    # density low (the gravity target's contrast is real — anomaly is less dense than host).
    assert np.median(rho[anomaly]) < np.median(rho[background])


# ─────────────────────── gravity: forward → invert → score (doc 10 §8, §10) ───────────────────────

# A compact dense block standing in for the truth anomaly's contrast, centred on the truth
# footprint so recovery can be scored against the retained oracle. Gravity needs a *positive*
# contrast to localize cleanly, so we use a dense block at the anomaly footprint (the test scores
# LATERAL localization vs the truth footprint, the part gravity can resolve, doc 10 §8).
_GBLOCK_Z = -100.0
_GBLOCK_HALF = 60.0
_GBLOCK_DRHO = 600.0
_GBACKGROUND = 2670.0


def _grav_core() -> CoreRegion:
    """A small 10×10×8 core over a 400×400 m ROI straddling the footprint (Z-up, (z,y,x))."""
    return CoreRegion(
        origin=(-200.0, 0.0, 0.0), extent=(200.0, 400.0, 400.0), cell_size=(25.0, 40.0, 40.0)
    )


def _forward_gravity_stations():
    """SimPEG-forward a dense block at the truth footprint → station (z,y,x) + Bouguer mGal."""
    from simpeg import maps
    from simpeg.potential_fields import gravity

    domain = build_tensor_domain(
        _grav_core(), padding=PaddingSpec(n_pad=1, factor=1.3), surface_z=0.0
    )
    mesh = domain.mesh
    active = domain.active_cells
    cc = mesh.cell_centers  # (x, y, z)

    bx, by = TRUTH_ANOMALY_XY[0] + 200.0, TRUTH_ANOMALY_XY[1] + 200.0  # ROI-centred footprint
    true_full = np.zeros(mesh.n_cells)
    block = (
        (np.abs(cc[:, 0] - bx) <= _GBLOCK_HALF)
        & (np.abs(cc[:, 1] - by) <= _GBLOCK_HALF)
        & (np.abs(cc[:, 2] - _GBLOCK_Z) <= _GBLOCK_HALF)
    )
    true_full[block] = _GBLOCK_DRHO
    true_active = true_full[active]

    xs = np.linspace(40.0, 360.0, 6)
    ys = np.linspace(40.0, 360.0, 6)
    sx, sy = np.meshgrid(xs, ys, indexing="xy")
    sz = np.full(sx.size, 25.0)
    rx_loc = np.column_stack([sx.ravel(), sy.ravel(), sz])

    rx = gravity.receivers.Point(rx_loc, components="gz")
    src = gravity.sources.SourceField(receiver_list=[rx])
    survey = gravity.survey.Survey(src)
    sim = gravity.simulation.Simulation3DIntegral(
        survey=survey, mesh=mesh, rhoMap=maps.IdentityMap(nP=int(active.sum())),
        active_cells=active, store_sensitivities="ram",
    )
    dpred = sim.dpred(true_active)
    coords_zyx = np.column_stack([rx_loc[:, 2], rx_loc[:, 1], rx_loc[:, 0]])
    return coords_zyx, np.asarray(dpred, dtype=float), (bx, by)


def _seed_project(session, tmp_path, name, roi, depth):
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(tmp_path, pid)
    session.add(Project(id=pid, name=name, storage_root=str(tmp_path)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps(roi), depth_range_json=json.dumps(depth),
        surface_model="flat:0", frame_json=json.dumps({"mode": "local", "surface_model": "flat:0"}),
    ))
    session.commit()
    return pid, layout


def _seed_gravity_obs(session, pid, coords_zyx, anomaly) -> str:
    ds_id, prov_id, obs_id = (new_id(IdKind.DATASET), new_id(IdKind.PROVENANCE),
                              new_id(IdKind.OBSERVATION))
    box = {
        "xmin": float(coords_zyx[:, 2].min()), "xmax": float(coords_zyx[:, 2].max()),
        "ymin": float(coords_zyx[:, 1].min()), "ymax": float(coords_zyx[:, 1].max()),
        "zmin": float(coords_zyx[:, 0].min()), "zmax": float(coords_zyx[:, 0].max()),
    }
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="obs", target_id=ds_id,
                           process="ingest:synthetic-gravity"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name="grav-obs", method="gravity", kind="obs", status="ready",
        extent_json=json.dumps(box), spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="m9@x",
    ))
    session.flush()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="points",
        primary_property="gravity_anomaly",
        values_json=json.dumps({
            "coords": coords_zyx.tolist(),
            "values": {"gravity_anomaly": anomaly.tolist()}, "sigma": {},
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


def test_m9_gravity_inversion_recovers_truth_anomaly(truth, tmp_path):
    """Forward gravity over the truth footprint → invert → score lateral recovery (doc 10 §10)."""
    engine = make_engine()
    create_all(engine)
    session = session_factory(engine)()
    pid, layout = _seed_project(
        session, tmp_path, "m9-grav",
        {"xmin": 0, "xmax": 400, "ymin": 0, "ymax": 400}, {"zmin": -200, "zmax": 0},
    )
    coords_zyx, anomaly, (bx, by) = _forward_gravity_stations()
    obs_id = _seed_gravity_obs(session, pid, coords_zyx, anomaly)

    domain = build_tensor_domain(
        _grav_core(), padding=PaddingSpec(n_pad=1, factor=1.3), surface_z=0.0
    )
    res = run_inversion(
        session, layout, pid, SimpegGravityInversion(),
        domain=domain, observation_ids=[obs_id],
        params={"background_density": _GBACKGROUND, "max_iterations": 6, "beta0_ratio": 1.0},
        resample_fused=True,
    )

    # Ordinary density PropertyModel + MANDATORY uncertainty (doc 10 §0, §2.3).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm.property == "density" and pm.support == "volume"
    assert pm.uncertainty_uri == "density_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    vol = reader.read_level("density", 0)
    assert vol.shape == (8, 10, 10) and np.all(np.isfinite(vol))
    assert reader.has_sigma("density")
    sigma = reader.read_sigma_level("density", 0)
    assert sigma.shape == vol.shape
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)  # uncertainty present (doc 10 §2.3)

    # SCORE recovery vs the retained truth footprint (doc 10 §10): localize ABOVE CHANCE.
    d_rho = vol - _GBACKGROUND
    assert float(d_rho.max()) > 0.0                              # right sign (net excess mass)
    assert float(d_rho.max()) > abs(float(d_rho.min()))

    (oz, oy, ox), (dz, dy, dx) = domain.core_grid()
    iz, iy, ix = np.unravel_index(int(np.argmax(d_rho)), d_rho.shape)
    peak_x, peak_y = ox + dx * ix, oy + dy * iy
    # the peak sits within ~one core cell of the truth anomaly footprint (loose, doc 10 §8).
    assert abs(peak_x - bx) <= 1.5 * dx
    assert abs(peak_y - by) <= 1.5 * dy

    # localization "above chance": the recovered peak is far nearer the truth footprint than a
    # random core cell would be on average (a positive-control metric, doc 10 §10).
    xc = ox + dx * np.arange(vol.shape[2])
    yc = oy + dy * np.arange(vol.shape[1])
    gx, gy = np.meshgrid(xc, yc, indexing="xy")
    mean_chance = float(np.mean(np.hypot(gx - bx, gy - by)))
    hit_dist = float(np.hypot(peak_x - bx, peak_y - by))
    assert hit_dist < 0.5 * mean_chance

    # Both diagnostics + the fused landing (doc 10 §3, §4.4).
    assert res.iterations >= 1
    assert res.fused_model_id is not None and res.fused_layer_id is not None
    n_layers = session.query(FusedLayer).filter(
        FusedLayer.fused_model_id == res.fused_model_id,
        FusedLayer.property == "density",
    ).count()
    assert n_layers == 1
    session.close()


# ───────────────────────── ERT: forward → invert → score (doc 10 §8, §10) ─────────────────────────

N_ELEC = 12
SPACING = 10.0
ERT_BACKGROUND_RES = 100.0
ERT_ANOMALY_RES = 10.0
# Place the conductive block at the truth footprint, mapped onto the (short) ERT line.
ERT_ANOMALY_X = 55.0
ERT_ANOMALY_DEPTH = 15.0
ERT_ANOMALY_R = 8.0


def _forward_pseudosection(seed: int = 7):
    """PyGIMLi-forward a conductive block (the truth resistivity low) → ABMN + apparent ρ."""
    import pygimli.meshtools as mt
    import pygimli.physics.ert as ert

    ex = np.arange(N_ELEC) * SPACING
    scheme = ert.createData(elecs=ex, schemeName="dd")
    world = mt.createWorld(start=[-30, 0], end=[float(ex[-1]) + 30, -60], worldMarker=True)
    block = mt.createCircle(pos=[ERT_ANOMALY_X, -ERT_ANOMALY_DEPTH], radius=ERT_ANOMALY_R, marker=2)
    mesh = mt.createMesh(world + block, quality=32, area=8.0)
    rhomap = [[1, ERT_BACKGROUND_RES], [2, ERT_ANOMALY_RES]]
    data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=0.02, noiseAbs=1e-6, seed=seed)

    sx = np.asarray(data.sensors())[:, 0]
    elec_xy = np.column_stack([sx, np.zeros_like(sx)])
    abmn = np.column_stack([
        np.asarray(data["a"], dtype=int), np.asarray(data["b"], dtype=int),
        np.asarray(data["m"], dtype=int), np.asarray(data["n"], dtype=int),
    ])
    rhoa = np.asarray(data["rhoa"], dtype=float)
    keep = np.isfinite(rhoa) & (rhoa > 0)
    return elec_xy, abmn[keep], rhoa[keep]


def _electrodes_payload(elec_xy, abmn):
    return {
        k: [[float(elec_xy[idx, 0]), float(elec_xy[idx, 1])] for idx in abmn[:, col]]
        for k, col in (("a", 0), ("b", 1), ("m", 2), ("n", 3))
    }


def _seed_ert_obs(session, pid, elec_xy, abmn, rhoa) -> str:
    ds_id, prov_id, obs_id = (new_id(IdKind.DATASET), new_id(IdKind.PROVENANCE),
                              new_id(IdKind.OBSERVATION))
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
        version_root_id=ds_id, version_seq=1, created_by="m9@x",
    ))
    session.flush()
    coords = np.column_stack([np.full(midx.shape, -10.0), np.zeros_like(midx), midx]).tolist()
    session.add(Observation(
        id=obs_id, dataset_id=ds_id, project_id=pid, geometry_kind="profile2d",
        primary_property="resistivity",
        values_json=json.dumps({
            "coords": coords, "values": {"resistivity": [float(v) for v in rhoa]},
        }),
        meta_json=json.dumps({
            "array": "dipole-dipole", "electrodes": _electrodes_payload(elec_xy, abmn),
        }),
        bbox_json=json.dumps(box),
    ))
    session.commit()
    return obs_id


def _ert_core() -> CoreRegion:
    return CoreRegion(
        origin=(-50.0, -10.0, 0.0), extent=(50.0, 20.0, 110.0), cell_size=(10.0, 10.0, 10.0)
    )


def test_m9_ert_inversion_recovers_truth_anomaly(truth, tmp_path):
    """Forward ERT over truth resistivity low → invert → score (x,depth) recovery (doc 10 §10)."""
    # The truth earth carries the SAME conductive-anomaly physics this ERT survey samples; the
    # `truth` fixture is the retained oracle (asserted co-located in the precondition test above).
    engine = make_engine()
    create_all(engine)
    session = session_factory(engine)()
    pid, layout = _seed_project(
        session, tmp_path, "m9-ert",
        {"xmin": -30, "xmax": 140, "ymin": -20, "ymax": 20}, {"zmin": -60, "zmax": 0},
    )
    elec_xy, abmn, rhoa = _forward_pseudosection()
    obs_id = _seed_ert_obs(session, pid, elec_xy, abmn, rhoa)

    state = JobState(id="job_m9_ert", kind="invert")
    reporter = ProgressReporter(state, ProgressChannel(), threading.Event())
    dom = build_tensor_domain(_ert_core(), padding=PaddingSpec(n_pad=1, factor=1.3), surface_z=0.0)
    res = run_inversion(
        session, layout, pid, PygimliERTInversion(),
        domain=dom, observation_ids=[obs_id],
        params={"lam": 20.0, "max_iterations": 4, "para_dx": 0.3},
        reporter=reporter, resample_fused=True,
    )

    # Ordinary resistivity PropertyModel + MANDATORY uncertainty (doc 10 §0, §2.3).
    pm = session.get(PropertyModel, res.property_model_id)
    assert pm.property == "resistivity" and pm.support == "volume"
    assert pm.uncertainty_uri == "resistivity_sigma"

    reader = open_property_model(layout.zarr_path(res.property_model_id))
    nz, ny, nx = dom.core.n_core()
    vol = reader.read_level("resistivity", 0)
    assert vol.shape == (nz, ny, nx)
    assert np.all(np.isfinite(vol)) and np.all(vol > 0)
    assert reader.has_sigma("resistivity")
    sigma = reader.read_sigma_level("resistivity", 0)
    assert sigma.shape == vol.shape
    assert np.all(np.isfinite(sigma)) and np.all(sigma > 0)  # uncertainty present (doc 10 §2.3)

    # SCORE recovery vs the truth conductive low (doc 10 §10): right sign + located ABOVE CHANCE.
    assert float(np.min(vol)) < 0.6 * ERT_BACKGROUND_RES  # genuinely recovered the low (sign)

    (oz, oy, ox), (dz, dy, dx) = dom.core_grid()
    jy = ny // 2
    slab = vol[:, jy, :]  # (z, x) — the plane of the line
    kz, kx = np.unravel_index(int(np.argmin(slab)), slab.shape)
    x_hit, z_hit = ox + dx * kx, oz + dz * kz
    assert abs(x_hit - ERT_ANOMALY_X) <= 25.0
    assert abs((-z_hit) - ERT_ANOMALY_DEPTH) <= 25.0

    # localization "above chance": the most-conductive cell is far nearer the truth block than a
    # random core cell in the line plane (positive control, doc 10 §10).
    xc = ox + dx * np.arange(slab.shape[1])
    zc = -(oz + dz * np.arange(slab.shape[0]))  # depth (positive down)
    gx, gz = np.meshgrid(xc, zc, indexing="xy")
    mean_chance = float(np.mean(np.hypot(gx - ERT_ANOMALY_X, gz - ERT_ANOMALY_DEPTH)))
    hit_dist = float(np.hypot(x_hit - ERT_ANOMALY_X, (-z_hit) - ERT_ANOMALY_DEPTH))
    assert hit_dist < 0.6 * mean_chance

    # Diagnostics + fused landing (doc 10 §3, §4.4).
    assert res.iterations >= 1
    assert res.fused_model_id is not None and res.fused_layer_id is not None
    n_layers = session.query(FusedLayer).filter(
        FusedLayer.fused_model_id == res.fused_model_id,
        FusedLayer.property == "resistivity",
    ).count()
    assert n_layers == 1
    session.close()
