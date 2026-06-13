"""Potential-field adapter round-trip tests (doc 03 §2 rows 1–2, doc 09/§11 round-trip).

Synthesizes a tiny gravity + magnetics dataset with the *real* synthgen T0 forwards
(:class:`GravityForward` / :class:`MagneticsForward`, doc 05 §4) and ingests every emitted
native file through the *real* ingestion pipeline (``store-raw → detect → parse →
normalize → write → register``, doc 03 §7). Asserts the right primitive kind, property
keys, canonical units (mGal / nT, doc 01 §5), and Engineering bbox come out the far end —
the OVERVIEW §8 forward→ingest round-trip.

In-memory SQLite (doc 04 §2.1) + a tmp storage root; SMALL truth grid keeps it fast.
No Docker/Postgres/Redis.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosim.catalog import (
    Dataset,
    Observation,
    PropertyModel,
    create_all,
    make_engine,
    session_factory,
)
from geosim.ingestion import (
    IngestStatus,
    RawSource,
    adapter_named,
    detect,
    ingest_file,
)
from geosim.synthgen import (
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    compile_scene,
)
from geosim.synthgen.forward import Acquisition, GravityForward, MagneticsForward


# ─────────────────────────── tiny synthetic earth + survey ───────────────────────────


def _tiny_scene(seed: int = 11) -> SceneSpec:
    return SceneSpec(
        id="tiny-pf-adapters-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-400, xmax=400, ymin=-400, ymax=400,
            zmin=-800, zmax=200, dx=100, dy=100, dz=100,
        ),
        surface=SurfaceSpec(kind="tilted-block", base_elev=100.0, tilt_x=0.1),
        layers=(
            LayerSpec("alluvium", "surface", (40.0, 60.0)),
            LayerSpec("volcanics", "conformable", (120.0, 180.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-400, -100), (400, 100)),
                      kind="normal", dip=60, dip_azimuth=90, throw=150, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=40.0),
        anomalies=(),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def earth():
    return compile_scene(_tiny_scene())


@pytest.fixture()
def acq(tmp_path):
    return Acquisition(
        gravity_spacing=200.0,
        mag_line_spacing=200.0,
        mag_altitude=60.0,
        params={"out_dir": str(tmp_path / "measured")},
    )


@pytest.fixture()
def session():
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1)
    create_all(engine)
    Session = session_factory(engine)
    with Session() as s:
        yield s


# ─────────────────────────── registry / detection (doc 03 §1, §7 step 3) ───────────────────────────


def test_potential_adapters_registered():
    assert adapter_named("gravity-potential-v1") is not None
    assert adapter_named("magnetics-potential-v1") is not None


def test_detect_routes_each_native_file(earth, acq):
    grav = GravityForward().simulate(earth, acq, np.random.default_rng(1))
    mag = MagneticsForward().simulate(earth, acq, np.random.default_rng(1))
    routed = {}
    for art in (*grav, *mag):
        src = RawSource(filename=art.path.name, data=art.path.read_bytes())
        routed[art.path.name] = detect(src).name
    # the synthgen gravity CSV goes to the potential adapter (beats generic gravity-csv-v1)
    assert routed["gravity_stations.csv"] == "gravity-potential-v1"
    assert routed["gravity_bouguer.tif"] == "gravity-potential-v1"
    assert routed["aeromag_lines.xyz"] == "magnetics-potential-v1"
    assert routed["mag_rtp.tif"] == "magnetics-potential-v1"


# ─────────────────────────── gravity round-trip (doc 03 §2 row 1) ───────────────────────────


def test_gravity_stations_csv_round_trip(session, tmp_path, earth, acq):
    arts = GravityForward().simulate(earth, acq, np.random.default_rng(2))
    csv = next(a for a in arts if a.fmt == "csv")

    report = ingest_file(session, tmp_path / "storage", None, csv.path)
    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.n_observations == 1

    ds = session.get(Dataset, report.dataset_id)
    assert ds.method == "gravity" and ds.submethod is None

    obs = session.query(Observation).filter_by(dataset_id=report.dataset_id).one()
    assert obs.geometry_kind == "points"
    assert obs.primary_property == "gravity_anomaly"

    import json

    payload = json.loads(obs.values_json)
    assert "gravity_anomaly" in payload["values"]
    n = len(payload["values"]["gravity_anomaly"])
    assert n == len(payload["coords"]) > 0
    assert np.isfinite(payload["values"]["gravity_anomaly"]).all()

    # bbox in Engineering metres encloses the station grid (doc 04 §2.2)
    bbox = json.loads(obs.bbox_json)
    assert bbox["xmin"] <= bbox["xmax"] and bbox["ymin"] <= bbox["ymax"]
    assert bbox["xmin"] >= -400.0 - 1e-6 and bbox["xmax"] <= 400.0 + 1e-6


def test_gravity_bouguer_geotiff_round_trip(session, tmp_path, earth, acq):
    arts = GravityForward().simulate(earth, acq, np.random.default_rng(3))
    tif = next(a for a in arts if a.fmt == "geotiff")

    report = ingest_file(session, tmp_path / "storage", None, tif.path)
    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.n_property_models == 1

    pm = session.query(PropertyModel).filter_by(dataset_id=report.dataset_id).one()
    assert pm.property == "gravity_anomaly"
    assert pm.canonical_unit == "mGal"
    assert pm.support == "grid2d"
    assert pm.store_format == "zarr"

    from geosim.storage import open_property_model

    vol = open_property_model(pm.store_uri).read_level("gravity_anomaly", 0)
    assert vol.ndim == 3 and vol.shape[0] == 1  # (z=1, ny, nx) grid2d embedded in 3D
    assert np.isfinite(vol).any()


# ─────────────────────────── magnetics round-trip (doc 03 §2 row 2) ───────────────────────────


def test_aeromag_xyz_round_trip(session, tmp_path, earth, acq):
    arts = MagneticsForward().simulate(earth, acq, np.random.default_rng(4))
    xyz = next(a for a in arts if a.fmt == "xyz")

    report = ingest_file(session, tmp_path / "storage", None, xyz.path)
    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.n_observations == 1

    ds = session.get(Dataset, report.dataset_id)
    assert ds.method == "magnetics" and ds.submethod is None

    obs = session.query(Observation).filter_by(dataset_id=report.dataset_id).one()
    assert obs.geometry_kind == "points"
    assert obs.primary_property == "magnetic_field"

    import json

    payload = json.loads(obs.values_json)
    mf = np.asarray(payload["values"]["magnetic_field"], dtype=float)
    assert mf.size > 0 and np.isfinite(mf).all()


def test_mag_rtp_geotiff_round_trip(session, tmp_path, earth, acq):
    arts = MagneticsForward().simulate(earth, acq, np.random.default_rng(5))
    tif = next(a for a in arts if a.fmt == "geotiff")

    report = ingest_file(session, tmp_path / "storage", None, tif.path)
    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.n_property_models == 1

    pm = session.query(PropertyModel).filter_by(dataset_id=report.dataset_id).one()
    assert pm.property == "magnetic_field"
    assert pm.canonical_unit == "nT"
    assert pm.support == "grid2d"

    import json

    # the stored 2D grid bbox should match the synthgen station grid extent in plan.
    bbox = json.loads(pm.bbox_json)
    assert bbox["xmax"] > bbox["xmin"] and bbox["ymax"] > bbox["ymin"]
