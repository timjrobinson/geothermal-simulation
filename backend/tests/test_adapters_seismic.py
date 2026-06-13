"""Round-trip tests for the seismic/wells/insar/microseismic adapters (doc 03 §2 rows 7,
9, 10, 11).

Each test runs the *matching* T0 forward (``geosim.synthgen.forward``) on a tiny truth
earth to emit a native file (SEG-Y / LAS+deviation / GeoTIFF time-series / QuakeML+CSV),
then feeds it through the adapter's :meth:`parse` and asserts the recovered doc-02
primitives are correct: kind, units (from ``geosim.spatial.REGISTRY``), the InSAR leading
``t`` axis (explicit ISO-8601 UTC, doc 02 §8), and the well-log ``wellId`` curve↔path join
(doc 03 §3d). SMALL truth grid keeps runtime trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosim.ingestion import RawSource, adapter_named
from geosim.ingestion.adapters.insar import InsarGeotiffAdapter
from geosim.ingestion.adapters.microseismic import MicroseismicQuakeMlAdapter
from geosim.ingestion.adapters.seismic import SeismicSegyAdapter
from geosim.ingestion.adapters.welllog import WellLogLasAdapter
from geosim.spatial import REGISTRY
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
from geosim.synthgen.forward import (
    Acquisition,
    InSARForward,
    MicroseismicForward,
    SeismicReflectionForward,
    WellLogForward,
)

# --------------------------------------------------------------- a tiny truth earth


def _tiny_scene(seed: int = 11) -> SceneSpec:
    return SceneSpec(
        id="tiny-adapter-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-500, xmax=500, ymin=-500, ymax=500,
            zmin=-900, zmax=200, dx=100, dy=100, dz=100,
        ),
        surface=SurfaceSpec(kind="flat", base_elev=100.0),
        layers=(
            LayerSpec("alluvium", "surface", (100.0, 150.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-500, -100), (500, 150)),
                      kind="normal", dip=60, dip_azimuth=90, throw=150, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=250.0,
                top_elev=100.0, bottom_elev=-700.0, controlled_by="range-front",
                temp_peak=200.0, alteration_frac=0.8, porosity_boost=0.04,
                salinity_tds=8000.0, fracture_density=0.5,
            ),
        ),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def earth():
    return compile_scene(_tiny_scene())


@pytest.fixture()
def acq(tmp_path):
    return Acquisition(
        seis_n_traces=16,
        seis_n_samples=128,
        ms_n_events=24,
        insar_n_epochs=4,
        insar_pixel=150.0,
        params={"out_dir": str(tmp_path)},
    )


def _rng():
    return np.random.default_rng(99)


# --------------------------------------------------------------- registration


def test_four_adapters_registered():
    for name in ("seismic-segy-v1", "welllog-las-v1",
                 "insar-geotiff-v1", "microseismic-quakeml-v1"):
        a = adapter_named(name)
        assert a is not None, name


def test_canonical_method_pairs():
    from geosim.plugins import is_canonical_pair

    for a in (SeismicSegyAdapter(), WellLogLasAdapter(),
              InsarGeotiffAdapter(), MicroseismicQuakeMlAdapter()):
        assert is_canonical_pair(a.method, a.submethod), (a.method, a.submethod)


# --------------------------------------------------------------- seismic SEG-Y


def test_seismic_segy_roundtrip(earth, acq):
    arts = SeismicReflectionForward().simulate(earth, acq, _rng())
    segy = next(a for a in arts if a.fmt == "segy")
    adp = SeismicSegyAdapter()
    assert adp.sniff(b"", segy.path.name) > 0.5

    res = adp.parse(RawSource(filename=segy.path.name, path=str(segy.path)))
    assert not res.warnings or all(w.severity.value != "high" for w in res.warnings)
    assert len(res.property_models) == 1
    pm = res.property_models[0]
    # a 2-D line → native vertical curtain (doc 02 §4 / doc 03 §3d).
    assert pm.support == "section"
    assert pm.property == "velocity_p"
    # values [time, along-line] = (n_samples, n_traces).
    assert pm.values.shape == (acq.seis_n_samples, acq.seis_n_traces)
    assert pm.meta["n_traces"] == acq.seis_n_traces
    assert pm.meta["n_samples"] == acq.seis_n_samples
    # time-axis spacing recovered from the SEG-Y sample interval (2 ms).
    assert pm.spacing[0] == pytest.approx(acq.seis_dt * 1000.0, abs=1e-6)
    # horizons GeoJSON joined as features (doc 03 §2 surfaces).
    assert any(f.feature_type == "horizon" for f in res.features)


# --------------------------------------------------------------- well log LAS


def test_welllog_las_roundtrip_units_and_join(earth, acq):
    arts = WellLogForward().simulate(earth, acq, _rng())
    las = next(a for a in arts if a.fmt == "las")
    adp = WellLogLasAdapter()
    assert adp.sniff(las.path.read_bytes()[:4096], las.path.name) > 0.5

    res = adp.parse(RawSource(filename=las.path.name, path=str(las.path)))
    assert len(res.observations) == 1
    obs = res.observations[0]
    assert obs.geometry_kind == "wellcurve"
    # canonical property_type mapping (doc 01 §5) for the registry-mapped curves.
    assert {"resistivity", "density", "velocity_p", "temperature"} <= set(obs.values)
    # native units declared per property (normalizer canonicalizes, doc 03 §3).
    assert res.units["resistivity"] in ("ohm.m", "ohm*m")
    assert res.units["velocity_p"] == "m/s"
    assert res.units["temperature"] == "degC"
    # GR has no registry key → carried as methodData (doc 02 §3).
    assert "GR" in obs.meta["methodData"]

    # wellPath feature joined to the curves by wellId (doc 03 §3d).
    assert len(res.features) == 1
    path = res.features[0]
    assert path.feature_type == "wellPath"
    assert path.props["wellId"] == obs.meta["wellId"]
    assert path.props["trajectory"] == "deviation_survey"  # sibling _deviation.csv found
    # trajectory is genuinely deviated (min curvature from the 35° survey, doc 01 §4).
    coords = np.asarray(path.geometry["coordinates"], dtype=float)
    assert coords.shape[1] == 3
    horiz_step = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    assert horiz_step.max() > 1.0  # not a vertical column

    # curve coords ride the trajectory: depth (Up) decreases monotonically-ish with MD.
    cz = np.asarray(obs.coords, dtype=float)[:, 2]
    assert cz[0] > cz[-1]  # shallow (top) above deep (bottom)


def test_welllog_vertical_assumption_without_survey(earth, acq, tmp_path):
    # Parse a LAS whose sibling deviation CSV is absent → vertical-well warning (doc 03 §5).
    arts = WellLogForward().simulate(earth, acq, _rng())
    las = next(a for a in arts if a.fmt == "las")
    lone = tmp_path / "LONE.las"
    lone.write_text(las.path.read_text(encoding="utf-8"), encoding="utf-8")

    res = WellLogLasAdapter().parse(RawSource(filename="LONE.las", path=str(lone)))
    assert any(w.code == "no_deviation_survey" for w in res.warnings)
    assert res.features[0].props["trajectory"] == "vertical_assumption"


# --------------------------------------------------------------- InSAR GeoTIFF


def test_insar_geotiff_timeseries_leading_t_axis(earth, acq):
    arts = InSARForward().simulate(earth, acq, _rng())
    assert len(arts) == acq.insar_n_epochs
    first = arts[0]
    adp = InsarGeotiffAdapter()
    assert adp.sniff(b"II*\x00", first.path.name) > 0.5

    res = adp.parse(RawSource(filename=first.path.name, path=str(first.path)))
    assert len(res.property_models) == 1
    pm = res.property_models[0]
    assert pm.property == "deformation"
    assert pm.support == "grid2d"
    # leading t axis: cube is [t, y, x] with one slice per epoch (doc 02 §8).
    assert pm.values.ndim == 3
    assert pm.values.shape[0] == acq.insar_n_epochs
    assert pm.meta["leading_axis"] == "t"
    # explicit ISO-8601 UTC epochs, not project-epoch offsets (doc 02 §8).
    epochs = pm.meta["timeAxis"]["epochs"]
    assert len(epochs) == acq.insar_n_epochs
    assert all(e.endswith("Z") and e.startswith("20") for e in epochs)
    # native units mm (the deformation registry canonical unit).
    assert res.units["deformation"] == REGISTRY.get("deformation").canonical_unit


# --------------------------------------------------------------- microseismic QuakeML


def test_microseismic_quakeml_pointcloud_4d(earth, acq):
    arts = MicroseismicForward().simulate(earth, acq, _rng())
    qml = next(a for a in arts if a.fmt == "quakeml")
    adp = MicroseismicQuakeMlAdapter()
    assert adp.sniff(qml.path.read_bytes()[:4096], qml.path.name) > 0.5

    res = adp.parse(RawSource(filename=qml.path.name, path=str(qml.path)))
    assert len(res.features) == 1
    feat = res.features[0]
    assert feat.feature_type == "pointCloud"
    assert feat.geometry["type"] == "MultiPoint"
    n = acq.ms_n_events
    assert feat.props["n_events"] == n
    coords = np.asarray(feat.geometry["coordinates"], dtype=float)
    assert coords.shape == (n, 3)  # x, y, z
    # 4-D: parallel ISO-8601 UTC time array + magnitudes (doc 02 §8, doc 03 §5).
    assert len(feat.props["time"]) == n
    assert all(t.endswith("Z") for t in feat.props["time"])
    assert len(feat.props["mag"]) == n
    assert set(feat.props["dims"]) == {"x", "y", "z", "t", "mag"}
    # x,y joined from the sibling catalog CSV (not the zeroed QuakeML lat/lon).
    assert np.any(coords[:, 0] != 0.0) or np.any(coords[:, 1] != 0.0)


def test_microseismic_without_catalog_csv_warns(earth, acq, tmp_path):
    arts = MicroseismicForward().simulate(earth, acq, _rng())
    qml = next(a for a in arts if a.fmt == "quakeml")
    iso_dir = tmp_path / "isolated"  # no sibling catalog CSV here
    iso_dir.mkdir()
    lone = iso_dir / "lonely.quakeml"
    lone.write_text(qml.path.read_text(encoding="utf-8"), encoding="utf-8")

    res = MicroseismicQuakeMlAdapter().parse(
        RawSource(filename="lonely.quakeml", path=str(lone))
    )
    assert any(w.code == "no_catalog_csv" for w in res.warnings)
    assert len(res.features) == 1
