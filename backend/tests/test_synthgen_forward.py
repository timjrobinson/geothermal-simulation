"""Tests for the per-method T0 forward models (doc 05 §4 + §6 T0, doc 08 §4d).

Every forward runs on a SMALL :class:`TruthEarth` and must (1) emit native-format file(s)
that re-read with their parsing library (rasterio/segyio/lasio/obspy/pandas + the custom
EDI/STG/XYZ text writers), with plausible shapes/values, and (2) honour the doc-05 §4
"only-sees-what-it-could" degradation (e.g. a magnetic *low* over the altered upflow, the
ERT/MT depth split, seismic blind to fluid). All grids are tiny and all I/O is to
``tmp_path`` — no Docker/Postgres/Redis (headless).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from geosim.spatial import convert
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
    FORWARD_MODELS,
    Acquisition,
    Artifact,
    ForwardModel,
    GeologyMapForward,
    GravityForward,
    HeatFlowForward,
    InSARForward,
    IPForward,
    MagneticsForward,
    MicroseismicForward,
    MTForward,
    SeismicReflectionForward,
    TDEMForward,
    WellLogForward,
    all_forwards,
    get_forward,
)

# --------------------------------------------------------------- a tiny truth earth


def _tiny_scene(seed: int = 5) -> SceneSpec:
    """A SMALL (12×12×12) Basin-&-Range scene with a fault-controlled altered plume."""
    return SceneSpec(
        id="tiny-fwd-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-600, xmax=600, ymin=-600, ymax=600,
            zmin=-1000, zmax=300, dx=100, dy=100, dz=100,
        ),  # → (nz=13, ny=12, nx=12)
        # tilted-block surface so the dip + fault expose >1 unit at the surface
        # (real mapped geology contacts; doc 05 §4 row 13).
        surface=SurfaceSpec(kind="tilted-block", base_elev=150.0, tilt_x=0.15),
        layers=(
            LayerSpec("alluvium", "surface", (50.0, 80.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-600, -100), (600, 200)),
                      kind="normal", dip=60, dip_azimuth=90, throw=200, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=300.0,
                top_elev=150.0, bottom_elev=-900.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.9, porosity_boost=0.04,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=100.0, clay_cap_thickness=150.0,
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
        gravity_spacing=150.0,
        mag_line_spacing=150.0,
        ert_n_electrodes=16,
        ert_spacing=80.0,
        em_n_soundings=4,
        mt_n_periods=12,
        seis_n_traces=24,
        seis_n_samples=256,
        ms_n_events=30,
        insar_n_epochs=3,
        insar_pixel=150.0,
        heat_n_points=16,
        params={"out_dir": str(tmp_path)},
    )


def _rng():
    return np.random.default_rng(123)


# --------------------------------------------------------------- contract & registry


def test_registry_covers_every_method():
    pairs = set(FORWARD_MODELS)
    # the twelve doc-05 §4 methods, canonical (method, submethod) pairs (doc 02 §2)
    expected = {
        ("gravity", None), ("magnetics", None),
        ("ert", "dc_resistivity"), ("ip", "ip_time"),
        ("em", "tdem"), ("mt", None),
        ("seismic", "reflection"), ("microseismic", None),
        ("insar", None), ("welllog", None),
        ("heatflow", None), ("geology", None),
    }
    assert expected <= pairs


def test_all_forwards_conform_to_protocol():
    for fwd in all_forwards():
        assert isinstance(fwd, ForwardModel)
        assert fwd.fidelity == "plausible"  # T0 tier (doc 05 §6)


def test_canonical_method_pairs():
    from geosim.plugins import is_canonical_pair

    for (method, submethod) in FORWARD_MODELS:
        assert is_canonical_pair(method, submethod), (method, submethod)


def test_get_forward_lookup_and_error():
    assert isinstance(get_forward("gravity"), GravityForward)
    with pytest.raises(KeyError):
        get_forward("nope")


def test_every_artifact_carries_synthetic_provenance(earth, acq):
    rng = _rng()
    for fwd in all_forwards():
        for art in fwd.simulate(earth, acq, rng):
            assert isinstance(art, Artifact)
            assert art.path.exists()
            prov = art.provenance.to_dict()
            assert prov["source"] == "synthgen"
            assert prov["sceneId"] == earth.spec.id
            assert prov["seed"] == earth.spec.seed
            assert prov["fidelity"] == "plausible"


# --------------------------------------------------------------- gravity


def test_gravity_csv_and_geotiff(earth, acq):
    import rasterio

    arts = GravityForward().simulate(earth, acq, _rng())
    fmts = {a.fmt for a in arts}
    assert fmts == {"csv", "geotiff"}
    csv = next(a for a in arts if a.fmt == "csv")
    df = pd.read_csv(csv.path)
    assert {"x", "y", "bouguer_mgal"} <= set(df.columns)
    assert np.isfinite(df["bouguer_mgal"]).all()
    # plausible Bouguer magnitude for a small model (well under tens of mGal)
    assert df["bouguer_mgal"].abs().max() < 50.0
    tif = next(a for a in arts if a.fmt == "geotiff")
    with rasterio.open(tif.path) as r:
        grid = r.read(1)
    assert grid.ndim == 2 and np.isfinite(grid).any()


# --------------------------------------------------------------- magnetics


def test_magnetics_low_over_altered_zone(earth, acq):
    """doc 05 §4.2: alteration sets χ→~0 → magnetics sees a LOW over the upflow."""
    import rasterio

    arts = MagneticsForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"xyz", "geotiff"}
    xyz = next(a for a in arts if a.fmt == "xyz")
    txt = xyz.path.read_text()
    assert txt.startswith("LINE")  # .xyz header (parseable line file)
    assert len(txt.strip().splitlines()) > 2
    tif = next(a for a in arts if a.fmt == "geotiff")
    with rasterio.open(tif.path) as r:
        grid = np.flipud(r.read(1))  # back to ascending-y
    # plausible nT magnitude
    assert np.nanmax(np.abs(grid)) < 5000.0
    ny, nx = grid.shape
    cy, cx = ny // 2, nx // 2
    center = float(grid[cy, cx])
    edge = float(np.nanmean(np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])))
    assert center < edge  # magnetic low over the altered plume


# --------------------------------------------------------------- ert / ip


def test_ert_pseudosection_stg(earth, acq):
    arts = get_forward("ert", "dc_resistivity").simulate(earth, acq, _rng())
    assert len(arts) == 1 and arts[0].fmt == "stg"
    lines = arts[0].path.read_text().splitlines()
    assert lines[0].startswith("AGI SuperSting")
    header = next(ln for ln in lines if ln.startswith("Idx,"))
    cols = header.split(",")
    assert "value" in cols and "pseudodepth" in cols
    data = [ln for ln in lines if ln and ln[0].isdigit()]
    assert len(data) > 3
    apparent = np.array([float(ln.split(",")[-1]) for ln in data])
    assert np.all(apparent > 0)  # apparent resistivity Ω·m positive


def test_ip_pseudosection_colocated(earth, acq):
    ert = get_forward("ert", "dc_resistivity").simulate(earth, acq, _rng())
    ip = IPForward().simulate(earth, acq, _rng())
    assert ip[0].fmt == "stg"
    ert_n = len([ln for ln in ert[0].path.read_text().splitlines() if ln and ln[0].isdigit()])
    ip_n = len([ln for ln in ip[0].path.read_text().splitlines() if ln and ln[0].isdigit()])
    assert ert_n == ip_n  # co-located with ERT (doc 05 §4 row 4)


# --------------------------------------------------------------- em / mt


def test_tdem_soundings_xyz(earth, acq):
    arts = TDEMForward().simulate(earth, acq, _rng())
    assert arts[0].fmt == "xyz"
    lines = arts[0].path.read_text().splitlines()
    assert "DEPTH_M" in lines[0] and "APP_COND" in lines[0]
    rows = [ln.split() for ln in lines[1:] if ln]
    depths = np.array([float(r[4]) for r in rows])
    conds = np.array([float(r[5]) for r in rows])
    assert (depths >= 0).all() and (conds > 0).all()
    # later (deeper) gates probe deeper — DOI grows with time (smoke ring)
    assert depths.max() > depths.min()


def test_mt_edi_per_station(earth, acq):
    arts = MTForward().simulate(earth, acq, _rng())
    assert all(a.fmt == "edi" for a in arts)
    assert len(arts) >= 1
    txt = arts[0].path.read_text()
    for tag in (">HEAD", ">FREQ", ">RHOXY", ">PHSXY", ">END"):
        assert tag in txt
    # parse the RHOXY block → positive apparent resistivities
    block = txt.split(">RHOXY")[1].split(">PHSXY")[0]
    vals = [float(v) for v in block.replace("\n", " ").split() if _isfloat(v)]
    assert len(vals) >= acq.mt_n_periods - 1
    assert all(v > 0 for v in vals)


def _isfloat(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def test_mt_resolves_deeper_than_ert(earth, acq):
    """doc 05 §4.2 MT-vs-ERT depth split: MT skin depth reaches below ERT DOI."""
    from geosim.synthgen.forward.electrical import build_pseudosection

    ps = build_pseudosection(earth, acq, "resistivity", log_average=True)
    ert_max_depth = ps["pseudodepth"].max()
    # MT skin depth at the longest period over a ~100 Ω·m halfspace
    longest_T = acq.mt_periods[1]
    mt_skin = 503.0 * np.sqrt(100.0 * longest_T)
    assert mt_skin > ert_max_depth


# --------------------------------------------------------------- seismic


def test_seismic_segy_and_horizons(earth, acq):
    import segyio

    arts = SeismicReflectionForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"segy", "geojson"}
    segy = next(a for a in arts if a.fmt == "segy")
    with segyio.open(str(segy.path), ignore_geometry=True) as f:
        assert f.tracecount == acq.seis_n_traces
        assert len(f.samples) == acq.seis_n_samples
        trace = f.trace[0]
        assert np.isfinite(trace).all()
        assert np.any(trace != 0.0)  # non-trivial reflectivity convolution
    gj = next(a for a in arts if a.fmt == "geojson")
    fc = json.loads(gj.path.read_text())
    assert fc["type"] == "FeatureCollection"
    assert fc["features"][0]["properties"]["kind"] == "horizon"


# --------------------------------------------------------------- microseismic


def test_microseismic_quakeml_and_catalog(earth, acq):
    import obspy

    arts = MicroseismicForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"quakeml", "csv"}
    qml = next(a for a in arts if a.fmt == "quakeml")
    cat = obspy.read_events(str(qml.path))
    assert len(cat) == acq.ms_n_events
    mags = np.array([ev.magnitudes[0].mag for ev in cat])
    # Gutenberg-Richter: small events vastly outnumber large ones
    assert (mags < np.median(mags) + 1.0).mean() > 0.7
    csv = next(a for a in arts if a.fmt == "csv")
    df = pd.read_csv(csv.path)
    assert {"x", "y", "elev", "mag"} <= set(df.columns)
    assert len(df) == acq.ms_n_events


# --------------------------------------------------------------- insar


def test_insar_los_timeseries_geotiff(earth, acq):
    import rasterio

    arts = InSARForward().simulate(earth, acq, _rng())
    assert len(arts) == acq.insar_n_epochs
    assert all(a.fmt == "geotiff" for a in arts)
    last_center = None
    first_center = None
    for k, art in enumerate(arts):
        with rasterio.open(art.path) as r:
            grid = np.flipud(r.read(1))
        assert np.isfinite(grid).all()
        ny, nx = grid.shape
        c = float(grid[ny // 2, nx // 2])
        if k == 0:
            first_center = c
        last_center = c
    # deformation accumulates over the time series (uplift grows)
    assert abs(last_center) >= abs(first_center)


# --------------------------------------------------------------- well logs


def test_welllog_las_curves_and_temperature(earth, acq):
    import lasio

    arts = WellLogForward().simulate(earth, acq, _rng())
    las_art = next(a for a in arts if a.fmt == "las")
    dev_art = next(a for a in arts if a.fmt == "csv")
    las = lasio.read(str(las_art.path))
    curves = set(las.curves.keys())
    assert {"DEPT", "RES", "GR", "DEN", "VP", "TEMP"} <= curves
    res = las["RES"]
    assert np.nanmin(res) > 0
    # temperature curve is in display °C and increases with depth (geotherm)
    temp = las["TEMP"]
    assert np.nanmax(temp) > np.nanmin(temp)
    assert np.nanmax(temp) < 400.0  # plausible °C
    dev = pd.read_csv(dev_art.path)
    assert {"MD", "INC", "AZI"} <= set(dev.columns)


# --------------------------------------------------------------- heat flow


def test_heatflow_temperature_points_kelvin(earth, acq):
    arts = HeatFlowForward().simulate(earth, acq, _rng())
    assert arts[0].fmt == "csv"
    df = pd.read_csv(arts[0].path)
    assert {"temperature_k", "temperature_degc"} <= set(df.columns)
    assert len(df) == acq.heat_n_points
    # canonical kelvin column consistent with the display °C column (doc 01 §5)
    expect_c = convert(df["temperature_k"].to_numpy(), "kelvin", "degC")
    assert np.allclose(df["temperature_degc"].to_numpy(), expect_c, atol=1e-3)
    assert (df["temperature_k"] > 273.0).all()


# --------------------------------------------------------------- geology


def test_geology_geojson_contacts_and_faults(earth, acq):
    arts = GeologyMapForward().simulate(earth, acq, _rng())
    assert arts[0].fmt == "geojson"
    fc = json.loads(arts[0].path.read_text())
    kinds = {f["properties"]["kind"] for f in fc["features"]}
    assert "fault" in kinds  # authored fault trace exported
    # at least one mapped surface contact between lithologies
    assert "contact" in kinds
    for f in fc["features"]:
        assert f["properties"]["uncertainty"] == "interpretive"


# --------------------------------------------------------------- determinism


def test_deterministic_given_seed(earth, acq):
    a1 = GravityForward().simulate(earth, acq, np.random.default_rng(7))
    csv1 = pd.read_csv(next(a for a in a1 if a.fmt == "csv").path)
    a2 = GravityForward().simulate(earth, acq, np.random.default_rng(7))
    csv2 = pd.read_csv(next(a for a in a2 if a.fmt == "csv").path)
    assert np.allclose(csv1["bouguer_mgal"], csv2["bouguer_mgal"])
