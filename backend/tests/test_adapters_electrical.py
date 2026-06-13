"""Round-trip tests for the electrical / EM adapters (doc 03 §2 ert/ip/em/mt rows).

Each adapter is exercised against a tiny :class:`~geosim.synthgen.truth.TruthEarth` run
through the *matching* synthgen T0 forward (doc 05 §4) — closing the OVERVIEW §8 forward→
ingest round-trip. We assert the doc-03 contract: the emitted primitive's ``geometry_kind``
(``profile2d`` for ert/ip pseudosections, ``soundings`` for em, ``tensor`` for mt sites),
the canonical doc-01 property keys + units (IP keys SPLIT per doc 01 §5 / doc 03 §2), and a
plausible bbox after normalization. All grids are SMALL and all I/O is to ``tmp_path`` —
no Docker/Postgres/Redis (headless).
"""

from __future__ import annotations

import numpy as np
import pytest

from geosim.ingestion import (
    RawSource,
    adapter_named,
    detect,
    normalize,
)
from geosim.spatial import FrameMode, SpatialFrame
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
    ERTForward,
    IPForward,
    MTForward,
    TDEMForward,
)


def _tiny_scene(seed: int = 7) -> SceneSpec:
    """A SMALL (13×12×12) altered-plume scene — enough physics for every forward."""
    return SceneSpec(
        id="elec-rt-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-600, xmax=600, ymin=-600, ymax=600,
            zmin=-1000, zmax=300, dx=100, dy=100, dz=100,
        ),
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
        ert_n_electrodes=16, ert_spacing=80.0,
        em_n_soundings=4, mt_n_periods=12,
        params={"out_dir": str(tmp_path)},
    )


def _rng():
    return np.random.default_rng(123)


def _local_frame() -> SpatialFrame:
    return SpatialFrame(mode=FrameMode.LOCAL)


def _forward_and_parse(fwd, earth, acq, *, idx: int = 0):
    """Run a forward, content-address the first artifact, detect+parse it (doc 03 §7)."""
    arts = fwd.simulate(earth, acq, _rng())
    art = arts[idx]
    src = RawSource(filename=art.path.name, data=art.path.read_bytes())
    return art, src


def _finite_bbox(bbox: dict[str, float]) -> bool:
    return bool(bbox) and all(np.isfinite(v) for v in bbox.values())


# ─────────────────────────────── registration / detection ───────────────────────────────


def test_all_four_adapters_registered():
    for name in ("ert-stg-v1", "ip-stg-v1", "em-xyz-v1", "mt-edi-v1"):
        assert adapter_named(name) is not None


def test_canonical_method_pairs():
    from geosim.plugins import is_canonical_pair

    for name, method, sub in (
        ("ert-stg-v1", "ert", "dc_resistivity"),
        ("ip-stg-v1", "ip", "ip_time"),
        ("em-xyz-v1", "em", "tdem"),
        ("mt-edi-v1", "mt", None),
    ):
        a = adapter_named(name)
        assert a.method == method and a.submethod == sub
        assert is_canonical_pair(method, sub)


# ─────────────────────────────── ERT (.stg → profile2d resistivity) ──────────────────────


def test_ert_roundtrip(earth, acq):
    art, src = _forward_and_parse(ERTForward(), earth, acq)
    chosen = detect(src)
    assert chosen.name == "ert-stg-v1"  # disambiguated from IP by the value label

    pr = chosen.parse(src)
    assert len(pr.observations) == 1 and not pr.property_models
    obs = pr.observations[0]
    assert obs.geometry_kind == "profile2d"
    assert obs.primary_property == "resistivity"
    assert list(obs.values) == ["resistivity"]
    assert pr.units["resistivity"] == "ohm*m"
    assert np.asarray(obs.coords).shape[1] == 3
    assert np.all(obs.values["resistivity"] > 0)

    bundle = normalize(pr, _local_frame())
    nobs = bundle.observations[0]
    assert nobs.geometry_kind == "profile2d"
    assert _finite_bbox(nobs.bbox)
    # pseudodepth → depth_below_surface → negative elevation band
    assert nobs.bbox["zmax"] <= 0.0


# ─────────────────────────────── IP (.stg → profile2d chargeability) ─────────────────────


def test_ip_roundtrip_split_keys(earth, acq):
    art, src = _forward_and_parse(IPForward(), earth, acq)
    chosen = detect(src)
    assert chosen.name == "ip-stg-v1"

    pr = chosen.parse(src)
    obs = pr.observations[0]
    assert obs.geometry_kind == "profile2d"
    # IP keys are SPLIT (doc 01 §5 / doc 03 §2) — the mV/V synthgen forward → chargeability_mv_v
    assert obs.primary_property == "chargeability_mv_v"
    assert "resistivity" not in obs.values
    assert pr.units == {"chargeability_mv_v": "mV/V"}

    bundle = normalize(pr, _local_frame())
    nobs = bundle.observations[0]
    assert "chargeability_mv_v" in nobs.values
    assert _finite_bbox(nobs.bbox)


def test_ip_stg_label_routing():
    """The IP adapter routes the .stg ``value:`` label to the right split key (doc 03 §2)."""
    from geosim.ingestion.adapters._stg import value_property_and_unit

    vpu = value_property_and_unit
    assert vpu("apparent_chargeability_mv_v", default_prop="chargeability_mv_v") == (
        "chargeability_mv_v", "mV/V"
    )
    assert vpu("ip_phase_mrad", default_prop="chargeability_mv_v") == ("phase_mrad", "mrad")
    assert vpu("chargeability_time_ms", default_prop="chargeability_mv_v") == (
        "chargeability_time_ms", "ms"
    )


# ─────────────────────────────── EM (.xyz → soundings conductivity) ──────────────────────


def test_em_roundtrip(earth, acq):
    art, src = _forward_and_parse(TDEMForward(), earth, acq)
    chosen = detect(src)
    assert chosen.name == "em-xyz-v1"

    pr = chosen.parse(src)
    obs = pr.observations[0]
    assert obs.geometry_kind == "soundings"
    assert obs.primary_property == "conductivity"
    assert pr.units["conductivity"] == "S/m"
    assert np.asarray(obs.coords).shape[1] == 3
    assert np.all(obs.values["conductivity"] > 0)
    # multiple soundings on the grid (doc 03 §4 stitching candidates)
    assert obs.meta["n_soundings"] >= 1

    bundle = normalize(pr, _local_frame())
    nobs = bundle.observations[0]
    assert nobs.geometry_kind == "soundings"
    assert _finite_bbox(nobs.bbox)


# ─────────────────────────────── MT (EDI → tensor resistivity + phase) ───────────────────


def test_mt_roundtrip(earth, acq):
    art, src = _forward_and_parse(MTForward(), earth, acq)
    chosen = detect(src)
    assert chosen.name == "mt-edi-v1"

    pr = chosen.parse(src)
    obs = pr.observations[0]
    assert obs.geometry_kind == "tensor"
    assert obs.primary_property == "resistivity"
    # app-res + phase, split into canonical keys (doc 03 §2 mt row)
    assert set(obs.values) == {"resistivity", "phase_mrad"}
    assert pr.units["resistivity"] == "ohm*m"
    assert pr.units["phase_mrad"] == "deg"  # EDI native is degrees
    assert np.all(obs.values["resistivity"] > 0)

    bundle = normalize(pr, _local_frame())
    nobs = bundle.observations[0]
    assert nobs.geometry_kind == "tensor"
    # degrees → canonical milliradians (doc 01 §5)
    deg = obs.values["phase_mrad"]
    mrad = nobs.values["phase_mrad"]
    assert np.allclose(mrad, np.deg2rad(deg) * 1000.0, rtol=1e-3)
    # the site bbox is a single plan point (tensor = one site)
    assert nobs.bbox["xmin"] == pytest.approx(nobs.bbox["xmax"])
    assert nobs.bbox["ymin"] == pytest.approx(nobs.bbox["ymax"])


def test_mt_handles_missing_refloc():
    """A REFLOC-less EDI still parses, placing the site at origin with a HIGH warning."""
    edi = (
        ">HEAD\n  DATAID=NOLOC\n\n>=MTSECT\n  NFREQ=2\n\n"
        ">FREQ ROT=ZROT // 2\n 1.0E+01  1.0E+00\n"
        ">RHOXY ROT=ZROT // 2\n 1.0E+02  2.0E+02\n"
        ">PHSXY ROT=ZROT // 2\n 4.5E+01  4.0E+01\n>END\n"
    )
    src = RawSource(filename="noloc.edi", data=edi.encode())
    pr = adapter_named("mt-edi-v1").parse(src)
    assert pr.observations[0].geometry_kind == "tensor"
    assert any(w.code == "no_location" for w in pr.warnings)
