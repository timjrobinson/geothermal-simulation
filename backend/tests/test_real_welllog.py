"""Real Utah FORGE Schlumberger LAS ingestion (doc 03 §2 welllog row, real-format branch).

The synthetic round-trip lives in ``test_adapters_seismic.py`` (Engineering-frame LAS +
sibling ``_deviation.csv``). This file exercises the *real* vendor LAS branch added to the
``welllog`` adapter: an MD-indexed log in FEET that carries its wellhead in the ~Well
``LATI``/``LONG`` headers (DMS glyphs OR plain decimal degrees) and no source CRS. The
adapter places the well from LATI/LONG (``EPSG:4326`` so the normalizer reprojects — no
"georeferenced project requires a source CRS" hard fail), converts depths ft→m, maps
temperature (degF→K), velocity_p (slowness us/ft→m/s), density and resistivity by their
canonical families, and keeps gamma as raw ``methodData``.

Real files:
- ``welllog/16A/16A-78-32_Spectral.las`` — DMS LATI/LONG, GTEM/CTEM temperature (degF).
- ``welllog/58-32/58-32_DSI_Sonic.las`` — decimal LATI/LONG, DTCO slowness, RHOZ density.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geosim.api.frame_io import frame_row_kwargs
from geosim.catalog.db import create_all, make_engine, session_factory
from geosim.catalog.ids import IdKind, new_id
from geosim.catalog.models import Project, SpatialFrameRow
from geosim.ingestion import ingest_file
from geosim.ingestion.adapters.welllog import (
    WellLogLasAdapter,
    _parse_latlon,
    _slowness_to_velocity,
    _to_metres,
)
from geosim.ingestion.base import IngestStatus, RawSource
from geosim.spatial import Aabb, DepthRange, SpatialFrame
from geosim.storage import ensure_project_layout

_REPO = Path(__file__).resolve().parents[2]
_FORGE = _REPO / "data" / "utah-forge" / "measured" / "welllog"
_LAS_16A = _FORGE / "16A" / "16A-78-32_Spectral.las"
_LAS_58 = _FORGE / "58-32" / "58-32_DSI_Sonic.las"

_real_las = pytest.mark.skipif(
    not (_LAS_16A.exists() and _LAS_58.exists()),
    reason="real Utah FORGE LAS files not present under data/utah-forge/measured/welllog",
)

# FORGE site wellhead (doc note): ~38.50 N / -112.89 W.
_SITE_LON, _SITE_LAT = -112.89, 38.50


def _forge_frame() -> SpatialFrame:
    return SpatialFrame.for_real_site(
        lon=_SITE_LON, lat=_SITE_LAT, surface_elev=1655,
        roi=Aabb(-15000, 15000, -15000, 15000),
        depth_range=DepthRange(-10000, 2000),
    )


# ──────────────────────────────── unit helpers ────────────────────────────────


def test_parse_latlon_dms_and_decimal():
    # DMS with degree/minute/second glyphs + hemisphere (16A encoding).
    lat = _parse_latlon('38° 30\' 14.447" N', is_lat=True)
    lon = _parse_latlon('112° 53\' 47.066" W', is_lat=False)
    assert lat == pytest.approx(38.5040, abs=1e-3)
    assert lon == pytest.approx(-112.8964, abs=1e-3)
    # plain decimal degrees, no hemisphere token — must NOT pick up an N/S/E/W from a
    # word like "degrees" (regression: 'degrees' ends in 's' → spurious South flip).
    assert _parse_latlon("38.500562 degrees", is_lat=True) == pytest.approx(38.500562)
    assert _parse_latlon("-112.88703 degrees", is_lat=False) == pytest.approx(-112.88703)


def test_depth_feet_to_metres():
    assert _to_metres(np.array([100.0]), "F")[0] == pytest.approx(30.48)
    assert _to_metres(np.array([100.0]), "ft")[0] == pytest.approx(30.48)
    assert _to_metres(np.array([100.0]), "m")[0] == pytest.approx(100.0)


def test_slowness_to_velocity():
    # 80 us/ft compressional slowness → ~3810 m/s (typical rock Vp).
    v, unit = _slowness_to_velocity(np.array([80.0]), "us/ft")
    assert unit == "m/s"
    assert v[0] == pytest.approx(3810.0, rel=1e-3)
    # NULL/zero slowness → NaN, never inf.
    v2, _ = _slowness_to_velocity(np.array([0.0]), "us/ft")
    assert np.isnan(v2[0])


# ─────────────────────────────── adapter (parse) ───────────────────────────────


@_real_las
def test_real_16a_wellhead_and_temperature():
    res = WellLogLasAdapter().parse(RawSource(filename=_LAS_16A.name, path=str(_LAS_16A)))
    # geographic wellhead → EPSG:4326 so the normalizer reprojects (no source-CRS fail).
    assert res.source.crs == "EPSG:4326"
    assert res.source.z_convention == "elevation_up"
    obs = res.observations[0]
    assert obs.geometry_kind == "wellcurve"
    # GTEM/CTEM borehole temperature mapped (degF declared → normalizer → kelvin).
    assert "temperature" in obs.values
    assert res.units["temperature"] == "degF"
    # gamma (GR_EDTC/HCGR) carried as raw methodData, not a registry curve.
    assert "GR" in obs.meta["methodData"]
    # depths in metres on the MD axis.
    assert obs.meta["md_unit"] == "m"
    # wellPath feature at the LATI/LONG wellhead near the FORGE site.
    wh = res.features[0].props["wellhead"]
    assert res.features[0].feature_type == "wellPath"
    assert wh[0] == pytest.approx(_SITE_LON, abs=0.02)
    assert wh[1] == pytest.approx(_SITE_LAT, abs=0.02)


@_real_las
def test_real_58_velocity_and_density():
    res = WellLogLasAdapter().parse(RawSource(filename=_LAS_58.name, path=str(_LAS_58)))
    assert res.source.crs == "EPSG:4326"
    obs = res.observations[0]
    # DTCO compressional slowness → velocity_p (m/s); RHOZ → density.
    assert "velocity_p" in obs.values
    assert res.units["velocity_p"] == "m/s"
    assert "density" in obs.values
    # finite velocity samples are physical rock Vp.
    vp = np.asarray(obs.values["velocity_p"])
    finite = vp[np.isfinite(vp)]
    assert finite.size > 0
    assert finite.min() > 1000.0 and finite.max() < 9000.0
    # wellhead decimal-degree LATI/LONG near the FORGE site (positive latitude).
    wh = res.features[0].props["wellhead"]
    assert wh[1] == pytest.approx(_SITE_LAT, abs=0.02)


# ──────────────────────────── full ingest pipeline ────────────────────────────


@_real_las
@pytest.mark.parametrize("las_path", [_LAS_16A, _LAS_58], ids=["16A", "58-32"])
def test_real_las_ingests_into_georeferenced_project(tmp_path, las_path):
    root = tmp_path
    engine = make_engine(f"sqlite:///{root / 'catalog.db'}")
    create_all(engine)
    Session = session_factory(engine)
    frame = _forge_frame()
    pid = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=pid, name="forge", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(project_id=pid, **frame_row_kwargs(frame))
        s.add(proj)
        s.commit()
    ensure_project_layout(root, pid)

    with Session() as s:
        rep = ingest_file(s, root, pid, las_path, method_hint="welllog")
        s.commit()

    # ok / ok_with_warnings — NOT failed (no more "requires a source CRS" hard fail).
    assert rep.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS), rep.message
    assert rep.n_observations == 1   # the wellcurve Observation
    assert rep.n_features == 1       # the wellPath feature at the wellhead
    assert rep.records_total > 0
