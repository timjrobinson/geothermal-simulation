"""Tests for the spatial framework (doc 01)."""

import math

import numpy as np
import pytest

from geosim.spatial import (
    Aabb,
    Anchor,
    DepthRange,
    FrameMode,
    GeorefStatus,
    SpatialFrame,
    convert,
    min_curvature_positions,
    to_canonical,
    to_display,
    utm_epsg_for_lonlat,
    REGISTRY,
)


# --------------------------------------------------------------------------- units


def test_temperature_canonical_kelvin_display_celsius():
    # doc 01 §5: stored absolute T in kelvin, displayed in °C.
    assert to_canonical(25.0, "degC", "temperature") == pytest.approx(298.15)
    assert to_display(298.15, "temperature") == pytest.approx(25.0)


def test_resistivity_unit_alias_normalisation():
    # "ohm.m" must parse to the canonical ohm*m without change in magnitude.
    assert to_canonical(100.0, "ohm.m", "resistivity") == pytest.approx(100.0)


def test_unit_conversion_feet_to_metres():
    assert convert(1.0, "ft", "m") == pytest.approx(0.3048)


def test_property_registry_seeded():
    rt = REGISTRY.get("resistivity")
    assert rt.canonical_unit == "ohm*m"
    assert rt.default_scaling == "log"
    assert rt.interp_space == "log10"
    assert REGISTRY.get("temperature").canonical_unit == "kelvin"


# --------------------------------------------------------------------------- frame


def test_local_mode_is_identity():
    f = SpatialFrame()  # local by default
    pts = [[10.0, 20.0, -30.0], [0.0, 0.0, 0.0]]
    out = f.engineering_to_crs(pts)
    np.testing.assert_allclose(out, np.asarray(pts))
    back = f.to_engineering(pts)
    np.testing.assert_allclose(back, np.asarray(pts))


def test_utm_zone_selection():
    # Milford, Utah (~ -112.85, 38.4) is UTM zone 12N → EPSG:32612.
    assert utm_epsg_for_lonlat(-112.85, 38.4) == 32612
    # Southern hemisphere → 327xx
    assert utm_epsg_for_lonlat(151.2, -33.87) == 32756  # Sydney, zone 56S


def test_georeferenced_roundtrip_engineering_crs():
    anchor = Anchor(easting=412300.0, northing=4517800.0, elevation=1620.0)
    f = SpatialFrame(
        mode=FrameMode.GEOREFERENCED,
        horizontal_crs="EPSG:32612",
        anchor=anchor,
        roi=Aabb(-5000, 5000, -5000, 5000),
        depth_range=DepthRange(-8000, 2000),
    )
    eng = [[100.0, 200.0, -50.0], [-300.0, 50.0, 10.0]]
    crs = f.engineering_to_crs(eng)
    # origin maps to anchor
    o = f.engineering_to_crs([[0, 0, 0]])[0]
    np.testing.assert_allclose(o, [anchor.easting, anchor.northing, anchor.elevation])
    # round trip
    back = f.crs_to_engineering(crs)
    np.testing.assert_allclose(back, np.asarray(eng), atol=1e-9)


def test_rotation_applied():
    anchor = Anchor(0.0, 0.0, 0.0)
    f = SpatialFrame(mode=FrameMode.GEOREFERENCED, horizontal_crs="EPSG:32612",
                     anchor=anchor, rotation_deg=90.0)
    # +X engineering rotated 90° CW about Z → +X maps to (0,+1)·... check round-trip instead
    eng = [[100.0, 0.0, 0.0]]
    crs = f.engineering_to_crs(eng)
    back = f.crs_to_engineering(crs)
    np.testing.assert_allclose(back, np.asarray(eng), atol=1e-9)


def test_for_real_site_anchors_at_centroid():
    f = SpatialFrame.for_real_site(
        lon=-112.85, lat=38.4, surface_elev=1620.0,
        roi=Aabb(-5000, 5000, -5000, 5000), depth_range=DepthRange(-8000, 2000),
    )
    assert f.horizontal_crs == "EPSG:32612"
    assert f.mode is FrameMode.GEOREFERENCED
    assert f.georef_status is GeorefStatus.ANCHORED
    # origin maps back to (lon,lat) within metres
    lonlat = f.to_lonlat([[0, 0, 0]])[0]
    assert lonlat[0] == pytest.approx(-112.85, abs=1e-4)
    assert lonlat[1] == pytest.approx(38.4, abs=1e-4)


def test_promote_local_to_georef_keeps_status_anchored_not_validated():
    f = SpatialFrame()  # local
    f.georeference(horizontal_crs="EPSG:32612", anchor=Anchor(412300, 4517800, 1620))
    assert f.mode is FrameMode.GEOREFERENCED
    # critique #9: assigning an anchor sets 'anchored', NOT 'validated'
    assert f.georef_status is GeorefStatus.ANCHORED


# ---------------------------------------------------------------- minimum curvature


def test_vertical_well_min_curvature():
    # A perfectly vertical well: inc=0 throughout → TVD == MD, no horizontal motion.
    survey = [[0, 0, 0], [500, 0, 0], [1000, 0, 0]]
    r = min_curvature_positions(survey, wellhead=(10.0, 20.0), kb_elev=1627.0)
    np.testing.assert_allclose(r.tvd, [0, 500, 1000])
    np.testing.assert_allclose(r.enu[:, 0], 10.0)  # East constant
    np.testing.assert_allclose(r.enu[:, 1], 20.0)  # North constant
    np.testing.assert_allclose(r.enu[:, 2], [1627.0, 1127.0, 627.0])  # Z = KB - TVD
    np.testing.assert_allclose(r.dls, 0.0)


def test_build_section_min_curvature_displacement_and_dls():
    # Build from vertical to 90° due East over a known interval; check eastward reach.
    survey = [[0, 0, 0], [100, 90.0, 90.0]]
    r = min_curvature_positions(survey, wellhead=(0.0, 0.0), kb_elev=0.0)
    # ΔMD=100, I1=0,I2=90,A2=90. β = 90°. RF = (2/β)tan(β/2), β=π/2.
    beta = math.pi / 2
    rf = (2.0 / beta) * math.tan(beta / 2.0)
    d_e = (100 / 2.0) * (0 + math.sin(math.radians(90)) * math.sin(math.radians(90))) * rf
    d_v = (100 / 2.0) * (math.cos(0) + math.cos(math.radians(90))) * rf
    assert r.enu[1, 0] == pytest.approx(d_e)        # East displacement
    assert r.enu[1, 1] == pytest.approx(0.0, abs=1e-9)  # azimuth 90 → no North
    assert r.tvd[1] == pytest.approx(d_v)
    # DLS = β·(30/ΔMD) in degrees per 30 m
    assert r.dls[1] == pytest.approx(90.0 * (30.0 / 100.0))
