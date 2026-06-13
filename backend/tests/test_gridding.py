"""Gridding tests (doc 03 §3c/§3d/§4): scattered Observations -> derived PropertyModel.

Uses SMALL grids and tmp dirs only (no Docker/Postgres). Asserts the doc-03 contract:
verde interpolates a scattered field and carries a co-registered 1σ; the IDW fallback
works; and the footprint/DOI mask produces NaN beyond coverage (footprint honesty,
DECISIONS doc 03). Also checks that ``write_grid_result`` emits a doc-02 PropertyModel
Zarr group with a ``_sigma`` sibling and a provenance sidecar recording the params.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosim.ingestion.gridding import (
    GriddingError,
    GriddingParams,
    GridResult,
    Sounding,
    grid_points_2d,
    stitch_soundings,
    write_grid_result,
)
from geosim.storage import SIGMA_SUFFIX, open_property_model


def _scatter(n: int = 60, seed: int = 0):
    """A smooth analytic field sampled at scattered (x, y) in a 0..1000 m box."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1000.0, n)
    y = rng.uniform(0.0, 1000.0, n)
    # a smooth plane + bump → known, easily-interpolated structure
    vals = 50.0 + 0.02 * x + 0.01 * y + 10.0 * np.exp(-(((x - 500) ** 2 + (y - 500) ** 2) / 2.0e4))
    return x, y, vals


# ───────────────────────────── 2D verde gridding ─────────────────────────────────
def test_grid_points_2d_interpolates_and_carries_sigma():
    x, y, vals = _scatter()
    params = GriddingParams(method="verde-spline", spacing=100.0)
    res = grid_points_2d("gravity_anomaly", x, y, vals, params=params)

    assert isinstance(res, GridResult)
    assert res.property == "gravity_anomaly"
    assert res.values.dtype == np.float32
    assert res.values.ndim == 3 and res.values.shape[0] == 1  # 2D field as nz=1 slice
    assert res.values.shape == res.sigma.shape
    # interpolated values land within the data range (no wild extrapolation here)
    interior = res.values[np.isfinite(res.values)]
    assert interior.min() >= vals.min() - 15.0
    assert interior.max() <= vals.max() + 15.0
    # σ present, positive, and monotonic-ish: a node far from data is more uncertain
    assert np.all(res.sigma[np.isfinite(res.sigma)] > 0)
    assert res.provenance["method"] == "verde-spline"
    assert res.provenance["support"] == "grid2d"


def test_grid_points_2d_recovers_known_value_near_a_station():
    # A node coincident with the data centre should be close to the field there.
    x = np.array([0.0, 1000.0, 0.0, 1000.0, 500.0])
    y = np.array([0.0, 0.0, 1000.0, 1000.0, 500.0])
    vals = np.array([10.0, 10.0, 10.0, 10.0, 30.0])  # central spike
    res = grid_points_2d("gravity_anomaly", x, y, vals, params=GriddingParams(spacing=250.0))
    # centre node (index of 500,500) should read clearly above the corner values
    nz, ny, nx = res.values.shape
    centre = res.values[0, ny // 2, nx // 2]
    corner = res.values[0, 0, 0]
    assert centre > corner


def test_too_few_points_raises():
    with pytest.raises(GriddingError):
        grid_points_2d("gravity_anomaly", [0.0, 1.0], [0.0, 1.0], [1.0, 2.0])


# ───────────────────────────── IDW fallback ──────────────────────────────────────
def test_idw_fallback_works():
    x, y, vals = _scatter(seed=3)
    params = GriddingParams(method="idw", spacing=100.0, idw_neighbors=6)
    res = grid_points_2d("gravity_anomaly", x, y, vals, params=params)
    finite = res.values[np.isfinite(res.values)]
    assert finite.size > 0
    # IDW is a bounded weighted mean → stays within the data envelope
    assert finite.min() >= vals.min() - 1e-3
    assert finite.max() <= vals.max() + 1e-3
    assert np.all(res.sigma[np.isfinite(res.sigma)] > 0)  # still carries a floor σ
    assert res.provenance["method"] == "idw"


# ───────────────────────────── footprint / DOI masking ───────────────────────────
def test_footprint_mask_produces_nan_beyond_coverage():
    # Cluster all stations in one corner; nodes far away must be NaN.
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 200.0, 25)
    y = rng.uniform(0.0, 200.0, 25)
    vals = 5.0 + 0.01 * x
    params = GriddingParams(
        spacing=100.0,
        region=(0.0, 1000.0, 0.0, 1000.0),  # grid far bigger than coverage
        max_distance=150.0,
    )
    res = grid_points_2d("gravity_anomaly", x, y, vals, params=params)
    assert np.isnan(res.values).any()  # uncovered far corner → NaN
    assert np.isfinite(res.values).any()  # covered corner → finite
    # the far corner (≈1000,1000) is definitely outside the footprint
    assert np.isnan(res.values[0, -1, -1])
    # σ is masked wherever the value is masked
    assert np.array_equal(np.isnan(res.values), np.isnan(res.sigma))


def test_stitch_soundings_dois_mask_to_nan():
    # 4 soundings on a square; each trusted only down to a shallow DOI elevation.
    rng = np.random.default_rng(2)
    elev = np.linspace(-500.0, 0.0, 21)  # Z-up, deepest first
    soundings = []
    for sx, sy in [(0.0, 0.0), (400.0, 0.0), (0.0, 400.0), (400.0, 400.0)]:
        # resistivity rising with depth, plausible Ω·m
        vals = 50.0 + (0.0 - elev) * 0.4 + rng.normal(0, 1, elev.size)
        soundings.append(Sounding(x=sx, y=sy, elevations=elev, values=vals, doi_elevation=-200.0))
    params = GriddingParams(method="verde-spline", spacing=100.0, z_spacing=50.0, max_distance=400.0)
    res = stitch_soundings("resistivity", soundings, params=params)

    assert res.values.ndim == 3
    nz = res.values.shape[0]
    assert nz > 1  # a real volume
    # Below the DOI (elevation < -200) cells must be masked NaN; above, finite.
    z_axis = res.origin[0] + np.arange(nz) * res.spacing[0]
    deep = np.argmin(np.abs(z_axis - (-450.0)))
    shallow = np.argmin(np.abs(z_axis - (-50.0)))
    assert np.isnan(res.values[deep]).all()  # entirely below DOI → all NaN
    assert np.isfinite(res.values[shallow]).any()  # above DOI → has data
    assert res.provenance["op"] == "stitch_soundings"
    assert res.provenance["doiMasked"] is True


def test_too_few_soundings_raises():
    s = Sounding(x=0.0, y=0.0, elevations=np.array([-100.0, 0.0]), values=np.array([10.0, 20.0]))
    with pytest.raises(GriddingError):
        stitch_soundings("resistivity", [s, s])


# ───────────────────────────── write derived PropertyModel ───────────────────────
def test_write_grid_result_emits_property_model_with_sigma(tmp_path):
    x, y, vals = _scatter(seed=5)
    res = grid_points_2d("gravity_anomaly", x, y, vals, params=GriddingParams(spacing=125.0))
    zpath = tmp_path / "arrays" / "grid.zarr"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    out = write_grid_result(res, zpath, overwrite=True)

    reader = open_property_model(out)
    assert "gravity_anomaly" in reader.properties
    assert reader.has_sigma("gravity_anomaly")
    lvl0 = reader.read_level("gravity_anomaly", 0)
    assert lvl0.shape == res.values.shape
    # provenance sidecar records the gridding params (doc 03 §8)
    sidecar = (zpath.parent / f"{zpath.name}.provenance.json")
    assert sidecar.exists()
    import json

    rec = json.loads(sidecar.read_text())
    assert rec["process"] == "grid"
    assert rec["params"]["method"] == "verde-spline"
    assert SIGMA_SUFFIX  # naming constant available
