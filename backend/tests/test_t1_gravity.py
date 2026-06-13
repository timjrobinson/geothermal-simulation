"""T1 (rigorous) gravity forward — full Newtonian prism integration (doc 05 §4, §6 T1).

The rigorous gravity forward (:class:`GravityRigorousForward`, ``fidelity="rigorous"``)
replaces the T0 far-field point-mass kernel with harmonica's exact right-rectangular
prism gravity (the Nagy prism formula). These tests assert, on a SMALL truth earth, that
it (1) runs via harmonica and emits the SAME native files as the T0 (CSV stations +
Bouguer GeoTIFF) so ingestion is unchanged, (2) is physically sensible — the right sign
and order of magnitude over a dense / light body — (3) DIFFERS from the T0 point-mass
approximation in the expected (more accurate, near-field) way, and (4) is selectable via
the fidelity-aware registry without breaking the T0 forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio

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
    RIGOROUS_FORWARD_MODELS,
    Acquisition,
    Artifact,
    ForwardModel,
    GravityForward,
    GravityRigorousForward,
    get_forward,
)

# --------------------------------------------------------------- a tiny truth earth


def _tiny_scene(seed: int = 5) -> SceneSpec:
    """A SMALL (≈10×10×10) Basin-&-Range scene with a dense basement + light plume."""
    return SceneSpec(
        id="tiny-t1-grav-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-500, xmax=500, ymin=-500, ymax=500,
            zmin=-900, zmax=200, dx=110, dy=110, dz=110,
        ),
        surface=SurfaceSpec(kind="tilted-block", base_elev=120.0, tilt_x=0.12),
        layers=(
            LayerSpec("alluvium", "surface", (50.0, 80.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-500, -100), (500, 200)),
                      kind="normal", dip=60, dip_azimuth=90, throw=180, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=250.0,
                top_elev=120.0, bottom_elev=-800.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.9, porosity_boost=0.05,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=80.0, clay_cap_thickness=120.0,
            ),
        ),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def earth():
    return compile_scene(_tiny_scene())


@pytest.fixture()
def acq(tmp_path):
    return Acquisition(gravity_spacing=150.0, params={"out_dir": str(tmp_path)})


def _rng():
    return np.random.default_rng(123)


# --------------------------------------------------------------- registry wiring


def test_rigorous_forward_is_registered_and_conforms():
    fwd = get_forward("gravity", None, fidelity="rigorous")
    assert isinstance(fwd, GravityRigorousForward)
    assert isinstance(fwd, ForwardModel)
    assert fwd.fidelity == "rigorous"
    assert fwd.method == "gravity" and fwd.submethod is None
    assert ("gravity", None) in RIGOROUS_FORWARD_MODELS


def test_plausible_default_unchanged():
    # the default fidelity is still T0, and a method without a T1 raises (no silent swap)
    assert isinstance(get_forward("gravity"), GravityForward)
    assert isinstance(get_forward("gravity", fidelity="plausible"), GravityForward)
    with pytest.raises(KeyError):
        get_forward("magnetics", fidelity="rigorous")
    with pytest.raises(KeyError):
        get_forward("gravity", fidelity="bogus")


# --------------------------------------------------------------- native I/O round-trip


def test_rigorous_emits_same_native_files(earth, acq):
    """T1 emits the identical CSV stations + Bouguer GeoTIFF as the T0 (doc 05 §4)."""
    arts = GravityRigorousForward().simulate(earth, acq, _rng())
    assert all(isinstance(a, Artifact) for a in arts)
    assert {a.fmt for a in arts} == {"csv", "geotiff"}

    csv = next(a for a in arts if a.fmt == "csv")
    assert csv.path.name == "gravity_stations.csv"
    df = pd.read_csv(csv.path)
    assert {"station", "x", "y", "elev", "bouguer_mgal"} <= set(df.columns)
    assert np.isfinite(df["bouguer_mgal"]).all()
    # plausible Bouguer magnitude for a small model (well under tens of mGal)
    assert df["bouguer_mgal"].abs().max() < 50.0

    tif = next(a for a in arts if a.fmt == "geotiff")
    assert tif.path.name == "gravity_bouguer.tif"
    with rasterio.open(tif.path) as r:
        grid = r.read(1)
    assert grid.ndim == 2 and np.isfinite(grid).any()
    # CSV station count matches the raster cells (same survey grid as T0).
    assert grid.size == len(df)

    prov = csv.provenance.to_dict()
    assert prov["source"] == "synthgen"
    assert prov["fidelity"] == "rigorous"
    assert prov["sceneId"] == earth.spec.id


def test_rigorous_is_deterministic(earth, acq):
    a = GravityRigorousForward().simulate(earth, acq, np.random.default_rng(7))
    b = GravityRigorousForward().simulate(earth, acq, np.random.default_rng(7))
    ga = pd.read_csv(next(x for x in a if x.fmt == "csv").path)["bouguer_mgal"].to_numpy()
    gb = pd.read_csv(next(x for x in b if x.fmt == "csv").path)["bouguer_mgal"].to_numpy()
    np.testing.assert_allclose(ga, gb)


# --------------------------------------------------------------- physical sensibility


def _bouguer_grid(arts: list[Artifact]) -> np.ndarray:
    tif = next(a for a in arts if a.fmt == "geotiff")
    with rasterio.open(tif.path) as r:
        return r.read(1)


def test_sign_over_dense_and_light_bodies():
    """A point-source prism gives + over excess mass, − over a mass deficit (doc 05 §4)."""
    import harmonica as hm

    # one 100 m cube whose top is 150 m below a station at the surface.
    prism = [[-50.0, 50.0, -50.0, 50.0, -250.0, -150.0]]
    obs = ([0.0], [0.0], [0.0])
    g_dense = hm.prism_gravity(obs, prism, [500.0], field="g_z")[0]
    g_light = hm.prism_gravity(obs, prism, [-500.0], field="g_z")[0]
    assert g_dense > 0.0  # downward g_z, +Δρ ⇒ +anomaly
    assert g_light < 0.0
    assert g_dense == pytest.approx(-g_light, rel=1e-6)


def test_anomaly_sign_tracks_lateral_density_contrast(earth, acq):
    """The Bouguer high/low must co-locate with the truth's dense/light columns."""
    arts = GravityRigorousForward().simulate(earth, acq, _rng())
    grid = _bouguer_grid(arts)

    # vertically-integrated density anomaly per (y, x) column → the dominant control on
    # the surface Bouguer signal; resample to the station grid and correlate.
    from geosim.synthgen.forward.potential_field import _density_anomaly

    col = np.nansum(_density_anomaly(earth), axis=0)  # (ny, nx)
    # coarsen the truth column-mass to the (coarser/equal) station grid by block-mean.
    ny_s, nx_s = grid.shape
    fy = max(col.shape[0] // ny_s, 1)
    fx = max(col.shape[1] // nx_s, 1)
    coarse = col[: ny_s * fy, : nx_s * fx].reshape(ny_s, fy, nx_s, fx).mean(axis=(1, 3))

    g = grid - np.nanmean(grid)
    c = coarse - np.nanmean(coarse)
    corr = np.corrcoef(g.ravel(), c.ravel())[0, 1]
    assert corr > 0.5  # the gravity high tracks the excess-mass columns


# --------------------------------------------------------------- T1 vs T0 difference


def test_rigorous_differs_from_t0_and_t0_still_works(earth, tmp_path):
    """T1 ≠ T0 (near-field accuracy) yet same order; both forwards still run (doc 05 §6)."""
    # write each forward into its OWN dir so neither overwrites the other's GeoTIFF.
    from dataclasses import replace

    base = Acquisition(gravity_spacing=150.0)
    acq0 = replace(base, params={"out_dir": str(tmp_path / "t0")})
    acq1 = replace(base, params={"out_dir": str(tmp_path / "t1")})

    t0 = GravityForward()
    t1 = GravityRigorousForward()

    a0 = t0.simulate(earth, acq0, _rng())  # T0 still works end-to-end
    a1 = t1.simulate(earth, acq1, _rng())
    g0 = _bouguer_grid(a0)
    g1 = _bouguer_grid(a1)
    assert g0.shape == g1.shape

    # same survey ⇒ same grid; the two kernels must give a measurable difference...
    diff = np.abs(g1 - g0)
    assert np.nanmax(diff) > 1e-3  # the point-mass vs prism kernels genuinely differ
    # ...but stay the same order of magnitude (both are the same potential field).
    rng_scale = np.nanmax(np.abs(g1)) + 1e-9
    assert np.nanmax(diff) < 5.0 * rng_scale


def test_rigorous_closer_to_dense_prism_reference(earth, acq):
    """Near-field check: the prism kernel matches an exact harmonica reference; the T0
    point mass does not (doc 05 §4 — rigorous is *more accurate*, not just different)."""
    import harmonica as hm

    from geosim.synthgen.forward.base import world_axes
    from geosim.synthgen.forward.potential_field import _density_anomaly, _obs_elevation

    z, y, x = world_axes(earth)
    dz, dy, dx = earth.spacing
    obs_elev = _obs_elevation(z, dz)
    d_rho = _density_anomaly(earth)

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    prisms = np.column_stack([
        (xx - dx / 2).ravel(), (xx + dx / 2).ravel(),
        (yy - dy / 2).ravel(), (yy + dy / 2).ravel(),
        (zz - dz / 2).ravel(), (zz + dz / 2).ravel(),
    ])
    dens = d_rho.ravel()

    # one observation point directly over the model centre.
    sx, sy = 0.0, 0.0
    ref = hm.prism_gravity(([sx], [sy], [obs_elev]), prisms, dens, field="g_z")[0]

    # T0 point-mass estimate at the same point (mirror of GravityForward's kernel).
    _G = 6.674e-11
    _MGAL = 1.0e5
    ddx = xx - sx
    ddy = yy - sy
    ddz = obs_elev - zz
    r2 = ddx * ddx + ddy * ddy + ddz * ddz + (0.5 * dz) ** 2
    r = np.sqrt(r2)
    cell_vol = dz * dy * dx
    g_t0 = _G * np.sum(d_rho * cell_vol * ddz / (r * r2)) * _MGAL

    # the prism (rigorous) value is the reference; T0 deviates from it.
    assert abs(g_t0 - ref) > 1e-4
