"""Gravity + magnetics T0 forwards — analytic potential-field sums (doc 05 §4 table).

Both are **degrade-the-truth** potential-field forwards (doc 05 §6 T0): an analytic
voxel sum of the source property anomaly onto an observation surface, then the three
universal degradations (acquisition geometry, depth/altitude low-pass = the field's own
DOI/smoothing, noise — doc 05 §4).

- **Gravity** (:class:`GravityForward`) — analytic prism/voxel sum of the *density
  anomaly* ``Δρ = ρ - ρ_background`` (the §4 "Nagy formula" reduced to the far-field
  point-mass kernel ``g = G·Δm·Δz / r³`` summed over voxels) sampled on a station grid →
  a Bouguer anomaly grid (mGal). Emits CSV stations + a GeoTIFF grid (``rasterio``).
- **Magnetics** (:class:`MagneticsForward`) — voxel sum of *susceptibility* induced
  magnetisation (vertical-field Poisson kernel) flown along aeromag lines at altitude,
  upward-continuation low-pass ∝ altitude, reduced-to-pole grid (nT). Emits a ``.xyz``
  line file + a GeoTIFF RTP grid. The "only-sees-what-it-could" hook (doc 05 §4.2): the
  altered plume has ``χ→~0``, so the magnetics sees a *low* over the upflow, not heat.

GeoTIFFs are written in the Engineering local frame: an identity-ish affine mapping pixel
→ Engineering metres (``rasterio`` ``Affine``), CRS left undefined (local frame, doc 01
§1) so the adapter reads pixel-space ground coordinates directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from ..truth import TruthEarth
from .base import (
    Acquisition,
    Artifact,
    T0Forward,
    add_gaussian_noise,
    gaussian_lowpass,
    world_axes,
)

__all__ = ["GravityForward", "MagneticsForward", "write_local_geotiff"]

_G = 6.674e-11  # gravitational constant (m³ kg⁻¹ s⁻²)
_MGAL = 1.0e5   # m/s² → mGal


def write_local_geotiff(path: Path, grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Write a single-band float32 GeoTIFF in the Engineering local frame (doc 01 §1).

    ``grid`` is ``(ny, nx)`` with ``grid[0]`` the *northmost* row (rasterio's top row);
    the affine maps pixel centres to Engineering metres. No CRS (local frame).
    """
    ny, nx = grid.shape
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    dy = float(y[1] - y[0]) if y.size > 1 else 1.0
    # rasterio row 0 = north; our y is ascending, so flip so top row = max y.
    transform = from_origin(float(x[0] - dx / 2), float(y[-1] + dy / 2), dx, dy)
    data = np.flipud(grid).astype(np.float32)
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="float32", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


def _station_grid(
    x: np.ndarray, y: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Regular station grid (1-D xs, ys) covering the ROI at ``spacing`` (doc 05 §4.3)."""
    xs = np.arange(x[0], x[-1] + 1e-6, spacing)
    ys = np.arange(y[0], y[-1] + 1e-6, spacing)
    if xs.size < 2:
        xs = np.array([x[0], x[-1]])
    if ys.size < 2:
        ys = np.array([y[0], y[-1]])
    return xs, ys


class GravityForward(T0Forward):
    """T0 gravity: analytic density-anomaly voxel sum → Bouguer grid (doc 05 §4 row 1)."""

    method = "gravity"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        dz, dy, dx = truth.spacing
        cell_vol = dz * dy * dx

        rho = truth.property_volume("density").astype(np.float64)
        # density anomaly relative to a depth-wise background (median per z-level)
        bg = np.nanmedian(rho, axis=(1, 2), keepdims=True)
        d_rho = rho - bg
        dm = d_rho * cell_vol  # excess mass per voxel (kg)

        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        obs_elev = float(np.max(z)) + max(dz, 1.0)  # just above the top of the model

        xs, ys = _station_grid(x, y, acq.gravity_spacing)
        gz = np.zeros((ys.size, xs.size), dtype=np.float64)
        for j, sy in enumerate(ys):
            for i, sx in enumerate(xs):
                ddx = xx - sx
                ddy = yy - sy
                ddz = obs_elev - zz  # +down from station to voxel
                r2 = ddx * ddx + ddy * ddy + ddz * ddz + (0.5 * dz) ** 2
                r = np.sqrt(r2)
                # vertical component of point-mass attraction (downward +)
                gz[j, i] = _G * np.sum(dm * ddz / (r * r2))
        gz_mgal = gz * _MGAL

        # resolution: gravity is inherently smooth — low-pass the station grid (deg. 2)
        gz_mgal = gaussian_lowpass(gz_mgal, sigma_cells=0.8)
        # noise: 0.02-0.05 mGal Gaussian (doc 05 §4)
        noisy = add_gaussian_noise(gz_mgal, 0.03, rng)

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)

        # CSV stations
        sxx, syy = np.meshgrid(xs, ys, indexing="xy")
        df = pd.DataFrame({
            "station": np.arange(noisy.size),
            "x": sxx.ravel(), "y": syy.ravel(), "elev": obs_elev,
            "bouguer_mgal": noisy.ravel(),
        })
        csv_path = out_dir / "gravity_stations.csv"
        df.to_csv(csv_path, index=False)

        tif_path = out_dir / "gravity_bouguer.tif"
        write_local_geotiff(tif_path, noisy, xs, ys)

        prov = self._prov(truth, units="mGal", obsElev=obs_elev)
        return [
            Artifact(csv_path, "csv", self.method, self.submethod, prov),
            Artifact(tif_path, "geotiff", self.method, self.submethod, prov),
        ]


class MagneticsForward(T0Forward):
    """T0 magnetics: susceptibility voxel sum + upward-cont low-pass → RTP (doc 05 §4)."""

    method = "magnetics"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        dz, dy, dx = truth.spacing
        cell_vol = dz * dy * dx

        chi = truth.property_volume("susceptibility").astype(np.float64)
        # Induced magnetisation in the ambient field B0 (vertical inducing field → RTP).
        # Magnetisation M = χ·H = χ·B0/μ0 (A/m); dipole moment per voxel m = M·V.
        # The vertical-field anomaly of a dipole is
        #   ΔBz = (μ0/4π)·m·(3·Δz² − r²)/r⁵   [Tesla];   ×1e9 → nT.
        # The μ0 cancels: ΔBz[nT] = (1/4π)·χ·B0[T]·V·(3Δz²−r²)/r⁵ · 1e9.
        b0_tesla = 50000.0e-9  # 50,000 nT ambient field in Tesla
        m = chi * b0_tesla * cell_vol  # χ·B0·V (Tesla·m³)

        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        flight_elev = float(np.max(z)) + acq.mag_altitude

        xs, ys = _station_grid(x, y, acq.mag_line_spacing)
        # dense along-line samples (aeromag lines), coarse across (line spacing)
        line_x = np.arange(x[0], x[-1] + 1e-6, max(dx, acq.mag_line_spacing / 8))
        grid = np.zeros((ys.size, xs.size), dtype=np.float64)
        for j, sy in enumerate(ys):
            for i, sx in enumerate(xs):
                ddx = xx - sx
                ddy = yy - sy
                ddz = flight_elev - zz  # +up sensor above source
                r2 = ddx * ddx + ddy * ddy + ddz * ddz + (0.5 * dz) ** 2
                r = np.sqrt(r2)
                # vertical-field (RTP) dipole anomaly summed over voxels → nT.
                grid[j, i] = (1.0 / (4 * np.pi)) * np.sum(
                    m * (3 * ddz * ddz - r2) / (r2 * r2 * r)
                ) * 1.0e9

        # upward-continuation low-pass ∝ altitude (deg. 2 resolution)
        sigma_cells = max(acq.mag_altitude / max(dy, 1.0) / 2.0, 0.6)
        rtp = gaussian_lowpass(grid, sigma_cells=sigma_cells)
        # noise: 1-3 nT + line leveling drift (doc 05 §4)
        noisy = add_gaussian_noise(rtp, 2.0, rng)
        drift = rng.normal(0.0, 1.0, size=ys.size)[:, None]  # per-line leveling error
        noisy = noisy + drift

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)

        # .xyz flight-line file (LINE x y alt tmi) — sample the grid along each line.
        rows = []
        for j, sy in enumerate(ys):
            for lx in line_x:
                # nearest grid column
                i = int(np.clip(np.searchsorted(xs, lx), 0, xs.size - 1))
                rows.append((j, float(lx), float(sy), flight_elev, float(noisy[j, i])))
        xyz_path = out_dir / "aeromag_lines.xyz"
        with open(xyz_path, "w", encoding="utf-8") as fh:
            fh.write("LINE X Y ALT TMI_RTP_nT\n")
            for ln, lx, ly, alt, v in rows:
                fh.write(f"{ln} {lx:.2f} {ly:.2f} {alt:.2f} {v:.4f}\n")

        tif_path = out_dir / "mag_rtp.tif"
        write_local_geotiff(tif_path, noisy, xs, ys)

        prov = self._prov(truth, units="nT", flightElev=flight_elev, product="RTP")
        return [
            Artifact(xyz_path, "xyz", self.method, self.submethod, prov),
            Artifact(tif_path, "geotiff", self.method, self.submethod, prov),
        ]
