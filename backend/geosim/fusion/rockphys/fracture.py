"""Fracture-density-target rock-physics transforms (doc 07 §4.2).

- ``microseismic_density`` — a **KDE of a microseismic event cloud** → a smoothed fracture
  density volume (a permeability proxy: more events ⇒ more active fracturing, doc 07 §4.2
  "Fracture density"). The grid input is a per-cell event *count* (an event cloud binned
  onto the fused grid — :func:`events_to_count_volume` does that binning); ``apply`` does
  the kernel smoothing (Gaussian KDE) and normalises to a [0, 1] density index.
- ``vp_vs_fracture_proxy`` — Vp/Vs ratio anomaly → fracture index: open / fluid-filled
  fractures lower Vp/Vs, so a low-Vp/Vs membership flags fractured/permeable zones (doc 07
  §4.2 "Fracture density (struct.)").

Both output **fracture_density** (dimensionless 0..1). Uncalibrated ⇒ proxy.
"""

from __future__ import annotations

import numpy as np

from geosim.fusion.grid import FusedGrid
from geosim.fusion.transform import (
    InputSpec,
    OutputSpec,
    Param,
    Transform,
    TransformContext,
)
from geosim.plugins import register

__all__ = ["MicroseismicDensity", "VpVsFractureProxy", "events_to_count_volume"]


def events_to_count_volume(
    events_xyz: np.ndarray, grid: FusedGrid
) -> np.ndarray:
    """Bin a microseismic event cloud (N×3 x,y,z) onto a fused grid → per-cell counts.

    The pre-step for :class:`MicroseismicDensity`: each event is assigned to its nearest
    voxel (cell-centered grid, doc 07 §1). Returns a ``grid.shape`` (nz, ny, nx) float
    count volume — the ``microseismic`` property model the transform then smooths.
    """
    events = np.asarray(events_xyz, dtype=float).reshape(-1, 3)
    # grid.origin/spacing are (z, y, x)-ordered to match the volume axes (doc 07 §1).
    oz, oy, ox = grid.origin
    dz, dy, dx = grid.spacing
    nz, ny, nx = grid.shape
    vol = np.zeros((nz, ny, nx), dtype=float)
    for x, y, z in events:
        ix = int(round((x - ox) / dx))
        iy = int(round((y - oy) / dy))
        iz = int(round((z - oz) / dz))
        if 0 <= iz < nz and 0 <= iy < ny and 0 <= ix < nx:
            vol[iz, iy, ix] += 1.0
    return vol


class MicroseismicDensity(Transform):
    """Microseismic event-count volume → smoothed fracture density (KDE, doc 07 §4.2).

    Applies an isotropic Gaussian kernel (a grid KDE) to the per-cell event counts, then
    normalises by the peak smoothed density to a [0, 1] fracture-density index. Bandwidth
    (kernel σ in cells) is the tunable smoothing param.
    """

    id = "rp.microseismic_density"
    version = "1.0.0"
    title = "Microseismic Events → Fracture Density (KDE)"
    target = "fracture"

    assumptions = [
        "microseismic event rate ∝ active fracturing / permeability creation",
        "isotropic Gaussian smoothing kernel (no preferred fracture orientation)",
        "output is a relative density index (peak-normalised), not an absolute count",
    ]
    calibration_status = "uncalibrated"

    inputs = [InputSpec("microseismic", unit="dimensionless", required=True)]
    output = OutputSpec(
        "fracture_density", unit="dimensionless", valid_range=(0.0, 1.0),
        colormap="hot", proxy_when_uncalibrated=True,
    )

    params = [
        Param("bandwidth_cells", float, default=1.0, range=(0.3, 5.0)),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        microseismic,
        *,
        bandwidth_cells,
    ):
        """Gaussian-smooth the event-count volume (KDE) and peak-normalise to [0, 1].

        ``microseismic`` arrives flattened over the valid cells (the harness flattens); we
        scatter it back to the full grid, smooth in 3D, then return the per-valid-cell
        values so the harness can re-scatter + write.
        """
        from scipy.ndimage import gaussian_filter

        counts_flat = np.asarray(microseismic, dtype=float)
        vol = np.zeros(ctx.grid.shape, dtype=float).reshape(-1)
        # The harness passes only valid cells; reconstruct positions by finite-count order.
        # When called whole-grid (n == grid size) this is a 1:1 fill; otherwise the missing
        # cells are zero (no events) which is the correct KDE background.
        if counts_flat.size == vol.size:
            vol[:] = counts_flat
        else:
            vol[: counts_flat.size] = counts_flat
        smoothed = gaussian_filter(
            vol.reshape(ctx.grid.shape), sigma=float(bandwidth_cells), mode="nearest"
        )
        peak = float(np.nanmax(smoothed))
        density = (smoothed / peak if peak > 0 else smoothed).reshape(-1)
        if counts_flat.size != vol.size:
            density = density[: counts_flat.size]
        return ctx.as_output(density)


class VpVsFractureProxy(Transform):
    """Vp/Vs ratio → fracture index (low Vp/Vs ⇒ open/fluid-filled fractures, doc 07 §4.2).

    Open or fluid-filled fractures depress the Vp/Vs ratio; this returns a smooth
    low-Vp/Vs membership ``σ((ratio_threshold − Vp/Vs)/width)`` as a fracture index ∈ [0, 1].
    """

    id = "rp.vp_vs_fracture_proxy"
    version = "1.0.0"
    title = "Vp/Vs → Fracture Index"
    target = "fracture"

    assumptions = [
        "open / fluid-filled fractures lower the Vp/Vs ratio",
        "single lithology baseline Vp/Vs; anomaly is fracture-driven not compositional",
        "threshold/width are heuristic params, calibratable to image/borehole logs",
    ]
    calibration_status = "uncalibrated"

    inputs = [
        InputSpec("velocity_p", unit="m/s", required=True),
        InputSpec("velocity_s", unit="m/s", required=True),
    ]
    output = OutputSpec(
        "fracture_density", unit="dimensionless", valid_range=(0.0, 1.0),
        colormap="hot", proxy_when_uncalibrated=True,
    )

    params = [
        Param("ratio_threshold", float, default=1.7, range=(1.4, 2.2)),
        Param("width", float, default=0.1, range=(0.02, 0.5)),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        velocity_p,
        velocity_s,
        *,
        ratio_threshold,
        width,
    ):
        """Low-Vp/Vs membership as a fracture index."""
        vp = np.asarray(velocity_p, dtype=float)
        vs = np.maximum(np.asarray(velocity_s, dtype=float), 1.0)
        ratio = vp / vs
        idx = 1.0 / (1.0 + np.exp(-(ratio_threshold - ratio) / width))
        return ctx.as_output(idx)


register.transform(MicroseismicDensity())
register.transform(VpVsFractureProxy())
