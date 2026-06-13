"""Fused-grid path/point sampling WITH uncertainty (doc 09 §3.2, §5.1).

The planner needs *"give me these properties at these points along this polyline, WITH
σ"* (doc 09 §5.1). :func:`geosim.fusion.sample_path` samples the resampled **value**
layers but not their σ companions; this helper reuses the SAME trilinear sampler
(:class:`scipy.interpolate.RegularGridInterpolator` over the fused axes) and additionally
samples each layer's sibling ``_sigma`` array from the fused Zarr group, returning aligned
``(value, sigma)`` arrays per property. It never re-derives any volume — it reads the
layers fusion already wrote (doc 09 decision 6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sqlalchemy.orm import Session

from geosim.catalog import FusedModel
from geosim.fusion import fused_grid_from_row
from geosim.fusion.grid import open_fused_group

__all__ = ["sample_layers_with_sigma"]


def sample_layers_with_sigma(
    session: Session,
    fem: FusedModel,
    points_zyx: np.ndarray,
    *,
    properties: list[str] | None = None,
    storage_root: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Trilinearly sample fused layers (value + σ) at ``points_zyx`` (doc 09 §5.1).

    ``points_zyx`` is ``(m, 3)`` Engineering ``(z, y, x)`` metres — the curved trajectory
    vertices (or a single target point). Returns ``{property: (values[m], sigma[m]|None)}``;
    out-of-grid points read NaN (footprint honesty, doc 07 §2.3). Last layer per property
    wins (mirrors :func:`geosim.fusion.analysis._layers_by_property`).
    """
    grid = fused_grid_from_row(fem)
    z, y, x = grid.axis_coords()
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    group = open_fused_group(fem, storage_root=storage_root)

    by_prop = {lay.property: lay for lay in fem.layers}
    wanted = properties if properties is not None else list(by_prop.keys())

    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for prop in wanted:
        lay = by_prop.get(prop)
        if lay is None:
            continue
        values = np.asarray(group[lay.id][...], dtype=float)
        v_interp = RegularGridInterpolator(
            (z, y, x), values, method="linear", bounds_error=False, fill_value=np.nan
        )
        v = v_interp(pts)

        sig = None
        if lay.sigma_array and lay.sigma_array in group:
            sigma = np.asarray(group[lay.sigma_array][...], dtype=float)
            s_interp = RegularGridInterpolator(
                (z, y, x), sigma, method="linear", bounds_error=False, fill_value=np.nan
            )
            sig = s_interp(pts)
        out[prop] = (v, sig)
    return out
