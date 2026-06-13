"""Vertical handling: elevation/depth/MD/TVD and the minimum-curvature integrator.

Doc 01 §4: internal canonical vertical = **orthometric elevation, metres, Z-up**.
Everything else (depth-from-surface, TVDSS, MD, TVD) is a *derived view* computed on
demand, never the source of truth.

The ``min_curvature_positions`` routine is the shared survey→position integrator that
both ingested wells (doc 02 §5) and the planner (doc 09 §4.3) use — it lives here, not
in the planner (doc 09 flag to doc 01/02 owners).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "elevation_to_depth",
    "depth_to_elevation",
    "tvd_to_elevation",
    "elevation_to_tvdss",
    "MinCurvatureResult",
    "min_curvature_positions",
]


def elevation_to_depth(z, surface_elev):
    """Depth below local ground surface: ``depth = surface_elev - z`` (doc 01 §4)."""
    return np.asarray(surface_elev, dtype=float) - np.asarray(z, dtype=float)


def depth_to_elevation(depth, surface_elev):
    """Elevation from depth-below-surface: ``z = surface_elev - depth`` (doc 01 §4)."""
    return np.asarray(surface_elev, dtype=float) - np.asarray(depth, dtype=float)


def elevation_to_tvdss(z):
    """TVDSS (depth below MSL/datum): ``depth = -z`` (doc 01 §4)."""
    return -np.asarray(z, dtype=float)


def tvd_to_elevation(tvd, ref_elev):
    """Elevation from true vertical depth below a well reference: ``z = ref_elev - tvd``."""
    return np.asarray(ref_elev, dtype=float) - np.asarray(tvd, dtype=float)


@dataclass
class MinCurvatureResult:
    md: np.ndarray  # measured depth per station (m)
    enu: np.ndarray  # (N,3) Engineering XYZ (East, North, Up) per station, m
    tvd: np.ndarray  # true vertical depth below MD datum (KB), m, +down
    dls: np.ndarray  # dogleg severity per interval, °/30 m (len N; dls[0]=0)


def min_curvature_positions(deviation_survey, wellhead, kb_elev: float | None = None) -> MinCurvatureResult:
    """Minimum-curvature integration of a deviation survey (doc 09 §4.3, industry standard).

    Parameters
    ----------
    deviation_survey : array-like (N,3)
        Ordered stations ``(MD, inclination_deg, azimuth_deg)``.
    wellhead : (x, y) or (x, y, elev)
        Engineering XY of the slot; elevation taken from ``kb_elev`` if given, else from
        a 3rd component, else 0.0. MD datum (MD=0) sits at this elevation.
    kb_elev : float, optional
        Kelly-bushing / MD-datum elevation (Engineering m). Overrides wellhead[2].

    Returns
    -------
    MinCurvatureResult with per-station Engineering XYZ, TVD below KB, and per-interval DLS.

    Math (per interval between stations 1 and 2):
        cos β = cos(I2−I1) − sinI1·sinI2·(1 − cos(A2−A1))
        RF    = (2/β)·tan(β/2),  RF→1 as β→0
        ΔN = (ΔMD/2)·(sinI1·cosA1 + sinI2·cosA2)·RF      # +North = +Y
        ΔE = (ΔMD/2)·(sinI1·sinA1 + sinI2·sinA2)·RF      # +East  = +X
        ΔV = (ΔMD/2)·(cosI1 + cosI2)·RF                  # +Down (TVD increment)
        DLS = β·(30/ΔMD)                                 # degrees per 30 m
    """
    surv = np.asarray(deviation_survey, dtype=float).reshape(-1, 3)
    n = surv.shape[0]
    wh = np.asarray(wellhead, dtype=float).reshape(-1)
    x0, y0 = float(wh[0]), float(wh[1])
    if kb_elev is not None:
        z0 = float(kb_elev)
    elif wh.size >= 3:
        z0 = float(wh[2])
    else:
        z0 = 0.0

    md = surv[:, 0]
    inc = np.radians(surv[:, 1])
    azi = np.radians(surv[:, 2])

    enu = np.zeros((n, 3), dtype=float)
    tvd = np.zeros(n, dtype=float)
    dls = np.zeros(n, dtype=float)
    enu[0] = [x0, y0, z0]
    tvd[0] = 0.0

    for i in range(1, n):
        d_md = md[i] - md[i - 1]
        i1, i2 = inc[i - 1], inc[i]
        a1, a2 = azi[i - 1], azi[i]
        cos_beta = math.cos(i2 - i1) - math.sin(i1) * math.sin(i2) * (1.0 - math.cos(a2 - a1))
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)
        rf = 1.0 if beta < 1e-7 else (2.0 / beta) * math.tan(beta / 2.0)

        d_n = (d_md / 2.0) * (math.sin(i1) * math.cos(a1) + math.sin(i2) * math.cos(a2)) * rf
        d_e = (d_md / 2.0) * (math.sin(i1) * math.sin(a1) + math.sin(i2) * math.sin(a2)) * rf
        d_v = (d_md / 2.0) * (math.cos(i1) + math.cos(i2)) * rf

        enu[i, 0] = enu[i - 1, 0] + d_e           # East  (+X)
        enu[i, 1] = enu[i - 1, 1] + d_n           # North (+Y)
        enu[i, 2] = enu[i - 1, 2] - d_v           # Up (Engineering Z); ΔV is downward
        tvd[i] = tvd[i - 1] + d_v
        dls[i] = math.degrees(beta) * (30.0 / d_md) if d_md > 0 else 0.0

    return MinCurvatureResult(md=md.copy(), enu=enu, tvd=tvd, dls=dls)
