"""Deterministic synthetic resistivity volume (doc 05 §2, scene ``unit-cube-v1``).

A single conductive blob embedded in a layered halfspace — the M1 geothermal-anomaly
stand-in (doc 05 §3.4 ``unit-cube-v1``: "single conductive cube in halfspace"). Values
are physically plausible per the doc-05 §2.2 property table: background resistivity
100–500 Ω·m, blob 5–20 Ω·m (hot + saline + altered → very conductive, doc 05 §2.2
resistivity row). A co-registered 1σ array is produced alongside (doc 02 §6) using the
property's default relative σ from the doc-01 registry.

Axis order is ``[z, y, x]`` Z-up to match the storage contract (doc 02 §10.2); ``origin``
and ``spacing`` are Engineering metres in ``(z, y, x)`` order (doc 01 §1, doc 02 §10.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geosim.spatial import REGISTRY

__all__ = ["VolumeResult", "build_resistivity_volume"]

# Physically plausible bounds (doc 05 §2.2 resistivity row).
_BG_TOP = 120.0  # Ω·m — shallow (clay-rich/weathered) cap, more conductive
_BG_BOTTOM = 450.0  # Ω·m — deeper, resistive basement halfspace
_BLOB_RESISTIVITY = 8.0  # Ω·m — the conductive geothermal anomaly (5–20 Ω·m band)


@dataclass(frozen=True)
class VolumeResult:
    """A synthetic resistivity volume + its co-registered 1σ (doc 02 §6).

    ``values``/``sigma`` are ``float32`` ``(nz, ny, nx)`` arrays in canonical Ω·m,
    Z-up. ``origin``/``spacing`` are Engineering metres in ``(z, y, x)`` order.
    """

    property: str
    canonical_unit: str
    values: np.ndarray  # (nz, ny, nx) Ω·m, Z-up
    sigma: np.ndarray  # (nz, ny, nx) 1σ in Ω·m
    origin: tuple[float, float, float]  # (z0, y0, x0) Engineering m
    spacing: tuple[float, float, float]  # (dz, dy, dx) Engineering m
    blob_center: tuple[float, float, float]  # (z, y, x) index of the blob centre
    blob_resistivity: float


def build_resistivity_volume(
    shape: tuple[int, int, int] = (32, 32, 32),
    spacing: tuple[float, float, float] = (25.0, 25.0, 25.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 42,
) -> VolumeResult:
    """Build a deterministic layered-halfspace resistivity volume with a conductive blob.

    ``shape`` is ``(nz, ny, nx)``; ``spacing``/``origin`` are ``(dz, dy, dx)`` /
    ``(z0, y0, x0)`` Engineering metres (Z-up). The background is a smooth depth-graded
    halfspace (conductive shallow cap → resistive basement); a smooth Gaussian-ish
    conductive blob is blended in near the centre. Seeded multiplicative noise adds
    texture (doc 05 §2.4) and is byte-identical for a given ``seed``.

    Returns a :class:`VolumeResult`; values/sigma are ``float32`` and finite everywhere
    (NaN is reserved for masked/outside-coverage cells, doc 02 §10.2 — none here).
    """
    nz, ny, nx = (int(s) for s in shape)
    if nz < 1 or ny < 1 or nx < 1:
        raise ValueError(f"shape must be positive (nz,ny,nx); got {shape}")

    pt = REGISTRY.get("resistivity")

    # ── layered halfspace: resistivity rises smoothly with depth ──────────────────
    # z index 0 is the deepest cell, index nz-1 the shallowest (Z-up). A geothermal
    # field is conductive shallow (clay cap/alteration) and resistive at depth, so
    # map the SHALLOW top to the conductive end and the DEEP bottom to resistive.
    if nz == 1:
        depth_frac = np.zeros(1, dtype=np.float64)
    else:
        # 0.0 at the shallowest layer, 1.0 at the deepest.
        depth_frac = 1.0 - np.arange(nz, dtype=np.float64) / (nz - 1)
    # Interpolate in log space (resistivity spans orders of magnitude, doc 01 §5).
    log_bg = np.log10(_BG_TOP) + depth_frac * (np.log10(_BG_BOTTOM) - np.log10(_BG_TOP))
    bg_profile = np.power(10.0, log_bg)  # (nz,)
    values = np.broadcast_to(bg_profile[:, None, None], (nz, ny, nx)).astype(np.float64).copy()

    # ── conductive blob: smooth low-resistivity anomaly near the centre ───────────
    cz, cy, cx = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    # radius ~ a quarter of the smallest dimension, at least 1 cell.
    radius = max(1.0, min(nz, ny, nx) / 4.0)
    zz, yy, xx = np.meshgrid(
        np.arange(nz, dtype=np.float64),
        np.arange(ny, dtype=np.float64),
        np.arange(nx, dtype=np.float64),
        indexing="ij",
    )
    r2 = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) / (radius**2)
    blob_weight = np.exp(-r2)  # 1.0 at centre, →0 outside (smooth, doc 05 §2.3 blend)
    # Blend toward the blob resistivity in log space so the core hits ~_BLOB_RESISTIVITY.
    log_vals = np.log10(values)
    log_blob = np.log10(_BLOB_RESISTIVITY)
    log_vals = (1.0 - blob_weight) * log_vals + blob_weight * log_blob
    values = np.power(10.0, log_vals)

    # ── seeded multiplicative texture (doc 05 §2.4 / §1 deterministic sub-stream) ──
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    texture = rng.normal(loc=0.0, scale=0.03, size=(nz, ny, nx))  # ±3% lognormal-ish
    values = values * np.exp(texture)
    values = np.clip(values, 1.0, 10000.0)  # registry display range guard (doc 01 §5)

    # ── co-registered 1σ (doc 02 §6): default relative σ from the registry ────────
    sigma = (values * float(pt.default_rel_sigma)).astype(np.float32)
    values = values.astype(np.float32)

    return VolumeResult(
        property="resistivity",
        canonical_unit=pt.canonical_unit,
        values=values,
        sigma=sigma,
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        blob_center=(cz, cy, cx),
        blob_resistivity=_BLOB_RESISTIVITY,
    )
