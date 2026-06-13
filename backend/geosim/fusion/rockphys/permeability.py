"""Permeability-target rock-physics transforms (doc 07 §4.2).

``fracture_to_permeability`` — a **heuristic, explicitly low-confidence** relative-perm
proxy mapping a fracture-density index (optionally damped by alteration, since clay gouge
*seals* fractures) to an intrinsic permeability in SI **m²** on a log scale (doc 07 §4.2
"Permeability"). This is a proxy, not a flow simulation: it is flagged low-confidence and
its output is a likelihood/proxy field (``calibration_status='uncalibrated'``).
"""

from __future__ import annotations

import numpy as np

from geosim.fusion.transform import (
    InputSpec,
    OutputSpec,
    Param,
    Transform,
    TransformContext,
)
from geosim.plugins import register

__all__ = ["FractureToPermeability"]


class FractureToPermeability(Transform):
    """Fracture density (+ alteration) → permeability proxy (heuristic, doc 07 §4.2).

    Log-linear interpolation between a tight-matrix floor ``k_min`` (zero fracturing) and a
    well-fractured ceiling ``k_max`` (full fracturing):

        ``log10 k = log10 k_min + fracture_density · (log10 k_max − log10 k_min)``

    then damped by ``(1 − alteration_seal · alteration)`` because clay-rich alteration gouge
    seals fractures and reduces permeability. Output is intrinsic permeability in **m²**
    (1 mD ≈ 9.869e-16 m²). Explicitly low-confidence / proxy.
    """

    id = "rp.fracture_to_permeability"
    version = "1.0.0"
    title = "Fracture Density (+Alteration) → Permeability (proxy)"
    target = "permeability"

    assumptions = [
        "HEURISTIC relative-perm index, NOT a flow/percolation simulation (low confidence)",
        "permeability is log-linear in the fracture-density index between k_min and k_max",
        "clay-rich alteration gouge seals fractures ⇒ damps permeability",
        "k_min/k_max/seal are first-class calibration anchors (core/well tests)",
    ]
    calibration_status = "uncalibrated"

    inputs = [
        InputSpec("fracture_density", unit="dimensionless", required=True),
        InputSpec("alteration", unit="dimensionless", required=False),
    ]
    output = OutputSpec(
        "permeability", unit="m**2", valid_range=(1e-18, 1e-11),
        colormap="viridis", proxy_when_uncalibrated=True,
    )

    params = [
        # tight matrix floor ~0.01 mD; well-fractured ceiling ~1 darcy.
        Param("k_min_m2", float, default=1e-17, range=(1e-20, 1e-15)),
        Param("k_max_m2", float, default=1e-12, range=(1e-15, 1e-11)),
        Param("alteration_seal", float, default=0.5, range=(0.0, 1.0)),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        fracture_density,
        alteration=None,
        *,
        k_min_m2,
        k_max_m2,
        alteration_seal,
    ):
        """Log-linear fracture→k, damped by an optional alteration-sealing term."""
        fd = np.clip(np.asarray(fracture_density, dtype=float), 0.0, 1.0)
        log_kmin = np.log10(k_min_m2)
        log_kmax = np.log10(k_max_m2)
        log_k = log_kmin + fd * (log_kmax - log_kmin)
        k = 10.0**log_k
        if alteration is not None:
            seal = np.clip(np.asarray(alteration, dtype=float), 0.0, 1.0)
            k = k * (1.0 - alteration_seal * seal)
        return ctx.as_output(k)


register.transform(FractureToPermeability())
