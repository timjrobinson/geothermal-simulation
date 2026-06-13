"""Porosity-target rock-physics transforms (doc 07 §4.2).

- ``velocity_to_porosity`` — P-wave velocity → porosity by the **Wyllie time-average** or
  **Raymer-Hunt-Gardner (RHG)** relation (doc 07 §4.2 "Porosity").
- ``density_to_porosity`` — bulk density → porosity by simple matrix/fluid mass balance
  ``φ = (ρ_matrix − ρ_b)/(ρ_matrix − ρ_fluid)`` (doc 07 §4.2 "Porosity (alt)").

Both output **porosity** (dimensionless fraction). Uncalibrated ⇒ proxy until a
core/well-log run (§4.8) promotes them. Matrix + fluid velocities/densities are the
first-class, user-tunable calibration anchors.
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

__all__ = ["VelocityToPorosity", "DensityToPorosity"]


class VelocityToPorosity(Transform):
    """P-wave velocity → porosity (Wyllie time-average / Raymer-Hunt-Gardner, doc 07 §4.2).

    ``model='wyllie'`` (default): the time-average equation ``1/Vp = (1−φ)/V_matrix +
    φ/V_fluid`` ⇒ ``φ = (1/Vp − 1/V_matrix)/(1/V_fluid − 1/V_matrix)``. ``model='rhg'``: the
    Raymer-Hunt-Gardner relation ``Vp = (1−φ)²·V_matrix + φ·V_fluid`` (more accurate at low
    porosity), solved for φ via the quadratic.
    """

    id = "rp.velocity_to_porosity"
    version = "1.0.0"
    title = "P-velocity → Porosity (Wyllie / Raymer-Hunt-Gardner)"
    target = "porosity"

    assumptions = [
        "fully brine-saturated, consolidated rock (Wyllie time-average regime)",
        "single matrix mineralogy with constant V_matrix",
        "no gas / fracture / scattering effects on Vp",
    ]
    calibration_status = "uncalibrated"

    inputs = [InputSpec("velocity_p", unit="m/s", required=True)]
    output = OutputSpec(
        "porosity", unit="dimensionless", valid_range=(0.0, 0.5),
        colormap="viridis", proxy_when_uncalibrated=True,
    )

    params = [
        Param("v_matrix_m_s", float, default=5500.0, range=(3000.0, 7000.0), sigma=200.0),
        Param("v_fluid_m_s", float, default=1500.0, range=(1400.0, 1600.0)),
        Param("model", str, default="wyllie"),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        velocity_p,
        *,
        v_matrix_m_s,
        v_fluid_m_s,
        model,
    ):
        """Invert Vp for φ via Wyllie time-average or the RHG quadratic."""
        vp = np.maximum(np.asarray(velocity_p, dtype=float), 1.0)
        if str(model).lower() == "rhg":
            # Vp = (1−φ)²·Vm + φ·Vf  ⇒  Vm·φ² − (2Vm − Vf)·φ + (Vm − Vp) = 0.
            vm, vf = v_matrix_m_s, v_fluid_m_s
            a = vm
            b = -(2.0 * vm - vf)
            c = vm - vp
            disc = np.maximum(b * b - 4.0 * a * c, 0.0)
            phi = (-b - np.sqrt(disc)) / (2.0 * a)  # physical root → φ↑ as Vp↓
        else:  # wyllie time-average
            phi = (1.0 / vp - 1.0 / v_matrix_m_s) / (1.0 / v_fluid_m_s - 1.0 / v_matrix_m_s)
        return ctx.as_output(phi)


class DensityToPorosity(Transform):
    """Bulk density → porosity by matrix/fluid mass balance (doc 07 §4.2 "Porosity (alt)").

    ``φ = (ρ_matrix − ρ_bulk) / (ρ_matrix − ρ_fluid)`` — the standard density-porosity log
    transform. ρ_matrix (e.g. 2650 kg/m³ quartz) and ρ_fluid (e.g. 1000 kg/m³ brine) are the
    calibration anchors.
    """

    id = "rp.density_to_porosity"
    version = "1.0.0"
    title = "Bulk Density → Porosity (mass balance)"
    target = "porosity"

    assumptions = [
        "two-component rock: single-mineral matrix + single pore fluid",
        "fully saturated; no gas effect on bulk density",
        "ρ_matrix, ρ_fluid constant params unless calibrated per-cell",
    ]
    calibration_status = "uncalibrated"

    inputs = [InputSpec("density", unit="kg/m**3", required=True)]
    output = OutputSpec(
        "porosity", unit="dimensionless", valid_range=(0.0, 0.5),
        colormap="viridis", proxy_when_uncalibrated=True,
    )

    params = [
        Param("rho_matrix_kg_m3", float, default=2650.0, range=(2300.0, 3100.0), sigma=50.0),
        Param("rho_fluid_kg_m3", float, default=1000.0, range=(800.0, 1200.0)),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        density,
        *,
        rho_matrix_kg_m3,
        rho_fluid_kg_m3,
    ):
        """φ = (ρ_matrix − ρ_b)/(ρ_matrix − ρ_fluid)."""
        rho_b = np.asarray(density, dtype=float)
        phi = (rho_matrix_kg_m3 - rho_b) / (rho_matrix_kg_m3 - rho_fluid_kg_m3)
        return ctx.as_output(phi)


register.transform(VelocityToPorosity())
register.transform(DensityToPorosity())
