"""Temperature-target rock-physics transforms (doc 07 §4.1 worked example, §4.2).

``resistivity_to_temperature`` (Arps) inverts bulk electrical resistivity for an
**absolute temperature** field (canonical **kelvin**, doc 01 §5) via two textbook steps:

1. **Archie** (clay-free, ``a=1``): bulk conductivity → pore-fluid conductivity
   ``σ_w = σ_bulk / φ^m`` (doc 07 §4.1 step 1).
2. **Arps fluid-conductivity-vs-temperature** at fixed salinity: brine conductivity rises
   ~2 %/°C, so ``σ_w(T) ≈ σ_w(T_ref)·(1 + α·(T − T_ref))`` inverts for ``T`` (doc 07 §4.1
   step 2 — kelvin out).

This is the canonical **uncalibrated** transform of doc 07 §4.1: its output is a
*temperature likelihood* (the harness retitles it + stamps ``tier='proxy'``) until a
well/core/geochem run promotes it (§4.8). Every param — porosity, cementation exponent,
salinity, Arps slope — is first-class and user-tunable.
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

__all__ = ["ResistivityToTemperature", "brine_conductivity"]


def brine_conductivity(salinity_ppm: float, temperature_K: float) -> float:
    """Approximate NaCl-brine electrical conductivity (S/m) at salinity + temperature.

    Textbook engineering approximation (doc 07 §4.1): conductivity scales ~linearly with
    NaCl concentration and rises ~2 %/°C with temperature (the Arps slope). Used only to
    anchor the reference fluid conductivity ``σ_w(T_ref)`` in the inversion below — the
    inversion's temperature sensitivity comes from the Arps slope, not this anchor.

    ``σ_w(T_ref) ≈ k · C_ppm`` with ``k ≈ 1.5e-4 S·m⁻¹/ppm`` at 25 °C (≈0.75 S/m at the
    default 5000 ppm), a standard NaCl-solution figure.
    """
    k_per_ppm = 1.5e-4  # S/m per ppm at the 25 °C reference (textbook NaCl figure)
    return float(k_per_ppm * salinity_ppm)


class ResistivityToTemperature(Transform):
    """Resistivity → temperature *likelihood* via Archie + Arps (doc 07 §4.1 worked example).

    Output is **kelvin** (canonical, doc 01 §5). ``uncalibrated`` ⇒ the harness retitles the
    layer "temperature likelihood" and stamps ``tier='proxy'`` (doc 07 §4.5 step 7) until a
    calibration run (§4.8) promotes it.
    """

    id = "rp.resistivity_to_temperature.arps"
    version = "1.0.0"
    title = "Resistivity → Temperature (Arps fluid-conductivity)"
    target = "temperature"

    assumptions = [
        "single liquid brine phase (no boiling / steam)",
        "porosity & salinity treated as constant params unless calibrated per-cell",
        "Archie a=1; bulk conduction only (use waxman_smits in clay/altered rock)",
        "fluid conductivity rises ~2 %/°C (Arps slope) at fixed salinity",
    ]
    calibration_status = "uncalibrated"

    inputs = [InputSpec("resistivity", unit="ohm*m", required=True)]
    output = OutputSpec(
        "temperature", unit="kelvin", valid_range=(273.0, 673.0),  # ≈0–400 °C
        colormap="thermal", proxy_when_uncalibrated=True,
    )

    params = [
        Param("porosity", float, default=0.10, range=(0.01, 0.5), sigma=0.03),
        Param("m_cementation", float, default=2.0, range=(1.3, 2.5)),
        Param("fluid_salinity_ppm", float, default=5000.0, range=(100.0, 250000.0)),
        Param("arps_slope_per_K", float, default=0.02, range=(0.005, 0.05)),
        Param("T_ref_K", float, default=298.15),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        resistivity,
        *,
        porosity,
        m_cementation,
        fluid_salinity_ppm,
        arps_slope_per_K,
        T_ref_K,
    ):
        """Archie (σ_w = σ_bulk/φ^m) then invert the Arps σ_w(T) line for T (kelvin)."""
        rho = np.maximum(np.asarray(resistivity, dtype=float), 1e-6)
        sigma_bulk = 1.0 / rho
        sigma_w = sigma_bulk / (porosity**m_cementation)  # a = 1
        sigma_w_ref = brine_conductivity(fluid_salinity_ppm, T_ref_K)
        # σ_w(T) = σ_w_ref · (1 + α·(T − T_ref))  ⇒  T = T_ref + (σ_w/σ_w_ref − 1)/α
        temperature_K = T_ref_K + (sigma_w / sigma_w_ref - 1.0) / arps_slope_per_K
        return ctx.as_output(temperature_K)


register.transform(ResistivityToTemperature())
