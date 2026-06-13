"""Fluid / saturation-target rock-physics transforms (doc 07 §4.2).

- ``archie_saturation`` — Archie's law water saturation
  ``Sw = ((a·ρ_w)/(φ^m·ρ_t))^{1/n}`` (doc 07 §4.2 "Fluid / saturation").
- ``dual_water`` / ``waxman_smits`` — clay-surface-conduction-corrected Sw for clay/altered
  rock, where bulk Archie *over-reads* water content because clay conducts independently of
  the pore brine (doc 07 §4.2 "Fluid / clay-conduction").

All output **water_saturation** (dimensionless, 0..1). Uncalibrated ⇒ proxy/likelihood
until a well-log run (§4.8) promotes them. Archie's law is nonlinear in φ, so prefer the
harness Monte-Carlo σ mode for tight φ uncertainty (doc 07 §5.2).
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

__all__ = ["ArchieSaturation", "WaxmanSmitsSaturation", "DualWaterSaturation"]


class ArchieSaturation(Transform):
    """Archie's-law water saturation from resistivity + porosity (doc 07 §4.2).

    ``Sw = ((a · ρ_w) / (φ^m · ρ_t))^{1/n}`` where ρ_t is the (true) formation resistivity,
    ρ_w the brine resistivity, ``a`` the tortuosity factor, ``m`` cementation, ``n``
    saturation exponent. Clay-free assumption — use :class:`WaxmanSmitsSaturation` in shaly
    rock.
    """

    id = "rp.archie_saturation"
    version = "1.0.0"
    title = "Resistivity + Porosity → Water Saturation (Archie)"
    target = "fluid"

    assumptions = [
        "clean (clay-free) formation — Archie over-reads Sw in shaly rock",
        "single brine phase; rock matrix is electrically non-conductive",
        "a, m, n are constant params unless calibrated per-cell",
    ]
    calibration_status = "uncalibrated"

    inputs = [
        InputSpec("resistivity", unit="ohm*m", required=True),
        InputSpec("porosity", unit="dimensionless", required=True),
    ]
    output = OutputSpec(
        "water_saturation", unit="dimensionless", valid_range=(0.0, 1.0),
        colormap="Blues", proxy_when_uncalibrated=True,
    )

    params = [
        Param("a_tortuosity", float, default=1.0, range=(0.5, 2.5)),
        Param("m_cementation", float, default=2.0, range=(1.3, 2.5)),
        Param("n_saturation", float, default=2.0, range=(1.5, 2.5)),
        Param("rho_w_ohm_m", float, default=0.5, range=(0.01, 100.0), sigma=0.1),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        resistivity,
        porosity,
        *,
        a_tortuosity,
        m_cementation,
        n_saturation,
        rho_w_ohm_m,
    ):
        """Solve Archie's law for Sw (clamped to a physical [0, 1] by the harness)."""
        rho_t = np.maximum(np.asarray(resistivity, dtype=float), 1e-6)
        phi = np.clip(np.asarray(porosity, dtype=float), 1e-4, 1.0)
        numer = a_tortuosity * rho_w_ohm_m
        denom = (phi**m_cementation) * rho_t
        sw = (numer / denom) ** (1.0 / n_saturation)
        return ctx.as_output(sw)


class WaxmanSmitsSaturation(Transform):
    """Waxman-Smits clay-corrected water saturation (doc 07 §4.2 "Fluid / clay-conduction").

    Adds a clay-surface-conduction term (``B·Qv``) in parallel with the brine path so Archie
    no longer over-reads Sw in shaly rock:

        ``1/ρ_t = (φ^m · Sw^n)/(a·ρ_w) + B·Qv·Sw^{n-1}``

    with the cation-exchange capacity per unit pore volume ``Qv`` taken proportional to clay
    volume (``Qv ≈ Qv_max · Vclay``). Solved for Sw by a few fixed-point iterations (the
    clay term is small relative to the brine path in moderately clay-bearing rock).
    """

    id = "rp.waxman_smits"
    version = "1.0.0"
    title = "Resistivity + Porosity + Clay → Water Saturation (Waxman-Smits)"
    target = "fluid"

    assumptions = [
        "clay conducts in parallel with brine (cation-exchange surface conduction)",
        "Qv (cation-exchange capacity / pore volume) ∝ clay volume",
        "B (equiv. counter-ion conductance) and Qv_max are constant params unless calibrated",
    ]
    calibration_status = "uncalibrated"

    inputs = [
        InputSpec("resistivity", unit="ohm*m", required=True),
        InputSpec("porosity", unit="dimensionless", required=True),
        InputSpec("clay_volume", unit="dimensionless", required=True),
    ]
    output = OutputSpec(
        "water_saturation", unit="dimensionless", valid_range=(0.0, 1.0),
        colormap="Blues", proxy_when_uncalibrated=True,
    )

    params = [
        Param("a_tortuosity", float, default=1.0, range=(0.5, 2.5)),
        Param("m_cementation", float, default=2.0, range=(1.3, 2.5)),
        Param("n_saturation", float, default=2.0, range=(1.5, 2.5)),
        Param("rho_w_ohm_m", float, default=0.5, range=(0.01, 100.0), sigma=0.1),
        # B = equivalent counter-ion conductance (S/m per (meq/mL)); ~4.6 at 25 °C.
        Param("B_counterion", float, default=4.6, range=(1.0, 12.0)),
        # Qv at 100 % clay (meq/mL of pore space) — calibration anchor.
        Param("Qv_max", float, default=1.0, range=(0.0, 5.0)),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        resistivity,
        porosity,
        clay_volume,
        *,
        a_tortuosity,
        m_cementation,
        n_saturation,
        rho_w_ohm_m,
        B_counterion,
        Qv_max,
    ):
        """Fixed-point solve of the Waxman-Smits conductivity equation for Sw."""
        rho_t = np.maximum(np.asarray(resistivity, dtype=float), 1e-6)
        phi = np.clip(np.asarray(porosity, dtype=float), 1e-4, 1.0)
        vclay = np.clip(np.asarray(clay_volume, dtype=float), 0.0, 1.0)
        ct = 1.0 / rho_t  # bulk conductivity
        qv = Qv_max * vclay
        f = (phi**m_cementation) / a_tortuosity  # formation-factor reciprocal F⁻¹
        # Iterate Ct = F⁻¹/ρ_w·Sw^n + B·Qv·Sw^{n-1}  ⇒  Sw = (Ct/(F⁻¹/ρ_w + B·Qv/Sw))^{1/n}
        sw = np.clip(
            (a_tortuosity * rho_w_ohm_m / ((phi**m_cementation) * rho_t)) ** (1.0 / n_saturation),
            1e-3,
            1.0,
        )
        for _ in range(12):
            denom = f / rho_w_ohm_m + B_counterion * qv / np.maximum(sw, 1e-3)
            sw = np.clip((ct / np.maximum(denom, 1e-12)) ** (1.0 / n_saturation), 1e-6, 1.0)
        return ctx.as_output(sw)


class DualWaterSaturation(WaxmanSmitsSaturation):
    """Dual-water clay-corrected water saturation (doc 07 §4.2 "Fluid / clay-conduction").

    The dual-water model splits pore water into *bound* (clay-associated, conductive,
    fraction ``Swb ≈ α·Vclay``) and *free* brine. Here it is expressed in the same
    surface-conduction algebra as Waxman-Smits with the clay term keyed off a bound-water
    fraction — a textbook-equivalent shaly-sand correction (doc 07 §4.2). Shares the
    Waxman-Smits solver; only the spec (id/title) differs so the UI lists both options.
    """

    id = "rp.dual_water"
    version = "1.0.0"
    title = "Resistivity + Porosity + Clay → Water Saturation (Dual-Water)"
    assumptions = [
        "pore water = bound (clay) + free brine; bound water is extra-conductive",
        "bound-water fraction ∝ clay volume (Swb ≈ α·Vclay)",
        "surface-conduction parameters are constant unless calibrated per-cell",
    ]


register.transform(ArchieSaturation())
register.transform(WaxmanSmitsSaturation())
register.transform(DualWaterSaturation())
