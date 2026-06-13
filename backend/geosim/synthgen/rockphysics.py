"""Rock physics — one geology, mutually-consistent properties (doc 05 §3).

A *named ruleset* maps ``(lithology unit, state)`` → each geophysical property through
simple, well-known petrophysics (doc 05 §3.1). Every rule is a **pure per-voxel
function** the compiler applies vectorised over the truth grid, so the same geology
drives density, susceptibility, resistivity, chargeability, Vp/Vs and temperature
*consistently* — a hot + saline + altered + porous voxel is simultaneously more
conductive, less dense (or slower), and magnetically suppressed (doc 05 §1 decision #1,
§4.2 worked examples).

Relationships (``default-v1``, doc 05 §3.1):

- **Resistivity** — modified Archie + clay term: ``1/ρ = φ^m·Sw^n/(a·ρw) +
  clayCond(alterationFrac, T)``, with ``ρw`` from salinity & temperature (Arps).
- **Density** — φ-mixing: ``ρ = (1−φ)·ρ_grain + φ·ρ_fluid``.
- **Susceptibility** — unit base, alteration-suppressed: ``χ = χ_base·(1 − alt)``
  (hydrothermal alteration destroys magnetite → magnetic *low*, doc 05 §4.2).
- **Vp/Vs** — unit base softened by porosity & fracture density; saturation stiffens Vp
  but barely Vs → Vp/Vs flags fluid (doc 05 §3.1).
- **Chargeability** — ``η = η0·(clayFrac + sulphideFrac)`` (alteration haloes chargeable).
- **Temperature** — taken directly from the state field (geotherm + plume, doc 05 §2.3).

The shipped per-unit base library (:data:`DEFAULT_UNIT_LIBRARY`) covers Basin-&-Range
lithologies (doc 05 §3.2). All outputs are in **canonical units** (doc 01 §5): density
kg/m³, χ dimensionless SI, resistivity Ω·m, chargeability mV/V, Vp/Vs m/s, temperature
kelvin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scene import UnitProps

__all__ = [
    "DEFAULT_UNIT_LIBRARY",
    "RockPhysicsResult",
    "RuleSet",
    "get_ruleset",
    "default_v1",
]

# Basin-&-Range base property library (doc 05 §3.2). Resistivity/η/Vs are DERIVED.
DEFAULT_UNIT_LIBRARY: dict[str, UnitProps] = {
    "alluvium": UnitProps(rho=2050.0, chi=0.0005, vp=1800.0, phi=0.30),
    "volcanics": UnitProps(rho=2450.0, chi=0.02, vp=3400.0, phi=0.12),
    "carbonate": UnitProps(rho=2680.0, chi=0.0001, vp=5200.0, phi=0.05),
    "basement_granite": UnitProps(rho=2670.0, chi=0.005, vp=5600.0, phi=0.01),
    "young_intrusive": UnitProps(rho=2750.0, chi=0.03, vp=5900.0, phi=0.01),
}

# Fluid / Archie constants (doc 05 §3.1).
_RHO_FLUID = 1000.0  # kg/m³ pore water
_ARCHIE_A = 1.0  # tortuosity factor
_ARCHIE_M = 2.0  # cementation exponent
_ARCHIE_N = 2.0  # saturation exponent
# Reference brine: ~ NaCl-equivalent. ρw at 25 °C ≈ 6.5 Ω·m for ~1000 ppm TDS, scaling
# inversely with salinity, and dropping with temperature (Arps, doc 05 §3.1).
_RHOW_REF = 6.5  # Ω·m at REF salinity & 25 °C
_TDS_REF = 1000.0  # ppm
_ARPS_T_REF = 298.15  # K (25 °C)
# Vp/Vs softening coefficients (doc 05 §3.1).
_K_PHI_VP = 1.5  # porosity softening of Vp
_K_FR_VP = 0.25  # fracture softening of Vp
_SAT_VP_STIFFEN = 0.08  # saturation re-stiffens Vp (Gassmann-lite); barely moves Vs
_K_PHI_VS = 1.6
_K_FR_VS = 0.30
# Chargeability scaling (doc 05 §3.1) — mV/V.
_ETA0 = 120.0


def _arps_rho_water(salinity_tds: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """Pore-water resistivity ρw (Ω·m) from salinity (ppm) & temperature (K), Arps-style.

    ρw scales inversely with TDS and falls with temperature: ``ρw = ρw_ref ·
    (TDS_ref/TDS) · (T_ref/T)`` clamped away from zero. Hot + saline → very low ρw →
    very conductive (the geothermal signature, doc 05 §3.1, §2.2).
    """
    tds = np.maximum(salinity_tds, 1.0)
    t = np.maximum(temperature_k, 250.0)
    rhow = _RHOW_REF * (_TDS_REF / tds) * (_ARPS_T_REF / t)
    return np.clip(rhow, 0.01, 1.0e4)


def _clay_conductivity(alteration_frac: np.ndarray, temperature_k: np.ndarray) -> np.ndarray:
    """Surface/clay conduction term (S/m) from alteration fraction & temperature.

    A clay/alteration halo adds a parallel conduction path (doc 05 §3.1 clay term),
    growing with alteration fraction and mildly with temperature. Returns S/m to be
    added to the Archie bulk conductivity.
    """
    t_factor = 1.0 + 0.004 * (np.maximum(temperature_k, 250.0) - _ARPS_T_REF)
    return 0.20 * np.clip(alteration_frac, 0.0, 1.0) * np.clip(t_factor, 0.5, 5.0)


@dataclass(frozen=True)
class RockPhysicsResult:
    """Co-located derived property volumes on the truth grid (canonical units).

    Every array is ``float32`` ``(nz, ny, nx)``, Z-up, co-registered with ``L`` and the
    state field ``S`` (doc 05 §2.1, §2.2 property set).
    """

    density: np.ndarray  # kg/m³
    susceptibility: np.ndarray  # dimensionless SI
    resistivity: np.ndarray  # Ω·m
    chargeability_mv_v: np.ndarray  # mV/V
    velocity_p: np.ndarray  # m/s
    velocity_s: np.ndarray  # m/s
    temperature: np.ndarray  # kelvin (taken directly from S)
    porosity: np.ndarray  # fraction (effective, after state boost)


@dataclass(frozen=True)
class RuleSet:
    """A named rock-physics ruleset (doc 05 §3) — pure ``(L, S) → properties``."""

    name: str

    def apply(
        self,
        *,
        unit_index: np.ndarray,  # int (nz,ny,nx) — index into `units`
        units: list[UnitProps],  # ordered library matching unit_index
        temperature_k: np.ndarray,
        porosity_state: np.ndarray,  # additive porosity boost from S (fraction)
        water_saturation: np.ndarray,
        salinity_tds: np.ndarray,  # ppm
        alteration_frac: np.ndarray,
        fracture_density: np.ndarray,
    ) -> RockPhysicsResult:
        raise NotImplementedError


def _gather_unit_field(unit_index: np.ndarray, units: list[UnitProps], attr: str) -> np.ndarray:
    """Vectorised per-voxel lookup of a base unit attribute (doc 05 §3.2 library)."""
    table = np.array([getattr(u, attr) for u in units], dtype=np.float64)
    return table[unit_index]


class _DefaultV1(RuleSet):
    """``default-v1`` ruleset (doc 05 §3.1)."""

    def apply(
        self,
        *,
        unit_index: np.ndarray,
        units: list[UnitProps],
        temperature_k: np.ndarray,
        porosity_state: np.ndarray,
        water_saturation: np.ndarray,
        salinity_tds: np.ndarray,
        alteration_frac: np.ndarray,
        fracture_density: np.ndarray,
    ) -> RockPhysicsResult:
        rho_grain = _gather_unit_field(unit_index, units, "rho")
        chi_base = _gather_unit_field(unit_index, units, "chi")
        vp_base = _gather_unit_field(unit_index, units, "vp")
        phi_matrix = _gather_unit_field(unit_index, units, "phi")
        sulphide = _gather_unit_field(unit_index, units, "chargeable_frac")
        vp_vs = _gather_unit_field(unit_index, units, "vp_vs_ratio")

        alt = np.clip(alteration_frac, 0.0, 1.0)
        frac = np.clip(fracture_density, 0.0, 1.0)
        sw = np.clip(water_saturation, 0.0, 1.0)
        phi = np.clip(phi_matrix + porosity_state, 1.0e-3, 0.6)

        # ── density: φ-mixing (doc 05 §3.1) ──────────────────────────────────────
        density = (1.0 - phi) * rho_grain + phi * _RHO_FLUID

        # ── susceptibility: alteration-suppressed (destroys magnetite) ───────────
        susceptibility = chi_base * (1.0 - alt)

        # ── resistivity: modified Archie + clay term (doc 05 §3.1) ───────────────
        rhow = _arps_rho_water(salinity_tds, temperature_k)
        archie_cond = (phi**_ARCHIE_M) * (sw**_ARCHIE_N) / (_ARCHIE_A * rhow)
        clay_cond = _clay_conductivity(alt, temperature_k)
        total_cond = np.maximum(archie_cond + clay_cond, 1.0e-6)  # S/m
        resistivity = 1.0 / total_cond

        # ── chargeability: clay + sulphide (doc 05 §3.1) ─────────────────────────
        chargeability = _ETA0 * np.clip(alt + sulphide, 0.0, 1.0)

        # ── Vp/Vs: porosity & fracture softening, saturation stiffens Vp ─────────
        soften_vp = np.clip(1.0 - _K_PHI_VP * phi - _K_FR_VP * frac, 0.1, 1.0)
        velocity_p = vp_base * soften_vp * (1.0 + _SAT_VP_STIFFEN * sw)
        vs_base = vp_base / np.maximum(vp_vs, 1.0e-3)
        soften_vs = np.clip(1.0 - _K_PHI_VS * phi - _K_FR_VS * frac, 0.1, 1.0)
        velocity_s = vs_base * soften_vs  # saturation barely moves Vs (doc 05 §3.1)

        f32 = np.float32
        return RockPhysicsResult(
            density=density.astype(f32),
            susceptibility=susceptibility.astype(f32),
            resistivity=resistivity.astype(f32),
            chargeability_mv_v=chargeability.astype(f32),
            velocity_p=velocity_p.astype(f32),
            velocity_s=velocity_s.astype(f32),
            temperature=temperature_k.astype(f32),
            porosity=phi.astype(f32),
        )


_RULESETS: dict[str, RuleSet] = {"default-v1": _DefaultV1("default-v1")}

# Convenience handle to the canonical ruleset (doc 05 §3.1).
default_v1: RuleSet = _RULESETS["default-v1"]


def get_ruleset(name: str) -> RuleSet:
    """Return the named rock-physics ruleset (doc 05 §3)."""
    try:
        return _RULESETS[name]
    except KeyError as e:
        raise KeyError(f"unknown rock-physics ruleset {name!r} (doc 05 §3.1)") from e
