"""Property-type registry (doc 01 §5, feeds doc 02 & doc 08).

A ``PropertyType`` pins, per physical property: canonical unit, default colourmap,
default log/linear scaling, sensible display range, and the interpolation space used by
fusion resampling (doc 07 §2.2 — orders-of-magnitude properties interpolate in log10).

This is the single place a new survey method declares its property once and the whole
stack (units, storage metadata, colour mapping, viewer defaults, fusion) knows how to
handle it. Plugins register new keys here (doc 08 §4b); doc 02 §1 reserves
``"<plugin-registered>"`` for exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .units import CANONICAL_UNITS

__all__ = ["PropertyType", "PropertyTypeRegistry", "REGISTRY"]


@dataclass(frozen=True)
class PropertyType:
    key: str
    canonical_unit: str
    default_colormap: str = "viridis"
    default_scaling: str = "linear"  # "linear" | "log"
    display_range: tuple[float, float] | None = None
    interp_space: str = "linear"  # "linear" | "log10" — fusion resampling space (doc 07 §2.2)
    default_rel_sigma: float = 0.15  # conservative relative σ when none supplied (doc 07 §5.1)
    description: str = ""
    categorical: bool = False  # lithology_class etc. (doc 02 §10.2)

    def __post_init__(self) -> None:
        if not self.categorical and self.canonical_unit != CANONICAL_UNITS.get(self.key, self.canonical_unit):
            # Keep the unit registry and property registry from silently disagreeing.
            object.__setattr__(self, "canonical_unit", CANONICAL_UNITS.get(self.key, self.canonical_unit))


class PropertyTypeRegistry:
    """In-process registry of property types (doc 08 §4b extension point)."""

    def __init__(self) -> None:
        self._by_key: dict[str, PropertyType] = {}

    def register(self, pt: PropertyType, *, replace: bool = False) -> PropertyType:
        if pt.key in self._by_key and not replace:
            existing = self._by_key[pt.key]
            if existing != pt:
                raise ValueError(f"property type {pt.key!r} already registered with different spec")
        self._by_key[pt.key] = pt
        return pt

    def get(self, key: str) -> PropertyType:
        try:
            return self._by_key[key]
        except KeyError as e:
            raise KeyError(f"unknown property type {key!r} — register it first (doc 08)") from e

    def __contains__(self, key: str) -> bool:
        return key in self._by_key

    def keys(self) -> list[str]:
        return list(self._by_key)

    def all(self) -> list[PropertyType]:
        return list(self._by_key.values())


REGISTRY = PropertyTypeRegistry()


def _seed() -> None:
    """Seed the canonical property types from doc 01 §5 / doc 02 §1."""
    defs = [
        PropertyType("resistivity", "ohm*m", "turbo", "log", (1, 10000), "log10",
                     description="electrical resistivity"),
        PropertyType("conductivity", "S/m", "turbo", "log", (1e-4, 1.0), "log10"),
        PropertyType("density", "kg/m**3", "viridis", "linear", (1800, 3200), "linear"),
        PropertyType("susceptibility", "dimensionless", "cividis", "linear", (0.0, 0.05), "linear"),
        PropertyType("velocity_p", "m/s", "viridis", "linear", (1500, 6500), "linear"),
        PropertyType("velocity_s", "m/s", "viridis", "linear", (800, 4000), "linear"),
        PropertyType("temperature", "kelvin", "thermal", "linear", (273.0, 673.0), "linear",
                     description="absolute temperature (canonical K; display °C)"),
        PropertyType("chargeability_time_ms", "ms", "magma", "linear", (0, 200), "linear"),
        PropertyType("chargeability_mv_v", "mV/V", "magma", "linear", (0, 100), "linear"),
        PropertyType("phase_mrad", "mrad", "magma", "linear", (0, 200), "linear"),
        PropertyType("gravity_anomaly", "mGal", "RdBu", "linear", (-50, 50), "linear"),
        PropertyType("magnetic_field", "nT", "RdBu", "linear", (-500, 500), "linear"),
        PropertyType("deformation", "mm", "RdBu", "linear", (-50, 50), "linear"),
        PropertyType("favorability", "dimensionless", "inferno", "linear", (0.0, 1.0), "linear"),
        PropertyType("porosity", "dimensionless", "viridis", "linear", (0.0, 0.4), "linear"),
        PropertyType("water_saturation", "dimensionless", "Blues", "linear", (0.0, 1.0), "linear"),
        PropertyType("fracture_density", "dimensionless", "hot", "linear", (0.0, 1.0), "linear"),
        PropertyType("lithology_class", "dimensionless", "tab20", "linear", None, "linear",
                     categorical=True, description="categorical lithology label / class probability"),
    ]
    for pt in defs:
        REGISTRY.register(pt, replace=True)


_seed()
