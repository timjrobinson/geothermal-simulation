"""Units registry (doc 01 §5).

Every numeric quantity carries an explicit unit; nothing is dimensionless-by-assumption.
We use a single ``pint`` registry, convert to **canonical internal units** on ingest,
store the canonical unit in metadata, and keep the original unit in provenance.

Critical conventions from doc 01 §5:

- **Temperature is canonical in kelvin, displayed in °C.** ``pint`` treats ``degC`` as an
  *offset* unit, so absolute temperatures and temperature *differences* are different
  quantities. We store absolute T in **K**, gradients in **K/km**, uncertainty/Δ in **K**,
  and convert to °C only for display. (Resolves critique #21.)
- **Chargeability is not one unit.** Time-domain IP (ms), frequency-domain IP (mV/V) and
  IP phase (mrad) are distinct ``PropertyTypeKey``s, never collapsed. (Resolves #22.)
"""

from __future__ import annotations

import functools

import pint

__all__ = [
    "ureg",
    "Q_",
    "CANONICAL_UNITS",
    "to_canonical",
    "to_display",
    "convert",
]


@functools.lru_cache(maxsize=1)
def _build_registry() -> pint.UnitRegistry:
    reg = pint.UnitRegistry()
    # mGal and nT are not in pint's default registry; define them.
    reg.define("Gal = cm / s**2")  # galileo, acceleration
    reg.define("mGal = 1e-3 Gal")
    # nanotesla for magnetic field (tesla is a base SI unit pint knows)
    # (nT resolves via the SI prefix automatically: reg.Unit("nT"))
    return reg


ureg: pint.UnitRegistry = _build_registry()
Q_ = ureg.Quantity

# Canonical internal unit per PropertyTypeKey (doc 01 §5 table). The property-type
# registry (``property_types.py``) is the authoritative owner; this dict is the unit
# half of it, kept here so unit conversion has no import cycle.
CANONICAL_UNITS: dict[str, str] = {
    # coordinates / length
    "length": "m",
    # electrical
    "resistivity": "ohm*m",
    "conductivity": "S/m",
    # potential fields
    "density": "kg/m**3",
    "susceptibility": "dimensionless",  # SI magnetic susceptibility
    "gravity_anomaly": "mGal",
    "magnetic_field": "nT",
    # seismic
    "velocity_p": "m/s",
    "velocity_s": "m/s",
    # IP (three distinct measurements — never one canonical unit)
    "chargeability_time_ms": "ms",
    "chargeability_mv_v": "mV/V",
    "phase_mrad": "mrad",
    # thermal — ABSOLUTE temperature in kelvin (display in °C)
    "temperature": "kelvin",
    "temperature_gradient": "kelvin/km",
    "temperature_sigma": "kelvin",
    # deformation
    "deformation": "mm",
    # derived
    "favorability": "dimensionless",
    "porosity": "dimensionless",
    "water_saturation": "dimensionless",
    "fracture_density": "dimensionless",
}

# Default display units that differ from canonical (UI edge concern, doc 01 §5).
DISPLAY_UNITS: dict[str, str] = {
    "temperature": "degC",
}


def convert(value, src_unit: str, dst_unit: str):
    """Convert a scalar/array ``value`` from ``src_unit`` to ``dst_unit`` via pint.

    Handles offset units (e.g. degC↔K) correctly for *absolute* quantities.
    """
    return (Q_(value, _norm(src_unit)).to(_norm(dst_unit))).magnitude


def to_canonical(value, unit: str, property_type: str):
    """Convert a measured ``value`` in ``unit`` to the canonical unit for ``property_type``.

    Returns the magnitude in canonical units. The source unit must be retained in
    provenance by the caller (doc 01 §5 / doc 02 §7).
    """
    canon = CANONICAL_UNITS.get(property_type)
    if canon is None:
        raise KeyError(f"no canonical unit registered for property_type {property_type!r}")
    return convert(value, unit, canon)


def to_display(value, property_type: str, display_unit: str | None = None):
    """Convert a canonical ``value`` for ``property_type`` to a display unit.

    Defaults to the registered display unit (e.g. °C for temperature) or the canonical
    unit when no display override exists.
    """
    canon = CANONICAL_UNITS.get(property_type)
    if canon is None:
        raise KeyError(f"no canonical unit registered for property_type {property_type!r}")
    dst = display_unit or DISPLAY_UNITS.get(property_type, canon)
    return convert(value, canon, dst)


def _norm(unit: str) -> str:
    """Normalise a few common geoscience unit spellings to pint-parseable strings."""
    aliases = {
        "ohm.m": "ohm*m",
        "ohmm": "ohm*m",
        "ohm-m": "ohm*m",
        "kg/m3": "kg/m**3",
        "mV/V": "mV/V",
        "degC": "degC",
        "C": "degC",
        "celsius": "degC",
        "K": "kelvin",
    }
    return aliases.get(unit, unit)
