"""Export unit profiles (doc 09 §9 "units honored via doc 01 §5").

An export is written either in **metric canonical** (the internal SI/locked convention —
m, dega, °/30 m, °C) or in **field units** (ft, °F, °/100 ft) per the export dialog. This
module is the single place those two profiles + their conversions live, so the CSV and
WITSML writers share one definition and the unit strings written into headers / WITSML
``uom`` attributes always match the numbers.

Conversions reuse :func:`geosim.spatial.convert` / the pint registry (doc 01 §5) — no
hand-rolled factors. Temperature is canonical **kelvin** internally; the predicted log
already carries the display °C, so a field export only needs °C→°F (an *offset* conversion,
handled by pint).
"""

from __future__ import annotations

from dataclasses import dataclass

from geosim.spatial import convert

__all__ = ["ExportUnits", "METRIC_UNITS", "FIELD_UNITS", "for_profile"]

# DLS / build / turn rates are normalized to a course length: metric °/30 m, field °/100 ft.
_M_PER_FT = 0.3048
_DLS_FIELD_PER_METRIC = (100.0 * _M_PER_FT) / 30.0  # (°/100ft) = (°/30m) · (100ft / 30m)


@dataclass(frozen=True)
class ExportUnits:
    """A coherent export unit profile + the conversions off the canonical metric values.

    All ``*_unit`` strings are what goes into CSV headers and WITSML ``uom`` attributes. The
    ``conv_*`` methods take a **canonical metric** value (m, dega, °/30 m, °C) and return it
    in this profile's units — angles are unit-invariant (degrees either way), so only
    length / DLS / temperature actually convert.
    """

    name: str  # "metric" | "field"
    length_unit: str  # m | ft  (CSV header label)
    length_uom: str  # m | ft  (WITSML EML uom)
    angle_unit: str  # deg | deg (display label)
    angle_uom: str  # dega     (WITSML EML uom for plane-angle)
    dls_unit: str  # deg/30m | deg/100ft (CSV header label)
    dls_uom: str  # 0.1 dega/m-ish — WITSML uses dega/30.m or dega/100.ft
    temperature_unit: str  # degC | degF (CSV header label)
    temperature_uom: str  # degC | degF (WITSML EML uom)

    def conv_length(self, value_m: float) -> float:
        """Canonical metres → this profile's length unit."""
        if self.length_unit == "m":
            return float(value_m)
        return float(convert(value_m, "m", "ft"))

    def conv_dls(self, value_deg30m: float) -> float:
        """Canonical °/30 m → this profile's DLS unit (°/30 m or °/100 ft)."""
        if self.dls_unit.endswith("30m"):
            return float(value_deg30m)
        return float(value_deg30m * _DLS_FIELD_PER_METRIC)

    def conv_temperature_c(self, value_c: float) -> float:
        """Display °C → this profile's temperature unit (°C or °F)."""
        if self.temperature_unit == "degC":
            return float(value_c)
        return float(convert(value_c, "degC", "degF"))

    def conv_temperature_delta_c(self, sigma_c: float) -> float:
        """A temperature *difference* (σ) in °C → this profile's unit.

        σ is a Δ, so it scales by 9/5 for °F (NOT the offset conversion). Pint's offset
        handling would mis-convert a difference, so we apply the ratio explicitly.
        """
        if self.temperature_unit == "degC":
            return float(sigma_c)
        return float(sigma_c * 9.0 / 5.0)


METRIC_UNITS = ExportUnits(
    name="metric",
    length_unit="m",
    length_uom="m",
    angle_unit="deg",
    angle_uom="dega",
    dls_unit="deg/30m",
    dls_uom="dega/30.m",
    temperature_unit="degC",
    temperature_uom="degC",
)

FIELD_UNITS = ExportUnits(
    name="field",
    length_unit="ft",
    length_uom="ft",
    angle_unit="deg",
    angle_uom="dega",
    dls_unit="deg/100ft",
    dls_uom="dega/100.ft",
    temperature_unit="degF",
    temperature_uom="degF",
)


def for_profile(units: str | ExportUnits) -> ExportUnits:
    """Resolve a ``"metric"`` / ``"field"`` name (or a passed :class:`ExportUnits`) to a profile."""
    if isinstance(units, ExportUnits):
        return units
    key = (units or "metric").lower()
    if key in ("metric", "si", "m"):
        return METRIC_UNITS
    if key in ("field", "imperial", "ft"):
        return FIELD_UNITS
    raise ValueError(f"unknown export units profile {units!r} (use 'metric' or 'field')")
