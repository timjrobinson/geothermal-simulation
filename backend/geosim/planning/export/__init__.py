"""Well-plan export (doc 09 §9).

Hand a real planning/drilling tool a **trajectory + predicted logs** in formats it already
reads (doc 09 §9 decided set):

- **CSV deviation survey** — ``MD, Inc, Azi, TVD, N, E, DLS`` (+ optional ``lat/lon/elev``
  & ``TVDSS`` when the project is georeferenced) via :func:`export_survey_csv`.
- **CSV predicted log** — ``MD, TVD, temperature(+σ), favorability, lithology, resistivity,
  fractureDensity, hazards, risk`` via :func:`export_log_csv`.
- **WITSML trajectory** — WITSML **2.0** (default) or **1.4.1.1** (legacy alt) via
  :func:`export_witsml_trajectory`, with the mandatory export→re-import round-trip guarded
  by :func:`parse_witsml_trajectory` (doc 09 §9.1).

Every export REUSES, never reinvents:

- :func:`geosim.spatial.min_curvature_positions` (via
  :meth:`geosim.planning.PlannedWell.positions`) — TVD/N/E/DLS are always derived from the
  survey, never re-implemented.
- :class:`geosim.spatial.SpatialFrame` ``engineering_to_crs`` / ``to_lonlat`` (doc 01 §7) —
  the georeferenced columns and the WITSML ``wellCRS`` round-trip through the project CRS.
- :func:`geosim.spatial.to_display` / the units registry (doc 01 §5) — metric canonical
  (m, dega, deg/30m, °C) or field units (ft, °F, °/100ft) written into every header.

WITSML XSD validation: the Energistics WITSML 2.0 / 1.4.1.1 XSDs and ``xmlschema`` are not
installed in this environment, so the writer validates the emitted XML **structurally**
(required objects/fields/uoms present, well-formed) and flags that the schema-validation step
is structural-only (doc 09 §9.1 "validate structurally without the XSD and note it").
"""

from .csv_export import export_log_csv, export_survey_csv
from .units import FIELD_UNITS, METRIC_UNITS, ExportUnits
from .witsml import (
    WitsmlValidationResult,
    export_witsml_trajectory,
    parse_witsml_trajectory,
    validate_witsml_trajectory,
)

__all__ = [
    # CSV (doc 09 §9, P0)
    "export_survey_csv",
    "export_log_csv",
    # units (doc 01 §5)
    "ExportUnits",
    "METRIC_UNITS",
    "FIELD_UNITS",
    # WITSML (doc 09 §9.1, P1)
    "export_witsml_trajectory",
    "parse_witsml_trajectory",
    "validate_witsml_trajectory",
    "WitsmlValidationResult",
]
