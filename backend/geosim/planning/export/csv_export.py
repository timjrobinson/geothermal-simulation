"""CSV exports — deviation survey + predicted log (doc 09 §9, both P0 in scope).

Two universal, spreadsheet-/script-friendly CSV writers:

- :func:`export_survey_csv` — ``MD, Inc, Azi, TVD, N, E, DLS`` per survey station, plus
  optional ``Lat, Lon, Elev, TVDSS`` columns when the project is **georeferenced** (doc 01
  §7 ``engineering_to_crs`` / ``to_lonlat``). The MD/inc/azi come straight from the survey;
  TVD/N/E/DLS are the **derived** min-curvature outputs (doc 09 §4.3, shared integrator).
- :func:`export_log_csv` — one row per predicted-log station: ``MD, TVD, Temperature, Sigma,
  Favorability, Lithology, Resistivity, FractureDensity, hazard columns…, Risk`` (doc 09
  §5.2 / §9 row 2). Temperature is the display °C the log already carries (canonical K
  internally), converted to °F under the field-unit profile.

Both writers prepend a **provenance block** (doc 09 §9 rule): tool + model version, design
method/constraints, sampling step, CRS / vertical datum, units profile, and a timestamp — so
the export is auditable and a downstream tool re-georeferences identically. Units are honored
via the :mod:`.units` profiles (metric canonical or field), with the unit written into every
column header.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime

import numpy as np

from .units import ExportUnits, for_profile

__all__ = ["export_survey_csv", "export_log_csv"]

_COMMENT = "# "


def _provenance_lines(
    *,
    title: str,
    well,
    units: ExportUnits,
    frame=None,
    model_version: str | None = None,
    md_step_m: float | None = None,
    extra: dict | None = None,
) -> list[str]:
    """The auditable provenance header block (doc 09 §9 rule: every export is reproducible)."""
    lines = [
        f"{_COMMENT}{title}",
        f"{_COMMENT}generatedBy: geosim well-planner (doc 09 §9)",
        f"{_COMMENT}generatedAt: {datetime.now(UTC).isoformat()}",
        f"{_COMMENT}wellId: {well.id}",
        f"{_COMMENT}wellName: {well.name}",
        f"{_COMMENT}units: {units.name} (length={units.length_unit}, angle={units.angle_unit}, "
        f"dls={units.dls_unit}, temperature={units.temperature_unit})",
    ]
    if model_version is not None:
        lines.append(f"{_COMMENT}modelVersion: {model_version}")
    if md_step_m is not None:
        lines.append(f"{_COMMENT}samplingStep_md: {md_step_m} m")
    if well.design is not None:
        d = well.design
        lines.append(
            f"{_COMMENT}designMethod: {d.method} (kop_md={d.kop_md_m} m, "
            f"buildRate={d.build_rate_deg30m} deg/30m)"
        )
    lines.append(
        f"{_COMMENT}constraints: maxDLS={well.constraints.max_dls_deg30m} deg/30m, "
        f"maxInc={well.constraints.max_inc_deg} deg"
    )
    # CRS / vertical datum round-trip (doc 09 §9 rule). Local-mode states "no CRS".
    if frame is not None and getattr(frame, "horizontal_crs", None):
        lines.append(f"{_COMMENT}horizontalCRS: {frame.horizontal_crs}")
        lines.append(f"{_COMMENT}verticalDatum: {frame.vertical_datum or 'unspecified'}")
    else:
        lines.append(f"{_COMMENT}horizontalCRS: local frame, no CRS")
    lines.append(
        f"{_COMMENT}mdDatum: {getattr(well, 'kb_elev_m', 0.0)} m "
        f"({getattr(well.design, 'method', None) and ''}depthReference=KB)"
    )
    for k, v in (extra or {}).items():
        lines.append(f"{_COMMENT}{k}: {v}")
    return lines


def _is_georeferenced(frame) -> bool:
    return frame is not None and getattr(frame, "horizontal_crs", None) is not None


def export_survey_csv(
    well,
    *,
    units: str | ExportUnits = "metric",
    frame=None,
) -> str:
    """Export the deviation survey to CSV (doc 09 §9 row 1).

    Columns: ``MD, Inc, Azi, TVD, N, E, DLS`` (units per the profile). When ``frame`` is a
    georeferenced :class:`~geosim.spatial.SpatialFrame`, four more columns — ``Lat, Lon,
    Elev, TVDSS`` — are appended via doc 01 §7 ``engineering_to_crs`` / ``to_lonlat`` so the
    survey carries real-world coordinates; otherwise they are omitted (local frame).

    MD/inc/azi are the survey's source-of-truth values; TVD/N/E/DLS are the **derived**
    min-curvature outputs (shared integrator, doc 09 §4.3) — never re-derived here.
    """
    u = for_profile(units)
    survey = np.asarray(well.deviation_survey, dtype=float).reshape(-1, 3)
    pos = well.positions()  # shared min-curvature integrator (doc 09 §4.3)
    enu = pos.enu  # (N,3) Engineering (E=x, N=y, Up=z)

    georef = _is_georeferenced(frame)
    latlon = elev_world = None
    if georef:
        crs_xyz = frame.engineering_to_crs(enu)  # easting/northing/elev in project CRS
        elev_world = crs_xyz[:, 2]
        try:
            latlon = frame.to_lonlat(enu)  # (lon, lat, elev)
        except Exception:  # noqa: BLE001 — projection unavailable → skip lat/lon, keep CRS elev
            latlon = None

    buf = io.StringIO()
    for line in _provenance_lines(
        title="GeoSim deviation-survey export (doc 09 §9)", well=well, units=u, frame=frame
    ):
        buf.write(line + "\n")

    w = csv.writer(buf)
    header = [
        f"MD_{u.length_unit}",
        f"Inc_{u.angle_unit}",
        f"Azi_{u.angle_unit}",
        f"TVD_{u.length_unit}",
        f"N_{u.length_unit}",
        f"E_{u.length_unit}",
        f"DLS_{u.dls_unit}",
    ]
    if georef:
        header += [f"Elev_{u.length_unit}", f"TVDSS_{u.length_unit}"]
        if latlon is not None:
            header += ["Lat_deg", "Lon_deg"]
    w.writerow(header)

    for i in range(survey.shape[0]):
        md, inc, azi = survey[i]
        e, n = enu[i, 0], enu[i, 1]
        row = [
            f"{u.conv_length(md):.4f}",
            f"{inc:.4f}",
            f"{azi:.4f}",
            f"{u.conv_length(pos.tvd[i]):.4f}",
            f"{u.conv_length(n):.4f}",
            f"{u.conv_length(e):.4f}",
            f"{u.conv_dls(pos.dls[i]):.6f}",
        ]
        if georef:
            elev = float(elev_world[i])
            row += [f"{u.conv_length(elev):.4f}", f"{u.conv_length(-elev):.4f}"]
            if latlon is not None:
                row += [f"{latlon[i, 1]:.8f}", f"{latlon[i, 0]:.8f}"]
        w.writerow(row)
    return buf.getvalue()


def _station_prop(station, key: str):
    """A predicted-log station's property value (the ``values`` dict carries {value,sigma,…})."""
    v = station.values.get(key)
    if v is None:
        return None
    return v.get("value")


def export_log_csv(
    log,
    well,
    *,
    units: str | ExportUnits = "metric",
    frame=None,
) -> str:
    """Export the predicted log to CSV (doc 09 §9 row 2).

    Columns: ``MD, TVD, Temperature(+Sigma), Favorability, Lithology, Resistivity,
    FractureDensity, <hazard columns>, Risk`` (doc 09 §5.2). Temperature is the display value
    the log already carries (°C; converted to °F under the field profile); its σ is a Δ
    (scaled, not offset). The hazard columns are the union of hazard keys present across all
    stations, so a station missing a hazard reads blank rather than mislabeled.
    """
    u = for_profile(units)

    # Stable union of hazard channel keys across stations (doc 09 §7.3 — may be proxy-named).
    hazard_keys: list[str] = []
    for s in log.stations:
        for k in s.hazards:
            if k not in hazard_keys:
                hazard_keys.append(k)

    buf = io.StringIO()
    for line in _provenance_lines(
        title="GeoSim predicted-log export (doc 09 §9)",
        well=well,
        units=u,
        frame=frame,
        model_version=log.model_version,
        md_step_m=log.md_step_m,
        extra={"riskWeights": log.risk_weights.normalized().to_payload()},
    ):
        buf.write(line + "\n")

    w = csv.writer(buf)
    header = [
        f"MD_{u.length_unit}",
        f"TVD_{u.length_unit}",
        f"Temperature_{u.temperature_unit}",
        f"TemperatureSigma_{u.temperature_unit}",
        "Favorability",
        "Lithology",
        "Resistivity_ohmm",
        "FractureDensity",
    ]
    header += [f"hazard_{k}" for k in hazard_keys]
    header.append("Risk")
    w.writerow(header)

    for s in log.stations:
        temp = s.values.get("temperatureC", {})
        tval = temp.get("value")
        tsig = temp.get("sigma")
        row = [
            f"{u.conv_length(s.md):.4f}",
            f"{u.conv_length(s.tvd):.4f}",
            ("" if tval is None else f"{u.conv_temperature_c(tval):.4f}"),
            ("" if tsig is None else f"{u.conv_temperature_delta_c(tsig):.4f}"),
            _fmt(_station_prop(s, "favorability")),
            ("" if s.lithology is None else str(s.lithology)),
            _fmt(_station_prop(s, "resistivity")),
            _fmt(_station_prop(s, "fracture_density")),
        ]
        row += [_fmt(s.hazards.get(k)) for k in hazard_keys]
        row.append(f"{s.risk:.4f}")
        w.writerow(row)
    return buf.getvalue()


def _fmt(value, ndigits: int = 4) -> str:
    """Format an optional float; ``None`` → blank cell."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return f"{float(value):.{ndigits}f}"
