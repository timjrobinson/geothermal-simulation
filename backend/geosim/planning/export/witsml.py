"""WITSML trajectory export + re-import (doc 09 §9.1, P1 in scope).

The industry interchange for a deviation survey. We emit a **trajectory-focused** WITSML
document — the minimum objects/fields of doc 09 §9.1 — not a full WITSML store:

``Well`` (name, uid, timeZone, CRS-referenced surface location) → ``Wellbore`` (name, uid,
parent ref, status) → ``Trajectory`` (name, uid, parent ref, MD datum → KB/GL elevation +
datum kind matching doc 01 ``depthReference``, serviceCompany = this tool + modelVersion) →
``TrajectoryStation[]`` (per station ``md, incl, azi`` + derived ``tvd, dispNs, dispEw, dls``;
``typeTrajStation=planned``). **Every quantity carries a ``uom``** (EML units-of-measure) and
the horizontal ``wellCRS`` is the project CRS (doc 01 §7) so a consumer re-georeferences
identically.

Two on-the-wire versions behind one writer via a ``version`` switch (doc 09 §9.1):

- **WITSML 2.0** (default; Energistics ETP-aligned) — element/attribute layout per the 2.0
  data model (``Trajectory`` with child ``TrajectoryStation`` elements, ``<Md uom=…>`` value
  elements, ``wellCRS`` reference).
- **WITSML 1.4.1.1** (legacy alt) — the older ``<witsml:trajectorys>`` envelope with
  ``<trajectoryStation>`` / lower-camel element names and ``uom`` attributes.

Derived geometry (TVD/N/E/DLS) is the **shared min-curvature integrator** output (doc 09
§4.3) — never re-implemented here. The writer and the :func:`parse_witsml_trajectory` reader
are a matched pair guarding the mandatory **export → re-import round-trip** test (doc 09
§9.1): re-parsing reproduces each station's (MD, inc, azi) and derived (TVD, N, E) within
tolerance, and the MD datum + CRS survive.

**XSD note (doc 09 §9.1).** The Energistics WITSML 2.0 / 1.4.1.1 XSDs and the ``xmlschema``
validator are not installed here, so :func:`validate_witsml_trajectory` validates the emitted
XML **structurally** (well-formed; required objects/fields/uoms present) and reports
``schema_validated=False`` with a note — rather than failing or silently claiming XSD
conformance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import numpy as np

from .units import ExportUnits, for_profile

__all__ = [
    "export_witsml_trajectory",
    "parse_witsml_trajectory",
    "validate_witsml_trajectory",
    "WitsmlValidationResult",
    "ParsedTrajectory",
    "ParsedStation",
]

WITSML_20_NS = "http://www.energistics.org/energyml/data/witsmlv2"
WITSML_141_NS = "http://www.witsml.org/schemas/1series"

_SERVICE_COMPANY = "geosim well-planner (doc 09)"


# ──────────────────────────────────────────────────────────────────────────
# parse results
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ParsedStation:
    """One re-imported trajectory station (canonical metric: m, dega, °/30 m)."""

    md: float
    inc: float
    azi: float
    tvd: float
    dispNs: float  # northing displacement from the well reference point (m)
    dispEw: float  # easting displacement from the well reference point (m)
    dls: float


@dataclass
class ParsedTrajectory:
    """A re-imported WITSML trajectory (doc 09 §9.1 round-trip target)."""

    well_uid: str
    well_name: str
    wellbore_uid: str
    trajectory_uid: str
    md_datum_elev_m: float
    md_datum_kind: str  # "KB" | "GL" | …  (doc 01 depthReference)
    well_crs: str | None
    version: str  # "2.0" | "1.4.1.1"
    stations: list[ParsedStation] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Station fields as aligned arrays for tolerance comparison in the round-trip test."""
        return {
            "md": np.array([s.md for s in self.stations]),
            "inc": np.array([s.inc for s in self.stations]),
            "azi": np.array([s.azi for s in self.stations]),
            "tvd": np.array([s.tvd for s in self.stations]),
            "dispNs": np.array([s.dispNs for s in self.stations]),
            "dispEw": np.array([s.dispEw for s in self.stations]),
            "dls": np.array([s.dls for s in self.stations]),
        }


@dataclass
class WitsmlValidationResult:
    """The result of validating an emitted WITSML document (doc 09 §9.1)."""

    well_formed: bool
    structural_ok: bool
    schema_validated: bool  # True only if an XSD validator actually ran
    n_stations: int
    errors: list[str] = field(default_factory=list)
    note: str = ""


# ──────────────────────────────────────────────────────────────────────────
# geometry helper (shared integrator → station displacements)
# ──────────────────────────────────────────────────────────────────────────


def _station_geometry(well) -> dict[str, np.ndarray]:
    """Derive MD/inc/azi + TVD/dispNs/dispEw/DLS from the survey (shared integrator, §4.3).

    ``dispNs``/``dispEw`` are WITSML displacements from the **well reference point** (the
    wellhead), so we subtract the wellhead from the Engineering E/N. TVD/DLS come straight
    from the min-curvature result.
    """
    survey = np.asarray(well.deviation_survey, dtype=float).reshape(-1, 3)
    pos = well.positions()
    wx, wy = float(well.wellhead[0]), float(well.wellhead[1])
    return {
        "md": survey[:, 0],
        "inc": survey[:, 1],
        "azi": survey[:, 2],
        "tvd": pos.tvd,
        "dispNs": pos.enu[:, 1] - wy,  # North displacement from wellhead
        "dispEw": pos.enu[:, 0] - wx,  # East displacement from wellhead
        "dls": pos.dls,
    }


def _datum_kind(well) -> str:
    """The MD-datum kind matching doc 01 ``depthReference`` (default KB)."""
    d = getattr(well, "design", None)
    ref = getattr(d, "depth_reference", None) if d is not None else None
    return ref or "KB"


# ──────────────────────────────────────────────────────────────────────────
# export
# ──────────────────────────────────────────────────────────────────────────


def export_witsml_trajectory(
    well,
    *,
    version: str = "2.0",
    units: str | ExportUnits = "metric",
    frame=None,
    model_version: str | None = None,
    well_uid: str | None = None,
    wellbore_uid: str | None = None,
    trajectory_uid: str | None = None,
    time_zone: str = "Z",
) -> str:
    """Export a planned well's deviation survey as a WITSML ``trajectory`` document (doc 09 §9.1).

    ``version`` selects WITSML **2.0** (default) or **1.4.1.1** (legacy alt). Units are honored
    via the :mod:`.units` profile (metric canonical m/dega/°·30m⁻¹ or field ft/dega/°·100ft⁻¹),
    written as EML ``uom`` on every quantity. The horizontal ``wellCRS`` is the project CRS
    when ``frame`` is georeferenced (doc 01 §7), else flagged local. Returns the XML string.
    """
    u = for_profile(units)
    geom = _station_geometry(well)
    well_uid = well_uid or well.id
    wellbore_uid = wellbore_uid or f"{well.id}_wb"
    trajectory_uid = trajectory_uid or f"{well.id}_traj"
    crs = getattr(frame, "horizontal_crs", None) if frame is not None else None
    datum_elev = float(getattr(well, "kb_elev_m", 0.0))
    datum_kind = _datum_kind(well)

    if version == "2.0":
        root = _build_20(
            well, u, geom, well_uid, wellbore_uid, trajectory_uid, crs,
            datum_elev, datum_kind, model_version, time_zone,
        )
    elif version in ("1.4.1.1", "1.4.1", "1.4"):
        root = _build_141(
            well, u, geom, well_uid, wellbore_uid, trajectory_uid, crs,
            datum_elev, datum_kind, model_version, time_zone,
        )
    else:
        raise ValueError(f"unsupported WITSML version {version!r} (use '2.0' or '1.4.1.1')")

    _indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _quom(parent, tag: str, value: float, uom: str, ndigits: int = 6):
    """Append a ``<tag uom=…>value</tag>`` quantity element (doc 09 §9.1: uom on everything)."""
    el = ET.SubElement(parent, tag)
    el.set("uom", uom)
    el.text = f"{float(value):.{ndigits}f}"
    return el


def _build_20(
    well, u, geom, well_uid, wellbore_uid, trajectory_uid, crs,
    datum_elev, datum_kind, model_version, time_zone,
):
    """WITSML 2.0 document (Energistics data model layout)."""
    root = ET.Element("Trajectorys")
    root.set("xmlns", WITSML_20_NS)

    # Well (surface location + CRS).
    welem = ET.SubElement(root, "Well")
    welem.set("uid", well_uid)
    _text(welem, "Name", well.name)
    _text(welem, "TimeZone", time_zone)
    loc = ET.SubElement(welem, "WellLocation")
    _quom(loc, "Easting", well.wellhead[0], u.length_uom)
    _quom(loc, "Northing", well.wellhead[1], u.length_uom)
    _text(welem, "WellCRS", crs or "local frame, no CRS")

    # Wellbore.
    wbelem = ET.SubElement(root, "Wellbore")
    wbelem.set("uid", wellbore_uid)
    _text(wbelem, "Name", f"{well.name} / WB")
    _text(wbelem, "WellUid", well_uid)
    _text(wbelem, "StatusWellbore", well.status)

    # Trajectory + MD datum.
    traj = ET.SubElement(root, "Trajectory")
    traj.set("uid", trajectory_uid)
    _text(traj, "Name", f"{well.name} planned trajectory")
    _text(traj, "WellboreUid", wellbore_uid)
    _text(traj, "ServiceCompany", _company(model_version))
    _text(traj, "GrowingStatus", "active")
    datum = ET.SubElement(traj, "MdDatum")
    _text(datum, "DatumKind", datum_kind)
    _quom(datum, "Elevation", u.conv_length(datum_elev), u.length_uom, ndigits=4)
    _text(traj, "WellCRS", crs or "local frame, no CRS")
    _quom(traj, "MdMn", u.conv_length(geom["md"][0]), u.length_uom, ndigits=4)
    _quom(traj, "MdMx", u.conv_length(geom["md"][-1]), u.length_uom, ndigits=4)

    for i in range(geom["md"].shape[0]):
        st = ET.SubElement(traj, "TrajectoryStation")
        st.set("uid", f"st_{i}")
        _text(st, "TypeTrajStation", "plan")
        _quom(st, "Md", u.conv_length(geom["md"][i]), u.length_uom, ndigits=4)
        _quom(st, "Tvd", u.conv_length(geom["tvd"][i]), u.length_uom, ndigits=4)
        _quom(st, "Incl", geom["inc"][i], u.angle_uom)
        _quom(st, "Azi", geom["azi"][i], u.angle_uom)
        _quom(st, "DispNs", u.conv_length(geom["dispNs"][i]), u.length_uom, ndigits=4)
        _quom(st, "DispEw", u.conv_length(geom["dispEw"][i]), u.length_uom, ndigits=4)
        _quom(st, "Dls", u.conv_dls(geom["dls"][i]), u.dls_uom)
    return root


def _build_141(
    well, u, geom, well_uid, wellbore_uid, trajectory_uid, crs,
    datum_elev, datum_kind, model_version, time_zone,
):
    """WITSML 1.4.1.1 legacy document (``trajectorys`` envelope, lower-camel names)."""
    root = ET.Element("trajectorys")
    root.set("xmlns", WITSML_141_NS)
    root.set("version", "1.4.1.1")

    traj = ET.SubElement(root, "trajectory")
    traj.set("uidWell", well_uid)
    traj.set("uidWellbore", wellbore_uid)
    traj.set("uid", trajectory_uid)
    _text(traj, "nameWell", well.name)
    _text(traj, "nameWellbore", f"{well.name} / WB")
    _text(traj, "name", f"{well.name} planned trajectory")
    _text(traj, "serviceCompany", _company(model_version))
    _text(traj, "wellCRS", crs or "local frame, no CRS")
    _text(traj, "timeZone", time_zone)
    _text(traj, "statusWellbore", well.status)

    datum = ET.SubElement(traj, "mdDatum")
    _text(datum, "datum", datum_kind)
    _quom(datum, "elevation", u.conv_length(datum_elev), u.length_uom, ndigits=4)
    _quom(traj, "mdMn", u.conv_length(geom["md"][0]), u.length_uom, ndigits=4)
    _quom(traj, "mdMx", u.conv_length(geom["md"][-1]), u.length_uom, ndigits=4)
    # Surface location for completeness (uid'd well refs are by attribute above).
    loc = ET.SubElement(traj, "wellLocation")
    _quom(loc, "easting", well.wellhead[0], u.length_uom)
    _quom(loc, "northing", well.wellhead[1], u.length_uom)

    for i in range(geom["md"].shape[0]):
        st = ET.SubElement(traj, "trajectoryStation")
        st.set("uid", f"st_{i}")
        _text(st, "typeTrajStation", "plan")
        _quom(st, "md", u.conv_length(geom["md"][i]), u.length_uom, ndigits=4)
        _quom(st, "tvd", u.conv_length(geom["tvd"][i]), u.length_uom, ndigits=4)
        _quom(st, "incl", geom["inc"][i], u.angle_uom)
        _quom(st, "azi", geom["azi"][i], u.angle_uom)
        _quom(st, "dispNs", u.conv_length(geom["dispNs"][i]), u.length_uom, ndigits=4)
        _quom(st, "dispEw", u.conv_length(geom["dispEw"][i]), u.length_uom, ndigits=4)
        _quom(st, "dls", u.conv_dls(geom["dls"][i]), u.dls_uom)
    return root


def _company(model_version: str | None) -> str:
    if model_version:
        return f"{_SERVICE_COMPANY} | modelVersion={model_version}"
    return _SERVICE_COMPANY


def _text(parent, tag: str, value: str):
    el = ET.SubElement(parent, tag)
    el.text = str(value)
    return el


# ──────────────────────────────────────────────────────────────────────────
# re-import (round-trip reader)
# ──────────────────────────────────────────────────────────────────────────


def parse_witsml_trajectory(xml: str) -> ParsedTrajectory:
    """Re-import a WITSML trajectory document → :class:`ParsedTrajectory` (doc 09 §9.1).

    Auto-detects 2.0 vs 1.4.1.1 from the root tag, reads every quantity **back through its
    ``uom``** into canonical metric (m, dega, °/30 m) — so a field-unit export round-trips to
    the same canonical numbers as the original — and recovers the MD datum + CRS. This is the
    reader half the round-trip test exercises against the writer.
    """
    root = ET.fromstring(xml)
    tag = _local(root.tag)
    if tag == "Trajectorys":
        return _parse_20(root)
    if tag == "trajectorys":
        return _parse_141(root)
    raise ValueError(f"unrecognized WITSML root element {tag!r}")


def _to_metric_length(value: float, uom: str) -> float:
    if uom in ("m", "meter", "metre"):
        return value
    if uom == "ft":
        return value * 0.3048
    raise ValueError(f"unsupported length uom {uom!r}")


def _to_metric_dls(value: float, uom: str) -> float:
    """Any DLS uom → canonical °/30 m."""
    if uom in ("dega/30.m", "deg/30m", "0.deg/30m"):
        return value
    if uom in ("dega/100.ft", "deg/100ft"):
        return value * 30.0 / (100.0 * 0.3048)
    raise ValueError(f"unsupported DLS uom {uom!r}")


def _quom_val(el) -> tuple[float, str]:
    return float(el.text), (el.get("uom") or "")


def _parse_20(root) -> ParsedTrajectory:
    well = _find(root, "Well")
    wb = _find(root, "Wellbore")
    traj = _find(root, "Trajectory")
    datum = _find(traj, "MdDatum")
    md_uom_el = _find(datum, "Elevation")
    md_datum_elev, elev_uom = _quom_val(md_uom_el)
    parsed = ParsedTrajectory(
        well_uid=well.get("uid", ""),
        well_name=_findtext(well, "Name"),
        wellbore_uid=wb.get("uid", ""),
        trajectory_uid=traj.get("uid", ""),
        md_datum_elev_m=_to_metric_length(md_datum_elev, elev_uom),
        md_datum_kind=_findtext(datum, "DatumKind"),
        well_crs=_findtext(traj, "WellCRS") or None,
        version="2.0",
    )
    for st in _findall(traj, "TrajectoryStation"):
        parsed.stations.append(_parse_station(st, "Md", "Incl", "Azi", "Tvd", "DispNs",
                                               "DispEw", "Dls"))
    return parsed


def _parse_141(root) -> ParsedTrajectory:
    traj = _find(root, "trajectory")
    datum = _find(traj, "mdDatum")
    md_datum_elev, elev_uom = _quom_val(_find(datum, "elevation"))
    parsed = ParsedTrajectory(
        well_uid=traj.get("uidWell", ""),
        well_name=_findtext(traj, "nameWell"),
        wellbore_uid=traj.get("uidWellbore", ""),
        trajectory_uid=traj.get("uid", ""),
        md_datum_elev_m=_to_metric_length(md_datum_elev, elev_uom),
        md_datum_kind=_findtext(datum, "datum"),
        well_crs=_findtext(traj, "wellCRS") or None,
        version="1.4.1.1",
    )
    for st in _findall(traj, "trajectoryStation"):
        parsed.stations.append(_parse_station(st, "md", "incl", "azi", "tvd", "dispNs",
                                               "dispEw", "dls"))
    return parsed


def _parse_station(st, md_t, inc_t, azi_t, tvd_t, ns_t, ew_t, dls_t) -> ParsedStation:
    md, md_uom = _quom_val(_find(st, md_t))
    inc, _ = _quom_val(_find(st, inc_t))
    azi, _ = _quom_val(_find(st, azi_t))
    tvd, tvd_uom = _quom_val(_find(st, tvd_t))
    ns, ns_uom = _quom_val(_find(st, ns_t))
    ew, ew_uom = _quom_val(_find(st, ew_t))
    dls, dls_uom = _quom_val(_find(st, dls_t))
    return ParsedStation(
        md=_to_metric_length(md, md_uom),
        inc=inc,  # dega — degrees either way
        azi=azi,
        tvd=_to_metric_length(tvd, tvd_uom),
        dispNs=_to_metric_length(ns, ns_uom),
        dispEw=_to_metric_length(ew, ew_uom),
        dls=_to_metric_dls(dls, dls_uom),
    )


# ──────────────────────────────────────────────────────────────────────────
# structural validation (doc 09 §9.1 — XSD unavailable → structural + note)
# ──────────────────────────────────────────────────────────────────────────

_REQUIRED_STATION_20 = ["Md", "Incl", "Azi", "Tvd", "DispNs", "DispEw", "Dls"]
_REQUIRED_STATION_141 = ["md", "incl", "azi", "tvd", "dispNs", "dispEw", "dls"]


def validate_witsml_trajectory(xml: str) -> WitsmlValidationResult:
    """Validate an emitted WITSML trajectory (doc 09 §9.1).

    Always checks the document is well-formed and **structurally** complete: the required
    Well/Wellbore/Trajectory/TrajectoryStation objects, the required per-station fields, and a
    ``uom`` on every quantity. If an Energistics XSD + ``xmlschema`` validator were installed
    we would additionally run XSD validation; they are not in this environment, so
    ``schema_validated`` is ``False`` and ``note`` records that the check is structural-only.
    """
    errors: list[str] = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        return WitsmlValidationResult(
            well_formed=False, structural_ok=False, schema_validated=False,
            n_stations=0, errors=[f"not well-formed: {e}"],
        )

    tag = _local(root.tag)
    if tag == "Trajectorys":
        version, station_tag, req = "2.0", "TrajectoryStation", _REQUIRED_STATION_20
        traj = _find_opt(root, "Trajectory")
        well_ok = _find_opt(root, "Well") is not None
        wb_ok = _find_opt(root, "Wellbore") is not None
        datum = _find_opt(traj, "MdDatum") if traj is not None else None
    elif tag == "trajectorys":
        version, station_tag, req = "1.4.1.1", "trajectoryStation", _REQUIRED_STATION_141
        traj = _find_opt(root, "trajectory")
        well_ok = traj is not None and _findtext(traj, "nameWell") != ""
        wb_ok = traj is not None and _findtext(traj, "nameWellbore") != ""
        datum = _find_opt(traj, "mdDatum") if traj is not None else None
    else:
        return WitsmlValidationResult(
            well_formed=True, structural_ok=False, schema_validated=False,
            n_stations=0, errors=[f"unexpected root {tag!r}"],
        )

    if not well_ok:
        errors.append("missing Well object / well name")
    if not wb_ok:
        errors.append("missing Wellbore object")
    if traj is None:
        errors.append("missing Trajectory object")
    if datum is None:
        errors.append("missing MD datum")

    stations = _findall(traj, station_tag) if traj is not None else []
    if not stations:
        errors.append("trajectory has no stations")
    for i, st in enumerate(stations):
        for fname in req:
            el = _find_opt(st, fname)
            if el is None:
                errors.append(f"station {i}: missing required field {fname}")
            elif not el.get("uom"):
                errors.append(f"station {i}: field {fname} missing uom")

    structural_ok = not errors
    return WitsmlValidationResult(
        well_formed=True,
        structural_ok=structural_ok,
        schema_validated=False,
        n_stations=len(stations),
        errors=errors,
        note=(
            f"WITSML {version}: structural validation only — Energistics XSD + xmlschema not "
            "installed in this environment (doc 09 §9.1); install them to add full schema "
            "validation."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# ElementTree helpers (namespace-agnostic by local name)
# ──────────────────────────────────────────────────────────────────────────


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find(parent, name: str):
    el = _find_opt(parent, name)
    if el is None:
        raise ValueError(f"missing required element {name!r}")
    return el


def _find_opt(parent, name: str):
    for child in parent:
        if _local(child.tag) == name:
            return child
    return None


def _findall(parent, name: str) -> list:
    return [c for c in parent if _local(c.tag) == name]


def _findtext(parent, name: str) -> str:
    el = _find_opt(parent, name)
    return (el.text or "").strip() if el is not None else ""


def _indent(elem, level: int = 0) -> None:
    """In-place pretty-print indentation (stdlib has no pretty serializer pre-3.9 ``indent``)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (child.tail and child.tail.strip()):
            child.tail = pad
    if level and not (elem.tail and elem.tail.strip()):
        elem.tail = pad
