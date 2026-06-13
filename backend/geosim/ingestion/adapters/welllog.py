"""Well-log LAS adapter (doc 03 §2 row 11, ``welllog``).

Parses a LAS file (``lasio``) into the two doc-02 primitives the well-log row mandates
(doc 03 §2, §3d, §5):

1. a :class:`~geosim.ingestion.base.RawObservation` of ``geometry_kind="wellcurve"`` —
   the immutable measured curves vs measured depth (MD), keyed by their canonical
   ``property_type`` (doc 01 §5: ``RES``→``resistivity``, ``DEN``→``density``,
   ``VP``→``velocity_p``, ``TEMP``→``temperature``; ``GR`` has no canonical key so it
   rides as ``methodData``); and
2. a separate ``wellPath`` :class:`~geosim.ingestion.base.RawFeature` — the borehole
   trajectory, **joined to the curves by ``wellId``** (doc 03 §3d: "there is no well_path
   support kind"). The trajectory comes from a deviation survey
   (``<well>_deviation.csv``: MD, INC, AZI) integrated to Engineering XYZ by
   :func:`geosim.spatial.min_curvature_positions` (doc 01 §4); when no survey is present a
   **vertical-well assumption** warning is emitted and MD=TVD below the wellhead (doc 03
   §5/§6).

Closes the OVERVIEW §8 round-trip against
:class:`geosim.synthgen.forward.WellLogForward`, whose ``<wid>.las`` carries DEPT (TVD),
MD, RES, GR, DEN, VP, TEMP alongside a ``<wid>_deviation.csv``. Coords/units stay native;
the normalizer (doc 03 §3) reprojects + canonicalizes.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

from geosim.spatial import min_curvature_positions

from ..base import (
    IngestWarning,
    ParseResult,
    Provenance,
    RawFeature,
    RawObservation,
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter

__all__ = ["WellLogLasAdapter"]

# LAS curve mnemonic → canonical property_type (doc 01 §5). GR has no canonical key.
_CURVE_TO_PROPERTY: dict[str, str] = {
    "RES": "resistivity",
    "RESISTIVITY": "resistivity",
    "DEN": "density",
    "RHOB": "density",
    "VP": "velocity_p",
    "DT": "velocity_p",
    "TEMP": "temperature",
    "TEMPERATURE": "temperature",
}
# curves carried as methodData (no registry key): gamma proxy etc.
_NON_REGISTRY = {"GR", "DEPT", "MD"}


@adapter
class WellLogLasAdapter:
    """``IngestionAdapter`` for LAS well logs (doc 03 §2 row 11, §3d, §5)."""

    method = "welllog"
    submethod = None
    name = "welllog-las-v1"
    version = "1.0"
    extensions = (".las",)
    media_types = ("text/plain",)
    formats = ["las"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is a LAS file (extension + ``~VERSION`` / ``~V`` section)."""
        low = filename.lower()
        try:
            text = sample.decode("utf-8", errors="replace").upper()
        except Exception:
            return 0.0
        has_version = "~VERSION" in text or "~V " in text or text.lstrip().startswith("~V")
        if low.endswith(".las"):
            return 0.95 if has_version else 0.8
        if has_version and "~CURVE" in text or has_version and "~W" in text:
            return 0.5
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse LAS → ``wellcurve`` observation + ``wellPath`` feature (joined by wellId)."""
        import lasio

        warnings: list[IngestWarning] = []
        text = _read_text(source)
        if text is None:
            return ParseResult(warnings=[IngestWarning(
                "no_data", Severity.HIGH, "LAS file has no bytes", source.filename
            )])
        try:
            las = lasio.read(io.StringIO(text))
        except Exception as exc:
            return ParseResult(warnings=[IngestWarning(
                "bad_las", Severity.HIGH, f"lasio failed to parse LAS: {exc}",
                source.filename,
            )])

        well_id = _well_id(las, source)
        curve_names = list(las.curves.keys())

        # locate MD: prefer an explicit MD curve, else the index/DEPT curve.
        md = _curve(las, "MD")
        if md is None:
            md = np.asarray(las.index, dtype=float)
        n = md.size
        total = int(n)

        values: dict[str, Any] = {}
        units: dict[str, str] = {}
        method_data: dict[str, Any] = {}
        for name in curve_names:
            up = name.upper()
            if up in _NON_REGISTRY and up != "MD":
                # carry non-registry curves (e.g. GR) as methodData (doc 02 §3)
                arr = np.asarray(las[name], dtype=float)
                method_data[up] = {
                    "values": arr.tolist(),
                    "unit": _curve_unit(las, name),
                }
                continue
            prop = _CURVE_TO_PROPERTY.get(up)
            if prop is None:
                continue
            arr = np.asarray(las[name], dtype=float)
            if arr.size != n:
                warnings.append(IngestWarning(
                    "curve_length_mismatch", Severity.LOW,
                    f"curve {name} has {arr.size} samples, expected {n}",
                    f"curve:{name}",
                ))
                continue
            values[prop] = arr
            units[prop] = _curve_unit(las, name)

        if not values:
            warnings.append(IngestWarning(
                "no_known_curves", Severity.MEDIUM,
                f"no registry-mapped curves among {curve_names}", source.filename,
            ))

        # ---- trajectory: deviation survey → Engineering XYZ (min curvature, doc 01 §4) ----
        survey = _load_deviation_survey(source)
        wellhead = _wellhead(las, source)
        if survey is not None:
            mc = min_curvature_positions(survey, (wellhead[0], wellhead[1]),
                                         kb_elev=wellhead[2])
            traj_md = mc.md
            traj_enu = mc.enu  # (N,3) East,North,Up
            traj_kind = "deviation_survey"
        else:
            # vertical-well assumption: MD=TVD straight down from the wellhead (doc 03 §5/§6)
            warnings.append(IngestWarning(
                "no_deviation_survey", Severity.MEDIUM,
                "no deviation survey — assuming a vertical well (MD=TVD)", well_id,
            ))
            traj_md = md
            traj_enu = np.column_stack([
                np.full(n, wellhead[0]),
                np.full(n, wellhead[1]),
                wellhead[2] - md,  # Up decreases with depth
            ])
            traj_kind = "vertical_assumption"

        # curve coords: place each MD sample on the trajectory (Engineering XYZ).
        if traj_md.size >= 2:
            cx = np.interp(md, traj_md, traj_enu[:, 0])
            cy = np.interp(md, traj_md, traj_enu[:, 1])
            cz = np.interp(md, traj_md, traj_enu[:, 2])
        else:
            cx = np.full(n, wellhead[0])
            cy = np.full(n, wellhead[1])
            cz = wellhead[2] - md
        coords = np.column_stack([cx, cy, cz])

        obs = RawObservation(
            geometry_kind="wellcurve",
            coords=coords,
            values=values,
            primary_property=next(iter(values), None),
            meta={
                "wellId": well_id,
                "md": md.tolist(),
                "methodData": method_data,
            },
        )

        # wellPath feature: trajectory polyline, joined to the curves by wellId.
        path_coords = [[float(e), float(n_), float(u)]
                       for e, n_, u in traj_enu]
        well_path = RawFeature(
            feature_type="wellPath",
            geometry={"type": "LineString", "coordinates": path_coords},
            props={
                "wellId": well_id,
                "trajectory": traj_kind,
                "wellhead": list(wellhead),
                "md_total": float(traj_md[-1]) if traj_md.size else 0.0,
            },
            store_format="geojson",
        )

        return ParseResult(
            observations=[obs],
            features=[well_path],
            source=SourceRef(
                crs=source.crs_hint,
                vertical_datum=None,
                horizontal_unit="m",
                z_convention="MD",
            ),
            units=units,
            provenance=Provenance(
                process="ingest:welllog-las-v1",
                params={"wellId": well_id, "curves": list(values.keys()),
                        "trajectory": traj_kind},
            ),
            warnings=warnings,
            records_total=total,
            records_dropped=0,
        )


def _read_text(source: RawSource) -> str | None:
    if source.data is not None:
        return source.data.decode("utf-8", errors="replace")
    if source.path is not None:
        return Path(source.path).read_text(encoding="utf-8", errors="replace")
    return None


def _curve(las: Any, name: str) -> np.ndarray | None:
    for key in las.curves.keys():
        if key.upper() == name.upper():
            return np.asarray(las[key], dtype=float)
    return None


def _curve_unit(las: Any, name: str) -> str:
    try:
        unit = las.curves[name].unit
    except Exception:
        unit = ""
    return unit or "dimensionless"


def _well_id(las: Any, source: RawSource) -> str:
    """Well identity for the curves↔path join (doc 03 §3d). LAS WELL header, else filename."""
    try:
        wid = las.well["WELL"].value
    except Exception:
        wid = None
    if wid:
        return str(wid)
    return Path(source.filename).stem


def _wellhead(las: Any, source: RawSource) -> tuple[float, float, float]:
    """Wellhead (x, y, kb_elev). LAS X/Y/ELEV headers if present, else origin at z=0."""
    def _hdr(key: str, default: float) -> float:
        try:
            v = las.well[key].value
            return float(v)
        except Exception:
            return default

    x = _hdr("XCOORD", _hdr("X", 0.0))
    y = _hdr("YCOORD", _hdr("Y", 0.0))
    elev = _hdr("EKB", _hdr("ELEV", 0.0))
    return (x, y, elev)


def _load_deviation_survey(source: RawSource) -> np.ndarray | None:
    """Find + parse a sibling ``<well>_deviation.csv`` (MD, INC, AZI) → (N,3) array."""
    if not source.path:
        return None
    p = Path(source.path)
    candidates = [
        p.with_name(p.stem + "_deviation.csv"),
        p.parent / f"{p.stem}_deviation.csv",
    ]
    dev_path = next((c for c in candidates if c.exists()), None)
    if dev_path is None:
        return None
    rows = list(csv.reader(io.StringIO(dev_path.read_text(encoding="utf-8"))))
    if not rows:
        return None
    header = [c.strip().upper() for c in rows[0]]
    try:
        i_md = header.index("MD")
        i_inc = header.index("INC")
        i_azi = header.index("AZI")
    except ValueError:
        return None
    out: list[list[float]] = []
    for r in rows[1:]:
        try:
            out.append([float(r[i_md]), float(r[i_inc]), float(r[i_azi])])
        except (ValueError, IndexError):
            continue
    return np.asarray(out, dtype=float) if out else None
