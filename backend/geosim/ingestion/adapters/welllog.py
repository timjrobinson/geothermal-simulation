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
import re
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

# ── real Schlumberger LAS curve families (doc 03 §2 welllog row, real-format branch) ──
# Picked deterministically by *ordered* preference per property: real logs carry several
# candidate curves at once (e.g. DTCO compressional + DTSM shear), so the first present
# canonical curve wins. Slowness curves (us/ft) are inverted to velocity_p (m/s) here.
_REAL_TEMPERATURE_PREF = ("GTEM", "CTEM", "TEMP", "WTEP")  # borehole / cartridge temp (degF)
_REAL_VELOCITY_PREF = ("DTCO", "DTC", "DT", "DTCOMP")       # compressional slowness (us/ft)
_REAL_DENSITY_PREF = ("RHOZ", "RHOB", "RHOM", "DEN")        # bulk density (g/cm3)
_REAL_RESISTIVITY_PREF = (                                  # deep-reading resistivity (ohm.m)
    "AT90", "AT60", "RLA5", "RLA4", "RT", "RD", "ILD", "RES", "RESISTIVITY",
)
# real gamma curves (no registry key) carried as methodData like synthetic GR.
_REAL_GAMMA = ("GR_EDTC", "HCGR", "HSGR", "GR", "ECGR", "SGR")


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

        # Real-format branch (doc 03 §2 welllog row): a vendor LAS that carries a geographic
        # wellhead (LATI/LONG in its ~Well header) is an MD-indexed log in FEET with no
        # deviation survey and no source CRS. We place the well from LATI/LONG (EPSG:4326)
        # and convert depths ft→m, instead of hard-failing on the missing CRS. The synthetic
        # round-trip (Engineering-frame LAS + sibling _deviation.csv) keeps the path below.
        lonlat = _wellhead_lonlat(las)
        if lonlat is not None:
            return self._parse_real(source, las, well_id, curve_names, lonlat, warnings)

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

    # ───────────────── real Schlumberger LAS branch (doc 03 §2 welllog row) ─────────────────

    def _parse_real(
        self,
        source: RawSource,
        las: Any,
        well_id: str,
        curve_names: list[str],
        lonlat: tuple[float, float],
        warnings: list[IngestWarning],
    ) -> ParseResult:
        """Parse a real MD-indexed (feet) vendor LAS placed by its LATI/LONG wellhead.

        Coords stay native — the wellhead lon/lat (EPSG:4326) with the MD samples hung
        vertically below it at *elevation in metres*; ``SourceRef.crs=EPSG:4326`` so the
        normalizer reprojects into the project frame (no source-CRS hard fail). Depths are
        FEET (STRT/STOP/STEP ``.F`` / the DEPT|MD curve unit ``F``) → metres. Slowness
        curves (us/ft) invert to ``velocity_p`` (m/s); temperature (degF) → kelvin via the
        declared unit; resistivity / density map by their canonical families.
        """
        lon, lat = lonlat

        # ── MD axis in feet → metres (units registry) ──
        md_name = _index_name(las)
        md_ft = _curve(las, md_name)
        if md_ft is None:
            md_ft = np.asarray(las.index, dtype=float)
        md_unit = _curve_unit(las, md_name)
        md_m = _to_metres(md_ft, md_unit)
        n = md_m.size
        total = int(n)

        # ── wellhead elevation (Elevation of Kelly Bushing, feet) → metres ──
        elev_ft, elev_unit = _wellhead_elev(las)
        wh_elev_m = _to_metres(np.asarray([elev_ft]), elev_unit)[0] if elev_ft is not None else 0.0
        if elev_ft is None:
            warnings.append(IngestWarning(
                "no_wellhead_elevation", Severity.LOW,
                "LAS has no EKB/elevation header; wellhead placed at elevation 0", well_id,
            ))

        # ── map registry curves by ordered family preference (real Schlumberger mnemonics) ──
        upper = {c.upper(): c for c in curve_names}
        values: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        for prop, pref in (
            ("temperature", _REAL_TEMPERATURE_PREF),
            ("velocity_p", _REAL_VELOCITY_PREF),
            ("density", _REAL_DENSITY_PREF),
            ("resistivity", _REAL_RESISTIVITY_PREF),
        ):
            name = next((upper[k] for k in pref if k in upper), None)
            if name is None:
                continue
            arr = np.asarray(las[name], dtype=float)
            if arr.size != n:
                continue
            unit = _curve_unit(las, name)
            if prop == "velocity_p" and _is_slowness(unit):
                # slowness (us/ft) → velocity (m/s): v = 1 / slowness.
                arr, unit = _slowness_to_velocity(arr, unit)
            values[prop] = arr
            units[prop] = _normalize_curve_unit(prop, unit)

        # ── gamma (no registry key) carried as methodData, like synthetic GR ──
        method_data: dict[str, Any] = {}
        gname = next((upper[k] for k in _REAL_GAMMA if k in upper), None)
        if gname is not None:
            garr = np.asarray(las[gname], dtype=float)
            if garr.size == n:
                method_data["GR"] = {
                    "values": garr.tolist(),
                    "unit": _curve_unit(las, gname),
                    "mnemonic": gname,
                }

        if not values:
            warnings.append(IngestWarning(
                "no_known_curves", Severity.MEDIUM,
                f"no registry-mapped curves among {curve_names}", source.filename,
            ))

        # ── no deviation survey in the real vendor LAS → vertical-well assumption ──
        warnings.append(IngestWarning(
            "no_deviation_survey", Severity.MEDIUM,
            "no deviation survey — assuming a vertical well (MD=TVD below the wellhead)",
            well_id,
        ))
        # native coords: lon/lat wellhead, elevation (m) decreasing with MD. The normalizer
        # reprojects EPSG:4326 → project CRS and drapes z into Engineering (z_convention up).
        cz = wh_elev_m - md_m
        coords = np.column_stack([np.full(n, lon), np.full(n, lat), cz])

        obs = RawObservation(
            geometry_kind="wellcurve",
            coords=coords,
            values=values,
            primary_property=next(iter(values), None),
            meta={
                "wellId": well_id,
                "md": md_m.tolist(),
                "md_unit": "m",
                "methodData": method_data,
            },
        )

        # wellPath feature at the LATI/LONG wellhead (lon/lat; normalizer reprojects).
        path_coords = [
            [float(lon), float(lat), float(wh_elev_m)],
            [float(lon), float(lat), float(wh_elev_m - md_m[-1])] if n else
            [float(lon), float(lat), float(wh_elev_m)],
        ]
        well_path = RawFeature(
            feature_type="wellPath",
            geometry={"type": "LineString", "coordinates": path_coords},
            props={
                "wellId": well_id,
                "trajectory": "vertical_assumption",
                "wellhead": [float(lon), float(lat), float(wh_elev_m)],
                "md_total": float(md_m[-1]) if n else 0.0,
            },
            store_format="geojson",
        )

        return ParseResult(
            observations=[obs],
            features=[well_path],
            source=SourceRef(
                # LATI/LONG wellhead → normalizer reprojects (no source-CRS hard fail).
                crs="EPSG:4326",
                vertical_datum=None,
                horizontal_unit="deg",
                # z is already resolved to metres elevation below the wellhead.
                z_convention="elevation_up",
            ),
            units=units,
            provenance=Provenance(
                process="ingest:welllog-las-v1",
                params={"wellId": well_id, "curves": list(values.keys()),
                        "trajectory": "vertical_assumption", "format": "real_las",
                        "wellhead_lonlat": [float(lon), float(lat)]},
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


# ── real-LAS helpers (LATI/LONG wellhead, feet depths, slowness, units) ──

# Foot-family + metre depth/elevation units seen in vendor LAS headers (~Well STRT/EKB,
# DEPT/MD curve unit). LAS is overwhelmingly feet ('F' / 'ft'); accept metres too.
_FEET_UNITS = {"f", "ft", "feet", "foot"}
_METRE_UNITS = {"m", "metre", "meter", "metres", "meters"}
# slowness units (Δt) that must be inverted to velocity_p (m/s).
_SLOWNESS_UNITS = {"us/ft", "usec/ft", "us/f", "us/m", "usec/m"}


def _index_name(las: Any) -> str:
    """Name of the MD/DEPT index curve (first curve), for unit + array lookup."""
    try:
        return las.curves[0].mnemonic
    except Exception:
        return "DEPT"


def _to_metres(arr: np.ndarray, unit: str) -> np.ndarray:
    """Depth/elevation array → metres. Feet (LAS default) → m; metres pass through."""
    u = (unit or "").strip().lower()
    if u in _FEET_UNITS:
        return np.asarray(arr, dtype=float) * 0.3048
    if u in _METRE_UNITS or u in ("", "dimensionless"):
        return np.asarray(arr, dtype=float)
    # unknown depth unit: assume feet (the LAS norm) rather than silently mixing units.
    return np.asarray(arr, dtype=float) * 0.3048


def _is_slowness(unit: str) -> bool:
    return (unit or "").strip().lower().replace(" ", "") in _SLOWNESS_UNITS


def _slowness_to_velocity(arr: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    """Interval slowness Δt → velocity_p (m/s): v = 1/Δt with the foot/metre factor.

    us/ft: v[m/s] = 0.3048 / (Δt·1e-6). us/m: v[m/s] = 1 / (Δt·1e-6). NULLs/zeros → NaN.
    """
    u = (unit or "").strip().lower().replace(" ", "")
    a = np.asarray(arr, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if u in ("us/m", "usec/m"):
            v = 1.0 / (a * 1e-6)
        else:  # us/ft family
            v = 0.3048 / (a * 1e-6)
    v[~np.isfinite(v)] = np.nan
    return v, "m/s"


def _normalize_curve_unit(prop: str, unit: str) -> str:
    """Map a vendor curve unit string to one the registry/pint understands (doc 01 §5).

    Temperature is declared so the normalizer converts to kelvin (degF→K); resistivity
    ohm.m spellings are kept (the normalizer aliases them); density g/cm3 → registry.
    """
    u = (unit or "").strip()
    if prop == "temperature":
        low = u.lower()
        if low in ("degf", "f", "deg f", "fahrenheit"):
            return "degF"
        if low in ("degc", "c", "deg c", "celsius"):
            return "degC"
        if low in ("k", "kelvin"):
            return "kelvin"
        return "degF"  # vendor borehole-temp curves are Fahrenheit
    if prop == "density":
        low = u.lower().replace(" ", "")
        if low in ("g/cm3", "g/cc", "gm/cc", "g/cm**3"):
            return "g/cm**3"
        return u or "kg/m**3"
    if prop == "velocity_p":
        return u or "m/s"
    if prop == "resistivity":
        return u or "ohm*m"
    return u or "dimensionless"


def _wellhead_lonlat(las: Any) -> tuple[float, float] | None:
    """Wellhead (lon, lat) decimal degrees from the ~Well LATI/LONG headers, else None.

    Handles both real vendor encodings: DMS with degree/minute/second glyphs + N/S/E/W
    (e.g. ``38° 30' 14.447" N`` / ``112° 53' 47.066" W``) and plain decimal degrees
    (e.g. ``38.500562 degrees`` / ``-112.88703 degrees``). None when absent/unparseable
    (keeps the synthetic Engineering-frame path, which carries no LATI/LONG, on its branch).
    """
    lat = _parse_latlon(_well_header(las, ("LATI", "LAT", "SLAT")), is_lat=True)
    lon = _parse_latlon(_well_header(las, ("LONG", "LON", "SLON")), is_lat=False)
    if lat is None or lon is None:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return (lon, lat)


def _well_header(las: Any, keys: tuple[str, ...]) -> str | None:
    """Raw value string of the first present ~Well header among ``keys``."""
    for k in keys:
        try:
            item = las.well[k]
        except Exception:
            continue
        # lasio stores the parsed value; fall back to descr for headers it mangles.
        for attr in (item.value, getattr(item, "descr", None)):
            if attr is None:
                continue
            s = str(attr).strip()
            if s:
                return s
    return None


def _parse_latlon(raw: str | None, *, is_lat: bool) -> float | None:
    """Parse a LATI/LONG header string (DMS or decimal degrees) → signed decimal degrees."""
    if not raw:
        return None
    s = raw.strip()
    # plain decimal degrees, e.g. '-112.88703 degrees' or '38.500562'
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    has_dms = bool(re.search(r"[°º'′\"″]", s)) or \
        bool(re.search(r"\d+\s+\d+\s+\d", s))
    # Hemisphere is a *standalone* N/S/E/W token (not a letter inside a word like
    # "degrees"): require it bounded by start/space/punct on both sides.
    hemi_m = re.search(r"(?:^|[\s,;])([NSEWnsew])(?:$|[\s,;.])", s)
    hemi = hemi_m.group(1).upper() if hemi_m else None

    if not has_dms and m is not None:
        val = float(m.group(0))
        if hemi in ("S", "W"):
            val = -abs(val)
        return val

    # DMS: split degrees/minutes/seconds tolerating the glyph variants (and spaces).
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if not nums:
        return None
    deg = float(nums[0])
    minutes = float(nums[1]) if len(nums) > 1 else 0.0
    seconds = float(nums[2]) if len(nums) > 2 else 0.0
    val = abs(deg) + minutes / 60.0 + seconds / 3600.0
    if hemi in ("S", "W") or deg < 0:
        val = -val
    return val


def _wellhead_elev(las: Any) -> tuple[float | None, str]:
    """Wellhead elevation value + unit (Elevation of Kelly Bushing, feet) from ~Well/~Param."""
    for key in ("EKB", "EDF", "ELEV", "EGL", "KB", "APD"):
        try:
            item = las.well[key]
        except Exception:
            try:
                item = las.params[key]
            except Exception:
                continue
        try:
            val = float(item.value)
        except (TypeError, ValueError):
            continue
        return val, (item.unit or "F")
    return None, "F"


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
