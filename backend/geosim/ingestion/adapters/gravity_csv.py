"""Gravity station CSV adapter (doc 03 §2 row 1) — a worked first-party example.

Parses a columnar gravity survey: ``x, y, z, gravity_anomaly`` (with optional sigma),
emitting one ``points`` :class:`~geosim.ingestion.base.RawObservation` carrying the
``gravity_anomaly`` measurement (doc 03 §2: gravity CSV → ``Observation(points)`` →
stations). Coordinates/units stay native; the normalizer reprojects + canonicalizes
(doc 03 §3). Bad rows are skipped + counted for the >10% partial-file rule (doc 03 §6).

Header convention (case-insensitive, common aliases accepted): an ``x``/``easting``,
``y``/``northing``, optional ``z``/``elev``/``elevation``, a ``gravity``/``gravity_anomaly``/
``ga``/``mgal`` value column, optional ``sigma``/``error`` column. A ``# unit: mGal`` or
``# crs: EPSG:32612`` comment line sets the source unit / CRS (doc 03 §3a/§3b).
"""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np

from ..base import (
    IngestWarning,
    ParseResult,
    Provenance,
    RawObservation,
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter

__all__ = ["GravityCsvAdapter"]

_X_KEYS = {"x", "easting", "east", "lon", "longitude"}
_Y_KEYS = {"y", "northing", "north", "lat", "latitude"}
_Z_KEYS = {"z", "elev", "elevation", "height"}
_G_KEYS = {"gravity", "gravity_anomaly", "ga", "mgal", "g", "anomaly"}
_S_KEYS = {"sigma", "error", "err", "std", "uncertainty"}


@adapter
class GravityCsvAdapter:
    """``IngestionAdapter`` for gravity station CSVs (doc 03 §1, §2)."""

    method = "gravity"
    submethod = None
    name = "gravity-csv-v1"
    version = "1.0"
    extensions = (".csv", ".txt")
    media_types = ("text/csv",)
    formats = ["csv"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is a gravity CSV (cheap header check, doc 03 §7 step 3)."""
        low = filename.lower()
        if not low.endswith((".csv", ".txt")):
            return 0.0
        try:
            text = sample.decode("utf-8", errors="replace")
        except Exception:
            return 0.0
        header = _first_data_line(text)
        if header is None:
            return 0.0
        cols = {c.strip().lower() for c in header.split(",")}
        has_xy = bool(cols & _X_KEYS) and bool(cols & _Y_KEYS)
        has_g = bool(cols & _G_KEYS)
        if has_xy and has_g:
            return 0.9
        if has_xy:
            return 0.4
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse the CSV → one ``points`` observation of ``gravity_anomaly`` (doc 03 §2)."""
        text = (source.data or b"").decode("utf-8", errors="replace")
        meta = _scan_comments(text)
        rows = list(csv.reader(io.StringIO(_strip_comments(text))))
        warnings: list[IngestWarning] = []
        if not rows:
            return ParseResult(warnings=[IngestWarning(
                "empty_file", Severity.HIGH, "no data rows", source.filename
            )])

        header = [c.strip().lower() for c in rows[0]]
        idx = _column_index(header)
        if idx.get("x") is None or idx.get("y") is None or idx.get("g") is None:
            return ParseResult(warnings=[IngestWarning(
                "bad_header", Severity.HIGH,
                f"missing x/y/gravity columns in header {header}", source.filename,
            )])

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        gs: list[float] = []
        sg: list[float] = []
        total = 0
        dropped = 0
        for n, raw in enumerate(rows[1:], start=2):
            total += 1
            try:
                x = float(raw[idx["x"]])
                y = float(raw[idx["y"]])
                g = float(raw[idx["g"]])
                z = float(raw[idx["z"]]) if idx.get("z") is not None else 0.0
                s = float(raw[idx["s"]]) if idx.get("s") is not None else np.nan
            except (ValueError, IndexError):
                dropped += 1
                warnings.append(IngestWarning(
                    "bad_row", Severity.LOW, f"unparseable row {n}", f"row {n}"
                ))
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
            gs.append(g)
            sg.append(s)

        coords = np.column_stack([xs, ys, zs]) if xs else np.zeros((0, 3))
        values = {"gravity_anomaly": np.asarray(gs, dtype=float)}
        sigma: dict[str, Any] = {}
        if idx.get("s") is not None and np.any(np.isfinite(sg)):
            sigma["gravity_anomaly"] = np.asarray(sg, dtype=float)

        obs = RawObservation(
            geometry_kind="points",
            coords=coords,
            values=values,
            sigma=sigma,
            primary_property="gravity_anomaly",
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                crs=meta.get("crs") or source.crs_hint,
                vertical_datum=meta.get("vertical_datum"),
                horizontal_unit=meta.get("horizontal_unit", "m"),
                z_convention=meta.get("z_convention", "elevation_up"),
            ),
            units={"gravity_anomaly": meta.get("unit", "mGal")},
            provenance=Provenance(process="ingest:gravity-csv-v1"),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )


def _first_data_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return None


def _strip_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))


def _scan_comments(text: str) -> dict[str, str]:
    """Pull ``# key: value`` metadata lines (unit/crs/vertical_datum, doc 03 §3)."""
    meta: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#") or ":" not in s:
            continue
        key, _, val = s.lstrip("#").strip().partition(":")
        meta[key.strip().lower()] = val.strip()
    return meta


def _column_index(header: list[str]) -> dict[str, int | None]:
    def find(keys: set[str]) -> int | None:
        for i, h in enumerate(header):
            if h in keys:
                return i
        return None

    return {
        "x": find(_X_KEYS),
        "y": find(_Y_KEYS),
        "z": find(_Z_KEYS),
        "g": find(_G_KEYS),
        "s": find(_S_KEYS),
    }
