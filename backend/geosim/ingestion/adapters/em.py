"""EM ``.xyz`` sounding adapter (doc 03 §2 em row, submethod ``tdem``).

Parses a TDEM/AEM ``.xyz`` sounding file into one ``soundings``
:class:`~geosim.ingestion.base.RawObservation` carrying ``conductivity`` (doc 03 §2:
em raw decay/CDI soundings → ``Observation(soundings)`` → soundings; later layered/CDI
inversion stitches a ``volume``, doc 03 §4). Each station contributes a vertical column
of apparent-conductivity-vs-depth samples; every sample is one observation record at
``(x, y, depth_below_surface)`` so the soundings ride as native columns the normalizer
places vertically (doc 03 §3d / §4 step 1). Coordinates/units stay native.

Targets the synthgen :class:`~geosim.synthgen.forward.em_mt.TDEMForward` ``.xyz`` writer:
a one-line header ``STATION X Y TIME_S DEPTH_M APP_COND_S_per_m`` then whitespace-separated
records, one per decay gate per station. The ``S/m`` apparent conductivity maps to the
canonical ``conductivity`` key (doc 01 §5). Bad rows are skipped + counted (doc 03 §6).
"""

from __future__ import annotations

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

__all__ = ["EmXyzAdapter"]

_REQUIRED_COLS = ("station", "x", "y", "app_cond")
# header-token aliases → canonical column role
_COL_ALIASES: dict[str, str] = {
    "station": "station", "sounding": "station", "id": "station",
    "x": "x", "easting": "x", "east": "x",
    "y": "y", "northing": "y", "north": "y",
    "time_s": "time", "time": "time", "t": "time",
    "depth_m": "depth", "depth": "depth", "z": "depth",
    "app_cond_s_per_m": "app_cond", "app_cond": "app_cond",
    "conductivity": "app_cond", "cond": "app_cond", "sigma_a": "app_cond",
}


@adapter
class EmXyzAdapter:
    """``IngestionAdapter`` for EM/TDEM ``.xyz`` conductivity soundings (doc 03 §1, §2)."""

    method = "em"
    submethod = "tdem"
    name = "em-xyz-v1"
    version = "1.0"
    extensions = (".xyz",)
    media_types = ("text/plain",)
    formats = ["xyz"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is an EM ``.xyz`` sounding file (cheap header check, doc 03 §7)."""
        if not filename.lower().endswith(".xyz"):
            return 0.0
        text = sample.decode("utf-8", errors="replace")
        header = _first_nonempty(text)
        if header is None:
            return 0.0
        toks = {t.strip().lower() for t in header.split()}
        roles = {_COL_ALIASES[t] for t in toks if t in _COL_ALIASES}
        has_xy = "x" in roles and "y" in roles
        has_cond = "app_cond" in roles
        if has_xy and has_cond:
            return 0.9
        if has_xy and "station" in roles:
            return 0.4
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse the ``.xyz`` → one ``soundings`` observation of conductivity (doc 03 §2)."""
        text = (source.data or b"").decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        warnings: list[IngestWarning] = []
        if not lines:
            return ParseResult(warnings=[IngestWarning(
                "empty_file", Severity.HIGH, "no data in .xyz", source.filename,
            )])

        header = lines[0].split()
        idx = _column_index(header)
        missing = [c for c in _REQUIRED_COLS if idx.get(c) is None]
        if missing:
            return ParseResult(warnings=[IngestWarning(
                "bad_header", Severity.HIGH,
                f"missing columns {missing} in header {header}", source.filename,
            )])

        xs: list[float] = []
        ys: list[float] = []
        depths: list[float] = []
        conds: list[float] = []
        stations: list[float] = []
        total = 0
        dropped = 0
        for n, line in enumerate(lines[1:], start=2):
            total += 1
            parts = line.split()
            try:
                station = float(parts[idx["station"]])
                x = float(parts[idx["x"]])
                y = float(parts[idx["y"]])
                cond = float(parts[idx["app_cond"]])
                depth = float(parts[idx["depth"]]) if idx.get("depth") is not None else 0.0
            except (ValueError, IndexError):
                dropped += 1
                warnings.append(IngestWarning(
                    "bad_row", Severity.LOW, f"unparseable row {n}", f"row {n}",
                ))
                continue
            stations.append(station)
            xs.append(x)
            ys.append(y)
            depths.append(depth)
            conds.append(cond)

        # one record per (station, depth) sample; Z carried as depth_below_surface
        coords = np.column_stack([xs, ys, depths]) if xs else np.zeros((0, 3))
        obs = RawObservation(
            geometry_kind="soundings",
            coords=coords,
            values={"conductivity": np.asarray(conds, dtype=float)},
            primary_property="conductivity",
            meta={
                "station_id": np.asarray(stations, dtype=float).tolist(),
                "n_soundings": int(len(set(stations))),
            },
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                crs=source.crs_hint,
                z_convention="depth_below_surface",
            ),
            units={"conductivity": "S/m"},
            provenance=Provenance(process="ingest:em-xyz-v1"),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )


def _first_nonempty(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line
    return None


def _column_index(header: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, tok in enumerate(header):
        role = _COL_ALIASES.get(tok.strip().lower())
        if role is not None and role not in out:
            out[role] = i
    return out
