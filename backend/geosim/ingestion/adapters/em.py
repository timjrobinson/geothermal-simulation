"""EM ``.xyz`` / Zonge ``.usf`` sounding adapter (doc 03 §2 em row, submethod ``tdem``).

Parses a TDEM/AEM sounding file into ``soundings``
:class:`~geosim.ingestion.base.RawObservation` carrying ``conductivity`` / the raw decay
(doc 03 §2: em raw decay/CDI soundings → ``Observation(soundings)`` → soundings; later
layered/CDI inversion stitches a ``volume``, doc 03 §4). Coordinates/units stay native.

Two input formats are handled (sniffed on header / extension):

- **synthgen ``.xyz``** — the :class:`~geosim.synthgen.forward.em_mt.TDEMForward` writer:
  a one-line header ``STATION X Y TIME_S DEPTH_M APP_COND_S_per_m`` then whitespace-
  separated records, one per decay gate per station. The ``S/m`` apparent conductivity
  maps to the canonical ``conductivity`` key (doc 01 §5). Each station contributes a
  vertical column of apparent-conductivity-vs-depth samples; every sample is one
  observation record at ``(x, y, depth_below_surface)`` so the soundings ride as native
  columns the normalizer places vertically (doc 03 §3d / §4 step 1).

- **real Zonge USF** (Universal Sounding Format, ``.usf``) — the FORGE WalkTEM transient-EM
  soundings. ``//``-prefixed file header (``//USF``, ``//EPSG``, ``//SOUNDINGS``),
  ``/``-prefixed per-sounding metadata (``/SOUNDING_NAME``, ``/LOCATION: easting, northing,
  elev`` in UTM 12N metres, ``/Z_DIRECTION``), then one or more ``TIME, VOLTAGE, QUALITY``
  decay tables. One ``soundings`` observation is emitted per sounding at its UTM ``(x, y, z)``
  carrying the stacked transient (time + voltage gates) in ``meta``; ``SourceRef.crs`` is
  taken from the ``//EPSG`` header so the normalizer reprojects into the project frame.

Bad rows are skipped + counted (doc 03 §6).
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
    extensions = (".xyz", ".usf")
    media_types = ("text/plain",)
    formats = ["xyz"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is an EM sounding file (cheap header check, doc 03 §7)."""
        low = filename.lower()
        text = sample.decode("utf-8", errors="replace")
        # Real Zonge USF transient-EM soundings (FORGE WalkTEM): `.usf` + `//USF` header.
        if text.lstrip()[:5].upper() == "//USF":
            return 0.95
        if low.endswith(".usf"):
            return 0.6
        if not low.endswith(".xyz"):
            return 0.0
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
        """Parse the file → ``soundings`` observation(s) (doc 03 §2). Routes USF vs xyz."""
        text = (source.data or b"").decode("utf-8", errors="replace")
        head = text.lstrip()
        if source.filename.lower().endswith(".usf") or head.startswith("//USF"):
            return self._parse_usf(source, text)
        return self._parse_xyz(source)

    def _parse_xyz(self, source: RawSource) -> ParseResult:
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

    # ───────────────────── real Zonge USF transient-EM soundings ─────────────────────

    def _parse_usf(self, source: RawSource, text: str) -> ParseResult:
        """Parse a Zonge ``.usf`` → one ``soundings`` observation per sounding (doc 03 §2).

        Reads the ``//EPSG`` file CRS, then each ``/SOUNDING_NAME`` block's ``/LOCATION``
        UTM position + ``/Z_DIRECTION``, stacks every ``TIME, VOLTAGE, QUALITY`` decay
        table belonging to that sounding, and emits an observation at the site carrying the
        raw transient (time + voltage gates) in ``meta`` (doc 03 §2: raw decay → soundings).
        """
        warnings: list[IngestWarning] = []
        lines = text.splitlines()

        # ── file-level `//` header → CRS from //EPSG (the project frame, doc 03 §1) ──
        epsg: str | None = None
        for ln in lines:
            s = ln.strip()
            if not s.startswith("//"):
                continue
            key, _, val = s[2:].partition(":")
            if key.strip().upper() == "EPSG":
                code = val.strip()
                if code and code not in {"-1", "0"}:
                    epsg = f"EPSG:{code}"
                break
        crs = source.crs_hint or epsg
        if crs is None:
            # FORGE USF coords are UTM 12N metres even when //EPSG is -1; assume it so the
            # normalizer can still reproject (doc 03 §3a), and flag the missing declaration.
            crs = "EPSG:32612"
            warnings.append(IngestWarning(
                "assumed_crs", Severity.MEDIUM,
                "no valid //EPSG header (or -1); assuming UTM 12N (EPSG:32612) for the "
                "FORGE WalkTEM soundings", source.filename,
            ))

        soundings = _split_usf_soundings(lines)
        if not soundings:
            return ParseResult(warnings=[IngestWarning(
                "no_soundings", Severity.HIGH, "USF file has no /SOUNDING blocks",
                source.filename,
            )])

        observations: list[RawObservation] = []
        z_conv = "depth_below_surface"
        total = 0
        dropped = 0
        for block in soundings:
            tags = _usf_tags(block)
            loc = _usf_location(tags.get("LOCATION"))
            name = tags.get("SOUNDING_NAME") or tags.get("SOUNDING_NUMBER") or source.filename
            z_dir = (tags.get("Z_DIRECTION") or "DOWN").strip().upper()
            # Z_DIRECTION DOWN → station elevation is up-positive metres (UTM 12N MSL);
            # the location's third field is ground elevation, so keep elevation_up.
            z_conv = "elevation_up"
            times, volts, n_gate, n_bad = _usf_decay(block)
            total += n_gate + n_bad
            dropped += n_bad
            if loc is None:
                warnings.append(IngestWarning(
                    "no_location", Severity.HIGH,
                    f"sounding {name!r} has no /LOCATION; skipped", source.filename,
                ))
                continue
            x, y, elev = loc
            coords = np.array([[x, y, elev]], dtype=float)
            obs = RawObservation(
                geometry_kind="soundings",
                coords=coords,
                values={},
                primary_property=None,
                meta={
                    "sounding_name": name,
                    "n_soundings": 1,
                    "z_direction": z_dir,
                    "transient": {
                        "time_s": times.tolist(),
                        "voltage": volts.tolist(),
                        "n_gates": int(times.size),
                    },
                    "voltage_units": tags.get("VOLTAGE_UNITS"),
                    "length_units": tags.get("LENGTH_UNITS"),
                    "array": tags.get("ARRAY"),
                },
            )
            observations.append(obs)

        if not observations:
            return ParseResult(
                warnings=warnings or [IngestWarning(
                    "no_located_soundings", Severity.HIGH,
                    "no USF sounding had a parseable /LOCATION", source.filename,
                )],
                records_total=total,
                records_dropped=dropped,
            )

        return ParseResult(
            observations=observations,
            source=SourceRef(
                crs=crs,
                horizontal_unit="m",
                z_convention=z_conv,
            ),
            units={},
            provenance=Provenance(process="ingest:em-usf-v1"),
            warnings=warnings,
            records_total=max(total, len(observations)),
            records_dropped=dropped,
        )


def _split_usf_soundings(lines: list[str]) -> list[list[str]]:
    """Split USF body into per-sounding blocks, keyed on ``/SOUNDING_NAME`` boundaries."""
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("/SOUNDING_NAME"):
            if cur is not None:
                blocks.append(cur)
            cur = [ln]
        elif cur is not None:
            cur.append(ln)
    if cur is not None:
        blocks.append(cur)
    return blocks


def _usf_tags(block: list[str]) -> dict[str, str]:
    """Collect ``/KEY: value`` metadata tags from a sounding block (first wins)."""
    tags: dict[str, str] = {}
    for ln in block:
        s = ln.strip()
        if not s.startswith("/") or s.startswith("//"):
            continue
        key, sep, val = s[1:].partition(":")
        if sep:
            k = key.strip().upper()
            if k not in tags:
                tags[k] = val.strip()
    return tags


def _usf_location(raw: str | None) -> tuple[float, float, float] | None:
    """Parse ``/LOCATION: easting, northing, elev`` → ``(x, y, z)`` (UTM 12N metres)."""
    if not raw:
        return None
    parts = [p for p in raw.replace(",", " ").split() if p]
    try:
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2]) if len(parts) > 2 else 0.0
    except (ValueError, IndexError):
        return None
    return x, y, z


def _usf_decay(block: list[str]) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Stack every ``TIME, VOLTAGE, QUALITY`` decay row in a sounding block.

    Returns ``(time_s, voltage, n_good, n_bad)``. Data rows are the comma-separated
    numeric rows following a ``TIME, VOLTAGE`` table header; non-numeric / short rows are
    counted as dropped (doc 03 §6).
    """
    times: list[float] = []
    volts: list[float] = []
    n_bad = 0
    in_table = False
    for ln in block:
        s = ln.strip()
        if not s or s.startswith("/"):
            in_table = False
            continue
        up = s.upper()
        if up.startswith("TIME") and "VOLTAGE" in up:
            in_table = True
            continue
        if not in_table:
            continue
        parts = [p for p in s.replace(",", " ").split() if p]
        try:
            t = float(parts[0])
            v = float(parts[1])
        except (ValueError, IndexError):
            n_bad += 1
            continue
        times.append(t)
        volts.append(v)
    return (
        np.asarray(times, dtype=float),
        np.asarray(volts, dtype=float),
        len(times),
        n_bad,
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
