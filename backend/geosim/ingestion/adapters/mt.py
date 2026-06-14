"""MT EDI adapter (doc 03 §2 mt row).

Parses a magnetotelluric **EDI** file (SEG/EMAP impedance-tensor exchange format) into one
``tensor`` :class:`~geosim.ingestion.base.RawObservation` carrying the apparent-resistivity
and phase sounding curves at the site (doc 03 §2: mt raw → ``Observation(tensor)`` →
sites; later inversion yields a resistivity ``volume``). The two curves are split into
canonical doc-01 keys: ``resistivity`` (apparent ρ, Ω·m) and ``phase_mrad`` (phase). Each
period sample is one record at the site ``(x, y, 0)`` so coords + value columns stay
aligned and the bbox lands on the station; the period/frequency axis rides in ``meta``.
Coordinates/units stay native — the normalizer reprojects + canonicalizes (doc 03 §3),
turning the EDI's degrees into canonical milliradians.

Targets the synthgen :class:`~geosim.synthgen.forward.em_mt.MTForward` ``write_edi``
writer: ``>HEAD`` (DATAID), ``>=DEFINEMEAS`` (``REFLOC=x,y``), ``>=MTSECT`` (NFREQ) then
``>FREQ`` / ``>RHOXY`` (apparent ρ Ω·m) / ``>PHSXY`` (phase °) numeric blocks. Missing
blocks degrade gracefully with a structured warning (doc 03 §6).
"""

from __future__ import annotations

import re

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

__all__ = ["MtEdiAdapter"]


@adapter
class MtEdiAdapter:
    """``IngestionAdapter`` for magnetotelluric EDI files (doc 03 §1, §2)."""

    method = "mt"
    submethod = None
    name = "mt-edi-v1"
    version = "1.0"
    extensions = (".edi",)
    media_types = ("text/plain",)
    formats = ["edi"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is an MT EDI file (cheap header check, doc 03 §7 step 3)."""
        if not filename.lower().endswith(".edi"):
            return 0.0
        low = sample.decode("utf-8", errors="replace").lower()
        if ">head" in low and (">=mtsect" in low or ">freq" in low):
            return 0.95
        if ">head" in low or ">=definemeas" in low:
            return 0.5
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse the EDI → one ``tensor`` site observation of app-ρ + phase (doc 03 §2)."""
        text = (source.data or b"").decode("utf-8", errors="replace")
        warnings: list[IngestWarning] = []

        dataid = _scalar(text, "DATAID") or source.filename
        refloc = _refloc(text)
        freq = _data_block(text, "FREQ")
        rho = _data_block(text, "RHOXY")
        phase = _data_block(text, "PHSXY")

        # Real EDIs often store only the impedance tensor; compute apparent resistivity +
        # phase from >ZXYR/>ZXYI when no precomputed >RHOXY block is present (Cagniard).
        if rho.size == 0:
            rho, phase = _impedance_app_res(text, freq)
            if rho.size == 0:
                return ParseResult(warnings=[IngestWarning(
                    "no_apparent_resistivity", Severity.HIGH,
                    "EDI has neither a >RHOXY block nor a >ZXYR/>ZXYI impedance tensor",
                    source.filename,
                )])
            warnings.append(IngestWarning(
                "computed_app_res", Severity.LOW,
                "apparent resistivity + phase computed from the impedance tensor "
                "(Cagniard); no >RHOXY block present", source.filename,
            ))
        if phase.size == 0:
            warnings.append(IngestWarning(
                "no_phase", Severity.MEDIUM, "EDI has no >PHSXY phase block",
                source.filename,
            ))

        n = rho.size
        # Site location: REFLOC=x,y (synthgen) or REFLAT/REFLONG lat/lon (real EDIs).
        geographic = False
        if refloc is not None:
            sx, sy = refloc
        else:
            lonlat = _reflatlon(text)
            if lonlat is not None:
                sx, sy = lonlat
                geographic = True
            else:
                sx, sy = 0.0, 0.0
                warnings.append(IngestWarning(
                    "no_location", Severity.HIGH,
                    "EDI has no REFLOC or REFLAT/REFLONG; site placed at origin",
                    source.filename,
                ))
        coords = np.column_stack([np.full(n, sx), np.full(n, sy), np.zeros(n)])

        values: dict[str, np.ndarray] = {"resistivity": rho}
        units = {"resistivity": "ohm*m"}
        if phase.size == n:
            values["phase_mrad"] = phase
            units["phase_mrad"] = "deg"  # EDI phase is degrees → mrad (normalizer)

        obs = RawObservation(
            geometry_kind="tensor",
            coords=coords,
            values=values,
            primary_property="resistivity",
            meta={
                "station": dataid,
                "frequency_hz": freq.tolist() if freq.size else None,
                "component": "xy",
            },
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                crs=source.crs_hint or ("EPSG:4326" if geographic else None),
                horizontal_unit="deg" if geographic else "m",
                z_convention="elevation_up",
            ),
            units=units,
            provenance=Provenance(process="ingest:mt-edi-v1"),
            warnings=warnings,
            records_total=n,
            records_dropped=0,
        )


def _scalar(text: str, key: str) -> str | None:
    """Pull a ``KEY=value`` scalar from an EDI ``>HEAD``/info block."""
    needle = key.upper() + "="
    for line in text.splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith(needle):
            return s.split("=", 1)[1].strip()
    return None


def _refloc(text: str) -> tuple[float, float] | None:
    """Parse ``REFLOC=x,y`` (synthgen writer) → ``(x, y)`` site plan position."""
    raw = _scalar(text, "REFLOC")
    if raw is None:
        return None
    parts = raw.replace(",", " ").split()
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None


def _impedance_app_res(text: str, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apparent resistivity (Ω·m) + phase (deg) from the off-diagonal impedance Z=Zxy when
    no precomputed ``>RHOXY`` block exists. Cagniard relation, EDI field units
    ``[(mV/km)/nT]``: ``ρ_a = 0.2·T·|Z|²`` with period ``T = 1/f``; ``φ = atan2(Im Z, Re Z)``.
    EDI no-data markers (≈1e32) are masked to NaN."""
    zr = _data_block(text, "ZXYR")
    zi = _data_block(text, "ZXYI")
    n = min(zr.size, zi.size, freq.size)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    zr, zi, f = zr[:n], zi[:n], freq[:n]
    bad = (np.abs(zr) > 1e30) | (np.abs(zi) > 1e30) | (f <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 0.2 * (1.0 / f) * (zr * zr + zi * zi)
        phase = np.degrees(np.arctan2(zi, zr))
    rho[bad] = np.nan
    phase[bad] = np.nan
    return rho, phase


def _reflatlon(text: str) -> tuple[float, float] | None:
    """Parse ``REFLAT``/``REFLONG`` (or ``LAT``/``LONG``) — DMS ``±DD:MM:SS.s`` or decimal —
    into ``(lon, lat)`` decimal degrees."""
    def dec(s: str | None) -> float | None:
        if s is None:
            return None
        m = re.match(r"([+-]?)(\d+):(\d+):([\d.]+)", s.strip())
        if not m:
            try:
                return float(s)
            except ValueError:
                return None
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (int(m.group(2)) + int(m.group(3)) / 60 + float(m.group(4)) / 3600)

    lat = dec(_scalar(text, "REFLAT") or _scalar(text, "LAT"))
    lon = dec(_scalar(text, "REFLONG") or _scalar(text, "REFLON") or _scalar(text, "LONG"))
    if lat is None or lon is None:
        return None
    return lon, lat


def _data_block(text: str, tag: str) -> np.ndarray:
    """Read an EDI ``>TAG ... // N`` numeric data block as a float array (doc 03 §2).

    Collects the free-form numeric lines after the ``>TAG`` header up to the next ``>``
    directive (EDI blocks are whitespace-separated reals, optionally 5 per line).
    """
    lines = text.splitlines()
    start: int | None = None
    needle = ">" + tag.upper()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith(needle):
            start = i
            break
    if start is None:
        return np.zeros(0)
    vals: list[float] = []
    for line in lines[start + 1:]:
        s = line.strip()
        if s.startswith(">"):
            break
        if not s:
            continue
        for tok in s.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    return np.asarray(vals, dtype=float)
