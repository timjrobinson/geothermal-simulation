"""AGI SuperSting ``.stg`` IP adapter (doc 03 §2 ip row, submethod ``ip_time``/``ip_freq``).

Parses an induced-polarization **apparent-chargeability / phase pseudosection** (acquired
paired with ERT) into one ``profile2d`` :class:`~geosim.ingestion.base.RawObservation`
(doc 03 §2: ip raw → ``Observation(profile2d)`` → pseudosection). The IP property keys are
**split** per doc 01 §5 / doc 03 §2: time-domain integral chargeability →
``chargeability_time_ms``, the mV/V apparent chargeability → ``chargeability_mv_v``,
frequency-domain phase → ``phase_mrad``. The reader picks the key from the ``.stg``
``value:`` header label so the canonical unit + colormap resolve automatically.

Shares the ``.stg`` byte parser with the ERT adapter (:mod:`._stg`); only the apparent
quantity differs (doc 03 §9 one-file-per-format). Targets the synthgen
:class:`~geosim.synthgen.forward.electrical.IPForward` ``.stg`` writer
(``value: apparent_chargeability_mv_v``). Bad rows are skipped + counted (doc 03 §6).
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
from ._stg import STG_COLUMNS, parse_stg, value_property_and_unit

__all__ = ["IpStgAdapter"]

_IP_PROPS = {"chargeability_time_ms", "chargeability_mv_v", "phase_mrad"}
# which submethod a given IP property implies (doc 02 §2)
_PROP_SUBMETHOD = {
    "chargeability_time_ms": "ip_time",
    "chargeability_mv_v": "ip_time",
    "phase_mrad": "ip_freq",
}


@adapter
class IpStgAdapter:
    """``IngestionAdapter`` for AGI SuperSting ``.stg`` IP pseudosections (doc 03 §1, §2)."""

    method = "ip"
    submethod = "ip_time"
    name = "ip-stg-v1"
    version = "1.0"
    extensions = (".stg",)
    media_types = ("text/plain",)
    formats = ["stg"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is an IP ``.stg`` pseudosection (cheap header check, doc 03 §7)."""
        if not filename.lower().endswith(".stg"):
            return 0.0
        low = sample.decode("utf-8", errors="replace").lower()
        has_cols = all(c.lower() in low for c in ("a_x", "m_x", "pseudodepth", "value"))
        if not has_cols:
            return 0.0
        if "charge" in low or "phase" in low or "mv_v" in low:
            return 0.95
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse the ``.stg`` → one ``profile2d`` IP observation (split keys, doc 03 §2)."""
        text = (source.data or b"").decode("utf-8", errors="replace")
        rows, label, warnings, total, dropped = parse_stg(text, source.filename)
        prop, unit = value_property_and_unit(label, default_prop="chargeability_mv_v")
        if prop not in _IP_PROPS:
            warnings.append(IngestWarning(
                "unexpected_quantity", Severity.MEDIUM,
                f"ip .stg declares {label!r}; not an IP quantity", f"property:{prop}",
            ))

        if not rows:
            return ParseResult(warnings=warnings or [IngestWarning(
                "empty_file", Severity.HIGH, "no .stg data records", source.filename,
            )])

        arr = np.asarray(rows, dtype=float)
        cols = {c: arr[:, i] for i, c in enumerate(STG_COLUMNS)}
        midx = 0.25 * (cols["a_x"] + cols["b_x"] + cols["m_x"] + cols["n_x"])
        midy = 0.25 * (cols["a_y"] + cols["b_y"] + cols["m_y"] + cols["n_y"])
        coords = np.column_stack([midx, midy, cols["pseudodepth"]])

        obs = RawObservation(
            geometry_kind="profile2d",
            coords=coords,
            values={prop: cols["value"]},
            primary_property=prop,
            meta={
                "array": "dipole-dipole",
                "submethod": _PROP_SUBMETHOD.get(prop, "ip_time"),
            },
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                crs=source.crs_hint,
                z_convention="depth_below_surface",
            ),
            units={prop: unit},
            provenance=Provenance(process="ingest:ip-stg-v1"),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )
