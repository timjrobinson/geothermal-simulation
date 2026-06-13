"""AGI SuperSting ``.stg`` ERT adapter (doc 03 §2 ert row, submethod ``dc_resistivity``).

Parses a DC-resistivity **apparent-resistivity pseudosection** into one ``profile2d``
:class:`~geosim.ingestion.base.RawObservation` carrying ``resistivity`` (the apparent ρ,
doc 03 §2: ert raw → ``Observation(profile2d)`` → pseudosection). Each record is one
dipole-dipole quadrupole ``A,B,M,N`` with its electrode XY positions and an apparent
value; the observation point is the quadrupole midpoint placed at the pseudodepth below
surface (``z_convention="depth_below_surface"``). Coordinates/units stay native — the
normalizer reprojects + canonicalizes (doc 03 §3).

The reader targets the synthgen :class:`~geosim.synthgen.forward.electrical.ERTForward`
``.stg`` writer (an AGI-style column layout): a few header lines, a column header
``Idx,A_x,A_y,B_x,B_y,M_x,M_y,N_x,N_y,pseudodepth,value`` and one comma-separated record
per measurement, with a ``value: <label>`` header line declaring the apparent quantity
(``apparent_resistivity_ohm_m``). Bad rows are skipped + counted for the >10% rule
(doc 03 §6).
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
from ._stg import (
    STG_COLUMNS,
    parse_stg,
    value_property_and_unit,
)

__all__ = ["ErtStgAdapter"]


@adapter
class ErtStgAdapter:
    """``IngestionAdapter`` for AGI SuperSting ``.stg`` ERT pseudosections (doc 03 §1, §2)."""

    method = "ert"
    submethod = "dc_resistivity"
    name = "ert-stg-v1"
    version = "1.0"
    extensions = (".stg",)
    media_types = ("text/plain",)
    formats = ["stg"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is an ERT ``.stg`` pseudosection (cheap header check, doc 03 §7)."""
        if not filename.lower().endswith(".stg"):
            return 0.0
        text = sample.decode("utf-8", errors="replace")
        low = text.lower()
        has_cols = all(c.lower() in low for c in ("a_x", "m_x", "pseudodepth", "value"))
        if not has_cols:
            return 0.0
        # disambiguate ERT vs IP by the declared apparent quantity (doc 03 §2)
        if "resistiv" in low and "charge" not in low:
            return 0.95
        if "charge" in low:
            return 0.2  # an IP .stg — let the IP adapter win
        return 0.6

    def parse(self, source: RawSource) -> ParseResult:
        """Parse the ``.stg`` → one ``profile2d`` observation of apparent ρ (doc 03 §2)."""
        text = (source.data or b"").decode("utf-8", errors="replace")
        rows, label, warnings, total, dropped = parse_stg(text, source.filename)
        prop, unit = value_property_and_unit(label, default_prop="resistivity")
        if prop != "resistivity":
            warnings.append(IngestWarning(
                "unexpected_quantity", Severity.MEDIUM,
                f"ert .stg declares {label!r}; expected apparent resistivity",
                f"property:{prop}",
            ))

        if not rows:
            return ParseResult(warnings=warnings or [IngestWarning(
                "empty_file", Severity.HIGH, "no .stg data records", source.filename,
            )])

        arr = np.asarray(rows, dtype=float)
        cols = {c: arr[:, i] for i, c in enumerate(STG_COLUMNS)}
        midx = 0.25 * (cols["a_x"] + cols["b_x"] + cols["m_x"] + cols["n_x"])
        midy = 0.25 * (cols["a_y"] + cols["b_y"] + cols["m_y"] + cols["n_y"])
        pseudodepth = cols["pseudodepth"]
        coords = np.column_stack([midx, midy, pseudodepth])

        obs = RawObservation(
            geometry_kind="profile2d",
            coords=coords,
            values={prop: cols["value"]},
            primary_property=prop,
            meta={
                "array": "dipole-dipole",
                "electrodes": {
                    "a": np.column_stack([cols["a_x"], cols["a_y"]]).tolist(),
                    "b": np.column_stack([cols["b_x"], cols["b_y"]]).tolist(),
                    "m": np.column_stack([cols["m_x"], cols["m_y"]]).tolist(),
                    "n": np.column_stack([cols["n_x"], cols["n_y"]]).tolist(),
                },
            },
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                crs=source.crs_hint,
                z_convention="depth_below_surface",
            ),
            units={prop: unit},
            provenance=Provenance(process="ingest:ert-stg-v1"),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )
