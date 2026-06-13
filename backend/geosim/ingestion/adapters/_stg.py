"""Shared AGI SuperSting ``.stg`` pseudosection parser (doc 03 §2 ert/ip rows).

Both the ERT (:mod:`.ert`) and IP (:mod:`.ip`) adapters read the *same* dipole-dipole
``.stg`` column layout — only the declared apparent quantity (the ``value:`` header
label) and the resulting ``property_type`` differ (doc 03 §2). This module owns the byte
parsing so the two adapters stay one-format-each (doc 03 §9). Underscore-prefixed so the
adapter-package auto-import skips it (it is not itself an adapter).

The layout (synthgen ``_write_stg``): a few free-text header lines, a ``value: <label>``
header noting the apparent quantity, a column-name header
``Idx,A_x,A_y,B_x,B_y,M_x,M_y,N_x,N_y,pseudodepth,value`` and one comma-separated record
per measurement. The label maps to a canonical doc-01 ``property_type`` + source unit.
"""

from __future__ import annotations

from ..base import IngestWarning, Severity

__all__ = ["STG_COLUMNS", "parse_stg", "value_property_and_unit"]

# the 9 numeric columns kept after the leading Idx (doc 05 §4 _write_stg layout)
STG_COLUMNS: tuple[str, ...] = (
    "a_x", "a_y", "b_x", "b_y", "m_x", "m_y", "n_x", "n_y", "pseudodepth", "value",
)

# label substring → (property_type, source unit). IP keys are SPLIT per doc 01 §5 /
# doc 03 §2: time-domain → chargeability_time_ms, mV/V → chargeability_mv_v,
# frequency-domain phase → phase_mrad.
_LABEL_MAP: tuple[tuple[str, tuple[str, str]], ...] = (
    ("resistiv", ("resistivity", "ohm*m")),
    ("chargeability_mv_v", ("chargeability_mv_v", "mV/V")),
    ("mv_v", ("chargeability_mv_v", "mV/V")),
    ("chargeability_time", ("chargeability_time_ms", "ms")),
    ("chargeability_ms", ("chargeability_time_ms", "ms")),
    ("phase_mrad", ("phase_mrad", "mrad")),
    ("phase", ("phase_mrad", "mrad")),
    ("chargeability", ("chargeability_mv_v", "mV/V")),
)


def value_property_and_unit(label: str | None, *, default_prop: str) -> tuple[str, str]:
    """Map a ``.stg`` ``value:`` label → ``(property_type, source_unit)`` (doc 03 §2).

    Unknown/missing labels fall back to ``default_prop`` and its canonical unit.
    """
    low = (label or "").lower()
    for needle, (prop, unit) in _LABEL_MAP:
        if needle in low:
            return prop, unit
    _defaults = {
        "resistivity": "ohm*m",
        "chargeability_mv_v": "mV/V",
        "chargeability_time_ms": "ms",
        "phase_mrad": "mrad",
    }
    return default_prop, _defaults.get(default_prop, "ohm*m")


def parse_stg(
    text: str, filename: str
) -> tuple[list[list[float]], str | None, list[IngestWarning], int, int]:
    """Parse a ``.stg`` body → ``(rows, value_label, warnings, total, dropped)``.

    ``rows`` are the 9-tuple float records in :data:`STG_COLUMNS` order. ``value_label``
    is the declared apparent quantity from a ``value: <label>`` header line (doc 03 §2).
    """
    warnings: list[IngestWarning] = []
    value_label: str | None = None
    header_idx: int | None = None
    lines = text.splitlines()

    for i, line in enumerate(lines):
        s = line.strip()
        low = s.lower()
        if value_label is None and "value:" in low:
            value_label = low.split("value:", 1)[1].strip()
        if header_idx is None and low.startswith("idx,") and "pseudodepth" in low:
            header_idx = i

    if header_idx is None:
        warnings.append(IngestWarning(
            "bad_header", Severity.HIGH,
            "no Idx,...,pseudodepth,value column header in .stg", filename,
        ))
        return [], value_label, warnings, 0, 0

    rows: list[list[float]] = []
    total = 0
    dropped = 0
    for n, line in enumerate(lines[header_idx + 1:], start=header_idx + 2):
        s = line.strip()
        if not s:
            continue
        total += 1
        parts = s.split(",")
        # drop the leading Idx; keep the 10 numeric columns
        nums = parts[1:]
        if len(nums) < len(STG_COLUMNS):
            dropped += 1
            warnings.append(IngestWarning(
                "bad_row", Severity.LOW, f"short record (line {n})", f"row {n}",
            ))
            continue
        try:
            rows.append([float(v) for v in nums[: len(STG_COLUMNS)]])
        except ValueError:
            dropped += 1
            warnings.append(IngestWarning(
                "bad_row", Severity.LOW, f"unparseable record (line {n})", f"row {n}",
            ))
    return rows, value_label, warnings, total, dropped
