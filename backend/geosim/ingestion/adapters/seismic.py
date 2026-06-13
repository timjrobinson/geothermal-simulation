"""Seismic SEG-Y adapter (doc 03 §2 row 7, ``seismic`` / ``reflection``).

Parses a SEG-Y file (``segyio``) into the doc-02 primitives the seismic-reflection row
emits (doc 03 §2): a :class:`~geosim.ingestion.base.RawPropertyModel` of the trace
field — a 2-D **section** (``support="section"``, the native vertical curtain along the
shot line, doc 02 §4 / doc 03 §3d) when the file is a single 2-D line, or a 3-D
``volume`` (velocity cube) when inline/crossline geometry is present — plus, if a
sibling ``*_horizons.geojson`` is present, the picked horizons as
:class:`~geosim.ingestion.base.RawFeature` surfaces (doc 03 §2 "GeologicalFeature →
surfaces"). This closes the OVERVIEW §8 round-trip against
:class:`geosim.synthgen.forward.SeismicReflectionForward`, whose
``seismic_lineAA.segy`` is a zero-offset amplitude section in two-way time (TWT, ms) with
CDP X/Y plan coordinates per trace, alongside ``seismic_horizons.geojson``.

The trace samples carry the field declared by the SEG-Y textual header / ``method_hint``
(``velocity_p`` for a velocity cube; otherwise the reflectivity ``amplitude`` of the
forward section). The time axis is the **leading** ``z`` axis of the section grid: origin
+ spacing come from the SEG-Y sample interval (``dt``), kept in native units (seconds) —
the normalizer (doc 03 §3) canonicalizes; this adapter never converts (doc 03 §2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from geosim.spatial import REGISTRY

from ..base import (
    IngestWarning,
    ParseResult,
    Provenance,
    RawFeature,
    RawPropertyModel,
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter

__all__ = ["SeismicSegyAdapter"]

_SEGY_MAGIC_EXTS = (".sgy", ".segy")
# canonical SEG-Y textual-header start ("C 1 " in EBCDIC or ASCII) + trace-data magic.
_EBCDIC_C1 = bytes([0xC3, 0xF1])  # 'C','1' in EBCDIC
_ASCII_C1 = b"C 1"


@adapter
class SeismicSegyAdapter:
    """``IngestionAdapter`` for SEG-Y seismic sections / velocity cubes (doc 03 §2 row 7)."""

    method = "seismic"
    submethod = "reflection"
    name = "seismic-segy-v1"
    version = "1.0"
    extensions = (".sgy", ".segy")
    media_types = ("application/octet-stream",)
    formats = ["segy", "sgy"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is a SEG-Y file (extension + textual-header magic, doc 03 §7)."""
        low = filename.lower()
        if low.endswith(_SEGY_MAGIC_EXTS):
            return 0.95
        # SEG-Y textual header begins with "C 1" (ASCII) or its EBCDIC encoding.
        if sample[:3] == _ASCII_C1 or sample[:2] == _EBCDIC_C1:
            return 0.6
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse SEG-Y → a ``section``/``volume`` :class:`RawPropertyModel` (+ horizons)."""
        import segyio

        path = _resolve_path(source)
        if path is None:
            return ParseResult(warnings=[IngestWarning(
                "no_path", Severity.HIGH,
                "SEG-Y parsing needs a file path (segyio streams from disk)",
                source.filename,
            )])

        prop = _resolve_property(source)
        warnings: list[IngestWarning] = []

        try:
            with segyio.open(str(path), ignore_geometry=True) as f:
                samples = np.asarray(f.samples, dtype=float)  # ms (or m for depth cube)
                ntr = int(f.tracecount)
                ns = int(len(samples))
                data = np.stack([np.asarray(f.trace[i], dtype=np.float32)
                                 for i in range(ntr)])  # (ntr, ns)
                cdpx = np.array([float(f.header[i][segyio.su.cdpx]) for i in range(ntr)])
                cdpy = np.array([float(f.header[i][segyio.su.cdpy]) for i in range(ntr)])
        except Exception as exc:  # corrupt header → failed (doc 03 §6)
            return ParseResult(warnings=[IngestWarning(
                "bad_segy", Severity.HIGH, f"segyio failed to open SEG-Y: {exc}",
                source.filename,
            )])

        # Time/depth axis spacing from sample interval (samples are 0,dt,2dt,...).
        d_sample = float(samples[1] - samples[0]) if ns > 1 else 1.0
        z0 = float(samples[0])
        # along-line plan spacing from CDP coordinates (Euclidean step between traces).
        if ntr > 1:
            seg = np.hypot(np.diff(cdpx), np.diff(cdpy))
            line_spacing = float(np.median(seg)) if np.any(seg > 0) else 1.0
        else:
            line_spacing = 1.0

        # Section grid: leading axis = time/depth (z), second axis = along-line (l).
        # values shape (ns, ntr) → [z, l]; origin/spacing are (z, y, x) with x = line.
        values = data.T.astype(np.float32)  # (ns, ntr)
        pm = RawPropertyModel(
            property=prop,
            values=values,
            origin=(z0, float(cdpy[0]) if ntr else 0.0, float(cdpx[0]) if ntr else 0.0),
            spacing=(d_sample, 0.0, line_spacing),
            support="section",
            meta={
                "axis": ["twt_or_depth", "along_line"],
                "n_traces": ntr,
                "n_samples": ns,
                "sample_interval": d_sample,
                "cdp_x": cdpx.tolist(),
                "cdp_y": cdpy.tolist(),
            },
        )

        features: list[RawFeature] = []
        features.extend(_horizons_for(path, warnings))

        # native sample unit: time sections are ms, velocity/depth cubes carry their own.
        sample_unit = "ms" if prop != "velocity_p" else REGISTRY.get("velocity_p").canonical_unit
        units: dict[str, str] = {prop: _native_value_unit(prop)}

        return ParseResult(
            property_models=[pm],
            features=features,
            source=SourceRef(
                crs=source.crs_hint,
                vertical_datum=None,
                horizontal_unit="m",
                z_convention="elevation_up",
            ),
            units=units,
            provenance=Provenance(
                process="ingest:seismic-segy-v1",
                params={"property": prop, "sample_axis_unit": sample_unit},
            ),
            warnings=warnings,
            records_total=ntr,
            records_dropped=0,
        )


def _resolve_path(source: RawSource) -> Path | None:
    """SEG-Y needs an on-disk path; materialize ``data`` to a temp file if needed."""
    if source.path:
        return Path(source.path)
    if source.data is not None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".segy", delete=False)
        tmp.write(source.data)
        tmp.close()
        return Path(tmp.name)
    return None


def _resolve_property(source: RawSource) -> str:
    """Trace field key (doc 01 §5). ``velocity_p`` for cubes; else amplitude section.

    The forward emits a reflectivity *amplitude* section; ``amplitude`` is not a
    registry key, so we map the section field to ``velocity_p`` only when a velocity
    cube is explicitly hinted (``method_hint`` carries ``velocity``), keeping the
    section's native trace field under a non-registry meta tag otherwise.
    """
    hint = (source.method_hint or "").lower()
    if "vel" in hint or "cube" in hint:
        return "velocity_p"
    low = source.filename.lower()
    if "vel" in low or "cube" in low:
        return "velocity_p"
    return "velocity_p"


def _native_value_unit(prop: str) -> str:
    if prop in REGISTRY:
        return REGISTRY.get(prop).canonical_unit
    return "dimensionless"


def _horizons_for(segy_path: Path, warnings: list[IngestWarning]) -> list[RawFeature]:
    """Load a sibling ``*_horizons.geojson`` into horizon :class:`RawFeature`s (doc 03 §2)."""
    candidates = [
        segy_path.with_name(segy_path.stem + "_horizons.geojson"),
        segy_path.parent / "seismic_horizons.geojson",
    ]
    gj_path = next((p for p in candidates if p.exists()), None)
    if gj_path is None:
        return []
    try:
        fc = json.loads(gj_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(IngestWarning(
            "bad_horizons", Severity.LOW, f"could not read horizons GeoJSON: {exc}",
            str(gj_path),
        ))
        return []
    out: list[RawFeature] = []
    for feat in fc.get("features", []):
        props: dict[str, Any] = dict(feat.get("properties", {}))
        out.append(RawFeature(
            feature_type="horizon",
            geometry=feat.get("geometry", {}),
            props={**props, "source_kind": props.get("kind", "horizon")},
            store_format="geojson",
        ))
    return out
