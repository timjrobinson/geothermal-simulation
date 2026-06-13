"""Microseismic QuakeML adapter (doc 03 §2 row 9, ``microseismic``).

Parses a QuakeML event catalog (``obspy``) into the doc-02 primitive the microseismic
row mandates (doc 03 §2): a **4-D point cloud** ``GeologicalFeature`` —
``(x, y, z, t, mag)`` per event — stored as a
:class:`~geosim.ingestion.base.RawFeature` of ``feature_type="pointCloud"`` whose
geometry is a GeoJSON ``MultiPoint`` of ``[x, y, z]`` and whose ``props`` carry the
parallel **explicit ISO-8601 UTC** time array (doc 02 §1/§8 — the leading ``t`` axis) and
the per-event magnitudes (doc 03 §5: "microseismic carry time").

Closes the OVERVIEW §8 round-trip against
:class:`geosim.synthgen.forward.MicroseismicForward`, whose ``microseismic.quakeml``
stores per-event ``time``/``depth``/``magnitude`` while the Engineering plan coordinates
(x, y) live in the sibling ``microseismic_catalog.csv`` (the forward zeroes QuakeML
lat/lon, which carry no Engineering frame). The adapter reads the QuakeML for the
canonical event identity + time + magnitude and joins the sibling CSV for ``(x, y, elev)``
so the 4-D cloud lands in the Engineering frame; with no CSV it falls back to the QuakeML
depth alone and warns.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from ..base import (
    IngestWarning,
    ParseResult,
    Provenance,
    RawFeature,
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter

__all__ = ["MicroseismicQuakeMlAdapter"]

_QML_EXTS = (".quakeml", ".xml", ".qml")


@adapter
class MicroseismicQuakeMlAdapter:
    """``IngestionAdapter`` for microseismic QuakeML catalogs (doc 03 §2 row 9)."""

    method = "microseismic"
    submethod = None
    name = "microseismic-quakeml-v1"
    version = "1.0"
    extensions = (".quakeml", ".xml", ".qml")
    media_types = ("application/xml",)
    formats = ["quakeml", "qml"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is QuakeML (extension + ``QuakeML`` / ``q:quakeml`` root XML)."""
        low = filename.lower()
        try:
            head = sample.decode("utf-8", errors="replace").lower()
        except Exception:
            head = ""
        is_quakeml = "quakeml" in head or "<eventparameters" in head
        if low.endswith(".quakeml") or low.endswith(".qml"):
            return 0.95 if is_quakeml else 0.85
        if low.endswith(".xml"):
            return 0.8 if is_quakeml else 0.0
        return 0.6 if is_quakeml else 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Parse QuakeML → a 4-D ``pointCloud`` feature (x, y, z, t, mag) (doc 03 §2/§5)."""
        from obspy import read_events

        warnings: list[IngestWarning] = []
        path = _resolve_path(source)
        if path is None:
            return ParseResult(warnings=[IngestWarning(
                "no_path", Severity.HIGH,
                "QuakeML parsing needs a file path or bytes", source.filename,
            )])

        try:
            cat = read_events(str(path))
        except Exception as exc:
            return ParseResult(warnings=[IngestWarning(
                "bad_quakeml", Severity.HIGH, f"obspy failed to read QuakeML: {exc}",
                source.filename,
            )])

        times: list[str] = []
        depths: list[float] = []
        mags: list[float] = []
        total = 0
        dropped = 0
        for ev in cat:
            total += 1
            try:
                origin = ev.preferred_origin() or ev.origins[0]
                mag = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
            except (IndexError, AttributeError):
                dropped += 1
                warnings.append(IngestWarning(
                    "incomplete_event", Severity.LOW,
                    "event missing origin/magnitude", f"event:{total}",
                ))
                continue
            # explicit ISO-8601 UTC epoch (doc 02 §8), not a project offset.
            times.append(origin.time.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"))
            depths.append(float(origin.depth) if origin.depth is not None else 0.0)
            mags.append(float(mag))

        n = len(times)
        # join the sibling CSV catalog for Engineering (x, y, elev) (doc 03 §5).
        xy = _load_catalog_xy(path, n, warnings)
        if xy is not None:
            xs, ys, zs = xy[:, 0], xy[:, 1], xy[:, 2]
        else:
            # QuakeML lat/lon carry no Engineering frame here → x=y=0; z from -depth.
            warnings.append(IngestWarning(
                "no_catalog_csv", Severity.MEDIUM,
                "no sibling catalog CSV — x/y unavailable from QuakeML, using depth only",
                source.filename,
            ))
            xs = np.zeros(n)
            ys = np.zeros(n)
            zs = -np.asarray(depths, dtype=float)  # depth below surface → elevation (approx)

        coords = [[float(xs[i]), float(ys[i]), float(zs[i])] for i in range(n)]

        feat = RawFeature(
            feature_type="pointCloud",
            geometry={"type": "MultiPoint", "coordinates": coords},
            props={
                "kind": "microseismic_events",
                "n_events": n,
                "time": times,            # leading t axis (ISO-8601 UTC, doc 02 §8)
                "mag": mags,
                "depth_m": depths,
                "dims": ["x", "y", "z", "t", "mag"],
            },
            store_format="geojson",
        )

        return ParseResult(
            features=[feat],
            source=SourceRef(
                crs=source.crs_hint,
                vertical_datum=None,
                horizontal_unit="m",
                z_convention="elevation_up",
            ),
            provenance=Provenance(
                process="ingest:microseismic-quakeml-v1",
                params={"n_events": n, "joined_catalog_csv": xy is not None},
            ),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )


def _resolve_path(source: RawSource) -> Path | None:
    if source.path:
        return Path(source.path)
    if source.data is not None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".quakeml", delete=False)
        tmp.write(source.data)
        tmp.close()
        return Path(tmp.name)
    return None


def _load_catalog_xy(
    qml_path: Path, n: int, warnings: list[IngestWarning]
) -> np.ndarray | None:
    """Join the sibling ``*_catalog.csv`` for Engineering ``(x, y, elev)`` (doc 03 §5).

    Returns an ``(n, 3)`` array of ``(x, y, elev)`` ordered by the catalog ``id`` column,
    or ``None`` when no usable CSV is found.
    """
    candidates = [
        qml_path.with_name(qml_path.stem + "_catalog.csv"),
        qml_path.parent / "microseismic_catalog.csv",
        qml_path.with_suffix(".csv"),
    ]
    csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        return None
    rows = list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    if len(rows) < 2:
        return None
    header = [c.strip().lower() for c in rows[0]]
    try:
        ix = header.index("x")
        iy = header.index("y")
        iz = header.index("elev")
    except ValueError:
        warnings.append(IngestWarning(
            "catalog_missing_xy", Severity.LOW,
            f"catalog CSV {csv_path.name} lacks x/y/elev columns", str(csv_path),
        ))
        return None
    out: list[list[float]] = []
    for r in rows[1:]:
        try:
            out.append([float(r[ix]), float(r[iy]), float(r[iz])])
        except (ValueError, IndexError):
            continue
    arr = np.asarray(out, dtype=float)
    if arr.shape[0] != n:
        warnings.append(IngestWarning(
            "catalog_count_mismatch", Severity.LOW,
            f"catalog has {arr.shape[0]} rows, QuakeML has {n} events", str(csv_path),
        ))
        # align by min length so the join still produces a usable cloud
        m = min(arr.shape[0], n)
        if m == 0:
            return None
        return arr[:m] if arr.shape[0] >= n else np.vstack(
            [arr, np.zeros((n - arr.shape[0], 3))]
        )
    return arr
