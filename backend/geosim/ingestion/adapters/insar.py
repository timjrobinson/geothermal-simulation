"""InSAR GeoTIFF time-series adapter (doc 03 §2 row 10, ``insar``).

Parses a line-of-sight (LOS) deformation **GeoTIFF time-series** (``rasterio``) into a
:class:`~geosim.ingestion.base.RawPropertyModel` of ``deformation`` with
``support="grid2d"`` and a **leading ``t`` axis** (doc 03 §2/§5, doc 02 §1/§8): the
4-D raster time-series the InSAR row mandates. Each epoch is one GeoTIFF (one band-file
per epoch, mm); the adapter stacks them in acquisition order into a ``[t, y, x]`` array
and attaches an explicit **ISO-8601 UTC** ``TimeAxis`` (doc 02 §8 — not project-epoch
offsets) in ``meta``.

Closes the OVERVIEW §8 round-trip against
:class:`geosim.synthgen.forward.InSARForward`, whose ``insar/los_NN.tif`` epoch rasters
are single-band float32 LOS deformation (mm) in the Engineering local frame (no CRS).
The pipeline supplies a single :class:`RawSource` per file; this adapter discovers the
sibling epochs in the same directory so one ingest yields the full time-series. Native
units (mm) stay native — the normalizer (doc 03 §3) canonicalizes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from geosim.spatial import REGISTRY

from ..base import (
    IngestWarning,
    ParseResult,
    Provenance,
    RawPropertyModel,
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter

__all__ = ["InsarGeotiffAdapter"]

_TIFF_EXTS = (".tif", ".tiff")
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")  # little/big-endian TIFF magic
# epoch index from a filename like ``los_03.tif`` (the forward's naming, doc 05 §4).
_EPOCH_RE = re.compile(r"(\d+)(?=\.\w+$)")
_EPOCH0 = "2026-01-01T00:00:00Z"
_EPOCH_STEP_S = 12 * 24 * 3600  # 12-day repeat pass (Sentinel-1-like) for the TimeAxis


@adapter
class InsarGeotiffAdapter:
    """``IngestionAdapter`` for InSAR LOS GeoTIFF time-series (doc 03 §2 row 10)."""

    method = "insar"
    submethod = None
    name = "insar-geotiff-v1"
    version = "1.0"
    extensions = (".tif", ".tiff")
    media_types = ("image/tiff",)
    formats = ["geotiff", "tif", "tiff"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is an InSAR LOS GeoTIFF (TIFF magic + ``los`` / ``insar`` hint)."""
        low = filename.lower()
        is_tiff = low.endswith(_TIFF_EXTS) or sample[:4] in _TIFF_MAGIC
        if not is_tiff:
            return 0.0
        if "los" in low or "insar" in low or "defo" in low:
            return 0.85
        return 0.5  # a generic GeoTIFF; gravity/mag grids also exist (lower confidence)

    def parse(self, source: RawSource) -> ParseResult:
        """Parse the GeoTIFF epoch series → ``deformation`` grid2d with leading t axis."""
        import rasterio

        path = _resolve_path(source)
        if path is None:
            return ParseResult(warnings=[IngestWarning(
                "no_path", Severity.HIGH,
                "GeoTIFF parsing needs a file path (rasterio reads from disk)",
                source.filename,
            )])

        warnings: list[IngestWarning] = []
        epoch_paths = _discover_epochs(path)
        slices: list[np.ndarray] = []
        transform = None
        crs = None
        for ep in epoch_paths:
            try:
                with rasterio.open(str(ep)) as r:
                    band = r.read(1).astype(np.float32)
                    if transform is None:
                        transform = r.transform
                        crs = r.crs
                slices.append(band)
            except Exception as exc:
                warnings.append(IngestWarning(
                    "bad_epoch", Severity.LOW, f"could not read epoch {ep.name}: {exc}",
                    str(ep),
                ))

        if not slices:
            return ParseResult(warnings=[IngestWarning(
                "no_rasters", Severity.HIGH, "no readable GeoTIFF epochs", source.filename
            )])

        shapes = {s.shape for s in slices}
        if len(shapes) > 1:
            return ParseResult(warnings=[IngestWarning(
                "epoch_shape_mismatch", Severity.HIGH,
                f"epochs have inconsistent shapes {shapes}", source.filename,
            )])

        # rasterio row 0 = north; flip each band so the y axis ascends (Engineering Z-up
        # plan convention used by the writer / forward GeoTIFF) before stacking on t.
        cube = np.stack([np.flipud(s) for s in slices], axis=0)  # [t, y, x]
        nt, ny, nx = cube.shape

        # grid origin/spacing in (z, y, x); a grid2d has no z extent → z spacing 0.
        dx = float(transform.a) if transform is not None else 1.0
        dy = float(-transform.e) if transform is not None else 1.0  # e is negative (north-up)
        x0 = float(transform.c) + dx / 2.0 if transform is not None else 0.0
        # after the flipud the first row is the southernmost (min y); top-left y is f:
        y_top = float(transform.f) if transform is not None else 0.0
        y0 = y_top - dy * (ny - 0.5)  # centre of southernmost row

        # explicit ISO-8601 UTC TimeAxis (doc 02 §8): one epoch per slice.
        from datetime import UTC, datetime, timedelta

        base = datetime(2026, 1, 1, tzinfo=UTC)
        epochs_iso = [
            (base + timedelta(seconds=i * _EPOCH_STEP_S)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(nt)
        ]

        pm = RawPropertyModel(
            property="deformation",
            values=cube.astype(np.float32),
            origin=(0.0, y0, x0),
            spacing=(0.0, dy, dx),
            support="grid2d",
            meta={
                "timeAxis": {"epochs": epochs_iso, "unit": "ISO-8601-UTC"},
                "n_epochs": nt,
                "los": "line_of_sight",
                "leading_axis": "t",
            },
        )

        return ParseResult(
            property_models=[pm],
            source=SourceRef(
                crs=str(crs) if crs else source.crs_hint,
                vertical_datum=None,
                horizontal_unit="m",
                z_convention="elevation_up",
            ),
            units={"deformation": REGISTRY.get("deformation").canonical_unit},
            provenance=Provenance(
                process="ingest:insar-geotiff-v1",
                params={"n_epochs": nt, "epochs": epochs_iso},
            ),
            warnings=warnings,
            records_total=nt,
            records_dropped=0,
        )


def _resolve_path(source: RawSource) -> Path | None:
    if source.path:
        return Path(source.path)
    if source.data is not None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.write(source.data)
        tmp.close()
        return Path(tmp.name)
    return None


def _discover_epochs(path: Path) -> list[Path]:
    """Sibling epoch rasters sharing the file's non-numeric stem, in epoch order.

    The forward writes ``los_00.tif … los_NN.tif`` in one directory; we gather them by
    the common prefix and sort by the trailing epoch index so the t axis is in
    acquisition order (doc 05 §4 / doc 02 §8). A lone file ingests as a single epoch.
    """
    stem = path.stem
    prefix = re.sub(r"\d+$", "", stem)  # strip the trailing epoch index
    siblings = [
        p for p in path.parent.iterdir()
        if p.suffix.lower() in _TIFF_EXTS and p.stem.startswith(prefix)
    ]
    if len(siblings) <= 1:
        return [path]

    def _key(p: Path) -> int:
        m = _EPOCH_RE.search(p.name)
        return int(m.group(1)) if m else 0

    return sorted(siblings, key=_key)
