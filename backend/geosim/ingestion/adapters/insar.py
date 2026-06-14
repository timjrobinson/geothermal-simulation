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

REAL FORGE FORMAT (added alongside the synthetic GeoTIFF path): the Utah FORGE InSAR
product ships **headerless 3-column point CSVs** in UTM 12 N metres (EPSG:32612) —
``<easting>,<northing>,<value>`` — where ``value`` is the mean LOS range-change **rate**
in m/yr (``avg_range_mperyr_utm.csv``) plus a sibling 1-sigma file
(``sig_range_mperyr_utm.csv``). Those are ~3-4 million scattered points (~8 m native
posting, ~105 MB). The CSV branch reads the 3 columns once (pandas), **bins** the
scattered points onto a regular grid (native posting coarsened to keep the cell count
tractable, <= ~1024 per axis), converts m/yr -> mm/yr (canonical ``deformation`` unit is
``mm``), leaves NaN where no points fall, and pairs the matching ``sig_*`` file as the
per-cell 1-sigma. The grid is emitted as a single-epoch ``deformation`` grid2d
:class:`RawPropertyModel` with ``SourceRef.crs="EPSG:32612"`` (coords already UTM; the
normalizer keeps grid origin/spacing native, doc 03 §3c).
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

# Real FORGE UTM point-CSV (headerless ``E,N,value``) — EPSG:32612 / metres / m-per-yr.
_CSV_EXTS = (".csv", ".txt")
_UTM_CSV_CRS = "EPSG:32612"
# Filename hints for the LOS range-change rate product (avg) vs its 1-sigma (sig).
_INSAR_CSV_HINTS = ("range_mperyr", "range_m", "mperyr", "los", "insar", "defo")
_SIG_PREFIXES = ("sig_", "sigma_", "std_", "err_")
# Keep the binned grid tractable: at most this many cells per axis (doc: <= ~1024).
_MAX_CELLS = 1024
# m/yr -> mm/yr: canonical ``deformation`` unit is mm (doc 01 §5 / spatial REGISTRY).
_M_TO_MM = 1000.0
# A line is a UTM point-CSV record when it is exactly 3 comma-separated finite floats
# whose first two columns look like UTM 12 N easting/northing (hundreds of km range).
_UTM_E_RANGE = (1.0e5, 9.0e5)
_UTM_N_RANGE = (0.0, 1.0e7)


@adapter
class InsarGeotiffAdapter:
    """``IngestionAdapter`` for InSAR LOS GeoTIFF time-series (doc 03 §2 row 10)."""

    method = "insar"
    submethod = None
    name = "insar-geotiff-v1"
    version = "1.1"
    extensions = (".tif", ".tiff", ".csv", ".txt")
    media_types = ("image/tiff", "text/csv")
    formats = ["geotiff", "tif", "tiff", "csv"]

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence it is InSAR: a LOS GeoTIFF, or the FORGE UTM range-change point-CSV."""
        low = filename.lower()
        is_tiff = low.endswith(_TIFF_EXTS) or sample[:4] in _TIFF_MAGIC
        if is_tiff:
            if "los" in low or "insar" in low or "defo" in low:
                return 0.85
            return 0.5  # generic GeoTIFF; gravity/mag grids also exist (lower confidence)
        if low.endswith(_CSV_EXTS) and _is_utm_point_csv(sample):
            # The headerless 3-col UTM point-CSV is unambiguous once content-sniffed; a
            # filename hint (range_mperyr/los/insar) lifts it above any generic CSV claim.
            return 0.9 if any(h in low for h in _INSAR_CSV_HINTS) else 0.55
        return 0.0

    def parse(self, source: RawSource) -> ParseResult:
        """Route to the GeoTIFF time-series path or the real UTM point-CSV binning path."""
        low = source.filename.lower()
        if low.endswith(_CSV_EXTS) and not low.endswith(_TIFF_EXTS):
            return self._parse_utm_csv(source)
        return self._parse_geotiff(source)

    def _parse_geotiff(self, source: RawSource) -> ParseResult:
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

    # ---- real FORGE UTM point-CSV → binned ``deformation`` grid2d (doc 03 §2) ----

    def _parse_utm_csv(self, source: RawSource) -> ParseResult:
        """Bin the headerless ``E,N,rate`` UTM point-CSV onto a regular ``deformation`` grid.

        Reads the 3 columns once (pandas C parser), bins ~M points by ``np.add.at`` into a
        coarsened regular grid (cell-mean = sum/count, NaN where a cell got no points),
        converts m/yr -> mm/yr, and pairs the sibling ``sig_*`` file as the per-cell 1σ.
        Coordinates stay native (UTM 12 N); ``SourceRef.crs="EPSG:32612"``.
        """
        warnings: list[IngestWarning] = []
        path = _resolve_csv_path(source)
        if path is None:
            return ParseResult(warnings=[IngestWarning(
                "no_path", Severity.HIGH,
                "UTM point-CSV parsing needs file bytes or a path", source.filename,
            )])

        try:
            e, n, v = _read_xyz_csv(path)
        except Exception as exc:  # noqa: BLE001 — an unreadable file is a hard fail here
            return ParseResult(warnings=[IngestWarning(
                "bad_csv", Severity.HIGH, f"unreadable UTM point-CSV: {exc}", source.filename,
            )])

        total = int(e.size)
        good = np.isfinite(e) & np.isfinite(n) & np.isfinite(v)
        dropped = int((~good).sum())
        e, n, v = e[good], n[good], v[good]
        if e.size == 0:
            return ParseResult(warnings=[IngestWarning(
                "no_points", Severity.HIGH, "no finite (E,N,value) rows", source.filename,
            )])
        if dropped:
            warnings.append(IngestWarning(
                "bad_row", Severity.LOW,
                f"{dropped} non-finite row(s) skipped", source.filename,
            ))

        # m/yr -> mm/yr (canonical deformation unit is mm); declare unit "mm" so the
        # normalizer passes the values through unchanged (doc 03 §3b).
        v = v * _M_TO_MM

        grid, cell, (x0, y0), counts = _bin_to_grid(e, n, v)
        ny, nx = grid.shape
        empty = int((counts == 0).sum())

        # Pair the sibling 1-sigma file (sig_<...>) onto the SAME grid when present.
        sigma_grid = None
        sig_path = _find_sigma_sibling(path)
        if sig_path is not None:
            try:
                se, sn, sv = _read_xyz_csv(sig_path)
                sg = np.isfinite(se) & np.isfinite(sn) & np.isfinite(sv)
                sigma_grid = _bin_onto(
                    se[sg], sn[sg], sv[sg] * _M_TO_MM, cell, x0, y0, ny, nx
                )
                warnings.append(IngestWarning(
                    "paired_sigma", Severity.INFO,
                    f"per-cell 1σ binned from sibling {sig_path.name}", source.filename,
                ))
            except Exception as exc:  # noqa: BLE001 — sigma is optional; degrade gracefully
                warnings.append(IngestWarning(
                    "sigma_unreadable", Severity.LOW,
                    f"could not pair sigma file {sig_path.name}: {exc}", source.filename,
                ))

        # grid2d embedded in 3D: (z=1, ny, nx) Z-up, ascending-y from the bin origin.
        values = grid.astype(np.float32)[np.newaxis, :, :]
        sigma_vals = (
            None if sigma_grid is None
            else sigma_grid.astype(np.float32)[np.newaxis, :, :]
        )
        pm = RawPropertyModel(
            property="deformation",
            values=values,
            origin=(0.0, float(y0 + cell / 2.0), float(x0 + cell / 2.0)),
            spacing=(max(cell, 1.0), float(cell), float(cell)),
            support="grid2d",
            sigma=sigma_vals,
            meta={
                "los": "line_of_sight",
                "product": "range_change_rate",
                "rate": True,
                "source_format": "utm_point_csv",
                "native_posting_m": float(cell),
                "n_cells": [int(ny), int(nx)],
                "n_empty_cells": empty,
                "n_points": int(e.size),
            },
        )
        return ParseResult(
            property_models=[pm],
            source=SourceRef(
                crs=_UTM_CSV_CRS,           # coords are already UTM 12 N metres
                vertical_datum=None,
                horizontal_unit="m",
                z_convention="elevation_up",
            ),
            # mm/yr declared as mm (canonical deformation); already converted above.
            units={"deformation": "mm"},
            provenance=Provenance(
                process="ingest:insar-utm-csv-v1",
                params={
                    "binning": "mean", "cell_m": float(cell),
                    "grid": [int(ny), int(nx)], "rate_unit": "mm/yr",
                    "sigma_paired": sig_path.name if sig_path else None,
                },
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


# ─────────────────────── real FORGE UTM point-CSV helpers ───────────────────────


def _is_utm_point_csv(sample: bytes) -> bool:
    """True if the first non-blank line is exactly 3 comma-separated floats whose first
    two look like a UTM 12 N easting/northing (cheap content sniff, doc 03 §7 step 3)."""
    text = sample.decode("utf-8", errors="replace")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) != 3:
            return False
        try:
            e, n, v = (float(p) for p in parts)
        except ValueError:
            return False
        return (
            _UTM_E_RANGE[0] <= e <= _UTM_E_RANGE[1]
            and _UTM_N_RANGE[0] <= n <= _UTM_N_RANGE[1]
            and np.isfinite(v)
        )
    return False


def _resolve_csv_path(source: RawSource) -> Path | None:
    """Prefer the on-disk path (read the 105 MB file once, not the in-memory copy)."""
    if source.path:
        return Path(source.path)
    if source.data is not None:
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        tmp.write(source.data)
        tmp.close()
        return Path(tmp.name)
    return None


def _read_xyz_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the 3 columns of a headerless ``E,N,value`` CSV efficiently (one pass).

    Uses the pandas C engine with explicit dtypes so the ~3-4 M-row / ~105 MB file is read
    a single time into three contiguous arrays (E,N float64 for grid maths; value float32).
    """
    import pandas as pd

    df = pd.read_csv(
        path, header=None, usecols=[0, 1, 2], names=["e", "n", "v"],
        dtype={0: np.float64, 1: np.float64, 2: np.float32},
        comment="#", engine="c", na_values=["NaN", "nan"],
    )
    return (
        df["e"].to_numpy(np.float64),
        df["n"].to_numpy(np.float64),
        df["v"].to_numpy(np.float64),
    )


def _native_posting(e: np.ndarray, n: np.ndarray) -> float:
    """Estimate the native grid posting (m) from the smallest positive coordinate step.

    The FORGE product is a resampled regular grid (~8 m), so the minimum gap between
    distinct sorted eastings/northings recovers the native cell size. Falls back to a
    coarse default if the points are too few/irregular to infer one.
    """
    def step(a: np.ndarray) -> float:
        u = np.unique(a)
        if u.size < 2:
            return 0.0
        d = np.diff(u)
        d = d[d > 1e-6]
        return float(d.min()) if d.size else 0.0

    s = max(step(e), step(n))
    return s if s > 0 else 10.0


def _grid_geometry(
    e: np.ndarray, n: np.ndarray
) -> tuple[float, float, float, int, int]:
    """Cell size + grid origin/shape: native posting coarsened so each axis <= _MAX_CELLS."""
    x0, x1 = float(e.min()), float(e.max())
    y0, y1 = float(n.min()), float(n.max())
    cell = _native_posting(e, n)
    span_x, span_y = max(x1 - x0, cell), max(y1 - y0, cell)
    # coarsen uniformly until both axes fit under the cap (keeps cells square)
    while (int(span_x / cell) + 1) > _MAX_CELLS or (int(span_y / cell) + 1) > _MAX_CELLS:
        cell *= 2.0
    nx = int(span_x / cell) + 1
    ny = int(span_y / cell) + 1
    return cell, x0, y0, nx, ny


def _bin_onto(
    e: np.ndarray, n: np.ndarray, v: np.ndarray,
    cell: float, x0: float, y0: float, ny: int, nx: int,
) -> np.ndarray:
    """Cell-mean of scattered ``v`` onto a fixed (ny, nx) grid; NaN where no points fall.

    ``np.add.at`` accumulates per-cell sum + count in one vectorised pass (no Python loop
    over the millions of points), then mean = sum / count.
    """
    ix = np.clip(((e - x0) / cell).astype(np.intp), 0, nx - 1)
    iy = np.clip(((n - y0) / cell).astype(np.intp), 0, ny - 1)
    flat = iy * nx + ix
    ssum = np.zeros(ny * nx, dtype=np.float64)
    cnt = np.zeros(ny * nx, dtype=np.float64)
    np.add.at(ssum, flat, v)
    np.add.at(cnt, flat, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / cnt
    mean[cnt == 0] = np.nan
    return mean.reshape(ny, nx)


def _bin_to_grid(
    e: np.ndarray, n: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, float, tuple[float, float], np.ndarray]:
    """Bin scattered points to a regular grid; return (mean grid, cell, (x0,y0), counts)."""
    cell, x0, y0, nx, ny = _grid_geometry(e, n)
    ix = np.clip(((e - x0) / cell).astype(np.intp), 0, nx - 1)
    iy = np.clip(((n - y0) / cell).astype(np.intp), 0, ny - 1)
    flat = iy * nx + ix
    ssum = np.zeros(ny * nx, dtype=np.float64)
    cnt = np.zeros(ny * nx, dtype=np.float64)
    np.add.at(ssum, flat, v)
    np.add.at(cnt, flat, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ssum / cnt
    mean[cnt == 0] = np.nan
    return mean.reshape(ny, nx), cell, (x0, y0), cnt.reshape(ny, nx)


def _find_sigma_sibling(path: Path) -> Path | None:
    """Match an ``avg_*`` rate file to its ``sig_*`` 1-sigma sibling by filename.

    The FORGE pair is ``avg_range_mperyr_utm.csv`` / ``sig_range_mperyr_utm.csv``. We map
    a leading ``avg`` (or no sig prefix) to ``sig`` and look for an existing sibling. A
    file that already IS a sigma product has no sigma of its own.
    """
    name = path.name
    low = name.lower()
    if low.startswith(_SIG_PREFIXES):
        return None
    candidates: list[str] = []
    if low.startswith("avg_"):
        candidates.append("sig_" + name[4:])
    if low.startswith("mean_"):
        candidates.append("sig_" + name[5:])
    candidates.append("sig_" + name)  # generic fallback prefix
    for cand in candidates:
        sib = path.with_name(cand)
        if sib.exists():
            return sib
    return None
