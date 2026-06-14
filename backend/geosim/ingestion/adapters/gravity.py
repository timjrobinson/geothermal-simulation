"""Gravity potential-field adapter (doc 03 §2 row 1) — parses the synthgen native pair.

The gravity T0 forward (:class:`geosim.synthgen.forward.GravityForward`, doc 05 §4 row 1)
emits two native files closing the OVERVIEW §8 round-trip:

- ``gravity_stations.csv`` — columnar stations ``station, x, y, elev, bouguer_mgal``
  (mGal). This adapter parses it into one ``points``
  :class:`~geosim.ingestion.base.RawObservation` carrying ``gravity_anomaly`` (doc 03 §2:
  *gravity CSV → ``Observation(points)`` → stations*).
- ``gravity_bouguer.tif`` — a single-band Bouguer GeoTIFF in the Engineering local frame
  (no CRS; affine maps pixel centres to Engineering metres, top row = max-y). This adapter
  parses it into a **pre-gridded** :class:`~geosim.ingestion.base.RawPropertyModel`
  ``gravity_anomaly`` with ``support="grid2d"`` (doc 03 §2: *if pre-gridded → ``PropertyModel``
  (gravity_anomaly, support.kind=grid2d)*).

This sits alongside the generic ``gravity-csv-v1`` adapter (a worked columnar example);
this one understands the synthgen ``bouguer_mgal`` column and the Bouguer GeoTIFF so the
forward → ingest round-trip works end to end. Coordinates/units stay native (Engineering
metres / mGal); the shared normalizer (doc 03 §3) canonicalizes. ``rasterio``/``pandas``
are the OVERVIEW §5 parse libraries.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from ..base import (
    IngestWarning,
    ParseResult,
    Provenance,
    RawObservation,
    RawPropertyModel,
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter

__all__ = ["GravityPotentialFieldAdapter"]

# Column aliases for the synthgen + common Bouguer station CSV (case-insensitive).
_X_KEYS = {"x", "easting", "east", "lon", "longitude"}
_Y_KEYS = {"y", "northing", "north", "lat", "latitude"}
_Z_KEYS = {"z", "elev", "elevation", "height", "alt", "altitude", "hae", "ngvd29"}
# Bouguer-anomaly columns — synthgen's `bouguer_mgal` plus real survey columns:
# gCBGA (complete Bouguer), gSBGA (simple Bouguer), gFA (free-air).
_G_KEYS = {
    "bouguer_mgal", "bouguer", "gravity_anomaly", "gravity", "ga", "mgal", "anomaly",
    "gcbga", "gsbga", "gfa", "cbga", "sbga",
}
_S_KEYS = {"sigma", "error", "err", "std", "uncertainty", "errg"}

# Ordered preferences — real files carry several coordinate/anomaly columns at once, so
# pick deterministically: geographic lon/lat first (self-describing as EPSG:4326), and
# the complete-Bouguer reduction over simpler ones.
_X_PREF = ("x", "lon", "longitude", "easting", "east")
_Y_PREF = ("y", "lat", "latitude", "northing", "north")
_G_PREF = ("bouguer_mgal", "gcbga", "gsbga", "gfa", "cbga", "sbga", "bouguer",
           "gravity_anomaly", "gravity", "ga", "mgal", "anomaly")
_GEO_COLS = {"lon", "longitude", "lat", "latitude"}

# GeoTIFF (Bouguer grid) magic + filename hints.
_TIF_EXT = (".tif", ".tiff")
_TIF_MAGIC = (b"II*\x00", b"MM\x00*")  # little/big-endian TIFF
_GRAV_NAME_HINTS = ("bouguer", "gravity", "grav")


@adapter
class GravityPotentialFieldAdapter:
    """``IngestionAdapter`` for synthgen gravity stations CSV + Bouguer GeoTIFF (doc 03 §2)."""

    method = "gravity"
    submethod = None
    name = "gravity-potential-v1"
    version = "1.0"
    extensions = (".csv", ".txt", ".tif", ".tiff")
    media_types = ("text/csv", "image/tiff")
    # Registry keys an adapter by ``method.formats[0]`` (doc 08 §3.2); the generic
    # ``gravity-csv-v1`` already holds ``gravity.csv`` — key this one on the GeoTIFF so
    # both coexist (detection still routes by ``sniff()`` across all adapters, doc 03 §7).
    formats = ["geotiff", "csv"]

    # ───────────────────────────── sniff (doc 03 §7 step 3) ─────────────────────────────

    def sniff(self, sample: bytes, filename: str) -> float:
        low = filename.lower()
        if low.endswith(_TIF_EXT):
            if not sample.startswith(_TIF_MAGIC):
                return 0.0
            # GeoTIFF content is just a float grid; lean on the filename to claim the
            # *gravity* Bouguer grid (mag RTP grids are claimed by the magnetics adapter).
            return 0.85 if any(h in low for h in _GRAV_NAME_HINTS) else 0.0
        if low.endswith((".csv", ".txt")):
            header = _first_data_line(_decode(sample))
            if header is None:
                return 0.0
            cols = {c.strip().lower() for c in header.split(",")}
            has_xy = bool(cols & _X_KEYS) and bool(cols & _Y_KEYS)
            has_bouguer = bool(cols & {"bouguer_mgal", "bouguer"})
            has_g = bool(cols & _G_KEYS)
            if has_xy and has_bouguer:
                return 0.95  # synthgen native column — beat the generic gravity-csv-v1
            if has_xy and has_g:
                return 0.6
        return 0.0

    # ───────────────────────────── parse (doc 03 §7 step 4) ─────────────────────────────

    def parse(self, source: RawSource) -> ParseResult:
        if source.filename.lower().endswith(_TIF_EXT):
            return self._parse_geotiff(source)
        return self._parse_csv(source)

    # ---- CSV stations → Observation(points, gravity_anomaly) (doc 03 §2) ----

    def _parse_csv(self, source: RawSource) -> ParseResult:
        text = _decode(source.data or b"")
        meta = _scan_comments(text)
        try:
            df = pd.read_csv(io.StringIO(text), comment="#")
        except Exception as e:  # noqa: BLE001 — a parse error is a hard fail for this file
            return ParseResult(warnings=[IngestWarning(
                "bad_csv", Severity.HIGH, f"unreadable gravity CSV: {e}", source.filename
            )])

        cols = {c.lower(): c for c in df.columns}
        xcol = _pick_ordered(cols, _X_PREF)
        ycol = _pick_ordered(cols, _Y_PREF)
        gcol = _pick_ordered(cols, _G_PREF)
        geographic = (xcol is not None and xcol.lower() in _GEO_COLS)
        if xcol is None or ycol is None or gcol is None:
            return ParseResult(warnings=[IngestWarning(
                "bad_header", Severity.HIGH,
                f"missing x/y/gravity columns in header {list(df.columns)}", source.filename,
            )])
        zcol = _pick(cols, _Z_KEYS)
        scol = _pick(cols, _S_KEYS)

        total = int(len(df))
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        g = pd.to_numeric(df[gcol], errors="coerce")
        z = pd.to_numeric(df[zcol], errors="coerce") if zcol else pd.Series(0.0, index=df.index)
        good = x.notna() & y.notna() & g.notna()
        dropped = int((~good).sum())

        warnings: list[IngestWarning] = []
        if dropped:
            warnings.append(IngestWarning(
                "bad_row", Severity.LOW, f"{dropped} unparseable row(s) skipped", source.filename
            ))

        x, y, g = x[good].to_numpy(), y[good].to_numpy(), g[good].to_numpy()
        z = z[good].fillna(0.0).to_numpy() if zcol else np.zeros(int(good.sum()))
        coords = np.column_stack([x, y, z]) if len(x) else np.zeros((0, 3))

        sigma: dict[str, np.ndarray] = {}
        if scol is not None:
            s = pd.to_numeric(df[scol], errors="coerce")[good].to_numpy()
            if np.any(np.isfinite(s)):
                sigma["gravity_anomaly"] = s

        obs = RawObservation(
            geometry_kind="points",
            coords=coords,
            values={"gravity_anomaly": np.asarray(g, dtype=float)},
            sigma=sigma,
            primary_property="gravity_anomaly",
            meta={"product": "bouguer"},
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                # geographic lon/lat columns are self-describing as EPSG:4326
                crs=meta.get("crs") or source.crs_hint or ("EPSG:4326" if geographic else None),
                vertical_datum=meta.get("vertical_datum"),
                horizontal_unit="deg" if geographic else meta.get("horizontal_unit", "m"),
                z_convention=meta.get("z_convention", "elevation_up"),
            ),
            units={"gravity_anomaly": meta.get("unit", "mGal")},
            provenance=Provenance(process=f"ingest:{self.name}"),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )

    # ---- Bouguer GeoTIFF → PropertyModel(grid2d, gravity_anomaly) (doc 03 §2) ----

    def _parse_geotiff(self, source: RawSource) -> ParseResult:
        pm, warnings = _read_local_grid(source, "gravity_anomaly", product="bouguer")
        if pm is None:
            return ParseResult(warnings=warnings)
        return ParseResult(
            property_models=[pm],
            source=SourceRef(crs=source.crs_hint, z_convention="elevation_up"),
            units={"gravity_anomaly": "mGal"},
            provenance=Provenance(process=f"ingest:{self.name}"),
            warnings=warnings,
            records_total=1,
            records_dropped=0,
        )


# ─────────────────────────────── shared helpers ───────────────────────────────


def _read_local_grid(
    source: RawSource, property_type: str, *, product: str
) -> tuple[RawPropertyModel | None, list[IngestWarning]]:
    """Read a single-band local-frame GeoTIFF → ``grid2d`` :class:`RawPropertyModel`.

    The synthgen writer (``write_local_geotiff``) stores row 0 = max-y with an affine
    mapping pixel centres to Engineering metres and no CRS. We flip back to ascending-y,
    recover ``(y0, x0)`` cell centres + spacing from the affine, and emit a Z-up
    ``(1, ny, nx)`` field at elevation 0 (a 2D field embedded in 3D, doc 02 §4 / §10.2).
    """
    import rasterio

    warnings: list[IngestWarning] = []
    path = source.path
    if path is None:
        # rasterio needs a path/handle; fall back to an in-memory file from bytes.
        from rasterio.io import MemoryFile

        with MemoryFile(source.data or b"") as mem, mem.open() as ds:
            return _grid_from_dataset(ds, property_type, product, warnings)
    with rasterio.open(path) as ds:
        return _grid_from_dataset(ds, property_type, product, warnings)


def _grid_from_dataset(ds, property_type, product, warnings):  # type: ignore[no-untyped-def]
    if ds.count < 1:
        warnings.append(IngestWarning("empty_raster", Severity.HIGH, "no raster bands", None))
        return None, warnings
    band = ds.read(1).astype(np.float32)
    if ds.nodata is not None and np.isfinite(ds.nodata):
        band = np.where(band == ds.nodata, np.nan, band)
    # rasterio row 0 = north (max y); our Engineering grid is ascending-y → flip.
    grid = np.flipud(band)
    ny, nx = grid.shape
    t = ds.transform
    dx = float(t.a)
    dy = float(-t.e)  # affine e is negative (north-up); spacing is +.
    # cell-centre origin (Z-up (z, y, x)): the bottom (min-y) row centre, left col centre.
    x0 = float(t.c + dx / 2.0)
    y_top = float(t.f - dy / 2.0)            # centre of the top (max-y) row
    y0 = y_top - (ny - 1) * dy                # centre of the bottom (min-y) row
    values = grid[np.newaxis, :, :]           # (z=1, ny, nx) Z-up
    pm = RawPropertyModel(
        property=property_type,
        values=values,
        origin=(0.0, y0, x0),
        spacing=(max(dy, dx, 1.0), dy, dx),  # nominal dz for the single grid2d layer
        support="grid2d",
        meta={"product": product},
    )
    return pm, warnings


def _decode(sample: bytes) -> str:
    return sample.decode("utf-8", errors="replace")


def _first_data_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return None


def _scan_comments(text: str) -> dict[str, str]:
    """Pull ``# key: value`` metadata lines (unit/crs/vertical_datum, doc 03 §3)."""
    meta: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#") or ":" not in s:
            continue
        key, _, val = s.lstrip("#").strip().partition(":")
        meta[key.strip().lower()] = val.strip()
    return meta


def _pick(cols: dict[str, str], keys: set[str]) -> str | None:
    """Return the original column name whose lowercase form is in ``keys``."""
    for low, orig in cols.items():
        if low in keys:
            return orig
    return None


def _pick_ordered(cols: dict[str, str], pref: tuple[str, ...]) -> str | None:
    """Return the first column (by preference order) present in ``cols``."""
    for k in pref:
        if k in cols:
            return cols[k]
    return None
