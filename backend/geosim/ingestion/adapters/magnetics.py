"""Magnetics potential-field adapter (doc 03 §2 row 2) — parses the synthgen native pair.

The magnetics T0 forward (:class:`geosim.synthgen.forward.MagneticsForward`, doc 05 §4
row 2) emits two native files closing the OVERVIEW §8 round-trip:

- ``aeromag_lines.xyz`` — whitespace-delimited flight lines ``LINE X Y ALT TMI_RTP_nT``
  (nT). This adapter parses it into one ``points``
  :class:`~geosim.ingestion.base.RawObservation` carrying ``magnetic_field`` (doc 03 §2:
  *magnetics CSV/.xyz → ``Observation(TMI/magnetic_field, points)`` → point set*). The line
  id rides in ``meta`` so per-line leveling stays inspectable.
- ``mag_rtp.tif`` — a single-band reduced-to-pole GeoTIFF in the Engineering local frame
  (no CRS; affine maps pixel centres to Engineering metres, top row = max-y). This adapter
  parses it into a **pre-gridded** :class:`~geosim.ingestion.base.RawPropertyModel`
  ``magnetic_field`` with ``support="grid2d"`` (doc 03 §2: *``PropertyModel(grid2d)``*).

Coordinates/units stay native (Engineering metres / nT); the shared normalizer (doc 03 §3)
canonicalizes. The Bouguer-vs-RTP GeoTIFF ambiguity (both are bare float grids) is broken
on the filename. ``rasterio``/``pandas`` are the OVERVIEW §5 parse libraries; the GeoTIFF
reader is reused from the sibling gravity adapter.
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
    RawSource,
    Severity,
    SourceRef,
)
from ..registry import adapter
from .gravity import _read_local_grid, _scan_comments

__all__ = ["MagneticsPotentialFieldAdapter"]

_X_KEYS = {"x", "easting", "east", "lon", "longitude"}
_Y_KEYS = {"y", "northing", "north", "lat", "latitude"}
_Z_KEYS = {"alt", "altitude", "z", "elev", "elevation", "height"}
_LINE_KEYS = {"line", "l", "fid"}
_M_KEYS = {
    "tmi_rtp_nt", "tmi_rtp", "tmi", "rtp", "magnetic_field", "mag", "nt", "tf", "tfa",
}

_XYZ_EXT = (".xyz", ".dat", ".txt")
_TIF_EXT = (".tif", ".tiff")
_TIF_MAGIC = (b"II*\x00", b"MM\x00*")
_MAG_NAME_HINTS = ("mag", "rtp", "tmi", "aeromag")


@adapter
class MagneticsPotentialFieldAdapter:
    """``IngestionAdapter`` for synthgen aeromag ``.xyz`` lines + RTP GeoTIFF (doc 03 §2)."""

    method = "magnetics"
    submethod = None
    name = "magnetics-potential-v1"
    version = "1.0"
    extensions = (".xyz", ".dat", ".txt", ".tif", ".tiff")
    media_types = ("text/plain", "image/tiff")
    formats = ["xyz", "geotiff"]

    # ───────────────────────────── sniff (doc 03 §7 step 3) ─────────────────────────────

    def sniff(self, sample: bytes, filename: str) -> float:
        low = filename.lower()
        if low.endswith(_TIF_EXT):
            if not sample.startswith(_TIF_MAGIC):
                return 0.0
            # GeoTIFF content is a bare float grid; claim the *RTP* grid by filename.
            return 0.85 if any(h in low for h in _MAG_NAME_HINTS) else 0.0
        if low.endswith(_XYZ_EXT):
            header = _first_data_line(_decode(sample))
            if header is None:
                return 0.0
            cols = {c.strip().lower() for c in header.split()}
            has_xy = bool(cols & _X_KEYS) and bool(cols & _Y_KEYS)
            has_m = bool(cols & _M_KEYS)
            has_tmi = bool(cols & {"tmi_rtp_nt", "tmi_rtp", "tmi", "rtp"})
            if has_xy and has_tmi:
                return 0.95  # synthgen native aeromag header
            if has_xy and has_m:
                return 0.55
        return 0.0

    # ───────────────────────────── parse (doc 03 §7 step 4) ─────────────────────────────

    def parse(self, source: RawSource) -> ParseResult:
        if source.filename.lower().endswith(_TIF_EXT):
            return self._parse_geotiff(source)
        return self._parse_xyz(source)

    # ---- .xyz lines → Observation(points, magnetic_field) (doc 03 §2) ----

    def _parse_xyz(self, source: RawSource) -> ParseResult:
        text = _decode(source.data or b"")
        meta = _scan_comments(text)
        try:
            df = pd.read_csv(
                io.StringIO(text), sep=r"\s+", comment="#", engine="python"
            )
        except Exception as e:  # noqa: BLE001
            return ParseResult(warnings=[IngestWarning(
                "bad_xyz", Severity.HIGH, f"unreadable aeromag .xyz: {e}", source.filename
            )])

        cols = {c.lower(): c for c in df.columns}
        xcol = _pick(cols, _X_KEYS)
        ycol = _pick(cols, _Y_KEYS)
        mcol = _pick(cols, _M_KEYS)
        if xcol is None or ycol is None or mcol is None:
            return ParseResult(warnings=[IngestWarning(
                "bad_header", Severity.HIGH,
                f"missing x/y/TMI columns in header {list(df.columns)}", source.filename,
            )])
        zcol = _pick(cols, _Z_KEYS)
        lcol = _pick(cols, _LINE_KEYS)

        total = int(len(df))
        x = pd.to_numeric(df[xcol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        m = pd.to_numeric(df[mcol], errors="coerce")
        z = pd.to_numeric(df[zcol], errors="coerce") if zcol else pd.Series(0.0, index=df.index)
        good = x.notna() & y.notna() & m.notna()
        dropped = int((~good).sum())

        warnings: list[IngestWarning] = []
        if dropped:
            warnings.append(IngestWarning(
                "bad_row", Severity.LOW, f"{dropped} unparseable row(s) skipped", source.filename
            ))

        x, y, m = x[good].to_numpy(), y[good].to_numpy(), m[good].to_numpy()
        z = z[good].fillna(0.0).to_numpy() if zcol else np.zeros(int(good.sum()))
        coords = np.column_stack([x, y, z]) if len(x) else np.zeros((0, 3))

        obs_meta: dict[str, object] = {"product": "RTP"}
        if lcol is not None:
            obs_meta["line"] = pd.to_numeric(df[lcol], errors="coerce")[good].tolist()

        obs = RawObservation(
            geometry_kind="points",
            coords=coords,
            values={"magnetic_field": np.asarray(m, dtype=float)},
            primary_property="magnetic_field",
            meta=obs_meta,
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(
                crs=meta.get("crs") or source.crs_hint,
                vertical_datum=meta.get("vertical_datum"),
                horizontal_unit=meta.get("horizontal_unit", "m"),
                z_convention=meta.get("z_convention", "elevation_up"),
            ),
            units={"magnetic_field": meta.get("unit", "nT")},
            provenance=Provenance(process=f"ingest:{self.name}"),
            warnings=warnings,
            records_total=total,
            records_dropped=dropped,
        )

    # ---- RTP GeoTIFF → PropertyModel(grid2d, magnetic_field) (doc 03 §2) ----

    def _parse_geotiff(self, source: RawSource) -> ParseResult:
        pm, warnings = _read_local_grid(source, "magnetic_field", product="RTP")
        if pm is None:
            return ParseResult(warnings=warnings)
        return ParseResult(
            property_models=[pm],
            source=SourceRef(crs=source.crs_hint, z_convention="elevation_up"),
            units={"magnetic_field": "nT"},
            provenance=Provenance(process=f"ingest:{self.name}"),
            warnings=warnings,
            records_total=1,
            records_dropped=0,
        )


# ─────────────────────────────── helpers ───────────────────────────────


def _decode(sample: bytes) -> str:
    return sample.decode("utf-8", errors="replace")


def _first_data_line(text: str) -> str | None:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return None


def _pick(cols: dict[str, str], keys: set[str]) -> str | None:
    for low, orig in cols.items():
        if low in keys:
            return orig
    return None
