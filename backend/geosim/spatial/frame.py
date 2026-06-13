"""The SpatialFrame and Engineering-Frame transforms (doc 01 §1–§3, §7).

Everything internal lives in one **Engineering Frame**: local right-handed ENU
(X=East, Y=North, Z=Up), metres, origin at a project-chosen anchor. Georeferencing is
*just* an optional rigid transform from the Engineering Frame to a real-world CRS:

    crs_easting  = anchor.E + ( x·cosθ − y·sinθ )
    crs_northing = anchor.N + ( x·sinθ + y·cosθ )
    crs_elev     = anchor.elevation + z            (θ = rotationDeg)

In **local mode** the transform is identity (Engineering == world). Bulk arrays are
*always* stored in Engineering coordinates, so promoting local→georeferenced never
reprocesses arrays (doc 01 §2, decision #2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np

__all__ = [
    "FrameMode",
    "GeorefStatus",
    "Anchor",
    "Aabb",
    "DepthRange",
    "SpatialFrame",
    "utm_epsg_for_lonlat",
]


class FrameMode(str, Enum):
    GEOREFERENCED = "georeferenced"
    LOCAL = "local"


class GeorefStatus(str, Enum):
    # georeferencing QUALITY, separate from `mode` (doc 01 §2, resolves critique #9)
    UNKNOWN = "unknown"
    ASSUMED_LOCAL = "assumed_local"
    ANCHORED = "anchored"
    VALIDATED = "validated"
    SURVEY_CONTROLLED = "survey_controlled"


@dataclass
class Anchor:
    easting: float
    northing: float
    elevation: float


@dataclass
class Aabb:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class DepthRange:
    zmin: float
    zmax: float  # Engineering elevation metres (zmax ≈ surface top)


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Auto-select the UTM zone EPSG code containing (lon, lat) — doc 01 §3 default.

    Returns 326## (N hemisphere) / 327## (S hemisphere). At 1–10 km extent distortion is
    negligible and the code is a standard EPSG every external tool understands. Basin/
    regional escalation to a custom ROI-centred Transverse Mercator is handled elsewhere.
    """
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class SpatialFrame:
    """Per-project spatial frame (small catalog metadata, not bulk data) — doc 01 §2."""

    mode: FrameMode = FrameMode.LOCAL

    # --- georeferencing (None in local mode) ---
    horizontal_crs: str | None = None  # projected CRS, e.g. "EPSG:32612" or WKT2
    vertical_datum: str | None = None  # e.g. "EPSG:3855" (EGM2008), "ellipsoidal", "local"
    anchor: Anchor | None = None
    rotation_deg: float = 0.0  # azimuth of Engineering +X CW from CRS East, about Z

    # --- always present ---
    axis_convention: Literal["ENU"] = "ENU"
    length_unit: str = "m"
    roi: Aabb = field(default_factory=lambda: Aabb(-5000, 5000, -5000, 5000))
    depth_range: DepthRange = field(default_factory=lambda: DepthRange(-8000, 2000))
    surface_model: str | None = "flat:0"  # "dem:copernicus-30m" | "flat:0" | "synthetic:<id>"
    georef_status: GeorefStatus = GeorefStatus.ASSUMED_LOCAL

    # ------------------------------------------------------------------ transforms

    def _rot(self) -> tuple[float, float]:
        th = math.radians(self.rotation_deg)
        return math.cos(th), math.sin(th)

    def engineering_to_crs(self, xyz):
        """Engineering XYZ (m) → projected CRS easting/northing/elev. doc 01 §2.

        ``xyz`` is an (N,3) array-like. In local mode this is identity.
        """
        pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if self.mode is FrameMode.LOCAL or self.anchor is None:
            return pts.copy()
        c, s = self._rot()
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        e = self.anchor.easting + (x * c - y * s)
        n = self.anchor.northing + (x * s + y * c)
        elev = self.anchor.elevation + z
        return np.column_stack([e, n, elev])

    def crs_to_engineering(self, enu):
        """Projected CRS easting/northing/elev → Engineering XYZ (m). Inverse of above."""
        pts = np.asarray(enu, dtype=float).reshape(-1, 3)
        if self.mode is FrameMode.LOCAL or self.anchor is None:
            return pts.copy()
        c, s = self._rot()
        de = pts[:, 0] - self.anchor.easting
        dn = pts[:, 1] - self.anchor.northing
        # inverse rotation (rotation matrix is orthogonal → transpose)
        x = de * c + dn * s
        y = -de * s + dn * c
        z = pts[:, 2] - self.anchor.elevation
        return np.column_stack([x, y, z])

    def to_engineering(self, points_xyz, src_crs: str | None = None,
                       src_vertical: str | None = None):
        """Transform incoming real-world coords into the Engineering Frame (doc 01 §7).

        ``src_crs`` is the source horizontal CRS (EPSG/WKT2). When the project is
        georeferenced and ``src_crs`` differs from the project CRS, we reproject with
        ``pyproj`` first, then apply the rigid Engineering transform. In local mode (or
        ``src_crs is None`` for a local project) coordinates are assumed already-Engineering.
        """
        pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
        if self.mode is FrameMode.LOCAL:
            return pts.copy()
        if self.horizontal_crs is None or self.anchor is None:
            raise ValueError("georeferenced frame missing horizontal_crs/anchor")
        if src_crs and src_crs != self.horizontal_crs:
            from pyproj import Transformer

            tr = Transformer.from_crs(src_crs, self.horizontal_crs, always_xy=True)
            e, n = tr.transform(pts[:, 0], pts[:, 1])
            pts = np.column_stack([e, n, pts[:, 2]])
        return self.crs_to_engineering(pts)

    def to_lonlat(self, points_xyz):
        """Engineering XYZ → (lon, lat, elev) for basemaps (doc 01 §2). Georeferenced only."""
        if self.mode is FrameMode.LOCAL or self.horizontal_crs is None:
            raise ValueError("to_lonlat requires a georeferenced frame")
        from pyproj import Transformer

        crs = self.engineering_to_crs(points_xyz)
        tr = Transformer.from_crs(self.horizontal_crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(crs[:, 0], crs[:, 1])
        return np.column_stack([lon, lat, crs[:, 2]])

    # ------------------------------------------------------------------ promotion

    def georeference(self, *, horizontal_crs: str, anchor: Anchor,
                     vertical_datum: str | None = None, rotation_deg: float = 0.0,
                     status: GeorefStatus = GeorefStatus.ANCHORED) -> "SpatialFrame":
        """Promote local → georeferenced (doc 01 §2). Bulk arrays are NOT reprocessed —
        only frame metadata changes (arrays are always Engineering). Assigning an anchor
        sets ``georef_status='anchored'``, NOT ``'validated'`` — it does not assert the
        data is physically correct in the world (doc 01 §2 note, critique #9).
        """
        self.mode = FrameMode.GEOREFERENCED
        self.horizontal_crs = horizontal_crs
        self.vertical_datum = vertical_datum or self.vertical_datum
        self.anchor = anchor
        self.rotation_deg = rotation_deg
        self.georef_status = status
        return self

    @classmethod
    def for_real_site(cls, *, lon: float, lat: float, surface_elev: float,
                      roi: Aabb, depth_range: DepthRange,
                      vertical_datum: str = "EPSG:3855") -> "SpatialFrame":
        """Build a georeferenced frame anchored at a real (lon, lat) with auto-UTM CRS.

        Anchor defaults to the ROI centroid at surface elevation, keeping Engineering
        coordinates centred on zero (doc 01 §3 step 4).
        """
        from pyproj import Transformer

        epsg = utm_epsg_for_lonlat(lon, lat)
        crs = f"EPSG:{epsg}"
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        easting, northing = tr.transform(lon, lat)
        return cls(
            mode=FrameMode.GEOREFERENCED,
            horizontal_crs=crs,
            vertical_datum=vertical_datum,
            anchor=Anchor(float(easting), float(northing), float(surface_elev)),
            roi=roi,
            depth_range=depth_range,
            surface_model="dem:copernicus-30m",
            georef_status=GeorefStatus.ANCHORED,
        )
