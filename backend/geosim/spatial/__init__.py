"""Spatial framework (doc 01) — the coordinate, datum, and units foundation.

Nothing downstream invents its own coordinate handling; it all flows through here:
the Engineering Frame, CRS/datum transforms, the units registry, the property-type
registry, and depth/MD/TVD + minimum-curvature helpers.
"""

from .frame import (
    Aabb,
    Anchor,
    DepthRange,
    FrameMode,
    GeorefStatus,
    SpatialFrame,
    utm_epsg_for_lonlat,
)
from .property_types import REGISTRY, PropertyType, PropertyTypeRegistry
from .units import CANONICAL_UNITS, Q_, convert, to_canonical, to_display, ureg
from .vertical import (
    MinCurvatureResult,
    depth_to_elevation,
    elevation_to_depth,
    elevation_to_tvdss,
    min_curvature_positions,
    tvd_to_elevation,
)

__all__ = [
    # frame
    "SpatialFrame", "FrameMode", "GeorefStatus", "Anchor", "Aabb", "DepthRange",
    "utm_epsg_for_lonlat",
    # units
    "ureg", "Q_", "CANONICAL_UNITS", "to_canonical", "to_display", "convert",
    # property types
    "PropertyType", "PropertyTypeRegistry", "REGISTRY",
    # vertical
    "elevation_to_depth", "depth_to_elevation", "tvd_to_elevation", "elevation_to_tvdss",
    "min_curvature_positions", "MinCurvatureResult",
]
