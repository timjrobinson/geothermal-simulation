"""SpatialFrame ↔ catalog row / JSON (de)serialization (doc 04 §9.2, doc 01 §2).

The catalog's :class:`~geosim.catalog.models.SpatialFrameRow` stores the doc-01
:class:`~geosim.spatial.frame.SpatialFrame` flattened into columns plus a canonical
``frame_json`` blob (doc 04 §2.4). This module is the single place that round-trips a
``SpatialFrame`` through that row and through the wire ``frame`` object the REST shapes
(doc 04 §9.2 ``POST /projects {name, frame?}``) exchange — so the frame contract lives in
exactly one spot.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from geosim.spatial import (
    Aabb,
    Anchor,
    DepthRange,
    FrameMode,
    GeorefStatus,
    SpatialFrame,
)

__all__ = [
    "frame_to_dict",
    "frame_from_dict",
    "frame_row_kwargs",
    "frame_from_row",
]


def frame_to_dict(frame: SpatialFrame) -> dict[str, Any]:
    """Serialize a :class:`SpatialFrame` to a JSON-safe dict (the wire ``frame`` shape)."""
    return {
        "mode": frame.mode.value,
        "horizontal_crs": frame.horizontal_crs,
        "vertical_datum": frame.vertical_datum,
        "anchor": asdict(frame.anchor) if frame.anchor is not None else None,
        "rotation_deg": frame.rotation_deg,
        "axis_convention": frame.axis_convention,
        "length_unit": frame.length_unit,
        "roi": asdict(frame.roi),
        "depth_range": asdict(frame.depth_range),
        "surface_model": frame.surface_model,
        "georef_status": frame.georef_status.value,
    }


def frame_from_dict(data: dict[str, Any] | None) -> SpatialFrame:
    """Build a :class:`SpatialFrame` from a wire ``frame`` dict (defaults → local frame)."""
    if not data:
        return SpatialFrame()
    anchor = data.get("anchor")
    roi = data.get("roi")
    depth = data.get("depth_range")
    return SpatialFrame(
        mode=FrameMode(data.get("mode", FrameMode.LOCAL.value)),
        horizontal_crs=data.get("horizontal_crs"),
        vertical_datum=data.get("vertical_datum"),
        anchor=Anchor(**anchor) if anchor else None,
        rotation_deg=float(data.get("rotation_deg", 0.0)),
        length_unit=data.get("length_unit", "m"),
        roi=Aabb(**roi) if roi else Aabb(-5000, 5000, -5000, 5000),
        depth_range=DepthRange(**depth) if depth else DepthRange(-8000, 2000),
        surface_model=data.get("surface_model", "flat:0"),
        georef_status=GeorefStatus(
            data.get("georef_status", GeorefStatus.ASSUMED_LOCAL.value)
        ),
    )


def frame_row_kwargs(frame: SpatialFrame) -> dict[str, Any]:
    """Flatten a :class:`SpatialFrame` into ``SpatialFrameRow`` column kwargs (doc 04 §2.4)."""
    return {
        "mode": frame.mode.value,
        "horizontal_crs": frame.horizontal_crs,
        "vertical_datum": frame.vertical_datum,
        "anchor_json": json.dumps(asdict(frame.anchor)) if frame.anchor else None,
        "rotation_deg": frame.rotation_deg,
        "axis_convention": frame.axis_convention,
        "length_unit": frame.length_unit,
        "roi_json": json.dumps(asdict(frame.roi)),
        "depth_range_json": json.dumps(asdict(frame.depth_range)),
        "surface_model": frame.surface_model,
        "frame_json": json.dumps(frame_to_dict(frame)),
    }


def frame_from_row(row: Any) -> SpatialFrame:
    """Rebuild a :class:`SpatialFrame` from a ``SpatialFrameRow`` (via its ``frame_json``)."""
    return frame_from_dict(json.loads(row.frame_json))
