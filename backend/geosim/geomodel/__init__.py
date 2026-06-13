"""Implicit geological modelling (GemPy) — OVERVIEW §6 L4 / doc 07 (M8).

Builds an **implicit** geomodel constrained by doc-02 horizon/fault surfaces + well
formation tops, over the project ``SpatialFrame`` ROI × depthRange (Engineering Frame,
Z-up, doc 01 §1), on the GemPy *numpy* backend. Exposes the result two ways (doc 02):

- a categorical **lithology PropertyModel** (``lithology_class``, doc 02 §10.2) — hard
  per-cell labels + a categories table, plus a smooth class-probability axis; and
- one **unitSolid GeologicalFeature per stratigraphic unit** (doc 02 §5) as a ``.glb``.

Sits beside the M7 fusion pipeline, not on its critical path. The HTTP surface is
``POST /projects/{pid}/geomodel`` (see :func:`build_geomodel_router`), wired into
:func:`geosim.api.create_app`.
"""

from __future__ import annotations

from .builder import (
    GeoModelResult,
    GeoModelSpec,
    GeoUnit,
    InterfacePoint,
    Orientation,
    build_geomodel,
    lith_to_zyx,
    spec_from_catalog_surfaces,
)
from .writer import PersistedGeoModel, persist_geomodel

__all__ = [
    "InterfacePoint",
    "Orientation",
    "GeoUnit",
    "GeoModelSpec",
    "GeoModelResult",
    "build_geomodel",
    "lith_to_zyx",
    "spec_from_catalog_surfaces",
    "PersistedGeoModel",
    "persist_geomodel",
]
