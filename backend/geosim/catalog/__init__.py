"""Catalog DB — the metadata index / source-of-truth (doc 04 §2; doc 02 §2–§11).

The catalog holds **metadata only** (doc 04 §1): kind-prefixed ULID rows, bounding
boxes in Engineering metres (doc 01 §1), provenance lineage, viewer layer/view
state, and job records — never bulk samples, only pointers into the array/raw
stores. Primary engine is PostgreSQL + PostGIS (doc 04 §2.1); an embedded SQLite
build is the lightweight fallback used by tests.

This package provides:
- ORM models for every doc 04 §2.4 table (``models``),
- a kind-prefixed ULID id helper (``ids``, doc 02 §1),
- an engine/session factory + ``create_all`` bootstrap (``db``, doc 04 §2.1),
- a portable Engineering-metre bbox-intersection query helper with a documented
  PostGIS GiST seam (``spatial``, doc 04 §2.5).
"""

from .db import (
    create_all,
    default_sqlite_url,
    drop_all,
    is_postgis,
    is_sqlite,
    make_engine,
    session_factory,
)
from .ids import IdKind, is_kind, new_id, prefix_of
from .models import (
    Base,
    Dataset,
    Feature,
    FusedLayer,
    FusedModel,
    Job,
    Layer,
    Observation,
    Project,
    PropertyModel,
    Provenance,
    ProvenanceInput,
    RawFile,
    SpatialFrameRow,
    View,
    now_ms,
)
from .spatial import (
    Bbox3D,
    aabb_from_json,
    boxes_intersect,
    query_artifacts_bbox,
    query_datasets_bbox,
)

__all__ = [
    # ids (doc 02 §1)
    "IdKind", "new_id", "prefix_of", "is_kind",
    # db (doc 04 §2.1)
    "make_engine", "session_factory", "create_all", "drop_all",
    "default_sqlite_url", "is_postgis", "is_sqlite", "now_ms",
    # models (doc 04 §2.4)
    "Base", "Project", "SpatialFrameRow", "Dataset", "PropertyModel",
    "FusedModel", "FusedLayer", "Observation", "Feature", "Provenance",
    "ProvenanceInput", "RawFile", "Layer", "View", "Job",
    # spatial (doc 04 §2.5)
    "Bbox3D", "aabb_from_json", "boxes_intersect",
    "query_datasets_bbox", "query_artifacts_bbox",
]
