"""Portable Engineering-metre bbox-intersection query helper (doc 04 §2.5).

Hot-path spatial queries ("which artifacts intersect this clip box / this
slice?") use a 3D bbox index on bounding boxes stored in **Engineering metres**
(doc 01 §1, doc 04 §2.2) — *not* lat/lon, so georeferencing never rewrites the
index. Doc 04 §2.5 specifies two backends behind one SQLAlchemy code path:

- **PostGIS GiST** on a ``box3d`` / ``geometry(...,3D)`` column built from
  ``extent_json`` — the primary engine. Gated by the ``is_postgis`` capability
  flag (doc 04 §2.1) and stubbed here as the documented seam (``_postgis_*``).
- **Portable numeric range query** on the ``*_extent``/``bbox`` JSON-derived
  columns — runs on SQLite (and any SQL engine). This is the path tests exercise.

Two AABBs in the same frame intersect iff they overlap on **every** axis
(``a.min <= b.max AND a.max >= b.min`` per axis). The JSON extent columns
(``extent_json`` on datasets, ``bbox_json`` on the typed artifact tables) are
parsed in Python for the SQLite fallback; the documented PostGIS seam pushes the
predicate into a GiST ``&&`` operator on the indexed geometry instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import is_postgis
from .models import Dataset, Feature, FusedModel, Observation, PropertyModel

__all__ = [
    "Bbox3D",
    "aabb_from_json",
    "boxes_intersect",
    "query_datasets_bbox",
    "query_artifacts_bbox",
]


@dataclass(frozen=True)
class Bbox3D:
    """An axis-aligned bounding box in Engineering metres (doc 01 §1)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)


def aabb_from_json(blob: str | dict) -> Bbox3D:
    """Parse a doc-02 ``Aabb`` JSON object/string into a :class:`Bbox3D`."""
    d = json.loads(blob) if isinstance(blob, str) else blob
    return Bbox3D(
        float(d["xmin"]), float(d["xmax"]),
        float(d["ymin"]), float(d["ymax"]),
        float(d["zmin"]), float(d["zmax"]),
    )


def boxes_intersect(a: Bbox3D, b: Bbox3D) -> bool:
    """True iff two Engineering-metre AABBs overlap on every axis (doc 04 §2.5).

    Touching boundaries count as intersecting (inclusive ``<=``/``>=``).
    """
    return (
        a.xmin <= b.xmax and a.xmax >= b.xmin
        and a.ymin <= b.ymax and a.ymax >= b.ymin
        and a.zmin <= b.zmax and a.zmax >= b.zmin
    )


def query_datasets_bbox(
    session: Session, project_id: str, query_box: Bbox3D
) -> list[Dataset]:
    """Return the project's datasets whose ``extent`` intersects ``query_box``.

    On the PostGIS primary the bbox predicate is pushed into the GiST 3D index
    (the seam in ``_postgis_dataset_query``); on the SQLite fallback we range-test
    the parsed ``extent_json`` columns in Python (doc 04 §2.5).
    """
    if is_postgis(session.get_bind()):  # pragma: no cover - needs a live PostGIS server
        return _postgis_dataset_query(session, project_id, query_box)
    rows = session.scalars(
        select(Dataset).where(Dataset.project_id == project_id)
    ).all()
    return [d for d in rows if boxes_intersect(aabb_from_json(d.extent_json), query_box)]


def query_artifacts_bbox(
    session: Session,
    project_id: str,
    query_box: Bbox3D,
    *,
    kinds: list[str] | None = None,
) -> list[object]:
    """Return artifact rows (property models, fused models, features, observations)
    whose ``bbox`` intersects ``query_box`` — the §2.5 ``/artifacts?bbox=`` query.

    ``kinds`` filters by doc 02 §2 kind (``propertyModel|fusedModel|feature|
    observation``); ``None`` searches all four typed tables. The SQLite fallback
    mirrors the PostGIS R-Tree-backed query by parsing each table's ``bbox_json``.
    """
    if is_postgis(session.get_bind()):  # pragma: no cover - needs a live PostGIS server
        return _postgis_artifact_query(session, project_id, query_box, kinds)

    wanted = set(kinds) if kinds else {"propertyModel", "fusedModel", "feature", "observation"}
    table_for_kind: dict[str, type] = {
        "propertyModel": PropertyModel,
        "fusedModel": FusedModel,
        "feature": Feature,
        "observation": Observation,
    }
    out: list[object] = []
    for kind in wanted:
        model = table_for_kind[kind]
        rows = session.scalars(
            select(model).where(model.project_id == project_id)
        ).all()
        out.extend(
            r for r in rows if boxes_intersect(aabb_from_json(r.bbox_json), query_box)
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# PostGIS GiST seam (doc 04 §2.5 / §2.1) — confined behind the capability flag.
# These are intentionally not exercised by the SQLite tests; they document the
# path the primary engine takes: a GiST ``&&`` (bbox-overlap) predicate on an
# indexed ``box3d`` / ``geometry(...,3D)`` column derived from ``extent_json``.
# ──────────────────────────────────────────────────────────────────────────


def _postgis_predicate_sql(column: str, box: Bbox3D) -> str:  # pragma: no cover
    """The GiST 3D bbox-overlap predicate for ``column`` (doc 04 §2.5 seam).

    On PostGIS the indexed column would be a ``box3d`` built once from the row's
    Engineering-metre extent; the query box becomes a literal ``box3d`` and the
    ``&&`` operator hits the GiST index directly.
    """
    qb = (
        f"box3d('BOX3D({box.xmin} {box.ymin} {box.zmin},"
        f"{box.xmax} {box.ymax} {box.zmax})')"
    )
    return f"{column} && {qb}"


def _postgis_dataset_query(  # pragma: no cover - needs a live PostGIS server
    session: Session, project_id: str, query_box: Bbox3D
) -> list[Dataset]:
    raise NotImplementedError(
        "PostGIS GiST dataset bbox query is the documented primary-engine seam "
        "(doc 04 §2.5); enable it behind is_postgis() once a box3d/geometry column "
        "is materialised from extent_json."
    )


def _postgis_artifact_query(  # pragma: no cover - needs a live PostGIS server
    session: Session, project_id: str, query_box: Bbox3D, kinds: list[str] | None
) -> list[object]:
    raise NotImplementedError(
        "PostGIS GiST artifact bbox query is the documented primary-engine seam "
        "(doc 04 §2.5)."
    )
