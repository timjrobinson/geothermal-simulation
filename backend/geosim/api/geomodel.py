"""Implicit-geomodel build endpoint (OVERVIEW §6 L4 / doc 07, M8).

``POST /projects/{pid}/geomodel`` builds a GemPy implicit model over the project's
``SpatialFrame`` ROI × depthRange (Engineering Frame, Z-up, doc 01 §1) from a supplied set
of horizon/fault **surface** contacts + optional well **formation tops** (doc 02 §5), then
catalogs the result: a categorical ``lithology_class`` **PropertyModel** (doc 02 §10.2) and
one **unitSolid** ``GeologicalFeature`` per stratigraphic unit (doc 02 §5). The grid is
small/coarse so the numpy-backend build runs synchronously in seconds (CLAUDE.md); the same
shape swaps to an :class:`~geosim.jobs.InlineJobRunner` job for larger grids.

The router shares the catalog session DI + ``storage_root`` off ``app.state`` (doc 04 §9),
reuses :func:`geosim.geomodel.spec_from_catalog_surfaces` / :func:`build_geomodel` /
:func:`persist_geomodel`, and never reimplements the spatial/storage primitives.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from geosim.catalog import Project as ProjectRow
from geosim.geomodel import build_geomodel, persist_geomodel, spec_from_catalog_surfaces
from geosim.storage import ProjectLayout

from .frame_io import frame_from_row

__all__ = ["build_geomodel_router"]


class SurfaceContact(BaseModel):
    """One horizon/fault contact with sampled interface points (doc 02 §5)."""

    name: str
    kind: str = Field(default="horizon", description="'horizon' | 'fault'")
    points: list[list[float]] = Field(description="Engineering XYZ interface points, Z-up")
    orientation: list[float] | None = Field(
        default=None, description="surface normal (gx,gy,gz); inferred flat/vertical if omitted"
    )
    color: str | None = None


class WellTop(BaseModel):
    """A formation top picked along a well path (doc 02 §5)."""

    surface: str
    x: float
    y: float
    z: float


class GeoModelBuildRequest(BaseModel):
    """``POST /projects/{pid}/geomodel`` body — contacts + tops + coarse resolution."""

    surfaces: list[SurfaceContact]
    wellTops: list[WellTop] = Field(default_factory=list)
    resolution: tuple[int, int, int] = (20, 20, 20)


class GeoModelBuildResponse(BaseModel):
    """The created catalog ids + a small summary (doc 04 §9.2 shape)."""

    datasetId: str
    propertyModelId: str
    provenanceId: str
    unitSolidFeatureIds: list[str]
    lithologyStoreUri: str
    categories: list[dict[str, Any]]
    shape: list[int]


def build_geomodel_router(session_dep: Any) -> APIRouter:
    """Build the geomodel router wired to the app's catalog + storage DI (doc 04 §9)."""
    router = APIRouter(tags=["geomodel"])

    @router.post(
        "/projects/{pid}/geomodel",
        response_model=GeoModelBuildResponse,
        status_code=201,
    )
    def build_project_geomodel(
        pid: str,
        body: GeoModelBuildRequest,
        request: Request,
        session: Session = session_dep,  # noqa: B008 — FastAPI DI marker
    ) -> GeoModelBuildResponse:
        row = session.get(ProjectRow, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="project not found")
        if not body.surfaces:
            raise HTTPException(status_code=422, detail="at least one surface contact required")

        frame = frame_from_row(row.spatial_frame)
        spec = spec_from_catalog_surfaces(
            frame,
            [s.model_dump() for s in body.surfaces],
            well_tops=[t.model_dump() for t in body.wellTops],
            resolution=tuple(body.resolution),
            project_name=f"geomodel-{pid}",
        )

        try:
            result = build_geomodel(spec)
        except Exception as exc:  # pragma: no cover - surfaced as a 422 to the caller
            raise HTTPException(status_code=422, detail=f"geomodel build failed: {exc}") from exc

        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, pid)
        persisted = persist_geomodel(
            session, layout, pid, result, extent=spec.extent(), created_by="api",
        )

        return GeoModelBuildResponse(
            datasetId=persisted.dataset_id,
            propertyModelId=persisted.property_model_id,
            provenanceId=persisted.provenance_id,
            unitSolidFeatureIds=persisted.unit_solid_feature_ids,
            lithologyStoreUri=persisted.lithology_store_uri,
            categories=result.categories,
            shape=list(result.shape_zyx),
        )

    return router
