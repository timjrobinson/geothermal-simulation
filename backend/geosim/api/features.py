"""Feature serving + 4D backend (doc 04 §9.2, doc 02 §5, doc 06 §5.3/§5.4/§9.4).

The HTTP read surface over ingested **GeologicalFeatures** (doc 02 §5) and the global
**time axis** (doc 02 §8) the viewer's time slider drives (doc 06 §9.4). Ingested features
land in the catalog ``features`` table with their GeoJSON geometry inline in ``props_json``
(``{"geometry", "props"}``, doc 03 §7 writer); this router resolves each feature kind to
the shape the M2/M4/M5 viewer can load (doc 06 §5.3):

- ``GET /projects/{pid}/features`` — list with ``featureKind`` / ``has_time`` filters
  (doc 04 §9.2). ``has_time`` is derived from the stored props (a microseismic point cloud
  carries its ISO-8601 ``time`` array even though the row flag is unset, doc 02 §8).
- ``GET /features/{id}`` — feature meta + detail (the parsed props, doc 02 §5).
- ``GET /features/{id}/geometry`` — the loadable geometry: a **glTF/.glb triangle mesh**
  for surfaces/faults/solids (server-side converted from the GeoJSON grid/polygon via
  :mod:`geosim.storage.gltf`, doc 06 §5.3) and **GeoJSON** for lines / well paths.
- ``GET /features/{id}/points?bbox=&t0=&t1=`` — a microseismic **4-D point cloud** filtered
  by an optional Engineering bbox + ISO time window, returned as a compact typed-array JSON
  (``x/y/z/t/magnitude``, doc 06 §5.4).
- ``GET /wells/{featureId}/trajectory`` — the min-curvature resolved Engineering polyline +
  per-station MD/TVD (reusing :func:`geosim.spatial.min_curvature_positions`) plus the
  joined well-log curve samples vs MD for tube colouring (doc 06 §5.3).
- ``GET /projects/{pid}/time-extent`` — the **union** of all time-bearing datasets' /
  features' epochs (ISO-8601 UTC) for the global slider (doc 02 §8, doc 06 §9.4).

The router shares the catalog session DI off ``app.state`` (doc 04 §9), mirrors the
property-model router's resolution pattern, and never reimplements the spatial/storage
primitives it reuses.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from geosim.catalog import Dataset as DatasetRow
from geosim.catalog import Feature as FeatureRow
from geosim.catalog import Project as ProjectRow
from geosim.spatial import min_curvature_positions
from geosim.storage import triangulate_grid, write_glb

__all__ = ["build_feature_router"]

# Feature kinds the viewer renders as triangle-mesh glTF surfaces (doc 06 §5.3); the rest
# (well paths, line interpretations) serve as GeoJSON, point clouds via /points.
_SURFACE_KINDS = {"horizon", "fault", "surface", "solid", "isosurface", "salt", "unit"}
# GeoJSON geometry types that genuinely form a 2-D surface mesh (vs a line/point).
_SURFACE_GEOM = {"Polygon", "MultiPolygon"}


# ──────────────────────────────── response models ────────────────────────────────
class FeatureSummary(BaseModel):
    """One row in the project feature list (doc 04 §9.2)."""

    id: str
    featureKind: str
    datasetId: str | None
    storeFormat: str
    bbox: dict[str, float]
    hasTime: bool
    geometryEndpoint: str  # 'gltf' | 'geojson' | 'points'
    props: dict[str, Any]


class FeatureDetail(BaseModel):
    """Full feature meta + detail (doc 02 §5)."""

    id: str
    featureKind: str
    datasetId: str | None
    projectId: str
    storeFormat: str
    bbox: dict[str, float]
    hasTime: bool
    geometryEndpoint: str
    geometryType: str | None
    props: dict[str, Any]


class TimeExtent(BaseModel):
    """The global 4-D time axis union (doc 02 §8, doc 06 §9.4)."""

    epochs: list[str]  # sorted unique ISO-8601 UTC
    t0: str | None
    t1: str | None
    count: int
    sources: list[dict[str, Any]]  # per-contributing artifact {id, kind, n}


# ──────────────────────────────── helpers ────────────────────────────────
def _feature_payload(row: FeatureRow) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Unpack the inline ``props_json`` → ``(geometry, props)`` (doc 03 §7 writer)."""
    if not row.props_json:
        return None, {}
    blob = json.loads(row.props_json)
    return blob.get("geometry"), dict(blob.get("props") or {})


def _geometry_type(geometry: dict[str, Any] | None) -> str | None:
    if isinstance(geometry, dict):
        return geometry.get("type")
    return None


def _times_of(props: dict[str, Any]) -> list[str]:
    """ISO-8601 epochs a feature carries (doc 02 §8): a point-cloud ``time`` array or a
    ``timeAxis.epochs`` block (4-D raster). Empty when the feature is static."""
    times = props.get("time")
    if isinstance(times, list) and times:
        return [str(t) for t in times]
    ta = props.get("timeAxis")
    if isinstance(ta, dict) and isinstance(ta.get("epochs"), list):
        return [str(t) for t in ta["epochs"]]
    return []


def _feature_has_time(row: FeatureRow, props: dict[str, Any]) -> bool:
    """A feature is 4-D if the row flag is set OR its props carry a time axis (doc 02 §8).

    Microseismic clouds ingest with ``time`` in props but the writer leaves ``has_time=0``;
    deriving it from props keeps the ``has_time`` filter correct without a writer change.
    """
    return bool(row.has_time) or bool(_times_of(props))


def _is_grid_surface(props: dict[str, Any]) -> bool:
    """A draped ``(ny, nx)`` height-grid of nodes is a surface mesh, not a cloud (doc 02 §5)."""
    grid = props.get("grid")
    return isinstance(grid, dict) and "ny" in grid and "nx" in grid


def _geometry_endpoint(
    row: FeatureRow, geometry: dict[str, Any] | None, props: dict[str, Any]
) -> str:
    """Which ``/features/{id}/...`` endpoint serves this feature's renderable geometry.

    Surfaces/faults/solids that store a draped point-grid (``props.grid``) or a polygon are
    meshed to glTF (doc 06 §5.3); a microseismic ``pointCloud`` / bare point geometry serves
    via ``/points``; everything else (lines, well paths) passes through as GeoJSON.
    """
    kind = row.feature_type
    gtype = _geometry_type(geometry)
    if kind in _SURFACE_KINDS and (_is_grid_surface(props) or gtype in _SURFACE_GEOM):
        return "gltf"
    if kind == "pointCloud" or gtype in ("MultiPoint", "Point"):
        return "points"
    return "geojson"


def _surface_mesh(
    geometry: dict[str, Any], props: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Build a triangle mesh (verts, tris) from a surface feature's geometry (doc 06 §5.3).

    Two storage shapes are supported (doc 02 §5): an explicit ``(ny, nx)`` draped grid of
    Engineering nodes (``props.grid = {ny, nx}``) which triangulates exactly, and a GeoJSON
    ``Polygon`` ring which fan-triangulates about its centroid.
    """
    gtype = geometry.get("type")
    grid = props.get("grid")
    if isinstance(grid, dict) and "ny" in grid and "nx" in grid:
        pts = np.asarray(geometry["coordinates"], dtype=float).reshape(-1, 3)
        return triangulate_grid(pts, int(grid["ny"]), int(grid["nx"]))

    # Polygon outer ring → fan triangulation about the centroid (planar surface patch).
    if gtype == "Polygon":
        ring = np.asarray(geometry["coordinates"][0], dtype=float)
    elif gtype == "MultiPolygon":
        ring = np.asarray(geometry["coordinates"][0][0], dtype=float)
    else:
        raise HTTPException(status_code=422, detail="feature geometry is not a surface")
    ring = _as_xyz(ring)
    # drop a duplicated closing vertex if present
    if ring.shape[0] >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if ring.shape[0] < 3:
        raise HTTPException(status_code=422, detail="surface ring needs >=3 vertices")
    centroid = ring.mean(axis=0, keepdims=True)
    verts = np.vstack([centroid, ring]).astype(np.float32)
    n = ring.shape[0]
    tris = np.asarray(
        [(0, i + 1, ((i + 1) % n) + 1) for i in range(n)], dtype=np.uint32
    )
    return verts, tris


def _as_xyz(pts: np.ndarray) -> np.ndarray:
    """Coerce ``(N,2|3)`` coords to ``(N,3)`` (pad a missing Z with 0)."""
    pts = np.asarray(pts, dtype=float).reshape(pts.shape[0], -1)
    if pts.shape[1] == 2:
        return np.column_stack([pts, np.zeros(pts.shape[0])])
    return pts[:, :3]


def _point_rows(geometry: dict[str, Any]) -> np.ndarray:
    """Engineering ``(N,3)`` points from a ``MultiPoint``/``Point`` geometry (doc 02 §5)."""
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Point":
        return _as_xyz(np.asarray([coords], dtype=float))
    if gtype == "MultiPoint":
        return _as_xyz(np.asarray(coords, dtype=float))
    raise HTTPException(status_code=422, detail="feature has no point geometry")


def _dataset_epochs(ds: DatasetRow) -> list[str]:
    """ISO epochs from a dataset's ``time_extent_json`` (doc 02 §8 TimeAxis)."""
    if not ds.time_extent_json:
        return []
    try:
        te = json.loads(ds.time_extent_json)
    except json.JSONDecodeError:
        return []
    if isinstance(te, dict):
        epochs = te.get("epochs")
        if isinstance(epochs, list):
            return [str(e) for e in epochs]
        t0, t1 = te.get("t0"), te.get("t1")
        return [str(t) for t in (t0, t1) if t]
    if isinstance(te, list):
        return [str(e) for e in te]
    return []


def build_feature_router(session_dep: Any) -> APIRouter:
    """Build the feature-serving + 4-D router wired to the app's catalog DI (doc 04 §9)."""
    router = APIRouter(tags=["features"])

    def _project_or_404(session: Session, pid: str) -> ProjectRow:
        row = session.get(ProjectRow, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="project not found")
        return row

    def _feature_or_404(session: Session, fid: str) -> FeatureRow:
        row = session.get(FeatureRow, fid)
        if row is None:
            raise HTTPException(status_code=404, detail="feature not found")
        return row

    # ──────────────────────── GET /projects/{pid}/features ────────────────────────
    @router.get("/projects/{pid}/features", response_model=list[FeatureSummary])
    def list_features(
        pid: str,
        session: Session = session_dep,
        featureKind: str | None = Query(default=None),
        has_time: bool | None = Query(default=None),
    ) -> list[FeatureSummary]:
        """List a project's features with optional ``featureKind``/``has_time`` filters."""
        _project_or_404(session, pid)
        q = session.query(FeatureRow).filter(FeatureRow.project_id == pid)
        if featureKind is not None:
            q = q.filter(FeatureRow.feature_type == featureKind)
        out: list[FeatureSummary] = []
        for row in q.order_by(FeatureRow.created_at).all():
            geometry, props = _feature_payload(row)
            ht = _feature_has_time(row, props)
            if has_time is not None and ht != has_time:
                continue
            out.append(FeatureSummary(
                id=row.id,
                featureKind=row.feature_type,
                datasetId=row.dataset_id,
                storeFormat=row.store_format,
                bbox=json.loads(row.bbox_json),
                hasTime=ht,
                geometryEndpoint=_geometry_endpoint(row, geometry, props),
                props=props,
            ))
        return out

    # ──────────────────────────── GET /features/{id} ────────────────────────────
    @router.get("/features/{fid}", response_model=FeatureDetail)
    def get_feature(fid: str, session: Session = session_dep) -> FeatureDetail:
        """Feature meta + detail: the parsed props + the geometry routing (doc 02 §5)."""
        row = _feature_or_404(session, fid)
        geometry, props = _feature_payload(row)
        return FeatureDetail(
            id=row.id,
            featureKind=row.feature_type,
            datasetId=row.dataset_id,
            projectId=row.project_id,
            storeFormat=row.store_format,
            bbox=json.loads(row.bbox_json),
            hasTime=_feature_has_time(row, props),
            geometryEndpoint=_geometry_endpoint(row, geometry, props),
            geometryType=_geometry_type(geometry),
            props=props,
        )

    # ──────────────────────── GET /features/{id}/geometry ────────────────────────
    @router.get("/features/{fid}/geometry")
    def get_feature_geometry(fid: str, session: Session = session_dep) -> Response:
        """The loadable geometry: ``.glb`` for surfaces, GeoJSON otherwise (doc 06 §5.3).

        Surfaces/faults/solids stored as a GeoJSON grid/polygon are converted **server-side**
        into a binary glTF triangle mesh (Engineering coords) the viewer's GLTFLoader
        streams; lines / well paths pass through as a GeoJSON ``Feature``.
        """
        row = _feature_or_404(session, fid)
        geometry, props = _feature_payload(row)
        if geometry is None:
            raise HTTPException(status_code=404, detail="feature has no geometry")
        endpoint = _geometry_endpoint(row, geometry, props)

        if endpoint == "gltf":
            verts, tris = _surface_mesh(geometry, props)
            glb = write_glb(
                verts, tris,
                extras={"featureId": row.id, "featureKind": row.feature_type},
            )
            return Response(
                content=glb,
                media_type="model/gltf-binary",
                headers={"Content-Disposition": f'inline; filename="{row.id}.glb"'},
            )

        # GeoJSON passthrough for lines / well paths / point sets.
        feature_geojson = {
            "type": "Feature",
            "id": row.id,
            "geometry": geometry,
            "properties": {**props, "featureKind": row.feature_type},
        }
        return Response(
            content=json.dumps(feature_geojson),
            media_type="application/geo+json",
        )

    # ──────────────────────── GET /features/{id}/points ────────────────────────
    @router.get("/features/{fid}/points")
    def get_feature_points(
        fid: str,
        session: Session = session_dep,
        bbox: str | None = Query(default=None, description="xmin,xmax,ymin,ymax,zmin,zmax"),
        t0: str | None = Query(default=None, description="ISO-8601 window start (inclusive)"),
        t1: str | None = Query(default=None, description="ISO-8601 window end (inclusive)"),
    ) -> dict[str, Any]:
        """A microseismic 4-D point cloud filtered by bbox + time window (doc 06 §5.4).

        Returns compact parallel typed-arrays ``x/y/z/t/magnitude`` (plus ``depth_m`` when
        present), so the viewer streams the cloud without per-point object overhead. ``t0``/
        ``t1`` are inclusive ISO-8601 UTC bounds; ``bbox`` is Engineering metres.
        """
        row = _feature_or_404(session, fid)
        geometry, props = _feature_payload(row)
        if geometry is None:
            raise HTTPException(status_code=404, detail="feature has no geometry")
        pts = _point_rows(geometry)
        n = pts.shape[0]
        times = _times_of(props)
        mags = props.get("mag") or props.get("magnitude") or []
        depth = props.get("depth_m") or []

        keep = np.ones(n, dtype=bool)
        if bbox is not None:
            box = _parse_bbox(bbox)
            keep &= (
                (pts[:, 0] >= box["xmin"]) & (pts[:, 0] <= box["xmax"])
                & (pts[:, 1] >= box["ymin"]) & (pts[:, 1] <= box["ymax"])
                & (pts[:, 2] >= box["zmin"]) & (pts[:, 2] <= box["zmax"])
            )
        if (t0 is not None or t1 is not None) and times:
            tt = np.array(times[:n], dtype=object)
            if t0 is not None:
                keep[: tt.size] &= np.array([str(t) >= t0 for t in tt])
            if t1 is not None:
                keep[: tt.size] &= np.array([str(t) <= t1 for t in tt])

        idx = np.nonzero(keep)[0]
        sel = pts[idx]
        return {
            "featureId": row.id,
            "count": int(idx.size),
            "x": sel[:, 0].astype(float).tolist(),
            "y": sel[:, 1].astype(float).tolist(),
            "z": sel[:, 2].astype(float).tolist(),
            "t": [times[i] for i in idx if i < len(times)],
            "magnitude": [float(mags[i]) for i in idx if i < len(mags)],
            "depth_m": [float(depth[i]) for i in idx if i < len(depth)],
            "window": {"t0": t0, "t1": t1, "bbox": bbox},
        }

    # ──────────────────────── GET /wells/{featureId}/trajectory ────────────────────────
    @router.get("/wells/{fid}/trajectory")
    def get_well_trajectory(fid: str, session: Session = session_dep) -> dict[str, Any]:
        """Resolved min-curvature Engineering polyline + MD/TVD + joined log samples.

        Re-integrates the deviation survey (``props.deviation_survey`` of ``[MD,INC,AZI]``
        rows) via :func:`geosim.spatial.min_curvature_positions` when present; otherwise
        falls back to the stored ``LineString`` trajectory (doc 02 §5 wellPath). The joined
        well-log curves (LAS samples vs MD, doc 06 §5.3 tube colouring) are pulled from the
        sibling ``wellcurve`` observation matched by ``wellId``.
        """
        row = _feature_or_404(session, fid)
        if row.feature_type != "wellPath":
            raise HTTPException(status_code=422, detail="feature is not a well path")
        geometry, props = _feature_payload(row)
        well_id = props.get("wellId")
        wellhead = props.get("wellhead") or [0.0, 0.0, 0.0]

        survey = props.get("deviation_survey")
        if survey:
            mc = min_curvature_positions(
                np.asarray(survey, dtype=float),
                (float(wellhead[0]), float(wellhead[1])),
                kb_elev=float(wellhead[2]) if len(wellhead) > 2 else 0.0,
            )
            polyline = mc.enu.astype(float).tolist()
            md = mc.md.astype(float).tolist()
            tvd = mc.tvd.astype(float).tolist()
            dls = mc.dls.astype(float).tolist()
        else:
            # No survey persisted: reuse the stored Engineering polyline + derive MD/TVD.
            coords = _as_xyz(np.asarray(geometry["coordinates"], dtype=float))
            polyline = coords.astype(float).tolist()
            seg = np.r_[0.0, np.linalg.norm(np.diff(coords, axis=0), axis=1)]
            md = np.cumsum(seg).astype(float).tolist()
            kb = float(coords[0, 2]) if coords.shape[0] else 0.0
            tvd = (kb - coords[:, 2]).astype(float).tolist()
            dls = [0.0] * len(md)

        logs = _well_logs(session, row.project_id, row.dataset_id, well_id)
        return {
            "featureId": row.id,
            "wellId": well_id,
            "trajectory": props.get("trajectory"),
            "wellhead": list(wellhead),
            "polyline": polyline,   # Engineering XYZ per station
            "md": md,               # measured depth per station (m)
            "tvd": tvd,             # true vertical depth below MD datum (m, +down)
            "dls": dls,             # dogleg severity per interval (°/30 m)
            "logs": logs,           # joined LAS curves vs MD for tube colouring
        }

    # ──────────────────────── GET /projects/{pid}/time-extent ────────────────────────
    @router.get("/projects/{pid}/time-extent", response_model=TimeExtent)
    def get_time_extent(pid: str, session: Session = session_dep) -> TimeExtent:
        """Union of every time-bearing dataset/feature's epochs for the slider (doc 06 §9.4).

        Aggregates dataset ``time_extent_json`` TimeAxes (4-D rasters/volumes, doc 02 §8)
        with time-bearing features' inline epochs (microseismic clouds), so the global slider
        spans the full project history regardless of which artifact carries the time.
        """
        _project_or_404(session, pid)
        epochs: set[str] = set()
        sources: list[dict[str, Any]] = []

        for ds in (
            session.query(DatasetRow).filter(DatasetRow.project_id == pid).all()
        ):
            ds_epochs = _dataset_epochs(ds)
            if ds_epochs:
                epochs.update(ds_epochs)
                sources.append({"id": ds.id, "kind": "dataset",
                                "method": ds.method, "n": len(ds_epochs)})

        for row in (
            session.query(FeatureRow).filter(FeatureRow.project_id == pid).all()
        ):
            _geom, props = _feature_payload(row)
            ft = _times_of(props)
            if ft:
                epochs.update(ft)
                sources.append({"id": row.id, "kind": "feature",
                                "featureKind": row.feature_type, "n": len(ft)})

        ordered = sorted(epochs)
        return TimeExtent(
            epochs=ordered,
            t0=ordered[0] if ordered else None,
            t1=ordered[-1] if ordered else None,
            count=len(ordered),
            sources=sources,
        )

    return router


def _well_logs(
    session: Session, project_id: str, dataset_id: str | None, well_id: str | None
) -> dict[str, Any]:
    """Joined well-log curves (LAS samples vs MD) for the path's ``wellId`` (doc 06 §5.3).

    The well-log adapter writes a ``wellcurve`` observation alongside the ``wellPath``
    feature in the SAME dataset, keyed by ``wellId`` (doc 03 §3d). We load that observation's
    inline ``values_json`` + ``meta.md`` so the viewer can colour the trajectory tube by any
    curve vs measured depth.
    """
    from geosim.catalog import Observation as ObservationRow

    q = session.query(ObservationRow).filter(
        ObservationRow.project_id == project_id,
        ObservationRow.geometry_kind == "wellcurve",
    )
    if dataset_id is not None:
        q = q.filter(ObservationRow.dataset_id == dataset_id)
    for obs in q.all():
        meta = json.loads(obs.meta_json) if obs.meta_json else {}
        if well_id is not None and meta.get("wellId") not in (None, well_id):
            continue
        vals = json.loads(obs.values_json) if obs.values_json else {}
        curves = {k: v for k, v in (vals.get("values") or {}).items()}
        return {
            "wellId": meta.get("wellId", well_id),
            "md": meta.get("md", []),
            "curves": curves,                 # {property: [samples vs MD]}
            "primaryProperty": obs.primary_property,
        }
    return {"wellId": well_id, "md": [], "curves": {}, "primaryProperty": None}


def _parse_bbox(spec: str) -> dict[str, float]:
    """Parse ``xmin,xmax,ymin,ymax,zmin,zmax`` → an Engineering-metre AABB (doc 04 §2.5)."""
    try:
        xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in spec.split(","))
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="bbox must be 'xmin,xmax,ymin,ymax,zmin,zmax'",
        ) from exc
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax}
