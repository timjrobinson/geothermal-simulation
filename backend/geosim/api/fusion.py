"""Fusion API router — fused-grid create/resample/get + artifact discovery (doc 07 §6, doc 04 §9.2).

Wired into :func:`geosim.api.create_app`. Endpoints (doc 07 §6 backend API sketch):

- ``POST /fused`` — create a regular-voxel FusedEarthModel container grid for a project
  (auto-resolution per doc 07 §1.1, or an explicit ``spacing``/``bbox`` override).
- ``POST /fused/{gridId}/resample`` — resample a native PropertyModel INTO the grid as a
  footprint-honest, σ-propagated :class:`~geosim.catalog.FusedLayer` (doc 07 §2).
- ``GET /fused/{gridId}`` — the fused-model handle: grid geometry + its resampled layers.
- ``POST /fused/{gridId}/sample`` — co-located multi-volume sampling → feature matrix +
  cell-index mask (doc 07 §3.1).
- ``POST /fused/{gridId}/crossplot`` — 2D/3D scatter or 2D density + histogram +
  correlation-matrix payloads for the analysis panels (doc 07 §3.2).
- ``POST /fused/{gridId}/cluster`` — k-means / GMM clustering → categorical class volume
  (+ GMM per-class probability volumes) written as derived PropertyModels; synchronous for
  small working sets, job-based for whole-grid runs (doc 07 §3.3/§3.4).
- ``GET /projects/{pid}/artifacts`` — catalog discovery for the frontend (doc 04 §7/§9.2):
  bbox/kind/method/property/time-filtered :class:`ArtifactSummary` list, via the
  Engineering-metre bbox helper :func:`geosim.catalog.query_artifacts_bbox` (doc 04 §2.5).

Shares the catalog session + ``storage_root`` injected on ``app.state`` by
:func:`create_app`; reuses :mod:`geosim.fusion` for all compute (never reimplements it).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from geosim.catalog import (
    Bbox3D,
    Dataset,
    FusedModel,
    Project,
    PropertyModel,
    query_artifacts_bbox,
)
from geosim.fusion import (
    SYNC_CELL_LIMIT,
    FavorabilitySpec,
    Probe,
    build_fused_model,
    calibrate_transform,
    cluster_fused,
    compute_favorability,
    correlation_matrix,
    crossplot,
    fused_grid_from_row,
    histogram,
    probes_from_deviation_survey,
    resample_to_fused,
    run_transform,
    sample_fused,
)
from geosim.fusion.transform import Transform
from geosim.jobs import JobRunner, ProgressReporter
from geosim.plugins import get_registry
from geosim.storage import ProjectLayout

__all__ = ["build_fusion_router"]


# ──────────────────────────────── wire shapes (doc 07 §6) ────────────────────────────────


class FusedCreate(BaseModel):
    """``POST /fused`` body (doc 07 §1.1)."""

    project_id: str
    name: str = "fused"
    source_property_model_ids: list[str] | None = None
    bbox: dict[str, float] | None = None
    spacing: list[float] | None = None  # explicit [dz,dy,dx] override of auto-resolution


class FusedLayerOut(BaseModel):
    layer_id: str
    property: str
    source_property_model_id: str
    source_version: str
    method: str
    sigma_array: str | None
    coverage_mask: str | None


class FusedModelOut(BaseModel):
    """``GET /fused/{gridId}`` (doc 07 §6)."""

    id: str
    project_id: str
    grid_type: str
    origin: list[float]
    spacing: list[float]
    shape: list[int]
    n_cells: int
    bbox: dict[str, float]
    layers: list[FusedLayerOut]


class ResampleRequest(BaseModel):
    """``POST /fused/{gridId}/resample`` body (doc 07 §2.4)."""

    property_model_id: str
    method: str = "auto"
    interp_space: str = "auto"
    respect_footprint: bool = True
    cache: bool = True


class ResampledLayerOut(BaseModel):
    layer_id: str
    fused_model_id: str
    property: str
    value_array: str
    sigma_array: str
    coverage_mask: str
    method: str
    interp_space: str
    cached: bool


class SampleRequest(BaseModel):
    """``POST /fused/{gridId}/sample`` body (doc 07 §3.1)."""

    properties: list[str] | None = None  # None ⇒ all resampled layers
    mode: str = "all"  # "all" (listwise) | "any" (for histograms)
    bbox: dict[str, float] | None = None  # Engineering-metre clip box (region of interest)


class SampleOut(BaseModel):
    """Co-located feature-matrix payload (doc 07 §3.1)."""

    properties: list[str]
    n: int
    features: list[list[float]]
    cell_index: list[int]
    coords: list[list[float]]
    grid_shape: list[int]
    mode: str


class CrossplotRequest(BaseModel):
    """``POST /fused/{gridId}/crossplot`` body (doc 07 §3.2)."""

    axes: list[str] | None = None  # 2 or 3 properties for the scatter/density
    color_by: str | None = None  # "depth" | a property name
    properties: list[str] | None = None  # sample subset; None ⇒ all
    bbox: dict[str, float] | None = None
    histogram_property: str | None = None  # if set, add a 1D histogram of this property
    kde: bool = False
    bins: int = 64
    correlation: bool = True  # include the correlation matrix


class ClusterRequest(BaseModel):
    """``POST /fused/{gridId}/cluster`` body (doc 07 §3.3)."""

    project_id: str
    algorithm: str = "kmeans"  # "kmeans" | "gmm"
    n_clusters: int = 3
    properties: list[str] | None = None
    bbox: dict[str, float] | None = None
    write_volumes: bool = True
    force_job: bool = False  # force the job-based path regardless of working-set size


class TransformRunRequest(BaseModel):
    """``POST /fused/{gridId}/transform`` body (doc 07 §6, §4.5)."""

    project_id: str
    transform_id: str
    version: str | None = None  # optional pin; None ⇒ the registered version (doc 07 §4.4)
    inputs: dict[str, str] | None = None  # property → native PropertyModel id to resample in
    params: dict[str, Any] | None = None
    uncertainty: str = "delta"  # "delta" (default) | "monte_carlo" (job-based, doc 07 §5.5)
    mc_samples: int = 64
    mc_seed: int = 0
    force_job: bool = False


class FavorabilityRequest(BaseModel):
    """``POST /fused/{gridId}/favorability`` body (doc 07 §4.6 ``FavorabilitySpec``).

    ``method`` defaults to fuzzy-conjunction (the non-compensatory default, critique #11);
    ``weighted`` is the exploratory mode; ``bayesian`` is deferred (→ 400). Each ``evidence``
    item is ``{source, target, transferFn, weight, role}`` (doc 07 §4.6 sketch).
    """

    project_id: str
    method: str = "fuzzy"  # "fuzzy" (default) | "weighted" (exploratory) | "bayesian" (deferred)
    fuzzy_and: str = "min"  # "min" | "product"
    missing_policy: str = "nodata"  # "nodata" | "neutral" | "drop"
    evidence: list[dict[str, Any]]
    force_job: bool = False


class CalibrateRequest(BaseModel):
    """``POST /fused/{gridId}/calibrate`` body (doc 07 §4.8).

    Calibrate a transform against ground-truth probes sampled ALONG a well path: fit
    ``fit_params`` to the (measured ↔ predicted) pairs → a parameter distribution, re-run the
    transform with the calibrated params over the grid, and promote the output to
    ``well_calibrated`` / ``quantitative`` WHERE the wells constrain it (within
    ``resolving_distance``), leaving distant cells ``proxy`` / "likelihood" (doc 07 §4.8 ④).

    Probes can be supplied directly as ``probes`` (Engineering-XYZ + measured value) or as a
    deviation survey (``deviation_survey`` ``(MD,inc°,azi°)`` rows + ``wellhead`` + a measured
    log ``measured_md``/``measured_values``), which is integrated to Engineering XYZ
    (minimum-curvature, doc 09 §4.3) and positioned along the path (doc 07 §3.1).
    """

    project_id: str
    transform_id: str
    version: str | None = None
    fit_params: list[str]
    resolving_distance: float
    inputs: dict[str, str] | None = None  # property → native PropertyModel id to resample in
    params: dict[str, Any] | None = None  # base params held fixed (non-fit)
    # Either supply probes directly …
    probes: list[dict[str, Any]] | None = None  # [{z,y,x,measured,unit,md?,sigma?}]
    # … or a deviation survey + measured log (the harness builds the probes).
    deviation_survey: list[list[float]] | None = None  # [[MD,inc°,azi°], …]
    wellhead: list[float] | None = None  # [x, y] or [x, y, elev]
    kb_elev: float | None = None
    measured_md: list[float] | None = None
    measured_values: list[float] | None = None
    measured_unit: str | None = None
    measured_sigma: list[float] | None = None
    force_job: bool = False


class ArtifactSummary(BaseModel):
    """A discoverable catalog artifact (doc 04 §7 ``/artifacts`` → frontend)."""

    id: str
    kind: str  # propertyModel|fusedModel|feature|observation
    property: str | None = None
    method: str | None = None
    bbox: dict[str, float]
    time_extent: dict[str, Any] | None = None


def _kind_of(row: object) -> str:
    if isinstance(row, PropertyModel):
        return "propertyModel"
    if isinstance(row, FusedModel):
        return "fusedModel"
    return type(row).__name__.lower()


def build_fusion_router(session_dep: Any) -> APIRouter:
    """Build the fusion router wired to the app's catalog + storage DI (doc 04 §9)."""
    router = APIRouter(tags=["fusion"])

    def _fem_or_404(session: Session, grid_id: str) -> FusedModel:
        fem = session.get(FusedModel, grid_id)
        if fem is None:
            raise HTTPException(status_code=404, detail="fused model not found")
        return fem

    def _layers_out(session: Session, fem: FusedModel) -> list[FusedLayerOut]:
        return [
            FusedLayerOut(
                layer_id=lay.id, property=lay.property,
                source_property_model_id=lay.source_property_model_id,
                source_version=lay.source_version,
                method=json.loads(lay.resample_op_json).get("method", ""),
                sigma_array=lay.sigma_array, coverage_mask=lay.valid_mask,
            )
            for lay in fem.layers
        ]

    def _fem_out(session: Session, fem: FusedModel) -> FusedModelOut:
        grid = fused_grid_from_row(fem)
        return FusedModelOut(
            id=fem.id, project_id=fem.project_id, grid_type=fem.grid_type,
            origin=list(grid.origin), spacing=list(grid.spacing), shape=list(grid.shape),
            n_cells=grid.n_cells, bbox=json.loads(fem.bbox_json),
            layers=_layers_out(session, fem),
        )

    # ──────────────────────────────── POST /fused ────────────────────────────────
    @router.post("/fused", response_model=FusedModelOut, status_code=201)
    def create_fused(body: FusedCreate, request: Request, session: Session = session_dep):
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        layout = ProjectLayout(request.app.state.storage_root, body.project_id)
        spacing = tuple(body.spacing) if body.spacing else None  # type: ignore[assignment]
        try:
            fem, _grid = build_fused_model(
                session, layout, body.project_id,
                source_property_model_ids=body.source_property_model_ids,
                bbox=body.bbox, spacing=spacing, name=body.name,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _fem_out(session, fem)

    # ──────────────────────── POST /fused/{gridId}/resample ───────────────────────
    @router.post("/fused/{grid_id}/resample", response_model=ResampledLayerOut, status_code=201)
    def resample(
        grid_id: str, body: ResampleRequest, request: Request, session: Session = session_dep
    ):
        fem = _fem_or_404(session, grid_id)
        try:
            ref = resample_to_fused(
                session, fem, body.property_model_id,
                method=body.method, interp_space=body.interp_space,  # type: ignore[arg-type]
                respect_footprint=body.respect_footprint, cache=body.cache,
                storage_root=request.app.state.storage_root,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return ResampledLayerOut(
            layer_id=ref.layer_id, fused_model_id=ref.fused_model_id, property=ref.property,
            value_array=ref.value_array, sigma_array=ref.sigma_array,
            coverage_mask=ref.coverage_mask, method=ref.method,
            interp_space=ref.interp_space, cached=ref.cached,
        )

    # ──────────────────────────────── GET /fused/{gridId} ────────────────────────────────
    @router.get("/fused/{grid_id}", response_model=FusedModelOut)
    def get_fused(grid_id: str, session: Session = session_dep):
        return _fem_out(session, _fem_or_404(session, grid_id))

    # ──────────────────────── POST /fused/{gridId}/sample (doc 07 §3.1) ───────────────────────
    @router.post("/fused/{grid_id}/sample", response_model=SampleOut)
    def sample(
        grid_id: str, body: SampleRequest, request: Request, session: Session = session_dep
    ):
        fem = _fem_or_404(session, grid_id)
        try:
            s = sample_fused(
                session, fem, body.properties, mode=body.mode, bbox=body.bbox,
                storage_root=request.app.state.storage_root,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return SampleOut(
            properties=s.properties, n=s.n, features=s.features.tolist(),
            cell_index=s.cell_index.tolist(), coords=s.coords.tolist(),
            grid_shape=list(s.grid_shape), mode=s.mode,
        )

    # ──────────────────── POST /fused/{gridId}/crossplot (doc 07 §3.2) ────────────────────
    @router.post("/fused/{grid_id}/crossplot")
    def crossplot_route(
        grid_id: str, body: CrossplotRequest, request: Request, session: Session = session_dep
    ):
        fem = _fem_or_404(session, grid_id)
        try:
            s = sample_fused(
                session, fem, body.properties, mode="all", bbox=body.bbox,
                storage_root=request.app.state.storage_root,
            )
            payload: dict[str, Any] = {"n": s.n, "properties": s.properties}
            if body.axes:
                payload["crossplot"] = crossplot(
                    s, body.axes, color_by=body.color_by, bins=body.bins
                )
            if body.histogram_property:
                payload["histogram"] = histogram(
                    s, body.histogram_property, bins=body.bins, kde=body.kde
                )
            if body.correlation:
                payload["correlation"] = correlation_matrix(s)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return payload

    # ──────────────────── POST /fused/{gridId}/cluster (doc 07 §3.3/§3.4) ────────────────────
    @router.post("/fused/{grid_id}/cluster")
    def cluster_route(
        grid_id: str, body: ClusterRequest, request: Request, session: Session = session_dep
    ):
        fem = _fem_or_404(session, grid_id)
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, body.project_id)
        grid = fused_grid_from_row(fem)

        # Sync for small working sets; job-based for whole-grid / forced (doc 07 §3.4).
        job_based = body.force_job or grid.n_cells > SYNC_CELL_LIMIT
        if not job_based:
            try:
                result = cluster_fused(
                    session, layout, fem,
                    properties=body.properties, algorithm=body.algorithm,
                    n_clusters=body.n_clusters, bbox=body.bbox,
                    write_volumes=body.write_volumes, storage_root=storage_root,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return {"mode": "sync", **result.to_payload()}

        runner: JobRunner = request.app.state.job_runner
        session_factory = request.app.state.session_factory

        def _job(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
            job_session = session_factory()
            try:
                job_fem = job_session.get(FusedModel, grid_id)
                result = cluster_fused(
                    job_session, layout, job_fem,
                    properties=params.get("properties"), algorithm=params["algorithm"],
                    n_clusters=params["n_clusters"], bbox=params.get("bbox"),
                    write_volumes=params["write_volumes"], storage_root=storage_root,
                    progress=reporter,
                )
                return result.to_payload()
            finally:
                job_session.close()

        job_id = runner.enqueue(
            "fuse:cluster",
            {
                "algorithm": body.algorithm, "n_clusters": body.n_clusters,
                "properties": body.properties, "bbox": body.bbox,
                "write_volumes": body.write_volumes,
            },
            _job,
            project_id=body.project_id,
        )
        return {"mode": "job", "job_id": job_id}

    # ──────────────────────────────── GET /transforms (doc 07 §6, §4.7) ────────────────────────
    @router.get("/transforms")
    def list_transforms() -> dict[str, Any]:
        """The transform registry palette (doc 07 §6 ``GET /transforms``, doc 08-backed).

        Returns every registered doc-07 :class:`~geosim.fusion.transform.Transform` with its
        full declarative spec (id/version/title/target, typed inputs/output, params, stated
        assumptions + calibration status) so the UI can build the transform palette + param
        controls (doc 07 §4.1/§4.7).
        """
        out = []
        for t in get_registry().transforms():
            if isinstance(t, Transform) and hasattr(t, "describe"):
                out.append(t.describe())
        return {"transforms": out}

    # ──────────────────── POST /fused/{gridId}/transform (doc 07 §6, §4.5) ────────────────────
    @router.post("/fused/{grid_id}/transform")
    def transform_route(
        grid_id: str, body: TransformRunRequest, request: Request, session: Session = session_dep
    ):
        fem = _fem_or_404(session, grid_id)
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        transform = _resolve_transform(body.transform_id, body.version)
        if transform is None:
            raise HTTPException(
                status_code=404, detail=f"transform {body.transform_id!r} not registered"
            )
        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, body.project_id)
        grid = fused_grid_from_row(fem)

        # Monte-Carlo / forced / whole-grid runs are job-based (doc 07 §4.5, §5.5, §6).
        job_based = (
            body.force_job
            or body.uncertainty == "monte_carlo"
            or grid.n_cells > SYNC_CELL_LIMIT
        )
        if not job_based:
            try:
                result = run_transform(
                    session, layout, fem, transform,
                    inputs=body.inputs, params=body.params,
                    uncertainty=body.uncertainty,
                    mc_samples=body.mc_samples, mc_seed=body.mc_seed,
                    storage_root=storage_root,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return {"mode": "sync", **result.to_payload()}

        runner: JobRunner = request.app.state.job_runner
        session_factory = request.app.state.session_factory

        def _job(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
            job_session = session_factory()
            try:
                job_fem = job_session.get(FusedModel, grid_id)
                result = run_transform(
                    job_session, layout, job_fem, transform,
                    inputs=params.get("inputs"), params=params.get("params"),
                    uncertainty=params["uncertainty"],
                    mc_samples=params["mc_samples"], mc_seed=params["mc_seed"],
                    storage_root=storage_root, progress=reporter,
                )
                return result.to_payload()
            finally:
                job_session.close()

        job_id = runner.enqueue(
            f"transform:{transform.id}",
            {
                "inputs": body.inputs, "params": body.params,
                "uncertainty": body.uncertainty,
                "mc_samples": body.mc_samples, "mc_seed": body.mc_seed,
            },
            _job,
            project_id=body.project_id,
        )
        return {"mode": "job", "job_id": job_id}

    # ──────────────── POST /fused/{gridId}/favorability (doc 07 §4.6) ────────────────
    @router.post("/fused/{grid_id}/favorability")
    def favorability_route(
        grid_id: str, body: FavorabilityRequest, request: Request, session: Session = session_dep
    ):
        """Compute the headline geothermal favorability volume + honesty diagnostics (doc 07 §4.6).

        Default = fuzzy-conjunction (heat AND fluid AND permeability, non-compensatory);
        ``weighted`` is the exploratory compensatory mode with the missing-required guard;
        ``bayesian`` is deferred. Writes a ``[0,1]`` favorability PropertyModel plus its
        confidence, evidence-overlap, and assumption-burden companion volumes.
        """
        fem = _fem_or_404(session, grid_id)
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, body.project_id)
        grid = fused_grid_from_row(fem)

        try:
            spec = FavorabilitySpec.from_payload({
                "method": body.method, "fuzzyAnd": body.fuzzy_and,
                "missingPolicy": body.missing_policy, "evidence": body.evidence,
            })
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        job_based = body.force_job or grid.n_cells > SYNC_CELL_LIMIT
        if not job_based:
            try:
                result = compute_favorability(
                    session, layout, fem, spec, storage_root=storage_root
                )
            except NotImplementedError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return {"mode": "sync", **result.to_payload()}

        runner: JobRunner = request.app.state.job_runner
        session_factory = request.app.state.session_factory

        def _job(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
            job_session = session_factory()
            try:
                job_fem = job_session.get(FusedModel, grid_id)
                job_spec = FavorabilitySpec.from_payload(params["spec"])
                result = compute_favorability(
                    job_session, layout, job_fem, job_spec,
                    storage_root=storage_root, progress=reporter,
                )
                return result.to_payload()
            finally:
                job_session.close()

        job_id = runner.enqueue(
            "fuse:favorability",
            {"spec": {
                "method": body.method, "fuzzyAnd": body.fuzzy_and,
                "missingPolicy": body.missing_policy, "evidence": body.evidence,
            }},
            _job,
            project_id=body.project_id,
        )
        return {"mode": "job", "job_id": job_id}

    # ──────────────── POST /fused/{gridId}/calibrate (doc 07 §4.8) ────────────────
    @router.post("/fused/{grid_id}/calibrate")
    def calibrate_route(
        grid_id: str, body: CalibrateRequest, request: Request, session: Session = session_dep
    ):
        """Calibrate a transform to well/core/geochem probes → a promoted derived volume (§4.8).

        Fits ``fit_params`` to the (measured ↔ predicted) pairs at probe locations (a param
        distribution, not a point fit), re-runs the transform with the calibrated params, and
        promotes the output to ``well_calibrated`` / ``quantitative`` within
        ``resolving_distance`` of the probes — distant cells stay ``proxy`` / "likelihood"
        (spatially honest, doc 07 §4.8). Sync for small grids; job-based for big/forced runs.
        """
        fem = _fem_or_404(session, grid_id)
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        transform = _resolve_transform(body.transform_id, body.version)
        if transform is None:
            raise HTTPException(
                status_code=404, detail=f"transform {body.transform_id!r} not registered"
            )
        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, body.project_id)
        grid = fused_grid_from_row(fem)

        try:
            probes = _build_probes(body)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        job_based = body.force_job or grid.n_cells > SYNC_CELL_LIMIT
        if not job_based:
            try:
                result = calibrate_transform(
                    session, layout, fem, transform, probes, body.fit_params,
                    resolving_distance=body.resolving_distance,
                    inputs=body.inputs, params=body.params,
                    storage_root=storage_root,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return {"mode": "sync", **result.to_payload()}

        runner: JobRunner = request.app.state.job_runner
        session_factory = request.app.state.session_factory
        probe_payload = [
            {"z": p.z, "y": p.y, "x": p.x, "measured": p.measured, "unit": p.unit,
             "md": p.md, "sigma": p.sigma}
            for p in probes
        ]

        def _job(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
            job_session = session_factory()
            try:
                job_fem = job_session.get(FusedModel, grid_id)
                job_probes = [Probe(**p) for p in params["probes"]]
                result = calibrate_transform(
                    job_session, layout, job_fem, transform, job_probes, params["fit_params"],
                    resolving_distance=params["resolving_distance"],
                    inputs=params.get("inputs"), params=params.get("params"),
                    storage_root=storage_root, progress=reporter,
                )
                return result.to_payload()
            finally:
                job_session.close()

        job_id = runner.enqueue(
            f"calibrate:{transform.id}",
            {
                "probes": probe_payload, "fit_params": body.fit_params,
                "resolving_distance": body.resolving_distance,
                "inputs": body.inputs, "params": body.params,
            },
            _job,
            project_id=body.project_id,
        )
        return {"mode": "job", "job_id": job_id}

    # ──────────────────────── GET /projects/{pid}/artifacts ───────────────────────
    @router.get("/projects/{pid}/artifacts", response_model=list[ArtifactSummary])
    def list_artifacts(
        pid: str,
        session: Session = session_dep,
        bbox: str | None = Query(default=None, description="xmin,xmax,ymin,ymax,zmin,zmax"),
        kind: str | None = Query(default=None),
        method: str | None = Query(default=None),
        property: str | None = Query(default=None),
        t: float | None = Query(default=None),
    ):
        """List project artifacts for frontend discovery (doc 04 §7/§9.2).

        ``bbox`` (Engineering metres) filters via the §2.5 helper; ``None`` returns the
        whole-project extent. ``kind``/``method``/``property``/``t`` are post-filters.
        """
        if session.get(Project, pid) is None:
            raise HTTPException(status_code=404, detail="project not found")
        query_box = _parse_bbox(bbox) if bbox else _project_box(session, pid)
        kinds = [kind] if kind else None
        rows = query_artifacts_bbox(session, pid, query_box, kinds=kinds)
        out: list[ArtifactSummary] = []
        for row in rows:
            summary = _to_summary(session, row)
            if property is not None and summary.property != property:
                continue
            if method is not None and summary.method != method:
                continue
            if t is not None and not _time_contains(summary.time_extent, t):
                continue
            out.append(summary)
        return out

    return router


def _build_probes(body: CalibrateRequest) -> list[Probe]:
    """Build calibration probes from the request body (direct probes OR a deviation survey)."""
    if body.probes:
        return [
            Probe(
                z=float(p["z"]), y=float(p["y"]), x=float(p["x"]),
                measured=float(p["measured"]), unit=str(p["unit"]),
                md=(float(p["md"]) if p.get("md") is not None else None),
                sigma=(float(p["sigma"]) if p.get("sigma") is not None else None),
            )
            for p in body.probes
        ]
    if (
        body.deviation_survey is not None
        and body.wellhead is not None
        and body.measured_md is not None
        and body.measured_values is not None
        and body.measured_unit is not None
    ):
        import numpy as np

        return probes_from_deviation_survey(
            np.asarray(body.deviation_survey, dtype=float),
            np.asarray(body.wellhead, dtype=float),
            np.asarray(body.measured_md, dtype=float),
            np.asarray(body.measured_values, dtype=float),
            unit=body.measured_unit, kb_elev=body.kb_elev,
            measured_sigma=(
                np.asarray(body.measured_sigma, dtype=float)
                if body.measured_sigma is not None else None
            ),
        )
    raise ValueError(
        "supply either `probes` or a `deviation_survey` + `wellhead` + "
        "`measured_md`/`measured_values`/`measured_unit`"
    )


def _resolve_transform(transform_id: str, version: str | None) -> Transform | None:
    """Resolve a registered doc-07 transform by id (+ optional version pin, doc 07 §4.4)."""
    for t in get_registry().transforms():
        if isinstance(t, Transform) and t.id == transform_id:
            if version is not None and t.version != version:
                continue
            return t
    return None


def _to_summary(session: Session, row: object) -> ArtifactSummary:
    kind = _kind_of(row)
    prop = getattr(row, "property", None)
    ds = session.get(Dataset, row.dataset_id)
    method = ds.method if ds is not None else None
    time_extent = None
    if ds is not None and ds.time_extent_json:
        time_extent = json.loads(ds.time_extent_json)
    return ArtifactSummary(
        id=row.id, kind=kind, property=prop, method=method,
        bbox=json.loads(row.bbox_json), time_extent=time_extent,
    )


def _parse_bbox(spec: str) -> Bbox3D:
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 6:
        raise HTTPException(status_code=400, detail="bbox must be xmin,xmax,ymin,ymax,zmin,zmax")
    return Bbox3D(*parts)


def _project_box(session: Session, pid: str) -> Bbox3D:
    project = session.get(Project, pid)
    frame = project.spatial_frame
    roi = json.loads(frame.roi_json)
    dr = json.loads(frame.depth_range_json)
    return Bbox3D(
        float(roi["xmin"]), float(roi["xmax"]),
        float(roi["ymin"]), float(roi["ymax"]),
        float(dr["zmin"]), float(dr["zmax"]),
    )


def _time_contains(time_extent: dict[str, Any] | None, t: float) -> bool:
    if not time_extent:
        return False
    lo = time_extent.get("tmin", time_extent.get("start"))
    hi = time_extent.get("tmax", time_extent.get("end"))
    if lo is None or hi is None:
        return True
    return float(lo) <= t <= float(hi)
