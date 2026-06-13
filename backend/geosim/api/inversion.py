"""Inversion API surface (doc 10 §3) — ``POST /property-models:invert`` + job reuse.

A thin HTTP layer over :mod:`geosim.inversion`: it validates the request, builds a
:class:`~geosim.inversion.domain.ModelDomain` over the project's Engineering Frame (core +
padding + topography active cells, doc 10 §4), validates user params against the engine's
``paramsSchema`` **before** enqueueing (doc 10 §3 — a bad request 400s, it never starts a
job), and enqueues the run on the shared :class:`~geosim.jobs.JobRunner`.

The run itself is the engine-agnostic :func:`geosim.inversion.run_inversion` harness; it
writes the recovered model as an ordinary PropertyModel + mandatory uncertainty +
provenance, then resamples onto a fused grid (doc 10 §0, §4.4). Progress / status /
cancellation reuse the doc-04 job endpoints (``GET /jobs/{id}``, ``WS /jobs/{id}/progress``,
``POST /jobs/{id}:cancel``) unchanged — inversion is *just another job* (doc 08 §4f).

Engines are resolved by id from the plugin registry; this router never imports SimPEG or
PyGIMLi (doc 10 §8).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from geosim.catalog import Project, SpatialFrameRow
from geosim.inversion import (
    CoreRegion,
    InversionEngine,
    PaddingSpec,
    ParamValidationError,
    build_tensor_domain,
    run_inversion,
    validate_params,
)
from geosim.inversion.cooperative import CooperativeStage, cooperative_invert
from geosim.inversion.engine import InversionEngineSpec
from geosim.jobs import JobRunner, ProgressReporter
from geosim.plugins import get_registry
from geosim.storage import ProjectLayout

__all__ = ["build_inversion_router"]


# ──────────────────────────────── wire shapes (doc 10 §3) ────────────────────────────────


class CoreSpec(BaseModel):
    """The inversion core region in Engineering metres ``(z, y, x)`` (doc 10 §4.1)."""

    origin: list[float]  # [z0, y0, x0] min corner
    extent: list[float]  # [dz, dy, dx] span
    cell_size: list[float]  # [cz, cy, cx] core cell edge


class InvertRequest(BaseModel):
    """``POST /property-models:invert`` body (doc 10 §3).

    ``engine_id`` selects a registered :class:`~geosim.inversion.engine.InversionEngine`;
    ``observation_ids`` are the doc-02 Observations to fit; ``core`` describes the resolved
    mesh core (auto-padded). ``params`` is validated against the engine ``paramsSchema``
    before the job is enqueued.
    """

    project_id: str
    engine_id: str
    observation_ids: list[str]
    core: CoreSpec
    params: dict[str, Any] = {}
    n_pad: int = 0
    pad_factor: float = 1.3
    name: str | None = None
    resample_fused: bool = True


class InvertResponse(BaseModel):
    """``{job_id}`` on the InlineJobRunner (doc 10 §3, doc 04 §9.4 job pattern)."""

    job_id: str
    engine_id: str


class CooperativeStageSpec(BaseModel):
    """One node of a cooperative (5b) DAG (doc 10 §6).

    A single-method inversion (``engine_id`` + ``observation_ids`` + ``core`` + ``params``).
    ``depends_on`` names an earlier stage whose recovered model seeds this one as a
    reference / structure weight (doc 10 §6 5b); ``None`` ⇒ a root (stage A).
    """

    name: str
    engine_id: str
    observation_ids: list[str]
    core: CoreSpec
    params: dict[str, Any] = {}
    depends_on: str | None = None
    n_pad: int = 0
    pad_factor: float = 1.3
    resample_fused: bool = True


class CooperativeRequest(BaseModel):
    """``POST /property-models:cooperative-invert`` body (doc 10 §6 5b).

    A list of ``stages`` run in order as a small DAG: invert method A, then feed A's
    recovered model into method B as a reference / structure weight (doc 10 §6) — an
    orchestration of ordinary §3 jobs, NOT a joint solver (5c stays roadmap).
    """

    project_id: str
    stages: list[CooperativeStageSpec]
    name: str | None = None


class CooperativeResponse(BaseModel):
    """``{job_id}`` for the parent cooperative DAG (doc 10 §6, doc 04 §9.4)."""

    job_id: str
    stages: list[str]


class EngineOut(BaseModel):
    """A discoverable inversion engine (doc 10 §2, doc 08 §7)."""

    id: str
    kind: str
    library: str
    methods: list[str]
    output_property: str
    mesh_types: list[str]
    coupling: str
    compute: str
    params_schema: dict[str, Any]


def _engine_spec(engine: InversionEngine) -> InversionEngineSpec:
    return engine.spec


def _resolve_engine(engine_id: str) -> InversionEngine | None:
    """Resolve a registered inversion engine by id (doc 08 §4f)."""
    for eng in get_registry().inversion_engines():
        spec = getattr(eng, "spec", None)
        if isinstance(spec, InversionEngineSpec) and spec.id == engine_id:
            return eng  # type: ignore[return-value]
    return None


def build_inversion_router(session_dep: Any) -> APIRouter:
    """Build the inversion router wired to the app's catalog + storage + jobs DI (doc 04 §9)."""
    router = APIRouter(tags=["inversion"])

    # ──────────────────────────────── GET /inversion-engines ────────────────────────────────
    @router.get("/inversion-engines", response_model=list[EngineOut])
    def list_engines() -> list[EngineOut]:
        """The registered inversion-engine palette (doc 10 §2, doc 08 §7)."""
        out: list[EngineOut] = []
        for eng in get_registry().inversion_engines():
            spec = getattr(eng, "spec", None)
            if not isinstance(spec, InversionEngineSpec):
                continue
            d = spec.to_dict()
            out.append(EngineOut(
                id=d["id"], kind=d["kind"], library=d["library"], methods=d["methods"],
                output_property=d["outputProperty"], mesh_types=d["meshTypes"],
                coupling=d["coupling"], compute=d["compute"], params_schema=d["paramsSchema"],
            ))
        return out

    # ──────────────────────── POST /property-models:invert (doc 10 §3) ───────────────────────
    @router.post("/property-models:invert", response_model=InvertResponse, status_code=202)
    def invert(body: InvertRequest, request: Request, session: Session = session_dep):
        """Enqueue an inversion → ``{job_id}`` (doc 10 §3, doc 04 §9.4).

        Validates the project, engine, observations, and (crucially) the params against
        the engine ``paramsSchema`` BEFORE enqueueing — a bad request 400s and no job is
        created (doc 10 §3). The run executes :func:`geosim.inversion.run_inversion` on the
        shared job runner; progress/cancel reuse the doc-04 job endpoints.
        """
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        engine = _resolve_engine(body.engine_id)
        if engine is None:
            raise HTTPException(
                status_code=404, detail=f"inversion engine {body.engine_id!r} not registered"
            )
        if not body.observation_ids:
            raise HTTPException(status_code=400, detail="observation_ids must be non-empty")

        # Validate params against paramsSchema BEFORE enqueue (doc 10 §3).
        try:
            validate_params(body.params, dict(engine.spec.params_schema))
        except ParamValidationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            core = _core_from_spec(body.core)
            padding = PaddingSpec(n_pad=body.n_pad, factor=body.pad_factor)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, body.project_id)
        surface_z = _surface_z(session, body.project_id)
        runner: JobRunner = request.app.state.job_runner
        session_factory = request.app.state.session_factory

        def _job(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
            job_session = session_factory()
            try:
                domain = build_tensor_domain(core, padding=padding, surface_z=surface_z)
                result = run_inversion(
                    job_session, layout, body.project_id, engine,
                    domain=domain, observation_ids=params["observation_ids"],
                    params=params["params"], name=body.name, reporter=reporter,
                    resample_fused=body.resample_fused, storage_root=storage_root,
                )
                return result.to_payload()
            finally:
                job_session.close()

        job_id = runner.enqueue(
            f"invert:{engine.spec.id}",
            {
                "engine_id": engine.spec.id,
                "observation_ids": body.observation_ids,
                "params": body.params,
            },
            _job,
            project_id=body.project_id,
        )
        return InvertResponse(job_id=job_id, engine_id=engine.spec.id)

    # ──────────── POST /property-models:cooperative-invert (doc 10 §6 5b) ────────────
    @router.post(
        "/property-models:cooperative-invert",
        response_model=CooperativeResponse,
        status_code=202,
    )
    def cooperative(body: CooperativeRequest, request: Request, session: Session = session_dep):
        """Launch a cooperative (5b) inversion DAG → ``{job_id}`` (doc 10 §6, doc 04 §9.4).

        Validates the project, every stage's engine + params + core + dependency graph
        BEFORE enqueueing (a bad request 400s, no job is created — doc 10 §3), then runs
        the stages in order on the shared job runner: each dependent stage is seeded with
        its parent's recovered model as a reference / structure weight (doc 10 §6 5b). The
        parent job id reuses the doc-04 job endpoints for progress / status / cancel.
        """
        if session.get(Project, body.project_id) is None:
            raise HTTPException(status_code=404, detail="project not found")
        if not body.stages:
            raise HTTPException(status_code=400, detail="stages must be non-empty")

        # Resolve engines + validate params + dependency graph BEFORE enqueue (doc 10 §3).
        resolved: list[InversionEngine] = []
        seen: set[str] = set()
        for st in body.stages:
            engine = _resolve_engine(st.engine_id)
            if engine is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"inversion engine {st.engine_id!r} not registered",
                )
            if not st.observation_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"stage {st.name!r}: observation_ids must be non-empty",
                )
            if st.name in seen:
                raise HTTPException(
                    status_code=400, detail=f"duplicate stage name {st.name!r}"
                )
            if st.depends_on is not None and st.depends_on not in seen:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"stage {st.name!r} depends on {st.depends_on!r} which is not an "
                        "earlier stage (stages run in order)"
                    ),
                )
            seen.add(st.name)
            try:
                validate_params(st.params, dict(engine.spec.params_schema))
                _core_from_spec(st.core)
                PaddingSpec(n_pad=st.n_pad, factor=st.pad_factor)
            except (ParamValidationError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"stage {st.name!r}: {e}") from e
            resolved.append(engine)

        storage_root = request.app.state.storage_root
        layout = ProjectLayout(storage_root, body.project_id)
        surface_z = _surface_z(session, body.project_id)
        runner: JobRunner = request.app.state.job_runner
        session_factory = request.app.state.session_factory
        specs = body.stages

        def _job(params: dict[str, Any], reporter: ProgressReporter) -> dict[str, Any]:
            job_session = session_factory()
            try:
                stages = [
                    CooperativeStage(
                        name=st.name,
                        engine=eng,
                        observation_ids=st.observation_ids,
                        core=_core_from_spec(st.core),
                        params=st.params,
                        depends_on=st.depends_on,
                        n_pad=st.n_pad,
                        pad_factor=st.pad_factor,
                        surface_z=surface_z,
                        resample_fused=st.resample_fused,
                    )
                    for st, eng in zip(specs, resolved, strict=True)
                ]
                result = cooperative_invert(
                    job_session, layout, body.project_id, stages,
                    reporter=reporter, storage_root=storage_root,
                )
                return result.to_payload()
            finally:
                job_session.close()

        job_id = runner.enqueue(
            "invert:cooperative",
            {"stages": [s.name for s in body.stages]},
            _job,
            project_id=body.project_id,
        )
        return CooperativeResponse(job_id=job_id, stages=[s.name for s in body.stages])

    return router


def _core_from_spec(spec: CoreSpec) -> CoreRegion:
    for name, vec in (("origin", spec.origin), ("extent", spec.extent),
                      ("cell_size", spec.cell_size)):
        if len(vec) != 3:
            raise ValueError(f"core.{name} must have 3 components (z, y, x)")
    return CoreRegion(
        origin=tuple(spec.origin),  # type: ignore[arg-type]
        extent=tuple(spec.extent),  # type: ignore[arg-type]
        cell_size=tuple(spec.cell_size),  # type: ignore[arg-type]
    )


def _surface_z(session: Session, project_id: str) -> float | None:
    """Flat-topography surface elevation from the project frame (doc 10 §4.3).

    A ``flat:<z>`` surface model gives a single elevation that masks air cells above it;
    anything else (DEM/synthetic) is left to the engine, so we return ``None`` (all cells
    active) here.
    """
    row = session.get(SpatialFrameRow, project_id)
    if row is None:
        return None
    sm = None
    if row.frame_json:
        sm = json.loads(row.frame_json).get("surface_model")
    sm = sm or row.surface_model
    if isinstance(sm, str) and sm.startswith("flat:"):
        try:
            return float(sm.split(":", 1)[1])
        except ValueError:
            return None
    return None
