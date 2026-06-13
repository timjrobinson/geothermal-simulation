"""Well-planning API router (doc 09 §10).

Wired into :func:`geosim.api.create_app`. Endpoints (doc 09 §10 backend surface):

- ``POST /projects/{pid}/targets`` — create + enrich a :class:`DrillTarget` (points-mode
  fused sample stamped with ``modelVersion``, doc 09 §3).
- ``POST /projects/{pid}/wells`` — create a :class:`PlannedWell` from an ``intent`` (a
  design solver runs) OR a directly-supplied ``survey`` (doc 09 §4).
- ``POST /wells/{wid}/solve`` — (re)solve the survey from design params → survey + DLS
  report (doc 09 §4.4).
- ``GET /wells/{wid}/positions`` — min-curvature Engineering XYZ / TVD / DLS per station
  (doc 09 §4.3).
- ``POST /wells/{wid}/predict`` — the predicted log + geothermal summary + per-station risk
  (doc 09 §5–§7).

Planning objects are doc-02 **features** (doc 09 §2): targets/wells persist in the
``features`` table (``feature_type`` ``drillTarget`` / ``plannedWell``) with their schema in
``props_json``. The router shares the catalog session + ``storage_root`` DI off
``app.state`` and reuses :mod:`geosim.planning` for all compute (never reimplements it).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from geosim.catalog import Feature, FusedModel, IdKind, Project, new_id
from geosim.planning import (
    DesignSpec,
    DrillabilityLimits,
    DrillTarget,
    PlannedWell,
    RiskWeights,
    TargetTolerance,
    TrajectoryConstraints,
    drillability_flag,
    enrich_target,
    predict_log,
    solve_survey,
)
from geosim.planning.export import (
    export_log_csv,
    export_survey_csv,
    export_witsml_trajectory,
)

from .frame_io import frame_from_row

__all__ = ["build_planning_router"]


# ──────────────────────────────── wire shapes (doc 09 §10) ────────────────────────────────


class TargetCreate(BaseModel):
    """``POST /projects/{pid}/targets`` body (doc 09 §3.3)."""

    fused_model_id: str
    name: str = "target"
    kind: str = "point"  # "point" | "zone"
    location: list[float]  # [x, y, z] Engineering metres
    tolerance_radius_m: float = 50.0
    tvd_window_m: float = 25.0
    desired_temperature_c: float | None = None
    min_temperature_c: float | None = None
    geological_unit: str | None = None
    rationale: str | None = None
    zone_ref: dict | None = None
    kb_elev_m: float = 0.0
    provenance: dict | None = None


class WellCreate(BaseModel):
    """``POST /projects/{pid}/wells`` body — ``intent`` OR ``survey`` (doc 09 §4.1)."""

    name: str = "well"
    wellhead: list[float]  # [x, y]
    kb_elev_m: float = 0.0
    target_ids: list[str] | None = None
    # intent mode (a solver runs):
    design: dict | None = None
    # direct-survey mode:
    survey: list[list[float]] | None = None  # [[MD, inc°, azi°], …]
    max_dls_deg30m: float = 5.0
    max_inc_deg: float = 92.0


class SolveRequest(BaseModel):
    """``POST /wells/{wid}/solve`` body (doc 09 §4.4)."""

    design: dict
    max_dls_deg30m: float = 5.0
    max_inc_deg: float = 92.0


class PredictRequest(BaseModel):
    """``POST /wells/{wid}/predict`` body (doc 09 §5–§7)."""

    fused_model_id: str
    md_step_m: float = 5.0
    target_id: str | None = None
    risk_weights: dict | None = None  # {tempConfidence, hazard, dlsExceedance, structuralUncertainty}  # noqa: E501
    favorability_threshold: float = 0.7
    fracture_threshold: float = 0.5


def _constraints(max_dls: float, max_inc: float) -> TrajectoryConstraints:
    return TrajectoryConstraints(max_dls_deg30m=max_dls, max_inc_deg=max_inc)


def _design_from_dict(d: dict) -> DesignSpec:
    target = d.get("target")
    return DesignSpec(
        method=d["method"],
        target=(tuple(float(c) for c in target) if target else None),
        kop_md_m=float(d.get("kop_md_m", d.get("kopMD_m", 0.0))),
        build_rate_deg30m=float(d.get("build_rate_deg30m", d.get("buildRate_deg30m", 3.0))),
        drop_rate_deg30m=float(d.get("drop_rate_deg30m", d.get("dropRate_deg30m", 3.0))),
        hold_inc_deg=(float(d["hold_inc_deg"]) if d.get("hold_inc_deg") is not None else None),
        landing_inc_deg=(
            float(d["landing_inc_deg"]) if d.get("landing_inc_deg") is not None else None
        ),
        station_step_m=float(d.get("station_step_m", 30.0)),
    )


def _solve_payload(res) -> dict:
    return {
        "survey": res.survey.tolist(),
        "maxDLS_deg30m": res.max_dls_deg30m,
        "dlsExceeded": res.dls_exceeded,
        "maxInc_deg": res.max_inc_deg,
        "incExceeded": res.inc_exceeded,
        "landingError_m": res.landing_error_m,
        "method": res.method,
    }


def _well_from_row(row: Feature) -> PlannedWell:
    props = json.loads(row.props_json) if row.props_json else {}
    p = props.get("props", props)
    design = _design_from_dict(p["design"]) if p.get("design") else None
    cons = p.get("constraints", {})
    return PlannedWell(
        id=row.id, name=p.get("name", "well"), project_id=row.project_id,
        wellhead=(float(p["wellhead"][0]), float(p["wellhead"][1])),
        kb_elev_m=float(p.get("kbElev_m", 0.0)),
        deviation_survey=np.asarray(p["deviationSurvey"], dtype=float),
        target_ids=list(p.get("targetIds", [])),
        design=design,
        constraints=TrajectoryConstraints(
            max_dls_deg30m=float(cons.get("maxDLS_deg30m", 5.0)),
            max_inc_deg=float(cons.get("maxInc_deg", 92.0)),
        ),
        status=p.get("status", "planned"),
    )


def _well_payload(well: PlannedWell, res=None) -> dict:
    p = {
        "name": well.name,
        "wellhead": list(well.wellhead),
        "kbElev_m": well.kb_elev_m,
        "targetIds": well.target_ids,
        "status": well.status,
        "deviationSurvey": np.asarray(well.deviation_survey, dtype=float).tolist(),
        "constraints": {
            "maxDLS_deg30m": well.constraints.max_dls_deg30m,
            "maxInc_deg": well.constraints.max_inc_deg,
        },
    }
    if well.design is not None:
        p["design"] = {
            "method": well.design.method,
            "target": list(well.design.target) if well.design.target else None,
            "kopMD_m": well.design.kop_md_m,
            "buildRate_deg30m": well.design.build_rate_deg30m,
        }
    out = {"id": well.id, "projectId": well.project_id, "props": p}
    if res is not None:
        out["solve"] = _solve_payload(res)
    return out


def build_planning_router(session_dep: Any) -> APIRouter:
    """Build the planning router wired to the app's catalog + storage DI (doc 04 §9)."""
    router = APIRouter(tags=["planning"])

    def _project_or_404(session: Session, pid: str) -> Project:
        row = session.get(Project, pid)
        if row is None:
            raise HTTPException(status_code=404, detail="project not found")
        return row

    def _fem_or_404(session: Session, fem_id: str) -> FusedModel:
        fem = session.get(FusedModel, fem_id)
        if fem is None:
            raise HTTPException(status_code=404, detail="fused model not found")
        return fem

    def _well_or_404(session: Session, wid: str) -> Feature:
        row = session.get(Feature, wid)
        if row is None or row.feature_type != "plannedWell":
            raise HTTPException(status_code=404, detail="planned well not found")
        return row

    def _persist_feature(
        session: Session, fid: str, pid: str, feature_type: str, payload: dict, bbox: dict
    ) -> None:
        session.add(Feature(
            id=fid, dataset_id=None, project_id=pid, feature_type=feature_type,
            store_format="geojson", bbox_json=json.dumps(bbox),
            props_json=json.dumps({"props": payload}),
        ))
        session.commit()

    # ──────────────────────── POST /projects/{pid}/targets ────────────────────────
    @router.post("/projects/{pid}/targets", status_code=201)
    def create_target(pid: str, body: TargetCreate, request: Request, session: Session = session_dep):  # noqa: E501
        _project_or_404(session, pid)
        fem = _fem_or_404(session, body.fused_model_id)
        if len(body.location) != 3:
            raise HTTPException(status_code=400, detail="location must be [x, y, z]")
        tgt = DrillTarget(
            id=new_id(IdKind.FEATURE), name=body.name, project_id=pid, kind=body.kind,
            location=(body.location[0], body.location[1], body.location[2]),
            tolerance=TargetTolerance(body.tolerance_radius_m, body.tvd_window_m),
            desired_temperature_c=body.desired_temperature_c,
            min_temperature_c=body.min_temperature_c,
            geological_unit=body.geological_unit, rationale=body.rationale,
            zone_ref=body.zone_ref, kb_elev_m=body.kb_elev_m,
            provenance=body.provenance or {},
        )
        enrich_target(session, fem, tgt, storage_root=request.app.state.storage_root)
        x, y, z = tgt.location
        bbox = {"xmin": x, "xmax": x, "ymin": y, "ymax": y, "zmin": z, "zmax": z}
        _persist_feature(session, tgt.id, pid, "drillTarget", tgt.to_payload(), bbox)
        return tgt.to_payload()

    # ──────────────────────── POST /projects/{pid}/wells ────────────────────────
    @router.post("/projects/{pid}/wells", status_code=201)
    def create_well(pid: str, body: WellCreate, session: Session = session_dep):
        _project_or_404(session, pid)
        if len(body.wellhead) < 2:
            raise HTTPException(status_code=400, detail="wellhead must be [x, y]")
        wellhead = (float(body.wellhead[0]), float(body.wellhead[1]))
        cons = _constraints(body.max_dls_deg30m, body.max_inc_deg)
        design = None
        res = None
        if body.survey is not None:
            survey = np.asarray(body.survey, dtype=float)
        elif body.design is not None:
            design = _design_from_dict(body.design)
            try:
                res = solve_survey(design, wellhead, body.kb_elev_m, cons)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            survey = res.survey
        else:
            raise HTTPException(status_code=400, detail="supply either `survey` or `design`")

        well = PlannedWell(
            id=new_id(IdKind.FEATURE), name=body.name, project_id=pid,
            wellhead=wellhead, kb_elev_m=body.kb_elev_m, deviation_survey=survey,
            target_ids=list(body.target_ids or []), design=design, constraints=cons,
        )
        pos = well.positions()
        bbox = _enu_bbox(pos.enu)
        _persist_feature(session, well.id, pid, "plannedWell", _well_payload(well)["props"], bbox)
        return _well_payload(well, res)

    # ──────────────────────── POST /wells/{wid}/solve ────────────────────────
    @router.post("/wells/{wid}/solve")
    def solve_well(wid: str, body: SolveRequest, session: Session = session_dep):
        row = _well_or_404(session, wid)
        well = _well_from_row(row)
        cons = _constraints(body.max_dls_deg30m, body.max_inc_deg)
        design = _design_from_dict(body.design)
        try:
            res = solve_survey(design, well.wellhead, well.kb_elev_m, cons)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        well.deviation_survey = res.survey
        well.design = design
        well.constraints = cons
        bbox = _enu_bbox(res.positions.enu)
        row.props_json = json.dumps({"props": _well_payload(well)["props"]})
        row.bbox_json = json.dumps(bbox)
        session.commit()
        return _well_payload(well, res)

    # ──────────────────────── GET /wells/{wid}/positions ────────────────────────
    @router.get("/wells/{wid}/positions")
    def well_positions_route(wid: str, session: Session = session_dep):
        well = _well_from_row(_well_or_404(session, wid))
        pos = well.positions()
        flag = drillability_flag(
            well.deviation_survey, pos,
            limits=DrillabilityLimits(
                max_dls_deg30m=well.constraints.max_dls_deg30m,
                max_inc_deg=well.constraints.max_inc_deg,
            ),
        )
        return {
            "wellId": well.id,
            "md": pos.md.tolist(),
            "tvd": pos.tvd.tolist(),
            "enu": pos.enu.tolist(),
            "dls": pos.dls.tolist(),
            "drillability": flag.to_payload(),
        }

    # ──────────────────────── POST /wells/{wid}/predict ────────────────────────
    @router.post("/wells/{wid}/predict")
    def predict_well(wid: str, body: PredictRequest, request: Request, session: Session = session_dep):  # noqa: E501
        well = _well_from_row(_well_or_404(session, wid))
        fem = _fem_or_404(session, body.fused_model_id)
        target = None
        if body.target_id is not None:
            trow = session.get(Feature, body.target_id)
            if trow is None or trow.feature_type != "drillTarget":
                raise HTTPException(status_code=404, detail="target not found")
            target = _target_from_row(trow)
        weights = RiskWeights()
        if body.risk_weights:
            rw = body.risk_weights
            weights = RiskWeights(
                temp_confidence=float(rw.get("tempConfidence", weights.temp_confidence)),
                hazard=float(rw.get("hazard", weights.hazard)),
                dls_exceedance=float(rw.get("dlsExceedance", weights.dls_exceedance)),
                structural_uncertainty=float(
                    rw.get("structuralUncertainty", weights.structural_uncertainty)
                ),
            )
        log = predict_log(
            session, fem, well, md_step_m=body.md_step_m, target=target,
            risk_weights=weights, favorability_threshold=body.favorability_threshold,
            fracture_threshold=body.fracture_threshold,
            storage_root=request.app.state.storage_root,
        )
        flag = drillability_flag(
            well.deviation_survey, well.positions(),
            hardness=log.hardness, hardness_md=log.hardness_md,
            limits=DrillabilityLimits(
                max_dls_deg30m=well.constraints.max_dls_deg30m,
                max_inc_deg=well.constraints.max_inc_deg,
            ),
        )
        payload = log.to_payload()
        payload["drillability"] = flag.to_payload()
        return payload

    # ──────────────────────── GET /wells/{wid}/export ────────────────────────
    @router.get("/wells/{wid}/export")
    def export_well(
        wid: str,
        request: Request,
        fmt: str = "csv-survey",
        version: str = "2.0",
        units: str = "metric",
        fused_model_id: str | None = None,
        md_step_m: float = 5.0,
        target_id: str | None = None,
        session: Session = session_dep,
    ):
        """Export a planned well (doc 09 §9 / §10 export surface).

        ``fmt`` selects ``csv-survey`` (deviation survey CSV), ``csv-log`` (predicted-log CSV,
        needs ``fused_model_id``), or ``witsml`` (trajectory; ``version`` 2.0 | 1.4.1.1).
        ``units`` is ``metric`` (canonical) or ``field``. Coordinates round-trip through the
        project CRS (doc 01 §7) when the frame is georeferenced. Returns the file as a download.
        """
        row = _well_or_404(session, wid)
        well = _well_from_row(row)
        frame = _frame_for_project(session, well.project_id)

        if fmt == "csv-survey":
            body = export_survey_csv(well, units=units, frame=frame)
            return _download(body, "text/csv", f"{well.id}_survey.csv")
        if fmt == "csv-log":
            log = _resolve_log(session, request, well, fused_model_id, md_step_m, target_id)
            body = export_log_csv(log, well, units=units, frame=frame)
            return _download(body, "text/csv", f"{well.id}_log.csv")
        if fmt == "witsml":
            mv = None
            if fused_model_id is not None:
                mv = _fem_or_404(session, fused_model_id).id
            try:
                body = export_witsml_trajectory(
                    well, version=version, units=units, frame=frame, model_version=mv,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            ext = "xml"
            return _download(body, "application/x-witsml+xml", f"{well.id}_trajectory.{ext}")
        raise HTTPException(
            status_code=400,
            detail="fmt must be one of: csv-survey, csv-log, witsml",
        )

    def _frame_for_project(session: Session, pid: str):
        row = session.get(Project, pid)
        if row is None or row.spatial_frame is None:
            return None
        return frame_from_row(row.spatial_frame)

    def _resolve_log(session, request, well, fused_model_id, md_step_m, target_id):
        if fused_model_id is None:
            raise HTTPException(status_code=400, detail="csv-log requires fused_model_id")
        fem = _fem_or_404(session, fused_model_id)
        target = None
        if target_id is not None:
            trow = session.get(Feature, target_id)
            if trow is None or trow.feature_type != "drillTarget":
                raise HTTPException(status_code=404, detail="target not found")
            target = _target_from_row(trow)
        return predict_log(
            session, fem, well, md_step_m=md_step_m, target=target,
            storage_root=request.app.state.storage_root,
        )

    return router


def _download(body: str, media_type: str, filename: str) -> Response:
    """A file-download response with a Content-Disposition attachment header."""
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _target_from_row(row: Feature) -> DrillTarget:
    props = json.loads(row.props_json) if row.props_json else {}
    p = props.get("props", props)
    loc = p["location"]
    return DrillTarget(
        id=row.id, name=p.get("name", "target"), project_id=row.project_id,
        kind=p.get("kind", "point"),
        location=(float(loc["x"]), float(loc["y"]), float(loc["z"])),
        tolerance=TargetTolerance(
            float(p.get("tolerance", {}).get("radius_m", 50.0)),
            float(p.get("tolerance", {}).get("tvd_window_m", 25.0)),
        ),
        desired_temperature_c=p.get("desiredTemperatureC"),
        min_temperature_c=p.get("minTemperatureC"),
        kb_elev_m=float(p.get("kb_elev_m", 0.0)),
    )


def _enu_bbox(enu: np.ndarray) -> dict:
    return {
        "xmin": float(enu[:, 0].min()), "xmax": float(enu[:, 0].max()),
        "ymin": float(enu[:, 1].min()), "ymax": float(enu[:, 1].max()),
        "zmin": float(enu[:, 2].min()), "zmax": float(enu[:, 2].max()),
    }
