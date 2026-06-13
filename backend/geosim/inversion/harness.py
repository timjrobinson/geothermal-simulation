"""The inversion run harness (doc 10 §3, §7) — validate → run → persist → resample.

This is the engine-agnostic orchestration that turns an :class:`~geosim.inversion.engine.
InversionEngine` into a finished catalog artifact, reusing **all** existing infrastructure
(doc 10 §0):

1. **validate** user params against the engine's ``paramsSchema`` BEFORE running (doc 10
   §3 — a job is never enqueued with bad params): :func:`validate_params`.
2. **build** the :class:`InversionContext` (Observations + ModelDomain + a progress shim
   over a :class:`~geosim.jobs.ProgressReporter` + a cooperative-cancel check).
3. **run** the engine; it returns a recovered CORE model + a MANDATORY uncertainty field
   (doc 10 §2.3). If the engine omitted σ a tier-B sensitivity/DOI-weighted default is
   substituted (:func:`default_uncertainty`).
4. **persist** the recovered model as an ORDINARY PropertyModel via
   :func:`geosim.storage.write_property_model` (value pyramid + sibling ``_sigma``), an
   :class:`~geosim.inversion.engine.InversionProvenance` row, and a ``property_models``
   catalog row (doc 10 §7) — inversion output is *just a PropertyModel* (doc 10 §0).
5. **resample** the core onto a fused grid via :func:`geosim.fusion.resample_to_fused`
   (doc 10 §4.4) so it co-locates with every other property.

Nothing here imports SimPEG/PyGIMLi; the engine owns all solver containers (doc 10 §8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from geosim.catalog import (
    Dataset,
    IdKind,
    Observation,
    Project,
    PropertyModel,
    Provenance,
    ProvenanceInput,
    new_id,
)
from geosim.fusion import build_fused_model, resample_to_fused
from geosim.jobs import ProgressReporter
from geosim.spatial import REGISTRY
from geosim.storage import (
    SIGMA_SUFFIX,
    GridSpec,
    ProjectLayout,
    write_property_model,
)

from .engine import InversionContext, InversionEngine, InversionProvenance, InversionResult

__all__ = [
    "ParamValidationError",
    "validate_params",
    "default_uncertainty",
    "InversionRunResult",
    "load_observations",
    "run_inversion",
]


# ──────────────────────────── params validation (doc 10 §3) ────────────────────────────


class ParamValidationError(ValueError):
    """User params failed the engine ``paramsSchema`` — raised BEFORE enqueue (doc 10 §3)."""


# JSON-Schema-subset keywords the harness understands (matching the manifest validator's
# self-contained style — no jsonschema dependency, doc 08 §5.1).
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def validate_params(params: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Validate + default ``params`` against a JSON-Schema-subset ``schema`` (doc 10 §3).

    Supported keywords: top-level ``required`` + ``properties`` with per-property
    ``type``, ``enum``, ``minimum``/``maximum`` (numbers), ``default``. Unknown properties
    are rejected when ``additionalProperties`` is ``False`` (the default). Returns a NEW
    dict with defaults applied; raises :class:`ParamValidationError` on the first failure
    so a bad request is rejected **before** the job is enqueued.
    """
    params = dict(params or {})
    props: dict[str, Any] = schema.get("properties", {})
    required: list[str] = list(schema.get("required", []))
    additional = schema.get("additionalProperties", False)

    if additional is False:
        extra = [k for k in params if k not in props]
        if extra:
            raise ParamValidationError(f"unknown param(s): {sorted(extra)}")

    for key in required:
        if key not in params:
            if key in props and "default" in props[key]:
                continue  # a default satisfies required
            raise ParamValidationError(f"missing required param {key!r}")

    out: dict[str, Any] = {}
    for key, pschema in props.items():
        if key not in params:
            if "default" in pschema:
                out[key] = pschema["default"]
            continue
        out[key] = _validate_one(key, params[key], pschema)
    # carry through any explicitly-allowed additional keys
    if additional is not False:
        for k, v in params.items():
            if k not in props:
                out[k] = v
    return out


def _validate_one(key: str, value: Any, pschema: dict[str, Any]) -> Any:
    t = pschema.get("type")
    if t is not None:
        allowed = _TYPE_MAP.get(t)
        if allowed is None:
            raise ParamValidationError(f"param {key!r}: unknown schema type {t!r}")
        # bool is an int subclass — exclude it from numeric/integer unless type=boolean.
        if t in ("number", "integer") and isinstance(value, bool):
            raise ParamValidationError(f"param {key!r} must be {t}, got boolean")
        if not isinstance(value, allowed):
            raise ParamValidationError(
                f"param {key!r} must be {t}, got {type(value).__name__}"
            )
    if "enum" in pschema and value not in pschema["enum"]:
        raise ParamValidationError(
            f"param {key!r}={value!r} not in enum {pschema['enum']}"
        )
    if t in ("number", "integer"):
        if "minimum" in pschema and value < pschema["minimum"]:
            raise ParamValidationError(
                f"param {key!r}={value} below minimum {pschema['minimum']}"
            )
        if "maximum" in pschema and value > pschema["maximum"]:
            raise ParamValidationError(
                f"param {key!r}={value} above maximum {pschema['maximum']}"
            )
    return value


# ──────────────────────────── default uncertainty (doc 10 §2.3) ────────────────────────────


def default_uncertainty(values: np.ndarray, output_property: str) -> np.ndarray:
    """Tier-B sensitivity/DOI-weighted default 1σ field (doc 10 §2.3).

    When an engine returns no native uncertainty the harness MUST still emit one (an
    inversion without uncertainty is invalid). The default is a depth-of-investigation
    proxy: a relative σ from the property registry, **inflated with depth** (cells deeper
    in the core are less constrained by surface data — a DOI proxy). ``values`` is the
    recovered ``(z, y, x)`` core (Z-up; index 0 = deepest).
    """
    try:
        rel = REGISTRY.get(output_property).default_rel_sigma
    except KeyError:
        rel = 0.15
    base = np.abs(np.asarray(values, dtype=float)) * float(rel)
    nz = values.shape[0]
    if nz > 1:
        # DOI inflation: 1.0 at the (shallow) top, growing toward the (deep) bottom.
        # Z-up → index 0 is deepest, so weight is largest at index 0.
        depth_frac = np.linspace(1.0, 0.0, nz)  # 1 at deepest, 0 at shallowest
        doi = 1.0 + depth_frac  # [1, 2] inflation factor
        base = base * doi[:, None, None]
    return base.astype(np.float32)


# ──────────────────────────── observation loading ────────────────────────────


def load_observations(session: Session, observation_ids: list[str]) -> list[dict[str, Any]]:
    """Load Observations into engine-agnostic dicts (doc 02 §3 normalized primitive).

    Each dict carries ``id`` / ``geometry_kind`` / ``primary_property`` / ``bbox`` plus the
    inline ``coords`` + per-property ``values``/``sigma`` (doc 04 §2.4 ``values_json``).
    No SimPEG/PyGIMLi types — the engine builds its own survey from these (doc 10 §8).
    """
    out: list[dict[str, Any]] = []
    for oid in observation_ids:
        row = session.get(Observation, oid)
        if row is None:
            raise ValueError(f"observation {oid!r} not found")
        payload = json.loads(row.values_json) if row.values_json else {}
        out.append(
            {
                "id": row.id,
                "geometry_kind": row.geometry_kind,
                "primary_property": row.primary_property,
                "bbox": json.loads(row.bbox_json),
                "coords": payload.get("coords", []),
                "values": payload.get("values", {}),
                "sigma": payload.get("sigma", {}),
                "meta": json.loads(row.meta_json) if row.meta_json else {},
            }
        )
    return out


# ──────────────────────────── run result ────────────────────────────


@dataclass(frozen=True)
class InversionRunResult:
    """Handle to a completed inversion run (doc 10 §3, §7)."""

    property_model_id: str
    dataset_id: str
    provenance_id: str
    property: str
    iterations: int
    final_phi_d: float | None
    final_phi_m: float | None
    fused_model_id: str | None = None
    fused_layer_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "propertyModelId": self.property_model_id,
            "datasetId": self.dataset_id,
            "provenanceId": self.provenance_id,
            "property": self.property,
            "iterations": self.iterations,
            "finalPhiD": self.final_phi_d,
            "finalPhiM": self.final_phi_m,
            "fusedModelId": self.fused_model_id,
            "fusedLayerId": self.fused_layer_id,
        }


# ──────────────────────────── the harness (doc 10 §3) ────────────────────────────


def run_inversion(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    engine: InversionEngine,
    *,
    domain: Any,  # ModelDomain
    observation_ids: list[str],
    params: dict[str, Any] | None = None,
    name: str | None = None,
    created_by: str = "system:inversion",
    reporter: ProgressReporter | None = None,
    resample_fused: bool = True,
    storage_root: str | Path | None = None,
) -> InversionRunResult:
    """Run an inversion end-to-end and persist it as a PropertyModel (doc 10 §3, §7).

    Params are validated against ``engine.spec.params_schema`` FIRST (raising
    :class:`ParamValidationError` before any work). The engine runs against an
    :class:`InversionContext` wired to ``reporter`` for progress + cooperative cancel; its
    recovered CORE model + MANDATORY uncertainty are written as an ordinary PropertyModel
    (value + ``_sigma`` pyramids, doc 02 §10.2) with a full :class:`InversionProvenance`
    lineage. When ``resample_fused`` the core resamples onto a fused grid (doc 10 §4.4).
    """
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError(f"unknown project {project_id!r}")

    # 1) validate params BEFORE running (doc 10 §3).
    validated = validate_params(params or {}, dict(engine.spec.params_schema))

    # 2) load observations + build the run context.
    observations = load_observations(session, observation_ids)

    def _report(frac: float, message: str, extra: dict[str, Any]) -> None:
        if reporter is None:
            return
        if extra:
            it = extra.get("iteration")
            pd, pm_ = extra.get("phi_d"), extra.get("phi_m")
            bits = [b for b in (
                f"it {it}" if it is not None else None,
                f"phi_d={pd:.3g}" if pd is not None else None,
                f"phi_m={pm_:.3g}" if pm_ is not None else None,
            ) if b]
            if bits:
                message = f"{message} ({', '.join(bits)})" if message else ", ".join(bits)
        reporter.report(frac, message or None)

    def _cancelled() -> bool:
        return reporter is not None and reporter.cancelled()

    ctx = InversionContext(
        spec=engine.spec,
        observations=observations,
        domain=domain,
        params=validated,
        report=_report,
        is_cancelled=_cancelled,
    )

    if reporter is not None:
        reporter.report(0.0, f"inversion {engine.spec.id} starting")

    # 3) run the engine (it builds its own solver containers internally, doc 10 §8).
    result: InversionResult = engine.run(ctx)

    # MANDATORY uncertainty (doc 10 §2.3): substitute the tier-B default if missing/empty.
    sigma = np.asarray(result.sigma, dtype=np.float32)
    if sigma.size == 0 or not np.any(np.isfinite(sigma)):
        sigma = default_uncertainty(result.values, engine.spec.output_property)

    if reporter is not None:
        reporter.report(0.9, "writing recovered property model")

    # 4) persist as an ordinary PropertyModel + provenance (doc 10 §0, §7).
    run = _persist(
        session, layout, project_id, engine, domain, result, sigma,
        observation_ids, validated, name=name, created_by=created_by,
    )

    # 5) resample the recovered core onto a fused grid (doc 10 §4.4).
    fem_id, layer_id = None, None
    if resample_fused:
        if reporter is not None:
            reporter.report(0.95, "resampling onto fused grid")
        fem, _grid = build_fused_model(
            session, layout, project_id,
            source_property_model_ids=[run.property_model_id],
            name=f"{engine.spec.id}-fused", created_by=created_by,
        )
        ref = resample_to_fused(
            session, fem, run.property_model_id, storage_root=storage_root,
        )
        fem_id, layer_id = fem.id, ref.layer_id

    if reporter is not None:
        reporter.report(1.0, "inversion complete")

    return InversionRunResult(
        property_model_id=run.property_model_id,
        dataset_id=run.dataset_id,
        provenance_id=run.provenance_id,
        property=engine.spec.output_property,
        iterations=result.iterations,
        final_phi_d=result.final_phi_d,
        final_phi_m=result.final_phi_m,
        fused_model_id=fem_id,
        fused_layer_id=layer_id,
    )


@dataclass(frozen=True)
class _PersistResult:
    property_model_id: str
    dataset_id: str
    provenance_id: str


def _persist(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    engine: InversionEngine,
    domain: Any,
    result: InversionResult,
    sigma: np.ndarray,
    observation_ids: list[str],
    params: dict[str, Any],
    *,
    name: str | None,
    created_by: str,
) -> _PersistResult:
    """Write the recovered model + provenance rows in one transaction (doc 10 §7).

    Provenance is written FIRST (doc 02 §7 — no dataset without provenance), with each
    input Observation recorded as a ``provenance_inputs`` edge; then the recovered model is
    written to a doc-02 Zarr group (value + ``_sigma`` pyramids) and a ``property_models``
    row inserted carrying the inversion uncertainty pointer (doc 02 §6).
    """
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    prop = engine.spec.output_property

    origin, spacing = domain.core_grid()
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    zarr_path = layout.zarr_path(pm_id)
    write_property_model(
        zarr_path, prop, result.values, grid=grid, sigma=sigma, overwrite=True
    )

    nz, ny, nx = result.values.shape
    oz, oy, ox = origin
    dz, dy, dx = spacing
    bbox = {
        "xmin": float(ox), "xmax": float(ox + dx * (nx - 1)),
        "ymin": float(oy), "ymax": float(oy + dy * (ny - 1)),
        "zmin": float(oz), "zmax": float(oz + dz * (nz - 1)),
    }
    bbox_json = json.dumps(bbox)

    prov = InversionProvenance(
        engine_id=engine.spec.id,
        engine_library=engine.spec.library,
        engine_kind=engine.spec.kind,
        process_version=str(result.metrics.get("processVersion", "1.0.0")),
        params=params,
        observation_ids=list(observation_ids),
        mesh={
            "type": domain.mesh_type,
            "nCells": domain.n_cells,
            "nActive": domain.n_active,
            "core": {
                "origin": list(domain.core.origin),
                "extent": list(domain.core.extent),
                "cellSize": list(domain.core.cell_size),
                "nCore": list(domain.core.n_core()),
            },
            "padding": {"nPad": domain.padding.n_pad, "factor": domain.padding.factor},
        },
        iterations=result.iterations,
        final_phi_d=result.final_phi_d,
        final_phi_m=result.final_phi_m,
        metrics=result.metrics,
    )

    method = engine.spec.methods[0] if engine.spec.methods else engine.spec.kind
    session.add(Provenance(
        id=prov_id, project_id=project_id, target_kind="propertyModel", target_id=pm_id,
        process=f"invert:{engine.spec.id}", process_version=prov.process_version,
        params_json=json.dumps(prov.to_params_json()),
    ))
    session.flush()
    for oid in observation_ids:
        session.add(ProvenanceInput(
            provenance_id=prov_id, input_kind="observation", input_id=oid
        ))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=project_id, name=name or f"{engine.spec.id}-inversion",
        method=method, kind="propertyModel", status="ready", extent_json=bbox_json,
        spatial_frame_id=project_id, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by=created_by,
    ))
    session.flush()
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=project_id, property=prop,
        canonical_unit=REGISTRY.get(prop).canonical_unit, support="volume",
        store_uri=str(zarr_path), store_format="zarr",
        shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(spacing)), origin_json=json.dumps(list(origin)),
        bbox_json=bbox_json, has_time=0, pyramid_levels=1,
        uncertainty_uri=f"{prop}{SIGMA_SUFFIX}",
    ))
    session.commit()

    return _PersistResult(property_model_id=pm_id, dataset_id=ds_id, provenance_id=prov_id)
