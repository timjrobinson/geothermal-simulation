"""Cooperative (sequential) inversion (doc 10 §6 stage 5b) — a small single-method DAG.

Doc 10 §6 distinguishes three multi-physics strategies:

- **5a** independent single-method inversions (the plain :func:`~geosim.inversion.harness.
  run_inversion` jobs);
- **5b** *cooperative / sequential* coupling — invert method **A**, then feed A's recovered
  model into method **B** as a **reference / starting model** or a **structure-guided
  weight** (this module);
- **5c** *joint / cross-gradient* coupling — a single monolithic objective solving both
  physics at once (roadmap; **not** built here).

The cooperative strategy is deliberately implemented as an **orchestration of ordinary
§3 single-method jobs** — a tiny dependency DAG — rather than a joint solver (doc 10 §6).
Stage A runs through the unmodified harness and persists an ordinary PropertyModel; the
orchestrator then reads A's recovered CORE model back and threads it into stage B's
:class:`~geosim.inversion.engine.InversionContext` as ``ctx.reference_model`` (and a
matching ``ctx.structure_weight``). Engine B opts in by reading those attributes; engines
that ignore them still run unchanged, so the DAG is robust across the whole engine palette.

Every coupling is recorded in the stage-B :class:`~geosim.inversion.engine.
InversionProvenance` under ``metrics['coupling']`` (``stage='5b'`` + the partner engine /
property / source PropertyModel), so a cooperative run is reproducible and the lineage
shows exactly which model guided which (doc 10 §7).

Nothing here imports SimPEG / PyGIMLi — the orchestrator only moves NumPy + ids between
unmodified §3 harness jobs (doc 10 §8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from geosim.jobs import ProgressReporter
from geosim.storage import ProjectLayout, open_property_model

from .domain import ModelDomain, PaddingSpec, build_tensor_domain
from .engine import InversionContext, InversionEngine, InversionResult
from .harness import InversionRunResult, run_inversion

__all__ = [
    "COUPLING_STAGE",
    "CooperativeStage",
    "CooperativeResult",
    "ReferenceModel",
    "ReferenceGuidedEngine",
    "cooperative_invert",
]

# The doc 10 §6 stage tag every cooperative coupling carries in provenance.
COUPLING_STAGE = "5b"


# ──────────────────────────── the reference handoff (doc 10 §6) ────────────────────────────


@dataclass(frozen=True)
class ReferenceModel:
    """A recovered CORE model handed from stage A to stage B (doc 10 §6 5b).

    Carries the recovered ``values`` on the **core** block as a Z-up ``(z, y, x)`` array,
    the property it represents (``property``), the partner engine id, and the source
    PropertyModel id for lineage. An engine that supports structure-guided coupling reads
    this off ``ctx.reference_model`` and uses it as a reference / starting model, while the
    derived :meth:`structure_weight` gives a normalised gradient-magnitude field a
    smoothness regulariser can use as a structure weight (doc 10 §6).
    """

    values: np.ndarray  # (z, y, x) recovered partner core model
    property: str  # the partner PropertyType (e.g. "density")
    engine_id: str  # the partner engine that produced it
    source_property_model_id: str  # lineage pointer (doc 10 §7)

    def structure_weight(self) -> np.ndarray:
        """Normalised ``[0, 1]`` gradient-magnitude structure weight (doc 10 §6).

        High where the partner model has sharp gradients (likely a geological boundary),
        low in smooth regions — exactly the cell weighting a structure-guided smoothness
        regulariser consumes so method B's boundaries are nudged toward A's (doc 10 §6 5b).
        """
        vals = np.asarray(self.values, dtype=float)
        if vals.ndim != 3 or vals.size == 0:
            return np.zeros_like(vals, dtype=np.float32)
        grad = np.gradient(vals)
        mag = np.sqrt(np.sum([g**2 for g in grad], axis=0))
        peak = float(np.max(mag))
        if peak <= 0.0:
            return np.zeros_like(mag, dtype=np.float32)
        return (mag / peak).astype(np.float32)


class ReferenceGuidedEngine:
    """Wrap stage-B's engine so its context carries stage-A's recovered model (doc 10 §6).

    A transparent decorator: it shares the wrapped engine's :attr:`spec` (so the harness
    persists / validates B exactly as a standalone run) but, on :meth:`run`, attaches the
    stage-A :class:`ReferenceModel` to the :class:`~geosim.inversion.engine.
    InversionContext` as ``ctx.reference_model`` + ``ctx.structure_weight`` before
    delegating to B. It then stamps the 5b coupling record into the returned
    :class:`~geosim.inversion.engine.InversionResult` ``metrics`` so the harness folds it
    into :class:`~geosim.inversion.engine.InversionProvenance` (doc 10 §7).

    Engines that don't read the reference still run unchanged — the coupling is advisory,
    and the provenance edge is recorded regardless, which is the contract stage 5b needs.
    """

    def __init__(self, engine: InversionEngine, reference: ReferenceModel) -> None:
        self._engine = engine
        self._reference = reference
        # Share B's spec verbatim so the harness validates + persists B unchanged.
        self.spec = engine.spec

    def run(self, ctx: InversionContext) -> InversionResult:
        # Attach the partner model to the context (opt-in: engines read it off ctx).
        ctx.reference_model = self._reference  # type: ignore[attr-defined]
        ctx.structure_weight = self._reference.structure_weight()  # type: ignore[attr-defined]
        result = self._engine.run(ctx)
        # Record the 5b coupling so provenance shows which model guided this run (doc 10 §7).
        coupling = {
            "stage": COUPLING_STAGE,
            "strategy": "cooperative",
            "partnerEngine": self._reference.engine_id,
            "partnerProperty": self._reference.property,
            "partnerPropertyModelId": self._reference.source_property_model_id,
            "use": "referenceModel+structureWeight",
        }
        result.metrics = {**result.metrics, "coupling": coupling}
        return result


# ──────────────────────────── DAG node + result (doc 10 §6) ────────────────────────────


@dataclass
class CooperativeStage:
    """One node of the cooperative DAG: a single §3 inversion job (doc 10 §6).

    Each stage is an ordinary single-method inversion (engine + observations + core +
    params). ``depends_on`` names an earlier stage whose recovered model seeds this one as
    a reference / structure weight (doc 10 §6 5b); ``None`` ⇒ a root (stage A). The DAG is
    run in list order, so a stage's dependency must precede it.
    """

    name: str
    engine: InversionEngine
    observation_ids: Sequence[str]
    core: Any  # CoreRegion
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: str | None = None
    n_pad: int = 0
    pad_factor: float = 1.3
    surface_z: float | None = None
    resample_fused: bool = True


@dataclass(frozen=True)
class CooperativeResult:
    """The outcome of a cooperative DAG run (doc 10 §6).

    ``stages`` maps each stage name → its :class:`~geosim.inversion.harness.
    InversionRunResult`. ``order`` is the executed sequence; ``couplings`` records each
    5b handoff (child stage → partner stage / property / source model) for the parent
    job payload (doc 10 §7).
    """

    stages: dict[str, InversionRunResult]
    order: list[str]
    couplings: list[dict[str, Any]]

    @property
    def final(self) -> InversionRunResult:
        """The last-executed stage's result (the cooperatively-guided model)."""
        return self.stages[self.order[-1]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "strategy": "cooperative",
            "stage": COUPLING_STAGE,
            "order": list(self.order),
            "stages": {name: r.to_payload() for name, r in self.stages.items()},
            "couplings": list(self.couplings),
            "final": self.final.to_payload(),
        }


# ──────────────────────────── the orchestrator (doc 10 §6 5b) ────────────────────────────


def _load_reference(
    layout: ProjectLayout,
    run: InversionRunResult,
    engine_id: str,
) -> ReferenceModel:
    """Read a finished stage's recovered CORE model back as a :class:`ReferenceModel`."""
    reader = open_property_model(layout.zarr_path(run.property_model_id))
    values = reader.read_level(run.property, 0)
    return ReferenceModel(
        values=np.asarray(values),
        property=run.property,
        engine_id=engine_id,
        source_property_model_id=run.property_model_id,
    )


def cooperative_invert(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    stages: Sequence[CooperativeStage],
    *,
    created_by: str = "system:inversion",
    reporter: ProgressReporter | None = None,
    storage_root: str | Path | None = None,
) -> CooperativeResult:
    """Run a cooperative (sequential) inversion DAG (doc 10 §6 stage 5b).

    Executes ``stages`` in order. A root stage (``depends_on is None``) runs as an ordinary
    §3 :func:`~geosim.inversion.harness.run_inversion` job. A dependent stage reads its
    parent's recovered CORE model back and threads it into its own engine as a
    reference / starting model + structure weight (via :class:`ReferenceGuidedEngine`),
    recording the 5b coupling in :class:`~geosim.inversion.engine.InversionProvenance`
    (doc 10 §7). Returns a :class:`CooperativeResult` whose ``final`` is the last,
    cooperatively-guided model.

    This is pure orchestration: NO monolithic joint solver (5c stays roadmap, doc 10 §6).
    Each stage is a real, separately-persisted PropertyModel, so the whole DAG reuses the
    existing storage / fusion / serving unchanged (doc 10 §0).
    """
    if not stages:
        raise ValueError("cooperative_invert needs at least one stage")

    results: dict[str, InversionRunResult] = {}
    couplings: list[dict[str, Any]] = []
    order: list[str] = []
    n = len(stages)

    for i, stage in enumerate(stages):
        if stage.name in results:
            raise ValueError(f"duplicate stage name {stage.name!r}")
        if stage.depends_on is not None and stage.depends_on not in results:
            raise ValueError(
                f"stage {stage.name!r} depends on {stage.depends_on!r} which has not run "
                "yet (stages run in list order; a dependency must precede it)"
            )

        if reporter is not None:
            reporter.report(i / n, f"cooperative stage {stage.name!r} ({stage.engine.spec.id})")

        domain: ModelDomain = build_tensor_domain(
            stage.core,
            padding=PaddingSpec(n_pad=stage.n_pad, factor=stage.pad_factor),
            surface_z=stage.surface_z,
        )

        engine: InversionEngine = stage.engine
        if stage.depends_on is not None:
            partner_run = results[stage.depends_on]
            partner = next(s for s in stages if s.name == stage.depends_on)
            reference = _load_reference(layout, partner_run, partner.engine.spec.id)
            engine = ReferenceGuidedEngine(stage.engine, reference)  # type: ignore[assignment]
            couplings.append({
                "stage": COUPLING_STAGE,
                "child": stage.name,
                "partner": stage.depends_on,
                "partnerEngine": partner.engine.spec.id,
                "partnerProperty": reference.property,
                "partnerPropertyModelId": reference.source_property_model_id,
            })

        run = run_inversion(
            session, layout, project_id, engine,
            domain=domain, observation_ids=list(stage.observation_ids),
            params=stage.params, name=f"{stage.name}-{stage.engine.spec.id}",
            created_by=created_by, reporter=None,
            resample_fused=stage.resample_fused, storage_root=storage_root,
        )
        results[stage.name] = run
        order.append(stage.name)

    if reporter is not None:
        reporter.report(1.0, "cooperative inversion complete")

    return CooperativeResult(stages=results, order=order, couplings=couplings)
