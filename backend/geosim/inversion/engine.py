"""The InversionEngine contract + context/result/provenance dataclasses (doc 10 §2, §7).

This is the **engine-agnostic** boundary. An inversion engine is *just another
contribution* (doc 08 §4f): it advertises an :class:`InversionEngineSpec` (id / kind /
library / methods / outputProperty / meshTypes / coupling / compute / paramsSchema) and
implements :meth:`InversionEngine.run` ``(ctx) -> InversionResult``. The harness
(:mod:`geosim.inversion.harness`) validates params against ``paramsSchema`` BEFORE the
engine runs, builds the :class:`InversionContext`, threads progress, and turns the
returned :class:`InversionResult` into an ordinary PropertyModel + uncertainty +
:class:`InversionProvenance` (doc 10 §3, §7).

SimPEG ``Survey``/``Data`` and PyGIMLi containers are constructed **inside** ``run`` and
never appear in these dataclasses — only NumPy + the engine-agnostic
:class:`~geosim.inversion.domain.ModelDomain` cross the boundary (doc 10 §8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .domain import MESH_TYPES, ModelDomain

__all__ = [
    "InversionEngineSpec",
    "InversionContext",
    "InversionResult",
    "InversionProvenance",
    "InversionEngine",
    "register_inversion_engine",
]


# ───────────────────────────── declarative spec (doc 10 §2) ─────────────────────────────


@dataclass(frozen=True)
class InversionEngineSpec:
    """Declarative descriptor of an inversion engine (doc 10 §2).

    All fields are serialisable so the engine palette is discoverable via the registry /
    capabilities (doc 08 §7). ``paramsSchema`` is a JSON-Schema-subset the harness
    validates user params against **before** enqueueing (doc 10 §3) — see
    :func:`geosim.inversion.harness.validate_params` for the supported keywords.
    """

    id: str  # "mock.linear", "simpeg.dc", "pygimli.ert"
    kind: str  # the inverse problem kind, e.g. "dc", "ert", "gravity", "mt"
    library: str  # "mock" | "simpeg" | "pygimli" — provenance + executionMode hint
    methods: Sequence[str]  # canonical MethodKeys it can invert (doc 02 §2)
    output_property: str  # PropertyType key the recovered model carries (doc 01 §5)
    mesh_types: Sequence[str] = ("tensor",)  # supported ModelDomain mesh kinds (doc 10 §4)
    coupling: str = "standalone"  # "standalone" | "joint" | "petrophysical" (doc 10 §2)
    compute: str = "in_process"  # ExecutionMode hint (doc 08 §2.1)
    params_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("InversionEngineSpec.id is required")
        if not self.output_property:
            raise ValueError("InversionEngineSpec.output_property is required")
        bad = [m for m in self.mesh_types if m not in MESH_TYPES]
        if bad:
            raise ValueError(f"unknown mesh_types {bad!r}; allowed {MESH_TYPES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "library": self.library,
            "methods": list(self.methods),
            "outputProperty": self.output_property,
            "meshTypes": list(self.mesh_types),
            "coupling": self.coupling,
            "compute": self.compute,
            "paramsSchema": dict(self.params_schema),
        }


# ──────────────────────────────── run context (doc 10 §2) ────────────────────────────────


# A progress callback: (fraction 0..1, message, extra-metrics). The harness adapts a
# geosim.jobs.ProgressReporter onto this so the engine never imports the jobs package.
ProgressFn = Callable[[float, str, dict[str, Any]], None]


@dataclass
class InversionContext:
    """Everything an engine needs to run, with NO SimPEG/PyGIMLi types (doc 10 §2, §8).

    - ``observations`` — the doc-02 Observations to fit (engine-agnostic dicts: coords +
      per-property values/σ + meta), exactly the normalized primitive ingestion produces.
    - ``domain`` — the :class:`~geosim.inversion.domain.ModelDomain` mesh in the
      Engineering Frame (doc 10 §4).
    - ``params`` — user params, already validated against the spec's ``paramsSchema``.
    - ``report`` — progress + iteration metrics (φ_d / φ_m), doc 10 §3.
    - ``is_cancelled`` — cooperative-cancel check the engine polls between iterations.
    """

    spec: InversionEngineSpec
    observations: list[dict[str, Any]]
    domain: ModelDomain
    params: dict[str, Any]
    report: ProgressFn = lambda frac, msg, extra: None  # noqa: E731
    is_cancelled: Callable[[], bool] = lambda: False  # noqa: E731

    def progress(
        self,
        frac: float,
        message: str = "",
        *,
        iteration: int | None = None,
        phi_d: float | None = None,
        phi_m: float | None = None,
    ) -> None:
        """Report progress + the classic inversion metrics (doc 10 §3).

        ``phi_d`` (data misfit) and ``phi_m`` (model regularisation) ride along in the
        ``extra`` dict so a UI can plot the Tikhonov tradeoff curve.
        """
        extra: dict[str, Any] = {}
        if iteration is not None:
            extra["iteration"] = int(iteration)
        if phi_d is not None:
            extra["phi_d"] = float(phi_d)
        if phi_m is not None:
            extra["phi_m"] = float(phi_m)
        self.report(float(frac), message, extra)


# ──────────────────────────────── result (doc 10 §2.3) ────────────────────────────────


@dataclass
class InversionResult:
    """What an engine returns: recovered CORE model + MANDATORY uncertainty (doc 10 §2.3).

    - ``values`` — the recovered property on the **core** cells as a Z-up ``(z, y, x)``
      array (use :meth:`ModelDomain.extract_core` to map a per-cell vector onto this).
    - ``sigma`` — the MANDATORY 1σ uncertainty field, same shape (doc 10 §2.3: an
      inversion without uncertainty is invalid). A tier-B sensitivity/DOI-weighted default
      is provided by :func:`geosim.inversion.harness.default_uncertainty` when an engine
      has no native estimate.
    - ``iterations`` / ``final_phi_d`` / ``final_phi_m`` — convergence record (doc 10 §3).
    - ``metrics`` — open per-engine diagnostics folded into provenance (doc 10 §7).
    """

    values: np.ndarray  # (z, y, x) recovered core model, canonical units
    sigma: np.ndarray  # (z, y, x) 1σ uncertainty — MANDATORY (doc 10 §2.3)
    iterations: int = 0
    final_phi_d: float | None = None
    final_phi_m: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float32)
        if self.sigma is None:
            raise ValueError(
                "InversionResult.sigma is MANDATORY — an inversion with no uncertainty is "
                "invalid (doc 10 §2.3)"
            )
        self.sigma = np.asarray(self.sigma, dtype=np.float32)
        if self.sigma.shape != self.values.shape:
            raise ValueError(
                f"sigma shape {self.sigma.shape} must match values {self.values.shape} "
                "(doc 10 §2.3)"
            )
        if self.values.ndim != 3:
            raise ValueError(f"values must be 3D (z,y,x); got {self.values.shape}")


# ──────────────────────────────── provenance (doc 10 §7) ────────────────────────────────


@dataclass
class InversionProvenance:
    """Full reproducibility record for an inversion run (doc 10 §7).

    Captures the engine identity + version, the exact params, the mesh fingerprint, the
    observation inputs, and the convergence record so a run is reproducible end-to-end.
    Serialised into the catalog ``provenance.params_json`` by the harness, with the
    Observation inputs recorded as ``provenance_inputs`` edges (doc 02 §7).
    """

    engine_id: str
    engine_library: str
    engine_kind: str
    process_version: str
    params: dict[str, Any]
    observation_ids: list[str]
    mesh: dict[str, Any]  # mesh fingerprint: type / n_cells / n_active / core / padding
    iterations: int
    final_phi_d: float | None
    final_phi_m: float | None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_params_json(self) -> dict[str, Any]:
        """The ``provenance.params_json`` payload (doc 02 §7, doc 10 §7)."""
        return {
            "engineId": self.engine_id,
            "engineLibrary": self.engine_library,
            "engineKind": self.engine_kind,
            "processVersion": self.process_version,
            "params": dict(self.params),
            "observationIds": list(self.observation_ids),
            "mesh": dict(self.mesh),
            "iterations": int(self.iterations),
            "finalPhiD": self.final_phi_d,
            "finalPhiM": self.final_phi_m,
            "metrics": dict(self.metrics),
        }


# ──────────────────────────────── the contract (doc 10 §2) ────────────────────────────────


@runtime_checkable
class InversionEngine(Protocol):
    """The doc-10 inversion engine plugin contract (doc 10 §2, doc 08 §4f).

    An engine carries a declarative :attr:`spec` and implements :meth:`run`. ``run``
    builds whatever solver containers it needs (SimPEG ``Survey``/``Data``, PyGIMLi
    managers) **internally** from the engine-agnostic :class:`InversionContext`, runs
    forward+inverse, and returns an :class:`InversionResult` with a MANDATORY uncertainty
    field (doc 10 §2.3, §8). It must poll ``ctx.is_cancelled()`` between iterations for
    cooperative cancellation (doc 10 §3).
    """

    spec: InversionEngineSpec

    def run(self, ctx: InversionContext) -> InversionResult:
        """Run forward + inverse for ``ctx`` → recovered core model + uncertainty."""
        ...


def register_inversion_engine(engine: InversionEngine) -> InversionEngine:
    """Register an inversion engine on the process-wide plugin registry (doc 08 §4f).

    A thin adapter over :meth:`geosim.plugins.PluginRegistry.register_inversion_engine`:
    the registry keys engines by a ``key`` attribute, so we surface ``spec.id`` as
    ``key`` and ``spec.methods`` as ``methods`` (the registry's conformance check) while
    keeping the richer doc-10 :class:`InversionEngineSpec` on ``spec``. Returns the engine
    unchanged so it can decorate a class/instance in place (doc 08 §3.1).
    """
    from geosim.plugins import get_registry

    # The registry's InversionEngine conformance check (doc 08 §8) wants ``key`` +
    # ``invert``; expose them as adapters onto our doc-10 surface without leaking solvers.
    if not hasattr(engine, "key"):
        engine.key = engine.spec.id  # type: ignore[attr-defined]
    if not hasattr(engine, "methods"):
        engine.methods = list(engine.spec.methods)  # type: ignore[attr-defined]
    if not hasattr(engine, "invert"):
        engine.invert = engine.run  # type: ignore[attr-defined]
    return get_registry().register_inversion_engine(engine)
