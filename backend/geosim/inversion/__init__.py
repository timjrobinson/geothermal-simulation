"""Inversion engine framework (doc 10) — the engine-agnostic inverse-modelling phase.

Doc 10 §0 core idea: an **inversion engine** consumes Observations + a
:class:`~geosim.inversion.domain.ModelDomain` (a mesh over the Engineering Frame), runs
forward + inverse, and emits an *ordinary* :class:`~geosim.catalog.PropertyModel` + a
**mandatory** uncertainty field + full provenance — reusing ALL existing storage / fusion
/ serving. SimPEG ``Survey``/``Data`` and PyGIMLi containers are built **inside** an engine
and never cross the plugin boundary (doc 10 §8); this package contains NO solver code.

Layers:

- :mod:`.domain` — :class:`ModelDomain` + the :func:`build_tensor_domain` ``discretize``
  TensorMesh builder (core region + geometric padding + topography active cells, doc 10
  §4).
- :mod:`.engine` — the :class:`InversionEngine` Protocol, its declarative
  :class:`InversionEngineSpec`, the :class:`InversionContext` / :class:`InversionResult` /
  :class:`InversionProvenance` dataclasses (doc 10 §2, §7), and the
  :func:`register_inversion_engine` plugin hook (doc 08 §4f).
- :mod:`.harness` — the run harness: validate params against ``paramsSchema`` BEFORE
  running (doc 10 §3), thread a :class:`~geosim.jobs.ProgressReporter`, run the engine,
  write the recovered model + uncertainty + provenance, then resample the core onto a
  fused grid (doc 10 §4.4).
- :mod:`.mock` — a trivial in-framework :class:`MockLinearEngine` (a linear toy) to test
  the harness without a heavy solver.
"""

from .domain import (
    MESH_TYPES,
    CoreRegion,
    ModelDomain,
    PaddingSpec,
    build_tensor_domain,
)
from .engine import (
    InversionContext,
    InversionEngine,
    InversionEngineSpec,
    InversionProvenance,
    InversionResult,
    register_inversion_engine,
)
from .harness import (
    InversionRunResult,
    ParamValidationError,
    default_uncertainty,
    load_observations,
    run_inversion,
    validate_params,
)
from .mock import MOCK_SPEC, MockLinearEngine

__all__ = [
    # domain (doc 10 §4)
    "MESH_TYPES",
    "CoreRegion",
    "PaddingSpec",
    "ModelDomain",
    "build_tensor_domain",
    # engine contract (doc 10 §2, §7)
    "InversionEngineSpec",
    "InversionContext",
    "InversionResult",
    "InversionProvenance",
    "InversionEngine",
    "register_inversion_engine",
    # harness (doc 10 §3)
    "ParamValidationError",
    "validate_params",
    "default_uncertainty",
    "InversionRunResult",
    "load_observations",
    "run_inversion",
    # mock engine
    "MockLinearEngine",
    "MOCK_SPEC",
]
