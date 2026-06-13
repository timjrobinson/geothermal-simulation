"""Well-planning core (doc 09) — target → trajectory → predicted log → risk → outputs.

The top of the fusion ladder (doc 09 §1): an engineer picks a **DrillTarget** in the
fused earth model, a design solver emits a **PlannedWell** deviation survey to it, the
curved minimum-curvature trajectory is sampled across the doc-07 fused volumes into a
**PredictedLog** (values + σ + per-station risk), and the geothermal outputs + a
transparent weighted risk score fall out of that log.

Everything here REUSES, never reinvents:

- :func:`geosim.spatial.min_curvature_positions` — THE shared survey→position
  integrator (doc 09 §4.3); a planned well IS a deviation survey identical to an
  ingested well (doc 09 §4.1), so TVD / Engineering-XYZ are always derived from it.
- :func:`geosim.fusion.sample_fused` / the fused-grid trilinear sampler — target
  enrichment (points mode) and the predicted-log path sampling read the SAME resampled
  layers (value + σ) the rest of fusion produced. This doc never re-derives temperature
  or favorability.

Modules:

- :mod:`.trajectory` — :class:`PlannedWell`, the design solvers (vertical /
  build-hold-land / S-curve) honouring ``maxDLS`` / ``maxInc``, min-curvature positions,
  and arc-faithful densification of the curved path (doc 09 §4).
- :mod:`.targets` — :class:`DrillTarget` (point|zone) + points-mode fused enrichment
  stamped with ``modelVersion`` for stale detection (doc 09 §3).
- :mod:`.drillability` — the crude, advisory ``ok|warn`` drillability flag (doc 09 §4.6).
- :mod:`.predict` — the curved-path predicted log with σ/confidence, the geothermal
  outputs (BHT, pay length, fracture intersections, in-window fraction), and the
  transparent weighted risk composite with its driver breakdown (doc 09 §5–§7).
"""

from .drillability import (
    DEFAULT_DRILLABILITY_LIMITS,
    DrillabilityCheck,
    DrillabilityFlag,
    DrillabilityLimits,
    drillability_flag,
)
from .predict import (
    DEFAULT_RISK_WEIGHTS,
    GeothermalSummary,
    PredictedLog,
    PredictedStation,
    RiskWeights,
    predict_log,
)
from .targets import (
    DrillTarget,
    SampledValue,
    TargetEnrichment,
    TargetTolerance,
    enrich_target,
)
from .trajectory import (
    DesignSpec,
    PlannedWell,
    SolveResult,
    TrajectoryConstraints,
    densify_survey,
    solve_survey,
    well_positions,
)

__all__ = [
    # trajectory (doc 09 §4)
    "PlannedWell",
    "DesignSpec",
    "TrajectoryConstraints",
    "SolveResult",
    "solve_survey",
    "well_positions",
    "densify_survey",
    # targets (doc 09 §3)
    "DrillTarget",
    "TargetTolerance",
    "SampledValue",
    "TargetEnrichment",
    "enrich_target",
    # drillability (doc 09 §4.6)
    "DrillabilityFlag",
    "DrillabilityCheck",
    "DrillabilityLimits",
    "DEFAULT_DRILLABILITY_LIMITS",
    "drillability_flag",
    # predicted log + geothermal outputs + risk (doc 09 §5–§7)
    "PredictedLog",
    "PredictedStation",
    "GeothermalSummary",
    "RiskWeights",
    "DEFAULT_RISK_WEIGHTS",
    "predict_log",
]
