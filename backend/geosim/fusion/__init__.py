"""Fusion core — the FusedEarthModel grid + resampling engine (doc 07 §1–§2, doc 02 §11).

This is the PROCESSING layer that puts every native property on a **shared support** so
cells are co-located and comparable, **without destroying native originals** (doc 07 §0).

- :mod:`.grid` — the :class:`FusedGrid` regular-voxel CONTAINER + auto-resolution
  (doc 07 §1.1: median native spacing, clamped, capped at ``256³``) + the
  ``provenance``/``datasets``/``fused_models`` catalog rows (doc 02 §11). A project may
  hold several fused grids (overview + zoomed target zone).
- :mod:`.resample` — :func:`resample_to_fused`: per-support interpolation (trilinear /
  block-mean / barycentric / spline-then-trilinear), interpolating in **log space** for
  log10-flagged properties (registry), **footprint-honest** (NaN outside the native
  footprint/DOI; emits a coverage mask), with σ propagated through the **same**
  interpolator + interpolation-variance inflation (doc 07 §5.2). Resampled arrays live in
  the fused Zarr group; layers are cached by ``(pmId, version, fusedGridId, method,
  params)``.

Derived/fused outputs remain ordinary catalog artifacts stored via
:mod:`geosim.storage` & :mod:`geosim.catalog` (doc 02 §11, doc 07 §4.3).
"""

from .analysis import (
    BIG_N_SCATTER,
    SYNC_CELL_LIMIT,
    ClusterResult,
    FusedSample,
    cluster_fused,
    correlation_matrix,
    crossplot,
    histogram,
    sample_fused,
    sample_path,
    selection_to_mask,
)
from .calibration import (
    CalibrationFit,
    CalibrationResult,
    Probe,
    calibrate_transform,
    fit_transform_params,
    probes_from_deviation_survey,
    promote_spatial,
    score_against_truth,
)
from .favorability import (
    Evidence,
    FavorabilityResult,
    FavorabilitySpec,
    TransferFn,
    compute_favorability,
    membership,
)
from .grid import (
    DEFAULT_CELL_CAP,
    FusedGrid,
    auto_resolution,
    build_fused_model,
    fused_grid_from_row,
    open_fused_group,
)
from .resample import ResampledLayerRef, resample_to_fused, resolve_method
from .transform import (
    CALIBRATION_STATUSES,
    TIER_ORDER,
    InputSpec,
    OutputSpec,
    Param,
    Transform,
    TransformContext,
    TransformResult,
    run_transform,
)

__all__ = [
    # grid (doc 07 §1)
    "FusedGrid",
    "DEFAULT_CELL_CAP",
    "auto_resolution",
    "build_fused_model",
    "fused_grid_from_row",
    "open_fused_group",
    # resample (doc 07 §2)
    "ResampledLayerRef",
    "resample_to_fused",
    "resolve_method",
    # analysis: cross-plot / stats / clustering (doc 07 §3)
    "SYNC_CELL_LIMIT",
    "BIG_N_SCATTER",
    "FusedSample",
    "ClusterResult",
    "sample_fused",
    "sample_path",
    "crossplot",
    "histogram",
    "correlation_matrix",
    "cluster_fused",
    "selection_to_mask",
    # transform engine (doc 07 §4)
    "Transform",
    "InputSpec",
    "OutputSpec",
    "Param",
    "TransformContext",
    "TransformResult",
    "run_transform",
    "CALIBRATION_STATUSES",
    "TIER_ORDER",
    # favorability (doc 07 §4.6)
    "TransferFn",
    "Evidence",
    "FavorabilitySpec",
    "FavorabilityResult",
    "membership",
    "compute_favorability",
    # calibration workflow (doc 07 §4.8)
    "Probe",
    "CalibrationFit",
    "CalibrationResult",
    "probes_from_deviation_survey",
    "fit_transform_params",
    "promote_spatial",
    "score_against_truth",
    "calibrate_transform",
]
