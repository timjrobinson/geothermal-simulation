"""Calibration workflow — well/core/geochem probes promote a proxy field to a measurement.

Calibration is the **centre** of the rock-physics workflow (doc 07 §4.8, critique #10/#14):
a transform is born ``uncalibrated`` and its output is a *likelihood/proxy* field; ground-
truth probes sampled ALONG a well path are what promote it to ``well_calibrated`` with a
``quantitative`` uncertainty tier — but only WHERE the wells constrain it (spatially honest).

The loop (doc 07 §4.8):

```
① INGEST ground truth      well logs / core / geochem sampled ALONG the well path (§3.1),
                           MD → Engineering XYZ via the deviation survey (doc 01/09), NOT
                           voxelized.
        ▼
② ESTIMATE site params     fit the transform's params (porosity, m_cementation, salinity…)
                           to the (measured ↔ predicted) pairs at the probe locations →
                           a site-specific PARAMETER DISTRIBUTION (mean + σ), not a point
                           fit (``scipy.optimize.least_squares``; σ from the fit covariance).
        ▼
③ RE-RUN transforms        push the calibrated param distribution through ``run_transform``
                           over the full fused grid (param σ now feeds §5.2 propagation —
                           often the dominant term once calibrated).
        ▼
④ PROMOTE / LABEL          calibrationStatus → well_calibrated; UncertaintySpec.tier
                           proxy → quantitative WHERE wells constrain it; cells beyond the
                           wells' resolving distance STAY proxy / "likelihood".
```

- **Parameter distributions, not point fits** (doc 07 §4.8). The fit emits a ``Param`` σ per
  calibrated param; :func:`calibrate_transform` writes those σ onto the transform's params so
  the re-run's delta-method propagation (:mod:`geosim.fusion.transform` §5.2) carries the
  residual spread of the fit.
- **Promotion is spatially honest** (doc 07 §4.8). A single well calibrates its neighbourhood,
  not the whole basin. :func:`promote_spatial` keeps cells beyond
  ``resolving_distance`` of every probe at ``tier="proxy"`` / their "likelihood" labelling.
- **Synthetic vs real** (doc 07 §4.8). When a synthetic truth field exists (doc 05) calibration
  quality can be *scored* against the oracle (:func:`score_against_truth`). **Real projects have
  no truth field** — only the sparse probes — so the calibrated transform is the best estimate,
  never a checked-against-truth value. This module never presents synthetic-only truth scoring
  as available on real data: scoring is opt-in and explicitly flagged ``synthetic_only``.

Canonical temperature is **KELVIN** end-to-end (doc 01 §5); probe measurements are converted to
the transform's declared output unit before fitting (:mod:`geosim.spatial`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from sqlalchemy.orm import Session

from geosim.catalog import Dataset, FusedModel, PropertyModel, Provenance
from geosim.spatial import REGISTRY
from geosim.spatial.units import convert
from geosim.spatial.vertical import min_curvature_positions
from geosim.storage import ProjectLayout

from .analysis import sample_path
from .grid import FusedGrid, fused_grid_from_row
from .transform import (
    Param,
    Transform,
    TransformContext,
    TransformResult,
    run_transform,
)

__all__ = [
    "Probe",
    "CalibrationFit",
    "CalibrationResult",
    "probes_from_deviation_survey",
    "fit_transform_params",
    "promote_spatial",
    "score_against_truth",
    "calibrate_transform",
]


# ──────────────────────────────────────────────────────────────────────────
# ① ground-truth probes (doc 07 §4.8 ①, §3.1)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Probe:
    """A single ground-truth measurement of the transform's TARGET property (doc 07 §4.8 ①).

    ``z``/``y``/``x`` are Engineering metres (Z-up) — the location the well log / core /
    geochem sample sits at. ``measured`` is the measured value of the transform's *output*
    property; ``unit`` its unit (converted to the transform's output unit before fitting).
    ``md`` is the optional measured depth along the well (for reporting). ``sigma`` is the
    optional 1σ measurement error of the probe (weights the fit; default = equal weights).
    """

    z: float
    y: float
    x: float
    measured: float
    unit: str
    md: float | None = None
    sigma: float | None = None

    @property
    def zyx(self) -> tuple[float, float, float]:
        return (self.z, self.y, self.x)


def probes_from_deviation_survey(
    deviation_survey,
    wellhead,
    measured_md: np.ndarray,
    measured_values: np.ndarray,
    *,
    unit: str,
    kb_elev: float | None = None,
    measured_sigma: np.ndarray | None = None,
) -> list[Probe]:
    """Turn a well deviation survey + a measured log into Engineering-XYZ probes (doc 07 §4.8 ①).

    The deviation survey (``(MD, inclination°, azimuth°)`` stations) is integrated by
    minimum curvature (:func:`geosim.spatial.vertical.min_curvature_positions`, doc 09 §4.3)
    to Engineering XYZ per station; the measured log (``measured_md`` ↔ ``measured_values``)
    is positioned by **linear interpolation of the station XYZ vs MD** (doc 07 §3.1 — logs
    are sampled along the path, not voxelized). Returns one :class:`Probe` per logged sample.

    ``measured_values`` is in ``unit`` (converted to the transform's output unit at fit time);
    ``measured_sigma`` (optional, same length) is the per-sample 1σ measurement error.
    """
    result = min_curvature_positions(deviation_survey, wellhead, kb_elev=kb_elev)
    # Station Engineering XYZ ordered (East=x, North=y, Up=z); MD increases monotonically.
    station_md = np.asarray(result.md, dtype=float)
    enu = np.asarray(result.enu, dtype=float)  # (N, 3) = (x, y, z)
    md = np.asarray(measured_md, dtype=float).reshape(-1)
    vals = np.asarray(measured_values, dtype=float).reshape(-1)
    if md.shape != vals.shape:
        raise ValueError("measured_md and measured_values must be the same length")
    sig = (
        np.asarray(measured_sigma, dtype=float).reshape(-1)
        if measured_sigma is not None
        else None
    )
    if sig is not None and sig.shape != vals.shape:
        raise ValueError("measured_sigma must match measured_values length")

    # Interpolate each XYZ component against MD (np.interp clamps outside the survey range).
    px = np.interp(md, station_md, enu[:, 0])
    py = np.interp(md, station_md, enu[:, 1])
    pz = np.interp(md, station_md, enu[:, 2])

    probes: list[Probe] = []
    for i in range(md.size):
        probes.append(
            Probe(
                z=float(pz[i]), y=float(py[i]), x=float(px[i]),
                measured=float(vals[i]), unit=unit, md=float(md[i]),
                sigma=(float(sig[i]) if sig is not None else None),
            )
        )
    return probes


# ──────────────────────────────────────────────────────────────────────────
# ② fit site params → parameter distribution (doc 07 §4.8 ②)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationFit:
    """The outcome of fitting a transform's params to the probe pairs (doc 07 §4.8 ②).

    A **parameter distribution**, not a point fit (doc 07 §4.8): ``params`` is the fitted
    mean of each calibrated param, ``param_sigma`` its 1σ (from the fit covariance — the
    residual spread that flows into σ propagation §5.2). ``n_probes`` is the number of usable
    (on-grid, finite-prediction) probe pairs; ``rms_residual`` is the post-fit RMS misfit in
    the output unit; ``predicted``/``measured`` are the co-located arrays at the probes.
    """

    fitted_params: list[str]
    params: dict[str, float]
    param_sigma: dict[str, float]
    n_probes: int
    rms_residual: float
    measured: np.ndarray
    predicted: np.ndarray
    converged: bool


def _predict_at_probes(
    transform: Transform,
    grid: FusedGrid,
    input_features: dict[str, np.ndarray],
    params: dict[str, Any],
) -> np.ndarray:
    """Run the pure ``apply()`` at the probe locations for a given param set (doc 07 §4.8 ②).

    ``input_features`` maps each input property → its trilinearly-sampled value at the probes
    (the §3.1 along-path view). Returns the predicted output (transform's output unit).
    """
    ctx = TransformContext(grid=grid, params=params)
    return np.asarray(transform.apply(ctx, **input_features, **params), dtype=float)


def fit_transform_params(
    session: Session,
    fem: FusedModel,
    transform: Transform,
    probes: list[Probe],
    fit_params: list[str],
    *,
    params: dict[str, Any] | None = None,
    storage_root: str | Path | None = None,
) -> CalibrationFit:
    """Fit ``fit_params`` of ``transform`` to the (measured ↔ predicted) probe pairs (§4.8 ②).

    ① The transform's INPUT layers are sampled along the probe locations
    (:func:`geosim.fusion.sample_path` — trilinear, NaN off-grid); probes whose inputs (or
    measured value) are non-finite are dropped (listwise).

    ② ``scipy.optimize.least_squares`` minimises the (optionally σ-weighted) residual
    ``predicted(θ) − measured`` over ``fit_params`` (bounded by each :class:`Param`'s
    ``range``), starting from the declared defaults / ``params``.

    ③ The fit covariance is estimated from the Jacobian at the optimum (``Cov ≈ σ²·(JᵀJ)⁻¹``,
    ``σ²`` = residual variance) → a **per-param 1σ**. This σ is the residual spread of the fit,
    which §5.2 propagation then carries (doc 07 §4.8 "parameter distributions, not point fits").

    Returns a :class:`CalibrationFit`. Raises ``ValueError`` if no usable probe pairs remain or
    a requested fit-param is not a tunable param of the transform.
    """
    by_name = {p.name: p for p in transform.params}
    for name in fit_params:
        if name not in by_name:
            raise ValueError(
                f"param {name!r} is not a tunable param of transform {transform.id!r}"
            )
    if not fit_params:
        raise ValueError("fit_params must name at least one param to calibrate")

    grid = fused_grid_from_row(fem)
    resolved = transform.resolve_params(params)

    # ① Sample the transform's inputs along the probe path (doc 07 §3.1 along-path view).
    input_props = [spec.name for spec in transform.inputs]
    pts = np.array([p.zyx for p in probes], dtype=float)  # (m, 3) Engineering (z,y,x)
    sample = sample_path(session, fem, pts, input_props, storage_root=storage_root)

    # Convert each sampled input from its canonical unit → the transform's declared input unit.
    input_features: dict[str, np.ndarray] = {}
    for spec in transform.inputs:
        col = sample.features[:, sample.properties.index(spec.name)]
        src_unit = REGISTRY.get(spec.name).canonical_unit
        input_features[spec.name] = (
            col if src_unit == spec.unit
            else np.asarray(convert(col, src_unit, spec.unit), dtype=float)
        )

    out_unit = transform.output.unit
    measured = np.array(
        [convert(p.measured, p.unit, out_unit) if p.unit != out_unit else p.measured
         for p in probes],
        dtype=float,
    )
    meas_sigma = np.array(
        [p.sigma if p.sigma is not None else np.nan for p in probes], dtype=float
    )

    # Listwise-drop probes off-grid (NaN input) or with a non-finite measurement.
    finite = np.isfinite(measured)
    for col in input_features.values():
        finite &= np.isfinite(col)
    keep = np.flatnonzero(finite)
    if keep.size == 0:
        raise ValueError(
            "no usable probe pairs: every probe is off the fused grid or has a "
            "non-finite input / measurement"
        )
    measured = measured[keep]
    feats = {name: col[keep] for name, col in input_features.items()}
    weights = meas_sigma[keep]
    use_weights = np.all(np.isfinite(weights)) and np.all(weights > 0)

    # ② Bounded least-squares over fit_params (other params held at `resolved`).
    x0 = np.array([float(resolved[name]) for name in fit_params], dtype=float)
    lo = np.array(
        [by_name[n].range[0] if by_name[n].range else -np.inf for n in fit_params]
    )
    hi = np.array(
        [by_name[n].range[1] if by_name[n].range else np.inf for n in fit_params]
    )
    # Nudge a start that sits on a bound just inside so least_squares has room to move.
    x0 = np.clip(x0, lo + 1e-9 * np.where(np.isfinite(lo), 1.0, 0.0),
                 hi - 1e-9 * np.where(np.isfinite(hi), 1.0, 0.0))

    def _residual(theta: np.ndarray) -> np.ndarray:
        params = dict(resolved)
        for name, val in zip(fit_params, theta, strict=True):
            params[name] = float(val)
        pred = _predict_at_probes(transform, grid, feats, params)
        r = pred - measured
        if use_weights:
            r = r / weights
        return r

    sol = least_squares(_residual, x0, bounds=(lo, hi), method="trf")
    theta_hat = sol.x

    fitted = dict(resolved)
    for name, val in zip(fit_params, theta_hat, strict=True):
        fitted[name] = float(val)
    pred_final = _predict_at_probes(transform, grid, feats, fitted)
    resid = pred_final - measured
    n = measured.size
    dof = max(n - len(fit_params), 1)
    rms = float(np.sqrt(np.mean(resid**2))) if n else 0.0

    # ③ Param σ from the Gauss-Newton covariance Cov ≈ s²·(JᵀJ)⁻¹ (doc 07 §4.8).
    param_sigma = _fit_param_sigma(sol.jac, resid, dof, fit_params, use_weights, weights)

    return CalibrationFit(
        fitted_params=list(fit_params),
        params={n: float(fitted[n]) for n in fit_params},
        param_sigma=param_sigma,
        n_probes=int(n),
        rms_residual=rms,
        measured=measured,
        predicted=pred_final,
        converged=bool(sol.success),
    )


def _fit_param_sigma(
    jac: np.ndarray,
    resid: np.ndarray,
    dof: int,
    fit_params: list[str],
    use_weights: bool,
    weights: np.ndarray,
) -> dict[str, float]:
    """Per-param 1σ from the least-squares Jacobian (``Cov ≈ s²·(JᵀJ)⁻¹``, doc 07 §4.8).

    ``jac`` is the residual Jacobian at the optimum (already in the weighted residual space
    when ``use_weights`` — then ``s² = 1`` since weights are the σ; otherwise
    ``s² = SSR/dof``). A singular ``JᵀJ`` (unidentifiable param) yields ``inf`` σ — the
    param is reported as un-pinned-down rather than spuriously tight.
    """
    jtj = jac.T @ jac
    try:
        cov_unit = np.linalg.inv(jtj)
    except np.linalg.LinAlgError:
        cov_unit = np.full((len(fit_params), len(fit_params)), np.inf)
    if use_weights:
        s2 = 1.0  # residuals already divided by σ → unit-variance
    else:
        s2 = float(resid @ resid) / dof if dof > 0 else 0.0
    cov = cov_unit * s2
    diag = np.diag(cov)
    out: dict[str, float] = {}
    for i, name in enumerate(fit_params):
        v = diag[i]
        out[name] = float(np.sqrt(v)) if np.isfinite(v) and v >= 0 else float("inf")
    return out


# ──────────────────────────────────────────────────────────────────────────
# ④ spatially-honest promotion (doc 07 §4.8 ④)
# ──────────────────────────────────────────────────────────────────────────


def _distance_to_nearest_probe(grid: FusedGrid, probes: list[Probe]) -> np.ndarray:
    """Per-cell Euclidean distance (Engineering m) to the nearest probe (doc 07 §4.8 ④).

    Returns a ``(nz, ny, nx)`` array. With no probes every cell is ``+inf`` (nothing
    constrains it → nothing is promoted).
    """
    z, y, x = grid.axis_coords()
    gz, gy, gx = np.meshgrid(z, y, x, indexing="ij")
    if not probes:
        return np.full(grid.shape, np.inf, dtype=float)
    dist = np.full(grid.shape, np.inf, dtype=float)
    for p in probes:
        d = np.sqrt((gz - p.z) ** 2 + (gy - p.y) ** 2 + (gx - p.x) ** 2)
        dist = np.minimum(dist, d)
    return dist


def promote_spatial(
    grid: FusedGrid,
    probes: list[Probe],
    *,
    resolving_distance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the spatially-honest promotion mask (doc 07 §4.8 ④).

    A cell within ``resolving_distance`` (Engineering m) of any probe is **well-calibrated**
    (``tier="quantitative"``); beyond that distance it STAYS ``proxy`` / its "likelihood"
    labelling — a single well calibrates its neighbourhood, not the whole basin (doc 07 §4.8).

    Returns ``(promoted_mask, stats)`` where ``promoted_mask`` is a ``(nz,ny,nx)`` boolean
    (True = promoted to quantitative) and ``stats`` reports the promoted fraction + the
    resolving distance used (the §4.6 assumption-burden indicator consumes this).
    """
    if resolving_distance <= 0:
        raise ValueError("resolving_distance must be positive (Engineering metres)")
    dist = _distance_to_nearest_probe(grid, probes)
    promoted = dist <= float(resolving_distance)
    n = int(np.prod(grid.shape))
    stats = {
        "resolving_distance_m": float(resolving_distance),
        "n_cells": n,
        "n_promoted": int(promoted.sum()),
        "promoted_fraction": float(promoted.sum()) / n if n else 0.0,
        "n_probes": len(probes),
    }
    return promoted, stats


# ──────────────────────────────────────────────────────────────────────────
# synthetic-only truth scoring (doc 07 §4.8 — NEVER presented as available on real data)
# ──────────────────────────────────────────────────────────────────────────


def score_against_truth(
    calibrated_value: np.ndarray,
    truth_value: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score a calibrated volume against a synthetic truth field (doc 07 §4.8 — synthetic ONLY).

    Returns RMSE / bias / correlation over the co-located finite cells (optionally restricted
    to ``mask``). The result is tagged ``synthetic_only=True`` so a caller can never surface it
    as a real-data quality metric — **real projects have no truth field** (doc 07 §4.8); only
    the synthetic earth (doc 05) carries truth volumes.
    """
    cal = np.asarray(calibrated_value, dtype=float)
    tru = np.asarray(truth_value, dtype=float)
    if cal.shape != tru.shape:
        raise ValueError("calibrated and truth volumes must share shape")
    finite = np.isfinite(cal) & np.isfinite(tru)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not finite.any():
        return {"synthetic_only": True, "n": 0, "rmse": float("nan"),
                "bias": float("nan"), "correlation": float("nan")}
    a = cal[finite]
    b = tru[finite]
    err = a - b
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 \
        else float("nan")
    return {
        "synthetic_only": True,
        "n": int(finite.sum()),
        "rmse": rmse,
        "bias": bias,
        "correlation": corr,
    }


# ──────────────────────────────────────────────────────────────────────────
# the full loop (doc 07 §4.8)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationResult:
    """The result of a full :func:`calibrate_transform` run (doc 07 §4.8).

    ``fit`` is the parameter distribution (②); ``transform_result`` is the re-run derived
    volume (③, an ordinary derived :class:`~geosim.fusion.TransformResult`); ``promotion``
    are the spatial-honesty stats (④). ``calibration_status`` is the PROMOTED status
    (``well_calibrated``) and ``promoted_fraction`` is the share of cells promoted to
    quantitative — the rest stay ``proxy`` / "likelihood". ``truth_score`` is present only
    when a synthetic truth was supplied (and is flagged ``synthetic_only``).
    """

    transform_id: str
    transform_version: str
    output_property: str
    calibration_status: str
    fit: CalibrationFit
    transform_result: TransformResult
    promotion: dict[str, Any]
    promoted_fraction: float
    far_cell_tier: str
    near_cell_tier: str
    truth_score: dict[str, float] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "output_property": self.output_property,
            "calibration_status": self.calibration_status,
            "fit": {
                "fitted_params": self.fit.fitted_params,
                "params": self.fit.params,
                "param_sigma": self.fit.param_sigma,
                "n_probes": self.fit.n_probes,
                "rms_residual": self.fit.rms_residual,
                "converged": self.fit.converged,
            },
            "transform_result": self.transform_result.to_payload(),
            "promotion": self.promotion,
            "promoted_fraction": self.promoted_fraction,
            "far_cell_tier": self.far_cell_tier,
            "near_cell_tier": self.near_cell_tier,
            "truth_score": self.truth_score,
        }


def calibrate_transform(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    transform: Transform,
    probes: list[Probe],
    fit_params: list[str],
    *,
    resolving_distance: float,
    inputs: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    truth_value: np.ndarray | None = None,
    created_by: str = "system:calibration",
    storage_root: str | Path | None = None,
    progress=None,
) -> CalibrationResult:
    """Run the full calibration loop → a promoted derived volume + diagnostics (doc 07 §4.8).

    ① probes are already ingested along the well path (see
    :func:`probes_from_deviation_survey`); ② :func:`fit_transform_params` fits ``fit_params``
    to the probe pairs → a param distribution (mean + σ); ③ the transform is re-run over the
    full fused grid with the calibrated params **and** their fitted σ written onto the params
    (so §5.2 propagation carries the calibration spread) and ``calibration_status`` promoted to
    ``well_calibrated``; ④ :func:`promote_spatial` marks the cells within
    ``resolving_distance`` of a probe as ``quantitative`` while the rest stay ``proxy`` /
    "likelihood".

    When ``truth_value`` (a synthetic doc-05 truth volume on the fused grid) is supplied the
    calibrated volume is scored against it (:func:`score_against_truth`, flagged
    ``synthetic_only``). **Real projects pass no truth** — the calibrated transform is then the
    best estimate, never a checked-against-truth value (doc 07 §4.8).
    """
    if progress is not None:
        progress.report(0.05, "fitting transform params to well probes")

    # The transform's inputs must be on the fused grid before we can sample them along the
    # well path; resample any named native models in first (idempotent via the §2.1 cache).
    if inputs:
        from .resample import resample_to_fused

        for _prop, pm_id in inputs.items():
            resample_to_fused(session, fem, pm_id, storage_root=storage_root)
        session.refresh(fem)

    # ② Fit → parameter distribution (mean + σ).
    fit = fit_transform_params(
        session, fem, transform, probes, fit_params,
        params=params, storage_root=storage_root,
    )

    # Promote a copy of the transform: write fitted means + σ onto its params, set status
    # well_calibrated (so the harness tier cap lifts to quantitative). We mutate a clone so the
    # registered transform spec is untouched (doc 07 §4.4 reproducibility).
    calibrated = _clone_with_calibrated_params(transform, fit)

    if progress is not None:
        progress.report(0.4, "re-running transform with calibrated params")

    # ③ Re-run over the full grid; param σ now feeds §5.2 propagation (often dominant, §4.8).
    rerun_params = dict(params or {})
    rerun_params.update(fit.params)
    transform_result = run_transform(
        session, layout, fem, calibrated,
        inputs=inputs, params=rerun_params, uncertainty="delta",
        created_by=created_by, storage_root=storage_root,
    )

    if progress is not None:
        progress.report(0.8, "promoting spatially (well resolving distance)")

    # ④ Spatially-honest promotion.
    grid = fused_grid_from_row(fem)
    _promoted, promo_stats = promote_spatial(
        grid, probes, resolving_distance=resolving_distance
    )

    # Optional synthetic-only scoring (NEVER surfaced as real-data quality, doc 07 §4.8).
    truth_score = None
    if truth_value is not None:
        from geosim.storage import open_property_model

        pm = session.get(PropertyModel, transform_result.model_id)
        reader = open_property_model(pm.store_uri)
        cal_value = reader.read_level(transform_result.output_property, 0)
        truth_score = score_against_truth(cal_value, truth_value)

    _stamp_calibration_provenance(
        session, transform_result, fit, promo_stats,
    )

    if progress is not None:
        progress.report(1.0, "done")

    return CalibrationResult(
        transform_id=transform.id,
        transform_version=transform.version,
        output_property=transform_result.output_property,
        calibration_status="well_calibrated",
        fit=fit,
        transform_result=transform_result,
        promotion=promo_stats,
        promoted_fraction=promo_stats["promoted_fraction"],
        far_cell_tier="proxy",            # beyond resolving distance → stays proxy/likelihood
        near_cell_tier=transform_result.tier,  # within → the re-run's (quantitative) tier
        truth_score=truth_score,
    )


def _clone_with_calibrated_params(transform: Transform, fit: CalibrationFit) -> Transform:
    """A shallow clone of ``transform`` with fitted means + σ on params + promoted status.

    The fitted σ (finite ones) is written onto the corresponding :class:`Param` so the re-run's
    delta-method propagation carries the calibration spread (doc 07 §5.2/§4.8 — "often
    dominant"). The registered transform is left untouched (doc 07 §4.4).
    """
    clone = type(transform).__new__(type(transform))
    clone.__dict__.update(transform.__dict__)
    new_params: list[Param] = []
    for p in transform.params:
        if p.name in fit.param_sigma and np.isfinite(fit.param_sigma[p.name]):
            new_params.append(
                replace(p, default=fit.params.get(p.name, p.default),
                        sigma=float(fit.param_sigma[p.name]))
            )
        else:
            new_params.append(p)
    clone.params = new_params
    clone.calibration_status = "well_calibrated"
    return clone


def _stamp_calibration_provenance(
    session: Session,
    transform_result: TransformResult,
    fit: CalibrationFit,
    promo_stats: dict[str, Any],
) -> None:
    """Record the calibration in the derived volume's provenance block (doc 07 §4.3/§4.8).

    Augments the §4.3 derivation block of the re-run derived PropertyModel with the fitted
    parameter distribution, probe count, RMS residual, and spatial-promotion stats so the
    calibrated volume is fully reproducible and its honesty (which cells are promoted) is
    auditable (doc 07 §4.4/§4.8).
    """
    pm = session.get(PropertyModel, transform_result.model_id)
    ds = session.get(Dataset, pm.dataset_id)
    prov = session.get(Provenance, ds.provenance_id)
    payload = json.loads(prov.params_json)
    deriv = payload.get("derivation", {})
    deriv["calibrationStatus"] = "well_calibrated"
    deriv["calibration"] = {
        "fittedParams": fit.fitted_params,
        "params": fit.params,
        "paramSigma": fit.param_sigma,
        "nProbes": fit.n_probes,
        "rmsResidual": fit.rms_residual,
        "converged": fit.converged,
        "promotion": promo_stats,
    }
    payload["derivation"] = deriv
    prov.params_json = json.dumps(payload)
    session.add(prov)
    session.commit()
