"""Predicted log, geothermal outputs, and risk (doc 09 §5–§7).

Given a resolved trajectory, **densify the CURVED path** (arc-faithful, doc 09 §5.1),
sample every relevant fused volume along it WITH uncertainty, and assemble a
:class:`PredictedLog` — per-station ``(md, tvd, z, x, y)`` + value/σ/confidence per
property + a per-station composite risk. From that log fall the geothermal outputs
(doc 09 §6: BHT ± σ, max-temp & MD, target/reservoir intersection length, productive
fracture intersections, in-window fraction) and the **transparent weighted risk** composite
(doc 09 §7.4) — always returned with its driver breakdown.

REUSE, never reinvent:
- the curved-path vertices come from :func:`geosim.planning.trajectory.densify_survey` +
  :func:`geosim.spatial.min_curvature_positions` (THE shared integrator);
- the property samples (value + σ) come from the fused layers fusion already wrote, read
  by :func:`geosim.planning._sampling.sample_layers_with_sigma` — temperature/favorability
  are NEVER re-derived here (doc 09 decision 6).

Risk (doc 09 §7.4) is a glass box: ``risk = w_T·(1−tempConfidence) + w_H·hazard +
w_D·dlsExceedance + w_U·structuralUncertainty``. Fault proximity is split into the four
channels of §7.2 — productivity raises favorability (NOT risk), drilling-hazard is its own
term, induced seismicity is deferred to a ``RiskPlugin``, structural uncertainty is its own
σ term — never collapsed into one "risk up near faults" scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from geosim.catalog import FusedModel
from geosim.spatial import REGISTRY, to_display

from ._sampling import sample_layers_with_sigma
from .trajectory import densify_survey, well_positions

__all__ = [
    "RiskWeights",
    "DEFAULT_RISK_WEIGHTS",
    "PredictedStation",
    "GeothermalSummary",
    "PredictedLog",
    "predict_log",
    "sigma_to_confidence",
]

# Properties the predicted log samples when present (doc 09 §5.1). Missing layers are simply
# skipped — the log only carries what the fused model actually has.
DEFAULT_LOG_PROPERTIES = [
    "temperature",
    "favorability",
    "resistivity",
    "lithology_class",
    "fracture_density",
    "water_saturation",
]
# Hazard-likelihood properties (doc 09 §7.3): a dedicated LCZ/overpressure volume if present,
# else proxied. We treat any of these (whatever the fused model carries) as hazard channels.
HAZARD_PROPERTIES = ["lost_circulation", "overpressure", "instability"]

FAULT_INFLUENCE_RADIUS_M = 250.0  # doc 09 §7.2 default fault-proximity influence radius


@dataclass
class RiskWeights:
    """Transparent, user-tunable risk weights (doc 09 §7.4; default = drilling-feasibility)."""

    temp_confidence: float = 0.40  # w_T
    hazard: float = 0.30  # w_H
    dls_exceedance: float = 0.10  # w_D
    structural_uncertainty: float = 0.20  # w_U

    def normalized(self) -> RiskWeights:
        s = self.temp_confidence + self.hazard + self.dls_exceedance + self.structural_uncertainty
        if s <= 0:
            return RiskWeights()
        return RiskWeights(
            self.temp_confidence / s, self.hazard / s,
            self.dls_exceedance / s, self.structural_uncertainty / s,
        )

    def to_payload(self) -> dict:
        return {
            "tempConfidence": self.temp_confidence, "hazard": self.hazard,
            "dlsExceedance": self.dls_exceedance,
            "structuralUncertainty": self.structural_uncertainty,
        }


DEFAULT_RISK_WEIGHTS = RiskWeights()


@dataclass
class PredictedStation:
    """One station of the predicted log (doc 09 §5.2)."""

    md: float
    tvd: float
    z: float
    x: float
    y: float
    values: dict[str, dict]  # property → {value, sigma, confidence}
    lithology: str | None
    hazards: dict[str, float]
    dist_to_nearest_fault_m: float | None
    risk: float
    risk_drivers: dict[str, float]

    def to_payload(self) -> dict:
        return {
            "md": self.md, "tvd": self.tvd, "z": self.z, "x": self.x, "y": self.y,
            "values": self.values, "lithology": self.lithology, "hazards": self.hazards,
            "distToNearestFault_m": self.dist_to_nearest_fault_m,
            "risk": self.risk, "riskDrivers": self.risk_drivers,
        }


@dataclass
class GeothermalSummary:
    """Geothermal outputs derived from the predicted log (doc 09 §6)."""

    bht_c: float | None
    bht_sigma_c: float | None
    bht_confidence: float | None
    max_temp_c: float | None
    max_temp_md_m: float | None
    max_temp_tvd_m: float | None
    target_intersection_length_m: float
    reservoir_intersection_length_m: float
    productive_fracture_intersections: int
    fracture_intersection_mds_m: list[float]
    in_window_fraction: float
    mean_risk: float
    peak_risk: float

    def to_payload(self) -> dict:
        return {
            "bhtC": self.bht_c, "bhtSigmaC": self.bht_sigma_c,
            "bhtConfidence": self.bht_confidence,
            "maxTempC": self.max_temp_c, "maxTempMD_m": self.max_temp_md_m,
            "maxTempTVD_m": self.max_temp_tvd_m,
            "targetIntersectionLength_m": self.target_intersection_length_m,
            "reservoirIntersectionLength_m": self.reservoir_intersection_length_m,
            "productiveFractureIntersections": self.productive_fracture_intersections,
            "fractureIntersectionMDs_m": self.fracture_intersection_mds_m,
            "inWindowFraction": self.in_window_fraction,
            "meanRisk": self.mean_risk, "peakRisk": self.peak_risk,
        }


@dataclass
class PredictedLog:
    """The full predicted log: stations + geothermal summary (doc 09 §5.2/§6)."""

    well_id: str
    model_version: str
    md_step_m: float
    stations: list[PredictedStation]
    summary: GeothermalSummary
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    hardness: np.ndarray | None = None  # per-station lithology-hardness (for drillability §4.6)
    hardness_md: np.ndarray | None = None

    def to_payload(self) -> dict:
        return {
            "wellId": self.well_id, "modelVersion": self.model_version,
            "mdStep_m": self.md_step_m,
            "stations": [s.to_payload() for s in self.stations],
            "summary": self.summary.to_payload(),
            "riskWeights": self.risk_weights.normalized().to_payload(),
        }


# ──────────────────────────────────────────────────────────────────────────
# σ → confidence (transparent relative-uncertainty mapping)
# ──────────────────────────────────────────────────────────────────────────


def sigma_to_confidence(prop: str, value: float | None, sigma: float | None) -> float | None:
    """Map a value+σ to a [0,1] confidence (doc 09 §7.1 surfacing doc-07 σ).

    A transparent relative-uncertainty mapping: ``confidence = 1 − clip(σ / scale, 0, 1)``,
    where ``scale`` is the property's display range span (registry) so the confidence is
    dimensionless and comparable across properties — large σ relative to the property's
    natural range reads as low confidence, σ≈0 reads as ~1. Returns ``None`` when σ is
    unavailable (a genuine unknown, not a silent 1.0).
    """
    if sigma is None or not np.isfinite(sigma):
        return None
    scale = _property_scale(prop, value)
    if scale <= 0:
        return None
    return float(np.clip(1.0 - abs(sigma) / scale, 0.0, 1.0))


def _property_scale(prop: str, value: float | None) -> float:
    """A natural σ scale for ``prop``: the registry display-range span, else |value|."""
    try:
        pt = REGISTRY.get(prop)
        rng = pt.display_range
        if rng is not None and rng[1] > rng[0]:
            return float(rng[1] - rng[0])
    except KeyError:
        pass
    if value is not None and abs(value) > 1e-9:
        return abs(value)
    return 1.0


# ──────────────────────────────────────────────────────────────────────────
# predicted log
# ──────────────────────────────────────────────────────────────────────────


def predict_log(
    session: Session,
    fem: FusedModel,
    well,  # geosim.planning.trajectory.PlannedWell
    *,
    md_step_m: float = 5.0,
    target=None,  # geosim.planning.targets.DrillTarget | None
    risk_weights: RiskWeights = DEFAULT_RISK_WEIGHTS,
    fault_points_xyz: np.ndarray | None = None,
    favorability_threshold: float = 0.7,
    fracture_threshold: float = 0.5,
    storage_root: str | Path | None = None,
) -> PredictedLog:
    """Predict the log along the CURVED trajectory + the geothermal outputs + risk (doc 09 §5–§7).

    Densifies the survey to ``md_step_m`` ON the min-curvature arc, integrates positions via
    the shared integrator, batch-samples the fused layers (value + σ) at the curved vertices,
    then assembles per-station value/σ/confidence, hazards, fault proximity, and the
    transparent weighted risk. The summary (doc 09 §6) reduces the stations into BHT, pay
    length, fracture intersections and in-window fraction.
    """
    dense = densify_survey(well.deviation_survey, md_step_m)
    pos = well_positions(dense, well.wellhead, well.kb_elev_m)
    md, tvd, enu, dls = pos.md, pos.tvd, pos.enu, pos.dls
    n = md.shape[0]
    pts_zyx = np.column_stack([enu[:, 2], enu[:, 1], enu[:, 0]])  # (z, y, x)

    props = [p for p in DEFAULT_LOG_PROPERTIES if any(lay.property == p for lay in fem.layers)]
    haz_props = [p for p in HAZARD_PROPERTIES if any(lay.property == p for lay in fem.layers)]
    sampled = sample_layers_with_sigma(
        session, fem, pts_zyx, properties=props + haz_props, storage_root=storage_root
    )

    # Per-interval DLS exceedance, normalized to [0,1] over the ceiling (risk term w_D).
    ceiling = max(well.constraints.max_dls_deg30m, 1e-6)
    dls_exceed = np.clip((dls - ceiling) / ceiling, 0.0, 1.0)

    # Fault proximity (doc 09 §7.2): geometric distance to the nearest fault vertex.
    fault_dist = _fault_distance(enu, fault_points_xyz)

    weights = risk_weights.normalized()
    stations: list[PredictedStation] = []
    hardness = np.full(n, np.nan)

    temp_c = np.full(n, np.nan)
    temp_conf = np.full(n, np.nan)
    fav = np.full(n, np.nan)

    for i in range(n):
        values: dict[str, dict] = {}
        lithology = None
        for prop in props:
            if prop not in sampled:
                continue
            vals, sig = sampled[prop]
            v = float(vals[i]) if np.isfinite(vals[i]) else None
            s = float(sig[i]) if (sig is not None and np.isfinite(sig[i])) else None
            conf = sigma_to_confidence(prop, v, s)
            if prop == "temperature":
                tc = None if v is None else float(to_display(v, "temperature"))
                values["temperatureC"] = {"value": tc, "sigma": s, "confidence": conf}
                if tc is not None:
                    temp_c[i] = tc
                if conf is not None:
                    temp_conf[i] = conf
            elif prop == "lithology_class":
                lithology = None if v is None else str(int(round(v)))
                values[prop] = {"value": v, "sigma": s, "confidence": conf}
            else:
                values[prop] = {"value": v, "sigma": s, "confidence": conf}
                if prop == "favorability" and v is not None:
                    fav[i] = v

        # Lithology-hardness proxy for the drillability flag (doc 09 §4.6 / §5).
        if "lithology_class" in sampled and np.isfinite(sampled["lithology_class"][0][i]):
            from .drillability import lithology_hardness
            hardness[i] = lithology_hardness(np.array([sampled["lithology_class"][0][i]]))[0]

        # Hazards (doc 09 §7.3): sampled hazard volumes, else a fracture∩favorability proxy.
        hazards = _station_hazards(sampled, haz_props, i, fav[i], fault_dist[i])

        # Structural-uncertainty term (doc 09 §7.2d / §7.4 w_U): mean normalized σ of the key
        # properties + the fault interpretation-uncertainty contribution.
        struct_unc = _structural_uncertainty(values, fault_dist[i])

        # Composite risk (doc 09 §7.4) — transparent weighted blend, drivers always returned.
        t_conf = temp_conf[i] if np.isfinite(temp_conf[i]) else 0.5
        hazard_lvl = max(hazards.values()) if hazards else 0.0
        drivers = {
            "tempConfidence": float(weights.temp_confidence * (1.0 - t_conf)),
            "hazard": float(weights.hazard * hazard_lvl),
            "dlsExceedance": float(weights.dls_exceedance * float(dls_exceed[i])),
            "structuralUncertainty": float(weights.structural_uncertainty * struct_unc),
        }
        risk = float(np.clip(sum(drivers.values()), 0.0, 1.0))

        stations.append(PredictedStation(
            md=float(md[i]), tvd=float(tvd[i]), z=float(enu[i, 2]),
            x=float(enu[i, 0]), y=float(enu[i, 1]),
            values=values, lithology=lithology, hazards=hazards,
            dist_to_nearest_fault_m=(None if fault_dist[i] is None else float(fault_dist[i])),
            risk=risk, risk_drivers=drivers,
        ))

    summary = _geothermal_summary(
        md, tvd, temp_c, temp_conf, sampled, target, well,
        favorability_threshold, fracture_threshold,
        np.array([s.risk for s in stations]),
    )

    return PredictedLog(
        well_id=well.id, model_version=fem.id, md_step_m=md_step_m,
        stations=stations, summary=summary, risk_weights=risk_weights,
        hardness=hardness, hardness_md=md,
    )


def _fault_distance(
    enu: np.ndarray, fault_points_xyz: np.ndarray | None
) -> np.ndarray | list[None]:
    """Per-station distance to the nearest fault vertex (doc 09 §7.2), or all-None if absent."""
    if fault_points_xyz is None or len(fault_points_xyz) == 0:
        return [None] * enu.shape[0]
    fp = np.asarray(fault_points_xyz, dtype=float).reshape(-1, 3)
    d = np.linalg.norm(enu[:, None, :] - fp[None, :, :], axis=2)
    return d.min(axis=1)


def _station_hazards(
    sampled: dict, haz_props: list[str], i: int, fav_i: float, fault_dist_i
) -> dict[str, float]:
    """Hazard likelihoods at a station (doc 09 §7.3): sampled volumes + a flagged proxy.

    Dedicated hazard volumes are sampled directly. Where none exist, an LCZ proxy = fracture
    density ∩ fault proximity (doc 09 §7.2b/§7.3) is added under ``lostCirculation_proxy`` so
    the channel is never silently zero, while being clearly flagged as a proxy.
    """
    hazards: dict[str, float] = {}
    for prop in haz_props:
        vals, _ = sampled[prop]
        if np.isfinite(vals[i]):
            hazards[prop] = float(np.clip(vals[i], 0.0, 1.0))
    if not haz_props:
        frac = sampled.get("fracture_density")
        f = float(frac[0][i]) if (frac is not None and np.isfinite(frac[0][i])) else 0.0
        prox = 0.0
        if fault_dist_i is not None:
            prox = float(np.clip(1.0 - fault_dist_i / FAULT_INFLUENCE_RADIUS_M, 0.0, 1.0))
        proxy = float(np.clip(max(f, 0.0) * 0.5 + prox * 0.5, 0.0, 1.0)) if (f or prox) else 0.0
        if proxy > 0:
            hazards["lostCirculation_proxy"] = proxy
    return hazards


def _structural_uncertainty(values: dict[str, dict], fault_dist_i) -> float:
    """Structural-uncertainty term (doc 09 §7.2d / §7.4 w_U).

    Mean of the per-property confidence-complements (``1 − confidence``) over the sampled
    properties that carry σ, plus a fault interpretation-uncertainty bump near a fault (the
    fault position is itself uncertain near the path — channel (d), NOT a hazard).
    """
    comps = []
    for v in values.values():
        c = v.get("confidence")
        if c is not None:
            comps.append(1.0 - float(c))
    base = float(np.mean(comps)) if comps else 0.5
    if fault_dist_i is not None:
        prox = float(np.clip(1.0 - fault_dist_i / FAULT_INFLUENCE_RADIUS_M, 0.0, 1.0))
        base = float(np.clip(base + 0.3 * prox, 0.0, 1.0))
    return base


def _contiguous_length(md: np.ndarray, mask: np.ndarray) -> float:
    """Total MD length covered by the True entries of ``mask`` (trapezoidal over stations)."""
    if md.size < 2:
        return 0.0
    seg = np.diff(md)
    both = mask[:-1] & mask[1:]
    return float(np.sum(seg[both]))


def _geothermal_summary(
    md, tvd, temp_c, temp_conf, sampled, target, well,
    favorability_threshold, fracture_threshold, risk,
) -> GeothermalSummary:
    """Reduce the predicted log into the geothermal outputs (doc 09 §6)."""
    finite_t = np.isfinite(temp_c)
    bht_c = float(temp_c[-1]) if finite_t.size and finite_t[-1] else None
    bht_sigma = None
    bht_conf = float(temp_conf[-1]) if (temp_conf.size and np.isfinite(temp_conf[-1])) else None
    if "temperature" in sampled and sampled["temperature"][1] is not None:
        sig = sampled["temperature"][1]
        if np.isfinite(sig[-1]):
            bht_sigma = float(sig[-1])

    max_temp_c = max_md = max_tvd = None
    if np.any(finite_t):
        j = int(np.nanargmax(np.where(finite_t, temp_c, -np.inf)))
        max_temp_c = float(temp_c[j])
        max_md = float(md[j])
        max_tvd = float(tvd[j])

    # Favorability pay length (doc 09 §6 "target intersection length" proxy).
    fav = sampled.get("favorability")
    fav_mask = (
        np.isfinite(fav[0]) & (fav[0] >= favorability_threshold)
        if fav is not None else np.zeros(md.shape, dtype=bool)
    )
    target_len = _contiguous_length(md, fav_mask)

    # Reservoir intersection: inside the target tolerance solid (point target proxy) when a
    # target is supplied, else the favorability pay.
    reservoir_len = target_len
    if target is not None:
        pos = well_positions(
            densify_survey(well.deviation_survey, md[1] - md[0] if md.size > 1 else 5.0),
            well.wellhead, well.kb_elev_m,
        )
        loc = np.asarray(target.location, dtype=float)
        d = np.linalg.norm(pos.enu - loc, axis=1)
        in_zone = d <= max(target.tolerance.radius_m, 1e-6)
        reservoir_len = _contiguous_length(pos.md, in_zone)

    # Productive fracture intersections (doc 09 §6): fractureDensity over threshold.
    frac = sampled.get("fracture_density")
    frac_mds: list[float] = []
    if frac is not None:
        over = np.isfinite(frac[0]) & (frac[0] >= fracture_threshold)
        # Count rising-edge crossings as discrete intersections.
        edges = np.flatnonzero((~over[:-1]) & over[1:]) + 1
        if over.size and over[0]:
            edges = np.r_[0, edges]
        frac_mds = [float(md[e]) for e in edges]
    n_frac = len(frac_mds)

    # In-window fraction (doc 09 §6): fraction of pay length within minTemperatureC AND fav.
    in_window = 0.0
    if md.size > 1:
        min_t = target.min_temperature_c if (target and target.min_temperature_c) else None
        t_ok = np.isfinite(temp_c) & (temp_c >= min_t) if min_t is not None else np.isfinite(temp_c)
        win_mask = t_ok & fav_mask if fav is not None else t_ok
        total = _contiguous_length(md, np.ones(md.shape, dtype=bool))
        win_len = _contiguous_length(md, win_mask)
        in_window = float(win_len / total) if total > 0 else 0.0

    return GeothermalSummary(
        bht_c=bht_c, bht_sigma_c=bht_sigma, bht_confidence=bht_conf,
        max_temp_c=max_temp_c, max_temp_md_m=max_md, max_temp_tvd_m=max_tvd,
        target_intersection_length_m=target_len,
        reservoir_intersection_length_m=reservoir_len,
        productive_fracture_intersections=n_frac,
        fracture_intersection_mds_m=frac_mds,
        in_window_fraction=in_window,
        mean_risk=float(np.mean(risk)) if risk.size else 0.0,
        peak_risk=float(np.max(risk)) if risk.size else 0.0,
    )
