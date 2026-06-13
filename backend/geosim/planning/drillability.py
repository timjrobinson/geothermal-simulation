"""Crude drillability flag (doc 09 §4.6) — explicitly NOT engineering-grade.

A lightweight **ok / warn** gate (never "fail") that catches obviously-impractical
geometry early. It is deliberately a sanity check, not a mechanics model: **no
torque-and-drag, no hydraulics/ECD, no BHA/buckling** — those remain the later
``TrajectoryPlugin`` (doc 09 §4.5/§11). It combines five cheap, transparent checks
(doc 09 §4.6 table), each emitting ``ok``/``warn`` + the offending MD interval:

1. **DLS exceedance** — max per-interval DLS vs ``max_dls_deg30m`` (already computed by
   min-curvature; also feeds the risk score, doc 09 §7.4);
2. **build/turn rate** — inclination- and azimuth-change rate per 30 m vs configurable
   limits;
3. **MD/TVD ratio** — total MD ÷ TVD vs a ceiling (step-out / horizontal-reach proxy);
4. **max inclination** — peak inclination vs ``max_inc_deg``;
5. **lithology-hardness proxy** — a hardness scalar along the path exceeding a soft
   threshold over a sustained interval → a hard-rock / slow-ROP warning (doc 09 §4.6).

Thresholds are configurable (defaults shipped). A ``warn`` is advisory metadata on the
plan; it does NOT gate export (doc 09 §4.6, unlike the DLS-vs-constraint export gate §4.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geosim.spatial import MinCurvatureResult

__all__ = [
    "DrillabilityLimits",
    "DEFAULT_DRILLABILITY_LIMITS",
    "DrillabilityCheck",
    "DrillabilityFlag",
    "drillability_flag",
    "lithology_hardness",
]

# A crude lithology-class → relative-hardness [0,1] proxy (doc 09 §4.6). Higher = harder /
# slower ROP. Keyed by the categorical ``lithology_class`` integer label when known; an
# unknown/continuous value falls back to itself clamped to [0,1].
LITHOLOGY_HARDNESS: dict[int, float] = {
    0: 0.2,  # sediment / soft
    1: 0.45,  # sandstone / volcaniclastic
    2: 0.7,  # granodiorite
    3: 0.85,  # granite / crystalline basement
}


@dataclass
class DrillabilityLimits:
    """Configurable per-project drillability thresholds (doc 09 §4.6, defaults shipped)."""

    max_dls_deg30m: float = 5.0
    max_build_rate_deg30m: float = 4.0
    max_turn_rate_deg30m: float = 4.0
    max_md_tvd_ratio: float = 2.5
    max_inc_deg: float = 92.0
    hardness_threshold: float = 0.7
    hardness_sustained_m: float = 60.0  # sustained interval over the threshold to warn


DEFAULT_DRILLABILITY_LIMITS = DrillabilityLimits()


@dataclass
class DrillabilityCheck:
    """One transparent check result (doc 09 §4.6 ``checks[]``)."""

    name: str
    verdict: str  # "ok" | "warn"
    value: float
    limit: float
    md_interval_m: tuple[float, float] | None = None

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "verdict": self.verdict,
            "value": self.value,
            "limit": self.limit,
            "mdInterval_m": list(self.md_interval_m) if self.md_interval_m else None,
        }


@dataclass
class DrillabilityFlag:
    """The advisory drillability verdict + its checks (doc 09 §4.6)."""

    verdict: str  # "ok" | "warn" — never "fail"
    checks: list[DrillabilityCheck] = field(default_factory=list)

    def to_payload(self) -> dict:
        return {"verdict": self.verdict, "checks": [c.to_payload() for c in self.checks]}


def lithology_hardness(values: np.ndarray) -> np.ndarray:
    """Map a per-station ``lithology_class`` / hardness value to a [0,1] hardness scalar.

    Integer-valued class labels look up :data:`LITHOLOGY_HARDNESS` (a crude proxy); any
    other finite value is treated as an already-normalized hardness and clamped to [0,1].
    NaN propagates (no coverage → no hardness signal).
    """
    out = np.full(values.shape, np.nan, dtype=float)
    for i, v in enumerate(np.asarray(values, dtype=float)):
        if not np.isfinite(v):
            continue
        key = int(round(v))
        if abs(v - key) < 1e-6 and key in LITHOLOGY_HARDNESS:
            out[i] = LITHOLOGY_HARDNESS[key]
        else:
            out[i] = float(np.clip(v, 0.0, 1.0))
    return out


def _interval_for(md: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    """MD span [first, last] of the True entries of ``mask`` (None if none)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    return float(md[idx[0]]), float(md[idx[-1]])


def drillability_flag(
    survey: np.ndarray,
    positions: MinCurvatureResult,
    *,
    hardness: np.ndarray | None = None,
    hardness_md: np.ndarray | None = None,
    limits: DrillabilityLimits = DEFAULT_DRILLABILITY_LIMITS,
) -> DrillabilityFlag:
    """Run the five crude drillability checks (doc 09 §4.6).

    ``survey`` is ``(N,3)`` ``(MD, inc°, azi°)``; ``positions`` is its min-curvature result
    (DLS/TVD reused, not recomputed). ``hardness``/``hardness_md`` optionally supply a
    per-station lithology-hardness scalar sampled along the path (doc 09 §5 predicted log)
    so check 5 can run; absent, the hardness check is reported ``ok`` with value 0.
    """
    surv = np.asarray(survey, dtype=float).reshape(-1, 3)
    md = surv[:, 0]
    inc = surv[:, 1]
    azi = surv[:, 2]
    checks: list[DrillabilityCheck] = []

    # 1. DLS exceedance (per-interval DLS already computed by min-curvature).
    dls = positions.dls
    max_dls = float(np.max(dls)) if dls.size else 0.0
    dls_interval = _interval_for(md, dls > limits.max_dls_deg30m + 1e-9) if dls.size else None
    checks.append(DrillabilityCheck(
        "dls", "warn" if max_dls > limits.max_dls_deg30m + 1e-9 else "ok",
        round(max_dls, 4), limits.max_dls_deg30m, dls_interval,
    ))

    # 2. build/turn rate per 30 m (inclination- and azimuth-change rate).
    d_md = np.diff(md)
    safe = d_md > 1e-9
    build_rate = np.zeros_like(d_md)
    turn_rate = np.zeros_like(d_md)
    build_rate[safe] = np.abs(np.diff(inc)[safe]) / d_md[safe] * 30.0
    # Azimuth difference wrapped to [-180,180] (a turn through North is small, not 350°).
    d_azi = (np.diff(azi) + 180.0) % 360.0 - 180.0
    # Heading is ill-defined near vertical; weight the turn rate by sin(inc) to avoid spurious
    # warns where a near-vertical section's azimuth flips.
    turn_w = np.sin(np.radians(0.5 * (inc[:-1] + inc[1:])))
    turn_rate[safe] = np.abs(d_azi)[safe] * turn_w[safe] / d_md[safe] * 30.0

    max_build = float(np.max(build_rate)) if build_rate.size else 0.0
    b_interval = _interval_for(md[1:], build_rate > limits.max_build_rate_deg30m + 1e-9)
    checks.append(DrillabilityCheck(
        "buildRate", "warn" if max_build > limits.max_build_rate_deg30m + 1e-9 else "ok",
        round(max_build, 4), limits.max_build_rate_deg30m, b_interval,
    ))
    max_turn = float(np.max(turn_rate)) if turn_rate.size else 0.0
    t_interval = _interval_for(md[1:], turn_rate > limits.max_turn_rate_deg30m + 1e-9)
    checks.append(DrillabilityCheck(
        "turnRate", "warn" if max_turn > limits.max_turn_rate_deg30m + 1e-9 else "ok",
        round(max_turn, 4), limits.max_turn_rate_deg30m, t_interval,
    ))

    # 3. MD/TVD ratio (step-out / horizontal-reach proxy).
    total_md = float(md[-1] - md[0]) if md.size else 0.0
    tvd_end = float(positions.tvd[-1]) if positions.tvd.size else 0.0
    ratio = total_md / tvd_end if tvd_end > 1e-9 else 0.0
    checks.append(DrillabilityCheck(
        "mdTvdRatio", "warn" if ratio > limits.max_md_tvd_ratio + 1e-9 else "ok",
        round(ratio, 4), limits.max_md_tvd_ratio,
    ))

    # 4. max inclination.
    max_inc = float(np.max(inc)) if inc.size else 0.0
    checks.append(DrillabilityCheck(
        "maxInc", "warn" if max_inc > limits.max_inc_deg + 1e-9 else "ok",
        round(max_inc, 4), limits.max_inc_deg,
    ))

    # 5. lithology-hardness proxy: a sustained interval over the threshold → warn.
    h_value = 0.0
    h_verdict = "ok"
    h_interval = None
    if hardness is not None and hardness_md is not None and len(hardness):
        h = np.asarray(hardness, dtype=float)
        hmd = np.asarray(hardness_md, dtype=float)
        over = np.isfinite(h) & (h > limits.hardness_threshold)
        h_value = float(np.nanmax(h)) if np.any(np.isfinite(h)) else 0.0
        if np.any(over):
            span = _interval_for(hmd, over)
            sustained = span is not None and (span[1] - span[0]) >= limits.hardness_sustained_m
            if sustained:
                h_verdict = "warn"
                h_interval = span
    checks.append(DrillabilityCheck(
        "hardness", h_verdict, round(h_value, 4), limits.hardness_threshold, h_interval,
    ))

    verdict = "warn" if any(c.verdict == "warn" for c in checks) else "ok"
    return DrillabilityFlag(verdict=verdict, checks=checks)
