"""Well trajectory model + design solvers (doc 09 §4).

A planned well is the **same deviation survey** doc 01 §4 mandates for boreholes: ordered
``(MD, inc°, azi°)`` stations. TVD, Engineering XYZ, N/E offsets and DLS are **derived**
from the survey + wellhead via the shared minimum-curvature integrator
(:func:`geosim.spatial.min_curvature_positions`) — never stored as source of truth
(doc 09 decisions §12.1–§12.2). A plan promotes to "as-drilled" with no schema change.

Design solvers (doc 09 §4.4) turn *intent* into a survey:

- **vertical** — straight down to the target TVD;
- **build-hold-land** — vertical to KOP, a constant-rate build arc, then a straight
  tangent that lands at the target's XYZ within tolerance. Closed-form heading from the
  wellhead→target horizontal bearing; the build rate / KOP are solved (a small 1-D numeric
  tightening on build rate) so the landing honours the DLS ceiling and lands in the
  tolerance window;
- **S-curve** — build to a tangent then drop back toward vertical (thread between hazards
  / stacked targets), still DLS-checked.

Every solver output is validated against :class:`TrajectoryConstraints` (max DLS, max
inc). A DLS exceedance is reported per-interval (it feeds the in-viewer red-tube flag and
the risk score, doc 09 §7.4) but the survey is still returned — the caller decides whether
to block export (doc 09 §4.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from geosim.spatial import MinCurvatureResult, min_curvature_positions

__all__ = [
    "PlannedWell",
    "DesignSpec",
    "TrajectoryConstraints",
    "SolveResult",
    "solve_survey",
    "well_positions",
    "densify_survey",
]

# DLS is canonical in degrees / 30 m (doc 09 §4.1 locked convention).
DLS_COURSE_M = 30.0


@dataclass
class TrajectoryConstraints:
    """Hard geometry constraints a solved survey must honour (doc 09 §4.1)."""

    max_dls_deg30m: float = 5.0
    max_inc_deg: float = 92.0
    min_md_m: float = 0.0


@dataclass
class DesignSpec:
    """Designer intent a solver consumes to emit a survey (doc 09 §4.1 ``design``)."""

    method: str  # "vertical" | "build-hold-land" | "S-curve" | "manual"
    target: tuple[float, float, float] | None = None  # Engineering (x, y, z) bullseye
    kop_md_m: float = 0.0  # kick-off point (MD where the build starts)
    build_rate_deg30m: float = 3.0  # build-up rate (°/30 m)
    drop_rate_deg30m: float = 3.0  # S-curve drop-back rate (°/30 m)
    hold_inc_deg: float | None = None  # S-curve tangent inclination before the drop
    landing_inc_deg: float | None = None  # desired inclination at landing (build-hold-land)
    station_step_m: float = 30.0  # MD spacing of the emitted tangent/vertical stations


@dataclass
class PlannedWell:
    """A planned well — a deviation survey + wellhead datum (doc 09 §4.1).

    ``deviation_survey`` is the source of truth: ``(N,3)`` ``(MD, inc°, azi°)`` rows.
    ``wellhead`` is ``(x, y)`` Engineering metres; ``kb_elev_m`` is the MD datum (MD=0)
    elevation (Engineering Z, +up). TVD / XYZ / DLS are derived via min-curvature.
    """

    id: str
    name: str
    project_id: str
    wellhead: tuple[float, float]
    kb_elev_m: float
    deviation_survey: np.ndarray  # (N,3) (MD, inc°, azi°)
    target_ids: list[str] = field(default_factory=list)
    design: DesignSpec | None = None
    constraints: TrajectoryConstraints = field(default_factory=TrajectoryConstraints)
    status: str = "planned"
    pad_id: str | None = None
    model_version: str | None = None

    def positions(self) -> MinCurvatureResult:
        """Min-curvature Engineering XYZ / TVD / DLS per station (doc 09 §4.3)."""
        return well_positions(self.deviation_survey, self.wellhead, self.kb_elev_m)


@dataclass
class SolveResult:
    """A solved survey + its per-interval DLS report (doc 09 §4.4)."""

    survey: np.ndarray  # (N,3) (MD, inc°, azi°)
    positions: MinCurvatureResult
    max_dls_deg30m: float
    dls_exceeded: bool  # any per-interval DLS over the ceiling
    max_inc_deg: float
    inc_exceeded: bool
    landing_error_m: float | None  # 3-D miss from the bullseye (None for vertical/manual)
    method: str


# ──────────────────────────────────────────────────────────────────────────
# positions (shared integrator) + arc-faithful densification
# ──────────────────────────────────────────────────────────────────────────


def well_positions(
    deviation_survey, wellhead: tuple[float, float], kb_elev_m: float
) -> MinCurvatureResult:
    """Resolve a survey to positions via the SHARED integrator (doc 09 §4.3 reuse).

    Thin wrapper over :func:`geosim.spatial.min_curvature_positions` so the planner never
    owns a second copy of the min-curvature math (doc 09 flag to doc 01/02 owners).
    """
    return min_curvature_positions(
        np.asarray(deviation_survey, dtype=float), wellhead, kb_elev=kb_elev_m
    )


def _interp_station(
    surv: np.ndarray, md: float
) -> tuple[float, float, float]:
    """Inclination/azimuth at an arbitrary MD by SLERP along the min-curvature arc.

    Between two survey stations the unit tangent rotates on a great circle (the circular
    arc of minimum curvature). Spherical-linear-interpolating the tangent unit vectors and
    converting back to ``(inc, azi)`` reproduces the arc faithfully (doc 09 §4.3), so a
    densified station's ``(inc,azi)`` — and hence its min-curvature position — lies ON the
    curve, not on the chord.
    """
    mds = surv[:, 0]
    if md <= mds[0]:
        return float(surv[0, 0]), float(surv[0, 1]), float(surv[0, 2])
    if md >= mds[-1]:
        return float(md), float(surv[-1, 1]), float(surv[-1, 2])
    i = int(np.searchsorted(mds, md, side="right"))
    md1, inc1, azi1 = surv[i - 1]
    md2, inc2, azi2 = surv[i]
    if md2 <= md1:
        return float(md), float(inc1), float(azi1)
    f = (md - md1) / (md2 - md1)
    t1 = _to_tangent(inc1, azi1)
    t2 = _to_tangent(inc2, azi2)
    dot = float(np.clip(np.dot(t1, t2), -1.0, 1.0))
    omega = math.acos(dot)
    if omega < 1e-7:
        t = t1 + f * (t2 - t1)
    else:
        s = math.sin(omega)
        t = (math.sin((1.0 - f) * omega) / s) * t1 + (math.sin(f * omega) / s) * t2
    inc, azi = _from_tangent(t)
    return float(md), inc, azi


def _to_tangent(inc_deg: float, azi_deg: float) -> np.ndarray:
    """Unit tangent (East, North, Down) for an ``(inc, azi)`` station (doc 09 §4.3 frame)."""
    i = math.radians(inc_deg)
    a = math.radians(azi_deg)
    return np.array([math.sin(i) * math.sin(a), math.sin(i) * math.cos(a), math.cos(i)])


def _from_tangent(t: np.ndarray) -> tuple[float, float]:
    """``(inc°, azi°)`` from a (not-necessarily-unit) tangent vector (East, North, Down)."""
    n = float(np.linalg.norm(t))
    if n == 0.0:
        return 0.0, 0.0
    e, nth, d = t / n
    inc = math.degrees(math.acos(max(-1.0, min(1.0, d))))
    azi = math.degrees(math.atan2(e, nth)) % 360.0
    return inc, azi


def densify_survey(deviation_survey, md_step_m: float) -> np.ndarray:
    """Densify a survey to ``md_step_m`` spacing ALONG the curved arc (doc 09 §5.1).

    Returns ``(M,3)`` ``(MD, inc°, azi°)`` rows at every multiple of ``md_step_m`` (plus the
    original station MDs and the TD), each ``(inc,azi)`` SLERP-interpolated on the arc — so
    the subsequent min-curvature positions follow the real wellbore, not its chord. These
    are the vertices the predicted-log path sampler walks (doc 09 §6 decision 6).
    """
    surv = np.asarray(deviation_survey, dtype=float).reshape(-1, 3)
    if surv.shape[0] < 2 or md_step_m <= 0:
        return surv.copy()
    md_min, md_max = float(surv[0, 0]), float(surv[-1, 0])
    grid = np.arange(md_min, md_max, md_step_m)
    mds = np.unique(np.concatenate([grid, surv[:, 0], [md_max]]))
    rows = [_interp_station(surv, float(m)) for m in mds]
    return np.asarray(rows, dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# DLS / constraint reporting
# ──────────────────────────────────────────────────────────────────────────


def _solve_result(
    survey: np.ndarray,
    wellhead: tuple[float, float],
    kb_elev_m: float,
    constraints: TrajectoryConstraints,
    method: str,
    target: tuple[float, float, float] | None,
) -> SolveResult:
    pos = well_positions(survey, wellhead, kb_elev_m)
    max_dls = float(np.max(pos.dls)) if pos.dls.size else 0.0
    max_inc = float(np.max(survey[:, 1])) if survey.size else 0.0
    landing_err = None
    if target is not None:
        td = pos.enu[-1]
        landing_err = float(np.linalg.norm(td - np.asarray(target, dtype=float)))
    return SolveResult(
        survey=survey,
        positions=pos,
        max_dls_deg30m=max_dls,
        dls_exceeded=max_dls > constraints.max_dls_deg30m + 1e-9,
        max_inc_deg=max_inc,
        inc_exceeded=max_inc > constraints.max_inc_deg + 1e-9,
        landing_error_m=landing_err,
        method=method,
    )


# ──────────────────────────────────────────────────────────────────────────
# design solvers (doc 09 §4.4)
# ──────────────────────────────────────────────────────────────────────────


def solve_survey(
    design: DesignSpec,
    wellhead: tuple[float, float],
    kb_elev_m: float,
    constraints: TrajectoryConstraints,
) -> SolveResult:
    """Emit a deviation survey from designer intent + constraints (doc 09 §4.4).

    Dispatches on ``design.method``. Every survey is integrated and DLS-/inc-checked before
    return so the caller sees the constraint report (doc 09 §4.4 validation).
    """
    method = design.method
    if method == "vertical":
        survey = _solve_vertical(design, wellhead, kb_elev_m)
        target = design.target
    elif method == "build-hold-land":
        survey = _solve_build_hold_land(design, wellhead, kb_elev_m, constraints)
        target = design.target
    elif method == "S-curve":
        survey = _solve_s_curve(design, wellhead, kb_elev_m, constraints)
        target = design.target
    else:
        raise ValueError(f"unknown design method {method!r}")
    return _solve_result(survey, wellhead, kb_elev_m, constraints, method, target)


def _target_tvd(design: DesignSpec, kb_elev_m: float) -> float:
    if design.target is None:
        raise ValueError(f"method {design.method!r} requires a target")
    return kb_elev_m - float(design.target[2])  # TVD below KB (Z is +up; target z below KB)


def _solve_vertical(
    design: DesignSpec, wellhead: tuple[float, float], kb_elev_m: float
) -> np.ndarray:
    """Straight down to the target TVD (doc 09 §4.4 vertical)."""
    tvd = _target_tvd(design, kb_elev_m)
    step = max(design.station_step_m, 1.0)
    mds = np.unique(np.concatenate([np.arange(0.0, tvd, step), [tvd]]))
    return np.column_stack([mds, np.zeros_like(mds), np.zeros_like(mds)])


def _horizontal_bearing(
    wellhead: tuple[float, float], target: tuple[float, float, float]
) -> float:
    """Azimuth (deg, from +North toward +East) of the wellhead→target horizontal vector."""
    de = float(target[0]) - wellhead[0]  # East
    dn = float(target[1]) - wellhead[1]  # North
    return math.degrees(math.atan2(de, dn)) % 360.0


def _build_hold_survey(
    kop_md: float,
    build_rate_deg30m: float,
    landing_inc: float,
    azi: float,
    final_md: float,
    step: float,
) -> np.ndarray:
    """Vertical→KOP, constant-rate build to ``landing_inc``, tangent hold to ``final_md``."""
    build_inc_per_m = build_rate_deg30m / DLS_COURSE_M
    build_len = landing_inc / build_inc_per_m if build_inc_per_m > 0 else 0.0
    eob_md = kop_md + build_len  # end-of-build MD

    rows: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    if kop_md > 0:
        rows.append((kop_md, 0.0, azi))
    # Build arc.
    b = kop_md + step
    while b < eob_md - 1e-9:
        rows.append((b, min(landing_inc, (b - kop_md) * build_inc_per_m), azi))
        b += step
    rows.append((eob_md, landing_inc, azi))
    # Tangent hold.
    h = eob_md + step
    while h < final_md - 1e-9:
        rows.append((h, landing_inc, azi))
        h += step
    if final_md > eob_md + 1e-9:
        rows.append((final_md, landing_inc, azi))
    return np.asarray(rows, dtype=float)


def _build_hold_geometry(
    kop: float, build_rate_deg30m: float, theta_deg: float, horiz: float, vert: float
) -> tuple[float, float]:
    """Closed-form build-hold landing for a planar build to inclination ``theta`` (doc 09 §4.4).

    Build arc radius ``R = (180/π)·(30/build_rate)``. Building 0→θ gains vertical ``R·sinθ``
    and horizontal ``R·(1−cosθ)``; a straight tangent of length ``T`` at θ gains ``T·cosθ``
    vertical and ``T·sinθ`` horizontal. Returns ``(horiz_miss, T)`` where the well lands at
    the target vertical depth ``vert`` (below KB): ``T`` is chosen to hit ``vert`` exactly,
    and ``horiz_miss`` is the residual horizontal error at that landing.
    """
    theta = math.radians(theta_deg)
    radius = (180.0 / math.pi) * (DLS_COURSE_M / build_rate_deg30m)
    v_build = kop + radius * math.sin(theta)
    h_build = radius * (1.0 - math.cos(theta))
    cos_t = math.cos(theta)
    if cos_t <= 1e-9:
        # Horizontal landing: tangent adds no vertical; reachable only if the build already
        # passed the target depth. Horizontal miss is whatever the tangent cannot fix.
        tangent = max(0.0, (horiz - h_build) / max(math.sin(theta), 1e-9))
        return abs(v_build - vert), tangent
    tangent = (vert - v_build) / cos_t
    if tangent < 0:
        return math.inf, 0.0  # build overshoots the target depth — θ too large for this KOP
    h_total = h_build + tangent * math.sin(theta)
    return abs(h_total - horiz), tangent


def _solve_build_hold_land(
    design: DesignSpec,
    wellhead: tuple[float, float],
    kb_elev_m: float,
    constraints: TrajectoryConstraints,
) -> np.ndarray:
    """Build-hold-land: land at the target XYZ within tolerance (doc 09 §4.4).

    The horizontal heading is the closed-form wellhead→target bearing (planar problem). With
    the build rate clamped to the DLS ceiling, this is a **2-parameter solve** (KOP, landing
    inclination θ) of the planar build-hold geometry: for each KOP we pick θ so the straight
    tangent off the build arc lands at the target's horizontal reach AND vertical depth, then
    keep the (KOP, θ) with the smallest landing miss (doc 09 §4.4 "2-parameter numeric solve
    when DLS-constrained"). Emits the resolved survey at ``station_step_m`` spacing.
    """
    if design.target is None:
        raise ValueError("build-hold-land requires a target")
    target = np.asarray(design.target, dtype=float)
    azi = _horizontal_bearing(wellhead, design.target)
    step = max(design.station_step_m, 1.0)

    horiz = math.hypot(target[0] - wellhead[0], target[1] - wellhead[1])
    vert = kb_elev_m - target[2]  # +down depth to land at

    build_rate = min(design.build_rate_deg30m, constraints.max_dls_deg30m)
    build_rate = max(build_rate, 0.1)
    max_inc = constraints.max_inc_deg

    # 2-parameter scan: KOP × landing inclination θ, minimizing the closed-form landing miss.
    kop0 = design.kop_md_m if design.kop_md_m > 0 else max(vert * 0.35, step)
    best = None  # (miss, kop, theta, tangent)
    kop_lo = max(step, kop0 * 0.3)
    kop_hi = min(vert * 0.95, kop0 * 1.8 + step)
    theta_hi = (
        design.landing_inc_deg + 1e-6 if design.landing_inc_deg is not None else max_inc
    )
    theta_lo = (
        design.landing_inc_deg if design.landing_inc_deg is not None else 1.0
    )
    for kop in np.linspace(kop_lo, max(kop_hi, kop_lo + step), 30):
        for theta in np.linspace(theta_lo, min(theta_hi, max_inc), 60):
            miss, tangent = _build_hold_geometry(float(kop), build_rate, float(theta), horiz, vert)
            if not math.isfinite(miss):
                continue
            if best is None or miss < best[0]:
                best = (miss, float(kop), float(theta), tangent)
    if best is None:
        # Fall back to a straight vertical to the target depth.
        return _solve_vertical(design, wellhead, kb_elev_m)

    _miss, kop, theta, tangent = best
    final_md = kop + (theta / (build_rate / DLS_COURSE_M)) + max(tangent, 0.0)
    return _build_hold_survey(kop, build_rate, theta, azi, final_md, step)


def _solve_s_curve(
    design: DesignSpec,
    wellhead: tuple[float, float],
    kb_elev_m: float,
    constraints: TrajectoryConstraints,
) -> np.ndarray:
    """S-curve: build to a hold inclination then drop back toward vertical (doc 09 §4.2).

    A symmetric build/drop about a tangent section — used to thread between hazards or hit
    stacked targets while landing near-vertical (doc 09 §4.2). DLS-checked by the caller.
    """
    target = design.target
    azi = _horizontal_bearing(wellhead, target) if target is not None else 0.0
    step = max(design.station_step_m, 1.0)
    hold_inc = design.hold_inc_deg if design.hold_inc_deg is not None else 30.0
    hold_inc = float(min(hold_inc, constraints.max_inc_deg))
    build_rate = min(design.build_rate_deg30m, constraints.max_dls_deg30m)
    drop_rate = min(design.drop_rate_deg30m, constraints.max_dls_deg30m)
    kop = design.kop_md_m if design.kop_md_m > 0 else max(step, 100.0)

    build_per_m = build_rate / DLS_COURSE_M
    drop_per_m = drop_rate / DLS_COURSE_M
    build_len = hold_inc / build_per_m if build_per_m > 0 else 0.0
    drop_len = hold_inc / drop_per_m if drop_per_m > 0 else 0.0
    hold_len = max(step, build_len)  # a short tangent between build and drop

    rows: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0), (kop, 0.0, azi)]
    eob = kop + build_len
    m = kop + step
    while m < eob - 1e-9:
        rows.append((m, (m - kop) * build_per_m, azi))
        m += step
    rows.append((eob, hold_inc, azi))
    sod = eob + hold_len  # start-of-drop
    rows.append((sod, hold_inc, azi))
    eod = sod + drop_len  # end-of-drop (back to vertical)
    m = sod + step
    while m < eod - 1e-9:
        rows.append((m, max(0.0, hold_inc - (m - sod) * drop_per_m), azi))
        m += step
    rows.append((eod, 0.0, azi))
    # A short vertical tail.
    rows.append((eod + max(step, 100.0), 0.0, azi))
    return np.asarray(rows, dtype=float)
