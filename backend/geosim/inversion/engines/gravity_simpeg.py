"""SimPEG linear gravity inversion engine (doc 10 §8, §9) — the plumbing-proof engine.

The doc 10 §9 "first engine = plumbing proof": a workstation-feasible, *linear* gravity
inversion that wires the whole doc-10 phase together end-to-end. It consumes the platform
gravity :class:`~geosim.catalog.Observation` stations + a
:class:`~geosim.inversion.domain.ModelDomain` (a ``discretize`` ``TensorMesh`` over the
Engineering Frame) and recovers a **density** ``PropertyModel`` on the active core cells,
with a MANDATORY tier-B sensitivity/DOI uncertainty + convergence diagnostics (doc 10
§2.3, §7).

Design (doc 10 §8 "SimPEG gravity → TensorMesh, gravity.Survey → density"):

1. Build a SimPEG ``gravity.Survey`` from the observation station coordinates + the
   measured Bouguer anomaly (``gravity_anomaly``, mGal) — **inside** ``run`` so no SimPEG
   type ever crosses the plugin boundary (doc 10 §8).
2. Build a ``Simulation3DIntegral`` over the domain's ``TensorMesh`` restricted to the
   active cells (air above topography is excluded, doc 10 §4.3); the model unknown is the
   density anomaly ``Δρ`` (kg/m³) via an ``IdentityMap``.
3. Run a linear **L2 Tikhonov** inversion (``WeightedLeastSquares`` smallness+smoothness)
   with the SimPEG directive stack: ``UpdateSensitivityWeights`` (depth weighting — gravity
   has no intrinsic depth resolution), ``BetaEstimate_ByEig`` + ``BetaSchedule`` (β cooling)
   and a ``TargetMisfit`` (χ ≈ n_data target). The recovered density anomaly is added to a
   background to yield an absolute density model on the core cells.
4. Emit the recovered core (Z-up ``(z, y, x)``) + a tier-B sensitivity/DOI σ field +
   ``phi_d``/``phi_m``/iteration diagnostics (doc 10 §3, §7).

``executionMode`` declares ``worker_process`` (doc 08 §2.1): SimPEG sensitivity matrices
are heavy enough to belong off the request thread.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from geosim.jobs import Cancelled

from ..engine import (
    InversionContext,
    InversionEngineSpec,
    InversionResult,
    register_inversion_engine,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, no SimPEG import at module load
    from ..domain import ModelDomain

__all__ = ["SimpegGravityInversion", "GRAVITY_SIMPEG_SPEC"]


# ───────────────────────────── declarative spec (doc 10 §2) ─────────────────────────────

GRAVITY_SIMPEG_SPEC = InversionEngineSpec(
    id="simpeg.gravity",
    kind="gravity",
    library="simpeg",
    methods=["gravity"],  # canonical MethodKey (doc 02 §2 / geosim.plugins.methods)
    output_property="density",  # recovers absolute density (kg/m³) on the core (doc 01 §5)
    mesh_types=("tensor",),
    coupling="standalone",
    compute="worker_process",  # SimPEG sensitivities are heavy — off the request thread
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Reference / background density the recovered anomaly Δρ is added to (kg/m³).
            "background_density": {
                "type": "number", "minimum": 0.0, "maximum": 10000.0, "default": 2670.0,
            },
            # Linear-inversion controls (kept SMALL/loose — doc 10 §9 workstation-feasible).
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
            "beta0_ratio": {"type": "number", "minimum": 1e-3, "default": 1.0},
            "cooling_factor": {"type": "number", "minimum": 1.0, "default": 2.0},
            "chi_target": {"type": "number", "minimum": 0.1, "default": 1.0},
            # Tikhonov α weights (smallness vs. smoothness, doc 10 §8).
            "alpha_s": {"type": "number", "minimum": 0.0, "default": 1e-4},
            "alpha_xyz": {"type": "number", "minimum": 0.0, "default": 1.0},
            # Density-anomaly bounds for the projected solver (kg/m³).
            "rho_min": {"type": "number", "default": -1000.0},
            "rho_max": {"type": "number", "default": 1000.0},
            # Relative noise floor used to build the data standard deviation when stations
            # carry no σ (fraction of the max |anomaly|), doc 10 §3.
            "rel_noise": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.02},
        },
    },
)


# ──────────────────────────────── the engine (doc 10 §8) ────────────────────────────────


class SimpegGravityInversion:
    """SimPEG linear-L2 Tikhonov gravity inversion → density (doc 10 §8, §9).

    Implements the :class:`~geosim.inversion.engine.InversionEngine` Protocol: a declarative
    :attr:`spec` + :meth:`run`. All SimPEG containers are constructed inside :meth:`run`
    (doc 10 §8); only NumPy + the :class:`~geosim.inversion.domain.ModelDomain` cross the
    boundary.
    """

    spec = GRAVITY_SIMPEG_SPEC

    def run(self, ctx: InversionContext) -> InversionResult:
        """Build the SimPEG gravity survey+sim, run the L2 inversion, return density.

        Heavy SimPEG modules are imported here (not at module load) so importing the engine
        package stays cheap and the solver types never leak across the boundary (doc 10 §8).
        """
        from simpeg import (
            data,
            data_misfit,
            directives,
            inverse_problem,
            inversion,
            maps,
            optimization,
            regularization,
        )
        from simpeg.potential_fields import gravity

        params = ctx.params
        domain: ModelDomain = ctx.domain
        background = float(params.get("background_density", 2670.0))

        ctx.progress(0.02, "building gravity survey")

        # 1) survey: station locations (x, y, z) + measured Bouguer anomaly (mGal). The
        #    platform Observation coords are Engineering (z, y, x) Z-up — SimPEG wants
        #    (easting=x, northing=y, up=z), so we re-order (doc 10 §8, doc 02 §10.2).
        rx_locs, dobs, dstd = self._collect_stations(ctx, default_rel=float(params["rel_noise"]))
        if rx_locs.shape[0] == 0:
            raise ValueError(
                "no gravity stations found in observations — the engine needs "
                "'gravity_anomaly' point values to invert (doc 10 §8)"
            )
        receivers = gravity.receivers.Point(rx_locs, components="gz")
        source_field = gravity.sources.SourceField(receiver_list=[receivers])
        survey = gravity.survey.Survey(source_field)

        # 2) simulation on the active core+pad mesh; the model unknown is Δρ on active cells.
        mesh = domain.mesh
        active = np.asarray(domain.active_cells, dtype=bool)
        n_active = int(active.sum())
        rho_map = maps.IdentityMap(nP=n_active)
        simulation = gravity.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            rhoMap=rho_map,
            active_cells=active,
            store_sensitivities="ram",  # TINY mesh in tests → in-RAM G is fine (doc 10 §9)
        )

        ctx.progress(0.1, "assembling data misfit + Tikhonov regularization")

        # 3) data misfit + L2 Tikhonov regularization (smallness + smoothness, doc 10 §8).
        survey_data = data.Data(survey, dobs=dobs, standard_deviation=dstd)
        dmis = data_misfit.L2DataMisfit(data=survey_data, simulation=simulation)
        reg = regularization.WeightedLeastSquares(
            mesh,
            active_cells=active,
            mapping=rho_map,
            alpha_s=float(params["alpha_s"]),
            alpha_x=float(params["alpha_xyz"]),
            alpha_y=float(params["alpha_xyz"]),
            alpha_z=float(params["alpha_xyz"]),
        )

        # 4) projected Gauss-Newton with density-anomaly bounds (doc 10 §8).
        opt = optimization.ProjectedGNCG(
            maxIter=int(params["max_iterations"]),
            lower=float(params["rho_min"]),
            upper=float(params["rho_max"]),
            maxIterLS=10,
            cg_maxiter=12,
            cg_rtol=1e-3,
        )
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # 5) directive stack: sensitivity (depth) weighting + β cooling + χ target (doc 10 §8).
        progress_directive = _ProgressDirective(ctx)
        directive_list = [
            directives.UpdateSensitivityWeights(every_iteration=False),
            directives.BetaEstimate_ByEig(beta0_ratio=float(params["beta0_ratio"])),
            directives.BetaSchedule(coolingFactor=float(params["cooling_factor"]), coolingRate=1),
            directives.TargetMisfit(chifact=float(params["chi_target"])),
            progress_directive,
        ]
        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)

        ctx.progress(0.15, "running gravity L2 inversion")
        m0 = np.zeros(n_active)
        try:
            m_rec = inv.run(m0)
        except Cancelled:
            raise
        except _CancelInversion as exc:  # cooperative cancel surfaced from the directive
            raise Cancelled from exc

        # 6) Δρ on active cells → full mesh → CORE sub-brick (Z-up (z, y, x)), then add
        #    the background to recover an ABSOLUTE density model (doc 10 §4.4, §8).
        full = np.zeros(mesh.n_cells, dtype=float)
        full[active] = m_rec
        d_rho_core = domain.extract_core(full)  # (nz, ny, nx)
        density_core = (d_rho_core + background).astype(np.float32)

        # 7) tier-B uncertainty: a sensitivity/DOI proxy on the CORE. The diagonal of the
        #    sensitivity-weight cell weights tells us how well each cell is constrained;
        #    we invert it into a depth-inflated σ (doc 10 §2.3). Falls back to a relative
        #    DOI σ when the weights are unavailable.
        sigma_core = self._sensitivity_sigma(domain, reg, d_rho_core)

        phi_d = float(inv_prob.phi_d) if inv_prob.phi_d is not None else None
        phi_m = float(inv_prob.phi_m) if inv_prob.phi_m is not None else None
        iterations = int(getattr(opt, "iter", 0) or 0)

        ctx.progress(0.88, "gravity inversion converged", iteration=iterations,
                     phi_d=phi_d, phi_m=phi_m)

        return InversionResult(
            values=density_core,
            sigma=sigma_core,
            iterations=iterations,
            final_phi_d=phi_d,
            final_phi_m=phi_m,
            metrics={
                "engine": "simpeg.gravity",
                "library": "simpeg",
                "nStations": int(rx_locs.shape[0]),
                "nActive": n_active,
                "backgroundDensity": background,
                "dRhoMin": float(np.nanmin(d_rho_core)),
                "dRhoMax": float(np.nanmax(d_rho_core)),
                "betaFinal": float(getattr(inv_prob, "beta", float("nan"))),
            },
        )

    # ──────────────────────────── helpers (build INSIDE the boundary) ────────────────────────────

    @staticmethod
    def _collect_stations(
        ctx: InversionContext, *, default_rel: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather gravity stations: SimPEG ``(x, y, z)`` locs + ``gravity_anomaly`` + σ.

        Reads the engine-agnostic Observation dicts (coords are Engineering ``(z, y, x)``
        Z-up; per-property ``values``/``sigma`` keyed by property, doc 02 §10.2). When a
        station has no σ a noise floor of ``default_rel × max|anomaly|`` is used so the χ
        target is well posed (doc 10 §3).
        """
        locs: list[list[float]] = []
        vals: list[float] = []
        sigs: list[float] = []
        for obs in ctx.observations:
            coords = obs.get("coords") or []
            values = obs.get("values") or {}
            sigma = obs.get("sigma") or {}
            ga = values.get("gravity_anomaly")
            if ga is None:
                continue
            ga = np.asarray(ga, dtype=float)
            ga_sigma = sigma.get("gravity_anomaly")
            ga_sigma = np.asarray(ga_sigma, dtype=float) if ga_sigma is not None else None
            for k, c in enumerate(coords):
                if len(c) < 3 or k >= ga.size:
                    continue
                z, y, x = float(c[0]), float(c[1]), float(c[2])
                v = float(ga[k])
                if not np.isfinite(v):
                    continue
                locs.append([x, y, z])  # SimPEG order (easting, northing, up)
                vals.append(v)
                s = (
                    float(ga_sigma[k])
                    if ga_sigma is not None and k < ga_sigma.size and np.isfinite(ga_sigma[k])
                    else np.nan
                )
                sigs.append(s)

        if not locs:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

        rx = np.asarray(locs, dtype=float)
        dobs = np.asarray(vals, dtype=float)
        std = np.asarray(sigs, dtype=float)
        # Floor: replace missing/zero σ with a relative noise floor (doc 10 §3).
        floor = max(default_rel * float(np.nanmax(np.abs(dobs)) or 1.0), 1e-6)
        std = np.where(np.isfinite(std) & (std > 0.0), std, floor)
        return rx, dobs, std

    @staticmethod
    def _sensitivity_sigma(
        domain: ModelDomain, reg: Any, d_rho_core: np.ndarray
    ) -> np.ndarray:
        """Tier-B sensitivity/DOI σ on the CORE from the regularization cell weights.

        SimPEG's ``UpdateSensitivityWeights`` stores per-cell sensitivity weights on the
        regularization mesh; a *low* weight ⇒ a poorly-constrained (deep / edge) cell ⇒ a
        *large* σ. We map those weights onto the core sub-brick and turn them into a
        relative 1σ field, normalised so the best-resolved cell gets ~10 % and the worst
        ~50 % of the recovered anomaly magnitude (doc 10 §2.3). Falls back to a flat
        depth-inflated proxy when no weights are available.
        """
        weights = _extract_cell_weights(reg)
        nz, ny, nx = d_rho_core.shape
        amp = np.abs(d_rho_core) + 1e-6 * float(np.nanmax(np.abs(d_rho_core)) + 1.0)
        if weights is None:
            # No sensitivity weights — depth-inflated relative σ (Z-up, index 0 deepest).
            rel = np.full((nz, ny, nx), 0.15, dtype=float)
            if nz > 1:
                rel = rel * (1.0 + np.linspace(1.0, 0.0, nz))[:, None, None]
            return (rel * amp).astype(np.float32)

        full = np.zeros(domain.mesh.n_cells, dtype=float)
        active = np.asarray(domain.active_cells, dtype=bool)
        w = np.asarray(weights, dtype=float)
        if w.size == active.sum():
            full[active] = w
        elif w.size == domain.mesh.n_cells:
            full = w
        else:  # unexpected length — fall back
            return SimpegGravityInversion._sensitivity_sigma(domain, None, d_rho_core)
        w_core = domain.extract_core(full)
        w_core = np.where(np.isfinite(w_core) & (w_core > 0), w_core, np.nan)
        wmax = np.nanmax(w_core)
        if not np.isfinite(wmax) or wmax <= 0:
            return SimpegGravityInversion._sensitivity_sigma(domain, None, d_rho_core)
        # Normalise weights to [0, 1]; resolution = w/wmax → rel σ in [0.1, 0.5].
        res = np.nan_to_num(w_core / wmax, nan=0.0)
        rel = 0.1 + 0.4 * (1.0 - np.clip(res, 0.0, 1.0))
        return (rel * amp).astype(np.float32)


# ──────────────────────────── progress + cancel directive (doc 10 §3) ────────────────────────────


class _CancelInversion(Exception):
    """Internal sentinel: the cooperative-cancel directive aborts the SimPEG run."""


def _make_progress_directive_base():  # noqa: ANN202 - returns a SimPEG directive base class
    """Import the SimPEG directive base lazily (kept out of module import, doc 10 §8)."""
    from simpeg.directives import InversionDirective

    return InversionDirective


class _ProgressDirective:
    """A SimPEG ``InversionDirective`` that reports φ_d/φ_m + polls cooperative cancel.

    This is a thin shim constructed per-run; it subclasses SimPEG's ``InversionDirective``
    dynamically (the base is imported lazily inside :meth:`run`) so importing this engine
    module never pulls SimPEG in (doc 10 §8). On each iteration it maps the SimPEG inner
    progress onto ``ctx.progress`` (0.15 → 0.85) carrying φ_d/φ_m, and raises
    :class:`_CancelInversion` when ``ctx.is_cancelled()`` so the harness records a cancelled
    job (doc 10 §3).
    """

    def __new__(cls, ctx: InversionContext):  # noqa: D102 - construct the dynamic subclass
        base = _make_progress_directive_base()

        class _Impl(base):  # type: ignore[misc, valid-type]
            def __init__(self, ctx: InversionContext) -> None:
                super().__init__()
                self._ctx = ctx
                self._max_iter = int(ctx.params.get("max_iterations", 8)) or 8

            def initialize(self) -> None:
                if self._ctx.is_cancelled():
                    raise _CancelInversion

            def endIter(self) -> None:
                ctx = self._ctx
                if ctx.is_cancelled():
                    raise _CancelInversion
                it = int(getattr(self.opt, "iter", 0) or 0)
                phi_d = getattr(self.invProb, "phi_d", None)
                phi_m = getattr(self.invProb, "phi_m", None)
                frac = 0.15 + 0.7 * min(1.0, it / max(self._max_iter, 1))
                ctx.progress(
                    frac, "gauss-newton",
                    iteration=it,
                    phi_d=float(phi_d) if phi_d is not None else None,
                    phi_m=float(phi_m) if phi_m is not None else None,
                )

        return _Impl(ctx)


def _extract_cell_weights(reg: Any) -> np.ndarray | None:
    """Best-effort extraction of per-cell sensitivity weights from a regularization object.

    ``UpdateSensitivityWeights`` writes weights onto each regularization objective-function
    term; the API has drifted across SimPEG versions, so we probe the common locations and
    return the first plausible per-cell vector, or ``None`` (the caller then uses a DOI
    fallback, doc 10 §2.3).
    """
    if reg is None:
        return None
    candidates = []
    objfcts = getattr(reg, "objfcts", None) or []
    for sub in objfcts:
        w = getattr(sub, "_weights", None)
        if isinstance(w, dict):
            for key, val in w.items():
                if "sensitivity" in str(key).lower():
                    candidates.append(np.asarray(val, dtype=float))
    for c in candidates:
        if c.ndim == 1 and c.size > 0:
            return c
    return None


# Self-register on import (doc 08 §4f) — exactly like the in-framework MockLinearEngine.
register_inversion_engine(SimpegGravityInversion())
