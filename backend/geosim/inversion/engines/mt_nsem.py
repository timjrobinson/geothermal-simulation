"""SimPEG 3-D natural-source MT inversion engine (doc 10 §8, §9) — MT → resistivity.

The geophysically-proper resistivity model (doc 10 §9): a 3-D magnetotelluric (MT)
inversion of the platform MT tensor soundings into an absolute **resistivity**
:class:`~geosim.catalog.PropertyModel`. Unlike the 2-D ERT section
(:mod:`geosim.inversion.engines.ert_pygimli`), MT recovers a full 3-D conductivity volume
from the natural electromagnetic field, so it is the right tool for a deep geothermal
resistivity image (the clay-cap / upflow structure FORGE-style surveys target, doc 05 §4).

Design (doc 10 §8 "SimPEG → TensorMesh, Survey → property", solvers stay LOCAL):

1. Build a SimPEG NSEM ``Survey`` from the platform MT Observations
   (:mod:`simpeg.electromagnetics.natural_source`): each site contributes an
   :class:`~simpeg.electromagnetics.natural_source.receivers.Impedance` receiver for the
   off-diagonal ``Z_xy`` / ``Z_yx`` elements, sampling **apparent resistivity + phase**;
   one :class:`~simpeg.electromagnetics.natural_source.sources.PlanewaveXYPrimary` plane
   wave per period drives both polarisations (doc 05 §4). The observed app-ρ / phase come
   straight off the per-site ``resistivity`` / ``phase_mrad`` sounding curves.
2. Build a :class:`~simpeg.electromagnetics.natural_source.simulation.Simulation3DPrimary
   Secondary` on the :class:`~geosim.inversion.domain.ModelDomain` ``TensorMesh``. The model
   unknown is **log-conductivity on the active cells** (``ExpMap ∘ InjectActiveCells``); air
   above topography is fixed at ``1e-8 S/m`` and never inverted (doc 10 §4.3). The
   primary/secondary split solves the secondary field against a 1-D background — a sparse
   PDE solve handled by a **direct** ``pymatsolver`` factorisation.
3. Run an L2 Tikhonov inversion (``WeightedLeastSquares`` smallness + smoothness) with the
   standard directive stack — ``BetaEstimate_ByEig`` + ``BetaSchedule`` (β cooling) and a
   ``TargetMisfit`` (χ ≈ n_data) — recovering log-σ, mapped back to **resistivity**
   (Ω·m = 1/σ) on the core cells. Per-observation σ (from the data error model) weights the
   misfit.
4. Emit the recovered core (Z-up ``(z, y, x)``) resistivity + a sensitivity/DOI σ field +
   ``phi_d`` / ``phi_m`` / iteration diagnostics (doc 10 §3, §7).

GPU NOTE (doc 10 §8, honest): NSEM uses a **sparse direct solver** (``pymatsolver`` LU /
Pardiso / MUMPS factorisation of the complex Maxwell system per frequency). That sparse
factorisation does **not** benefit from a GPU the way the dense potential-field / 1-D
sensitivity products do — the GPU acceleration in this platform helps gravity/magnetics and
1-D EM, **not** this 3-D sparse PDE solve. This engine therefore runs on the CPU direct
solver on both the CI box and the user's RTX-4090 workstation; the win on the 4090 machine
is its 96 GB of RAM (the factorisation is memory-bound), not the GPU. See module
:data:`MT_NSEM_SPEC` ``compute`` = ``worker_process``.

``executionMode`` declares ``worker_process`` (doc 08 §2.1): the NSEM factorisations are a
heavy, long-running native solve that must never block the request thread (doc 10 §8).
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

__all__ = ["SimpegMTInversion", "MT_NSEM_SPEC"]

PROCESS_VERSION = "1.0.0"

# Air conductivity assigned to inactive (above-topography) cells — fixed, never inverted
# (doc 10 §4.3). NSEM needs a finite, tiny σ for the air half-space.
_AIR_SIGMA = 1e-8  # S/m


# ───────────────────────────── declarative spec (doc 10 §2) ─────────────────────────────

MT_NSEM_SPEC = InversionEngineSpec(
    id="simpeg.mt.nsem",
    kind="mt",
    library="simpeg",
    methods=["mt"],  # canonical MethodKey (doc 02 §2)
    output_property="resistivity",  # recovers absolute resistivity (Ω·m) on the core (doc 01 §5)
    mesh_types=("tensor",),
    coupling="standalone",
    compute="worker_process",  # heavy sparse PDE factorisations — off the request thread
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Background / starting resistivity (Ω·m): the 1-D primary half-space AND the
            # log-σ reference model the recovered anomaly departs from (doc 10 §8).
            "background_resistivity": {
                "type": "number", "minimum": 1e-3, "maximum": 1e6, "default": 100.0,
            },
            # Which off-diagonal impedance orientations to fit (doc 05 §4). ``xy`` + ``yx``
            # is the standard 3-D MT data set; a single orientation keeps tests tiny.
            "orientations": {
                "type": "array", "default": ["xy", "yx"],
            },
            # L2 inversion controls (kept SMALL — NSEM forward solves are expensive, doc 10 §9).
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 30, "default": 6},
            "beta0_ratio": {"type": "number", "minimum": 1e-3, "default": 1.0},
            "cooling_factor": {"type": "number", "minimum": 1.0, "default": 2.0},
            "chi_target": {"type": "number", "minimum": 0.1, "default": 1.0},
            # Tikhonov α weights (smallness vs. smoothness, doc 10 §8).
            "alpha_s": {"type": "number", "minimum": 0.0, "default": 1e-3},
            "alpha_xyz": {"type": "number", "minimum": 0.0, "default": 1.0},
            # Log-resistivity bounds for the projected solver (Ω·m → log10 internally).
            "rho_min": {"type": "number", "minimum": 1e-3, "default": 1.0},
            "rho_max": {"type": "number", "minimum": 1e-3, "default": 1e4},
            # Data error model (doc 10 §3): relative floors on app-ρ + an absolute phase
            # floor (degrees), used to build the per-obs standard deviation when the
            # Observation carries no σ. MT app-ρ noise is multiplicative (~5 %); phase ~2°.
            "rel_error_rho": {"type": "number", "minimum": 1e-4, "maximum": 1.0, "default": 0.05},
            "abs_error_phase": {"type": "number", "minimum": 1e-4, "default": 2.0},
            # σ model: σ_ρ = rel·ρ inflated with depth / where sensitivity is low (doc 10 §2.3).
            "rel_sigma": {"type": "number", "minimum": 0.0, "default": 0.15},
        },
    },
)


# ──────────────────────────────── the engine (doc 10 §8) ────────────────────────────────


class SimpegMTInversion:
    """SimPEG 3-D NSEM L2 Tikhonov MT inversion → resistivity (doc 10 §8, §9).

    Implements the :class:`~geosim.inversion.engine.InversionEngine` Protocol: a declarative
    :attr:`spec` + :meth:`run`. All SimPEG NSEM containers (``Survey`` / ``Simulation`` /
    solver) are constructed inside :meth:`run` (doc 10 §8); only NumPy + the
    :class:`~geosim.inversion.domain.ModelDomain` cross the boundary.
    """

    spec = MT_NSEM_SPEC

    def run(self, ctx: InversionContext) -> InversionResult:
        """Build the NSEM survey + Simulation3DPrimarySecondary, invert, return resistivity.

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
        from simpeg.electromagnetics import natural_source as nsem

        params = ctx.params
        domain: ModelDomain = ctx.domain
        background_rho = float(params["background_resistivity"])
        background_sigma = 1.0 / background_rho

        ctx.progress(0.02, "collecting MT soundings")

        # 1) gather per-site app-ρ + phase soundings keyed by period (doc 05 §4). Coords are
        #    Engineering (z, y, x) Z-up (doc 02 §10.2) — re-ordered to SimPEG (x, y, z).
        sites = self._collect_sites(ctx.observations)
        if not sites:
            raise ValueError(
                "simpeg.mt.nsem: no MT soundings found in observations — the engine needs "
                "per-site 'resistivity' (apparent ρ) + 'phase_mrad' curves with a "
                "meta.frequency_hz axis (doc 05 §4)"
            )

        orientations = [str(o) for o in (params.get("orientations") or ["xy", "yx"])]
        survey, dobs, dstd = self._build_survey(
            nsem, sites, orientations, params
        )
        if dobs.size == 0:
            raise ValueError(
                "simpeg.mt.nsem: MT soundings present but no usable (period, app-ρ, phase) "
                "samples after alignment — check the frequency axis (doc 05 §4)"
            )

        ctx.progress(0.1, "building NSEM 3-D simulation")

        # 2) simulation on the active core+pad mesh; the unknown is log-σ on active cells.
        #    InjectActiveCells fixes air at log(_AIR_SIGMA); ExpMap makes the model log-σ.
        mesh = domain.mesh
        active = np.asarray(domain.active_cells, dtype=bool)
        n_active = int(active.sum())
        act_map = maps.InjectActiveCells(
            mesh=mesh, active_cells=active, value_inactive=np.log(_AIR_SIGMA)
        )
        sigma_map = maps.ExpMap(mesh) * act_map  # model = log(σ) on active cells

        # 1-D background σ for the primary field (air masked, doc 10 §4.3).
        sigma_primary = np.full(mesh.n_cells, background_sigma)
        sigma_primary[~active] = _AIR_SIGMA

        simulation = nsem.simulation.Simulation3DPrimarySecondary(
            mesh,
            survey=survey,
            sigmaPrimary=sigma_primary,
            sigmaMap=sigma_map,
        )
        simulation.solver = self._direct_solver()

        ctx.progress(0.15, "assembling data misfit + Tikhonov regularization")

        # 3) data misfit + L2 Tikhonov (smallness + smoothness) on the log-σ model (doc 10 §8).
        survey_data = data.Data(survey, dobs=dobs, standard_deviation=dstd)
        dmis = data_misfit.L2DataMisfit(data=survey_data, simulation=simulation)
        reg = regularization.WeightedLeastSquares(
            mesh,
            active_cells=active,
            mapping=maps.IdentityMap(nP=n_active),
            reference_model=np.full(n_active, np.log(background_sigma)),
            alpha_s=float(params["alpha_s"]),
            alpha_x=float(params["alpha_xyz"]),
            alpha_y=float(params["alpha_xyz"]),
            alpha_z=float(params["alpha_xyz"]),
        )

        # 4) projected Gauss-Newton with log-σ bounds derived from the ρ bounds (doc 10 §8).
        #    ρ in [rho_min, rho_max] ⇒ σ in [1/rho_max, 1/rho_min] ⇒ log-σ bounds.
        rho_min = float(params["rho_min"])
        rho_max = float(params["rho_max"])
        log_sigma_lo = float(np.log(1.0 / rho_max))
        log_sigma_hi = float(np.log(1.0 / rho_min))
        opt = optimization.ProjectedGNCG(
            maxIter=int(params["max_iterations"]),
            lower=log_sigma_lo,
            upper=log_sigma_hi,
            maxIterLS=10,
            cg_maxiter=12,
            cg_rtol=1e-3,
        )
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

        # 5) directive stack: β cooling + χ target + progress/cancel (doc 10 §8). NSEM
        #    sensitivities are formed implicitly, so we do NOT use UpdateSensitivityWeights.
        progress_directive = _ProgressDirective(ctx)
        directive_list = [
            directives.BetaEstimate_ByEig(beta0_ratio=float(params["beta0_ratio"])),
            directives.BetaSchedule(
                coolingFactor=float(params["cooling_factor"]), coolingRate=1
            ),
            directives.TargetMisfit(chifact=float(params["chi_target"])),
            progress_directive,
        ]
        inv = inversion.BaseInversion(inv_prob, directiveList=directive_list)

        ctx.progress(0.2, "running NSEM L2 inversion")
        m0 = np.full(n_active, np.log(background_sigma))
        try:
            m_rec = inv.run(m0)
        except Cancelled:
            raise
        except _CancelInversion as exc:  # cooperative cancel surfaced from the directive
            raise Cancelled from exc

        # 6) log-σ on active cells → full mesh σ → CORE sub-brick → resistivity (Ω·m = 1/σ).
        sigma_full = sigma_map * m_rec  # full-mesh conductivity (air = _AIR_SIGMA)
        sigma_core = domain.extract_core(np.asarray(sigma_full, dtype=float))  # (nz, ny, nx)
        sigma_core = np.clip(sigma_core, 1.0 / rho_max, 1.0 / rho_min)
        rho_core = (1.0 / sigma_core).astype(np.float32)

        # 7) tier-B sensitivity/DOI σ on the CORE (doc 10 §2.3): a depth-inflated relative σ
        #    on the recovered resistivity (MT resolution degrades with depth + away from
        #    sites; the harness default is the same DOI proxy, reused here on log-ρ scale).
        sigma_uncert = self._depth_sigma(rho_core, float(params["rel_sigma"]))

        phi_d = float(inv_prob.phi_d) if inv_prob.phi_d is not None else None
        phi_m = float(inv_prob.phi_m) if inv_prob.phi_m is not None else None
        iterations = int(getattr(opt, "iter", 0) or 0)

        ctx.progress(0.9, "MT inversion converged", iteration=iterations,
                     phi_d=phi_d, phi_m=phi_m)

        return InversionResult(
            values=rho_core,
            sigma=sigma_uncert,
            iterations=iterations,
            final_phi_d=phi_d,
            final_phi_m=phi_m,
            metrics={
                "engine": "simpeg.mt.nsem",
                "library": "simpeg",
                "processVersion": PROCESS_VERSION,
                "nSites": len(sites),
                "nPeriods": int(sum(len(s["periods"]) for s in sites) // max(len(sites), 1)),
                "nData": int(dobs.size),
                "nActive": n_active,
                "orientations": orientations,
                "backgroundResistivity": background_rho,
                "rhoMin": float(np.nanmin(rho_core)),
                "rhoMax": float(np.nanmax(rho_core)),
                "betaFinal": float(getattr(inv_prob, "beta", float("nan"))),
                "solver": self._solver_name(),
                "gpuAccelerated": False,  # sparse direct factorisation — CPU (doc 10 §8)
            },
        )

    # ──────────────────────────── survey assembly (doc 10 §8) ────────────────────────────

    @staticmethod
    def _collect_sites(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Gather per-site MT soundings: location ``(x, y, z)`` + period / app-ρ / phase.

        The platform MT Observation (doc 03 §2 / doc 05 §4) is a ``tensor`` site carrying the
        apparent-resistivity (``values.resistivity``, Ω·m) and phase (``values.phase_mrad``,
        milliradians after normalisation) sounding curves, with the period axis in
        ``meta.frequency_hz`` (Hz). Coords are Engineering ``(z, y, x)`` Z-up (doc 02 §10.2);
        every period sample sits at the same site location, so the site XY/Z is taken from the
        first coord row. No SimPEG types are touched here.
        """
        sites: list[dict[str, Any]] = []
        for obs in observations:
            values = obs.get("values") or {}
            rho = values.get("resistivity")
            if rho is None:
                continue
            rho = np.asarray(rho, dtype=float)
            phase_mrad = values.get("phase_mrad")
            phase_deg = (
                np.degrees(np.asarray(phase_mrad, dtype=float) / 1000.0)
                if phase_mrad is not None
                else None
            )
            meta = obs.get("meta") or {}
            freq = meta.get("frequency_hz")
            coords = obs.get("coords") or []
            if freq is None or not coords:
                continue
            freq = np.asarray(freq, dtype=float)
            periods = np.divide(
                1.0, freq, out=np.full_like(freq, np.nan), where=freq != 0.0
            )

            # First coord row → site location. Engineering (z, y, x) → SimPEG (x, y, z).
            c0 = coords[0]
            if len(c0) < 3:
                continue
            z, y, x = float(c0[0]), float(c0[1]), float(c0[2])

            n = min(periods.size, rho.size)
            if phase_deg is not None:
                n = min(n, phase_deg.size)
            ok = np.isfinite(periods[:n]) & np.isfinite(rho[:n]) & (rho[:n] > 0)
            if phase_deg is not None:
                ok &= np.isfinite(phase_deg[:n])
            if not np.any(ok):
                continue

            sites.append({
                "loc": (x, y, z),
                "periods": periods[:n][ok],
                "rho": rho[:n][ok],
                "phase": (phase_deg[:n][ok] if phase_deg is not None else None),
            })
        return sites

    def _build_survey(
        self,
        nsem: Any,
        sites: list[dict[str, Any]],
        orientations: list[str],
        params: dict[str, Any],
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """Build the NSEM ``Survey`` + matching ``dobs`` / ``standard_deviation`` vectors.

        One :class:`PlanewaveXYPrimary` source per UNIQUE period drives both polarisations;
        each site at that period contributes an ``Impedance`` apparent-resistivity receiver
        (and a phase receiver when the site has phase) per requested off-diagonal orientation.
        The observed vector ``dobs`` and per-obs ``standard_deviation`` are assembled in the
        SAME receiver order SimPEG predicts (source-major, then receiver-major), so the data
        misfit lines up element-for-element (doc 10 §8). All NSEM types stay local (doc 10 §8).
        """
        rx = nsem.receivers
        src = nsem.sources

        # Unique period grid across all sites (sources are per-frequency, doc 05 §4).
        all_periods = np.unique(np.concatenate([s["periods"] for s in sites]))
        rel_rho = float(params["rel_error_rho"])
        abs_phase = float(params["abs_error_phase"])

        source_list: list[Any] = []
        dobs_parts: list[np.ndarray] = []
        dstd_parts: list[np.ndarray] = []

        for period in all_periods:
            freq = 1.0 / float(period)
            rx_list: list[Any] = []
            for s in sites:
                idx = np.where(np.isclose(s["periods"], period))[0]
                if idx.size == 0:
                    continue
                k = int(idx[0])
                loc = np.asarray([s["loc"]], dtype=float)  # (1, 3) SimPEG (x, y, z)
                rho_v = float(s["rho"][k])
                for orient in orientations:
                    rx_list.append(
                        rx.Impedance(loc, orientation=orient, component="apparent_resistivity")
                    )
                    dobs_parts.append(np.array([rho_v]))
                    dstd_parts.append(np.array([max(rel_rho * abs(rho_v), 1e-6)]))
                    if s["phase"] is not None:
                        phase_v = float(s["phase"][k])
                        rx_list.append(
                            rx.Impedance(loc, orientation=orient, component="phase")
                        )
                        dobs_parts.append(np.array([phase_v]))
                        dstd_parts.append(np.array([abs_phase]))
            if rx_list:
                source_list.append(src.PlanewaveXYPrimary(rx_list, freq))

        survey = nsem.survey.Survey(source_list)
        dobs = (
            np.concatenate(dobs_parts) if dobs_parts else np.zeros(0)
        )
        dstd = (
            np.concatenate(dstd_parts) if dstd_parts else np.zeros(0)
        )
        return survey, dobs, dstd

    # ──────────────────────── uncertainty + solver (doc 10 §2.3, §8) ────────────────────────

    @staticmethod
    def _depth_sigma(rho_core: np.ndarray, rel_sigma: float) -> np.ndarray:
        """Tier-B depth-inflated relative σ on the recovered resistivity (doc 10 §2.3).

        MT resolution is best near the surface + sites and degrades with depth (the deep
        target is constrained only by the longest periods). We emit a relative 1σ that grows
        with depth: ``rel_sigma`` at the shallow top, up to ``2×`` at the deep bottom (Z-up,
        index 0 = deepest). Never zero — an inversion with no uncertainty is invalid.
        """
        rho_core = np.asarray(rho_core, dtype=float)
        nz, ny, nx = rho_core.shape
        rel = np.full((nz, ny, nx), float(rel_sigma), dtype=float)
        if nz > 1:
            # Z-up: index 0 deepest → largest inflation (1+1=2×), shallowest → 1×.
            depth_frac = np.linspace(1.0, 0.0, nz)
            rel = rel * (1.0 + depth_frac)[:, None, None]
        sigma = rel * np.abs(rho_core)
        sigma = np.maximum(sigma, 1e-3 * np.abs(rho_core) + 1e-6)
        return sigma.astype(np.float32)

    @staticmethod
    def _direct_solver() -> Any:
        """The sparse DIRECT solver for the NSEM factorisation (CPU-bound, doc 10 §8).

        Prefer Pardiso / MUMPS when the optional native packages are installed (faster
        factorisations on the user's workstation); otherwise fall back to the always-present
        scipy LU (``pymatsolver.SolverLU``). GPU plays no role here — the factorisation of the
        complex Maxwell system is sparse-direct (doc 10 §8 gpu note).
        """
        import pymatsolver

        avail = getattr(pymatsolver, "AvailableSolvers", {})
        if avail.get("Pardiso"):
            return pymatsolver.Pardiso
        if avail.get("Mumps"):
            return pymatsolver.Mumps
        return pymatsolver.SolverLU

    @staticmethod
    def _solver_name() -> str:
        """Name of the chosen direct solver, for provenance (doc 10 §7)."""
        try:
            import pymatsolver

            avail = getattr(pymatsolver, "AvailableSolvers", {})
            if avail.get("Pardiso"):
                return "pymatsolver.Pardiso"
            if avail.get("Mumps"):
                return "pymatsolver.Mumps"
        except Exception:  # pragma: no cover
            pass
        return "pymatsolver.SolverLU"


# ──────────────────────────── progress + cancel directive (doc 10 §3) ────────────────────────────


class _CancelInversion(Exception):
    """Internal sentinel: the cooperative-cancel directive aborts the SimPEG run."""


def _make_progress_directive_base():  # noqa: ANN202 - returns a SimPEG directive base class
    """Import the SimPEG directive base lazily (kept out of module import, doc 10 §8)."""
    from simpeg.directives import InversionDirective

    return InversionDirective


class _ProgressDirective:
    """A SimPEG ``InversionDirective`` that reports φ_d/φ_m + polls cooperative cancel.

    A thin per-run shim subclassing SimPEG's ``InversionDirective`` dynamically (the base is
    imported lazily inside :meth:`run`) so importing this engine module never pulls SimPEG in
    (doc 10 §8). On each iteration it maps the SimPEG inner progress onto ``ctx.progress``
    (0.2 → 0.88) carrying φ_d/φ_m, and raises :class:`_CancelInversion` when
    ``ctx.is_cancelled()`` so the harness records a cancelled job (doc 10 §3).
    """

    def __new__(cls, ctx: InversionContext):  # noqa: D102 - construct the dynamic subclass
        base = _make_progress_directive_base()

        class _Impl(base):  # type: ignore[misc, valid-type]
            def __init__(self, ctx: InversionContext) -> None:
                super().__init__()
                self._ctx = ctx
                self._max_iter = int(ctx.params.get("max_iterations", 6)) or 6

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
                frac = 0.2 + 0.68 * min(1.0, it / max(self._max_iter, 1))
                ctx.progress(
                    frac, "gauss-newton",
                    iteration=it,
                    phi_d=float(phi_d) if phi_d is not None else None,
                    phi_m=float(phi_m) if phi_m is not None else None,
                )

        return _Impl(ctx)


# Self-register on the process-wide plugin registry at import time (doc 08 §4f), exactly like
# the in-framework mock engine + the other concrete engines. Importing this module is enough
# to add ``simpeg.mt.nsem`` to the engine palette served at ``GET /inversion-engines`` (doc 10 §2).
register_inversion_engine(SimpegMTInversion())
