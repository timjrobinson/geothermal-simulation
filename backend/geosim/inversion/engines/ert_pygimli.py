"""PyGIMLi ERT inversion engine (doc 10 §8, §9) — apparent resistivity → resistivity.

The first *local-feasible, geothermally-meaningful* inversion engine (doc 10 §9): it consumes
a platform ERT Observation (a dipole-dipole apparent-resistivity **pseudosection** plus its
electrode geometry, doc 03 §2 ert row) and runs a PyGIMLi 2D ERT inversion to recover a
**resistivity** :class:`~geosim.catalog.PropertyModel`, with PyGIMLi's model **coverage** as
the tier-B uncertainty + diagnostics (doc 10 §2.3, §8).

Design (doc 10 §8): the engine builds the PyGIMLi ``DataContainerERT`` / ``ERTManager``
**entirely inside** :meth:`PygimliERTInversion.run` from the engine-agnostic NumPy inputs on
the :class:`~geosim.inversion.engine.InversionContext`; no PyGIMLi type ever crosses the
plugin boundary. PyGIMLi solves on a topography-conforming triangular ``SimplexMesh`` (its
``paraDomain``); the recovered field is then resampled onto the regular CORE block of the
:class:`~geosim.inversion.domain.ModelDomain` (doc 10 §4.4) so it lands as an ordinary
gridded PropertyModel that fuses + serves like any other (doc 10 §0).

ERT is intrinsically a *line* survey: the inversion lives in the vertical section under the
electrode line (distance-along-line ``s`` × elevation ``z``). The 2D section is swept across
the (thin) ``y`` extent of the core when resampling — the recovered model is only trusted in
the plane of the line, which the coverage-derived σ reflects (cells off-section / poorly
covered get inflated σ, doc 10 §2.3).

PyGIMLi 1.6 API notes (adapted where the installed API differs from older docs):
- the survey is assembled as a ``pg.DataContainerERT`` with explicit ``a/b/m/n`` sensor
  indices + ``rhoa``; geometric factors via :func:`ert.createGeometricFactors` and a data
  error model via :func:`ert.estimateError` (relative + abs floor).
- :meth:`ert.ERTManager.invert` builds + meshes the inversion domain itself (``paraDX`` /
  ``paraMaxCellSize`` / ``paraDepth`` control the auto-mesh); ``mgr.paraDomain`` is the
  recovered mesh and ``mgr.coverage()`` the (logarithmic) cumulative sensitivity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..engine import (
    InversionContext,
    InversionEngineSpec,
    InversionResult,
    register_inversion_engine,
)

__all__ = ["PygimliERTInversion", "ERT_PYGIMLI_SPEC"]

_log = logging.getLogger(__name__)

PROCESS_VERSION = "1.0.0"


# Declarative spec (doc 10 §2). ``params_schema`` is the JSON-Schema-subset the harness
# validates BEFORE the engine runs (doc 10 §3). executionMode worker_process: a PyGIMLi
# inversion is a heavy, long-running native solve and must not block the event loop
# (doc 08 §2.1 / doc 10 §8).
ERT_PYGIMLI_SPEC = InversionEngineSpec(
    id="pygimli.ert",
    kind="ert",
    library="pygimli",
    methods=["ert"],  # canonical MethodKey (doc 02 §2)
    output_property="resistivity",
    mesh_types=("tensor",),  # CORE comes from a TensorMesh ModelDomain; PyGIMLi meshes internally
    coupling="standalone",
    compute="worker_process",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Tikhonov regularisation strength (λ): larger ⇒ smoother model (doc 10 §3).
            "lam": {"type": "number", "minimum": 1e-3, "maximum": 1e6, "default": 20.0},
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 50, "default": 6},
            # relative data error model (3 % typical for ERT, doc 03 §2) + an abs floor.
            "relative_error": {"type": "number", "minimum": 1e-4, "maximum": 1.0, "default": 0.03},
            "abs_error": {"type": "number", "minimum": 0.0, "default": 1e-4},
            # auto-mesh controls (PyGIMLi paraDX/paraMaxCellSize); coarse keeps tests fast.
            "para_dx": {"type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.3},
            "para_max_cell_size": {"type": "number", "minimum": 0.0, "default": 0.0},
            "para_depth": {"type": "number", "minimum": 0.0, "default": 0.0},
            # background ρ assigned to CORE cells with no PyGIMLi coverage (doc 10 §4.4).
            "background_resistivity": {"type": "number", "minimum": 1e-6, "default": 100.0},
            # σ model: σ = rel·ρ inflated where coverage is low (doc 10 §2.3).
            "rel_sigma": {"type": "number", "minimum": 0.0, "default": 0.15},
        },
    },
)


class PygimliERTInversion:
    """PyGIMLi ERT inversion engine (doc 10 §8, §9) — pseudosection → resistivity volume."""

    spec = ERT_PYGIMLI_SPEC

    def run(self, ctx: InversionContext) -> InversionResult:
        """Build a PyGIMLi ERT survey from the Observations, invert, resample to the core.

        Steps (all PyGIMLi types stay local, doc 10 §8):

        1. assemble a ``DataContainerERT`` from the ERT Observation electrode geometry +
           apparent resistivity (:meth:`_build_data_container`).
        2. run :meth:`ert.ERTManager.invert` (auto-meshing a topography-conforming
           ``paraDomain``), polling ``ctx.is_cancelled`` and reporting φ_d via the χ² record.
        3. resample the recovered 2D section onto the regular CORE block of the
           :class:`~geosim.inversion.domain.ModelDomain` (doc 10 §4.4) and derive a
           coverage-weighted σ (doc 10 §2.3).
        """
        # Heavy PyGIMLi imports live INSIDE run so importing the module is cheap and the
        # solver never leaks across the boundary (doc 10 §8).
        import pygimli as pg
        import pygimli.physics.ert as ert

        params = ctx.params
        ctx.progress(0.05, "building ERT survey")

        elec_xy, abmn, rhoa = self._collect_measurements(ctx.observations)
        if abmn.shape[0] == 0:
            raise ValueError(
                "pygimli.ert: no ERT measurements found in observations (need electrode "
                "quadrupoles A/B/M/N + apparent resistivity, doc 03 §2)"
            )

        data = self._build_data_container(pg, ert, elec_xy, abmn, rhoa, params)

        if ctx.is_cancelled():
            from geosim.jobs import Cancelled

            raise Cancelled

        ctx.progress(0.2, "pygimli ERT inversion")
        mgr = ert.ERTManager(sr=False)

        invert_kwargs: dict[str, Any] = {
            "lam": float(params["lam"]),
            "maxIter": int(params["max_iterations"]),
            "verbose": False,
            "paraDX": float(params["para_dx"]),
        }
        if float(params["para_max_cell_size"]) > 0:
            invert_kwargs["paraMaxCellSize"] = float(params["para_max_cell_size"])
        if float(params["para_depth"]) > 0:
            invert_kwargs["paraDepth"] = float(params["para_depth"])

        model = np.asarray(mgr.invert(data, **invert_kwargs), dtype=float)

        para = mgr.paraDomain
        try:
            coverage = np.asarray(mgr.coverage(), dtype=float)
        except Exception:  # pragma: no cover - coverage is optional diagnostics
            coverage = np.zeros_like(model)

        iterations, phi_d = self._convergence(mgr)
        ctx.progress(0.85, "resampling to core grid", iteration=iterations, phi_d=phi_d)

        cell_centers = np.asarray(para.cellCenters())  # (n_cells, 3): (x', z, 0)
        values, sigma = self._resample_to_core(
            ctx.domain, elec_xy, cell_centers, model, coverage, params
        )

        metrics = {
            "engine": "pygimli.ert",
            "processVersion": PROCESS_VERSION,
            "pygimliVersion": _pygimli_version(pg),
            "nMeasurements": int(abmn.shape[0]),
            "nElectrodes": int(elec_xy.shape[0]),
            "paraDomainCells": int(para.cellCount()),
            "chi2": float(phi_d) if phi_d is not None else None,
            "modelResistivityRange": [float(np.min(model)), float(np.max(model))],
            "coverageRange": [float(np.min(coverage)), float(np.max(coverage))],
        }

        return InversionResult(
            values=values.astype(np.float32),
            sigma=sigma.astype(np.float32),
            iterations=iterations,
            final_phi_d=phi_d,
            final_phi_m=None,
            metrics=metrics,
        )

    # ──────────────────────────── survey assembly (doc 10 §8) ────────────────────────────

    def _collect_measurements(
        self, observations: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gather electrode XY + ABMN sensor-index quadrupoles + apparent ρ from the Obs.

        The platform ERT Observation (doc 03 §2) carries the per-measurement electrode XY in
        ``meta.electrodes`` (``a/b/m/n`` lists of ``[x, y]``) and the apparent resistivity in
        ``values.resistivity``. Electrodes are de-duplicated into a unique sensor list and each
        measurement becomes a quadrupole of sensor indices — exactly what a PyGIMLi
        ``DataContainerERT`` wants. No PyGIMLi types are touched here.
        """
        sensor_index: dict[tuple[float, float], int] = {}
        sensors: list[tuple[float, float]] = []

        def sid(x: float, y: float) -> int:
            key = (round(float(x), 4), round(float(y), 4))
            if key not in sensor_index:
                sensor_index[key] = len(sensors)
                sensors.append((float(x), float(y)))
            return sensor_index[key]

        quads: list[tuple[int, int, int, int]] = []
        rho: list[float] = []
        for obs in observations:
            meta = obs.get("meta") or {}
            elecs = meta.get("electrodes")
            vals = (obs.get("values") or {}).get("resistivity")
            if not elecs or vals is None:
                continue
            a, b, m, n = (elecs.get(k) for k in ("a", "b", "m", "n"))
            if not (a and b and m and n):
                continue
            count = min(len(a), len(b), len(m), len(n), len(vals))
            for i in range(count):
                v = float(vals[i])
                if not np.isfinite(v) or v <= 0:
                    continue
                quads.append((
                    sid(a[i][0], a[i][1]), sid(b[i][0], b[i][1]),
                    sid(m[i][0], m[i][1]), sid(n[i][0], n[i][1]),
                ))
                rho.append(v)

        elec_xy = np.asarray(sensors, dtype=float) if sensors else np.zeros((0, 2))
        abmn = np.asarray(quads, dtype=int) if quads else np.zeros((0, 4), dtype=int)
        rhoa = np.asarray(rho, dtype=float) if rho else np.zeros((0,))
        return elec_xy, abmn, rhoa

    def _build_data_container(
        self,
        pg: Any,
        ert: Any,
        elec_xy: np.ndarray,
        abmn: np.ndarray,
        rhoa: np.ndarray,
        params: dict[str, Any],
    ) -> Any:
        """Build a PyGIMLi ``DataContainerERT`` (local type, never leaked — doc 10 §8).

        Sensors are placed along the survey-line distance ``s`` from the first electrode (a
        2D ERT inversion is parameterised by along-line distance × depth); flat topography is
        assumed (z=0 at surface). Geometric factors + a relative/abs data-error model are
        added so :meth:`ert.ERTManager.invert` has the ``k``/``err`` it needs.
        """
        s = self._along_line(elec_xy, elec_xy)  # distance of each sensor along the line

        data = pg.DataContainerERT()
        for si in s:
            data.createSensor([float(si), 0.0, 0.0])
        data.resize(int(abmn.shape[0]))
        data["a"] = abmn[:, 0]
        data["b"] = abmn[:, 1]
        data["m"] = abmn[:, 2]
        data["n"] = abmn[:, 3]
        data["rhoa"] = rhoa
        data["valid"] = np.ones(abmn.shape[0])

        data["k"] = ert.createGeometricFactors(data)
        data["err"] = ert.estimateError(
            data,
            relativeError=float(params["relative_error"]),
            absoluteError=float(params["abs_error"]),
        )
        return data

    # ──────────────────────────── resample → core grid (doc 10 §4.4) ────────────────────────────

    @staticmethod
    def _along_line(elec_xy: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """Signed distance of ``pts_xy`` projected onto the electrode-line direction.

        The survey line is the first→last electrode direction; distance is measured from the
        first electrode. This collapses the XY survey to the 1D along-line coordinate the 2D
        ERT section is parameterised by.
        """
        first = elec_xy[0]
        last = elec_xy[-1]
        d = last - first
        norm = float(np.hypot(d[0], d[1]))
        if norm < 1e-9:
            # Degenerate line (single point) — fall back to x distance.
            return np.asarray(pts_xy, dtype=float)[:, 0] - float(first[0])
        u = d / norm
        rel = np.asarray(pts_xy, dtype=float) - first
        return rel @ u

    def _resample_to_core(
        self,
        domain: Any,
        elec_xy: np.ndarray,
        cell_centers: np.ndarray,
        model: np.ndarray,
        coverage: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbour resample the 2D PyGIMLi section onto the regular CORE block.

        The recovered model lives on the PyGIMLi ``paraDomain`` in ``(s, z)`` (along-line
        distance × elevation). Each CORE cell centre ``(z, y, x)`` is projected onto the
        survey line to get its ``s`` and matched to the nearest PyGIMLi cell in ``(s, z)``;
        cells outside the section's convex span fall back to ``background_resistivity`` with a
        strongly-inflated σ (doc 10 §4.4, §2.3). The 2D section is swept across ``y`` — the
        model is only trusted in the plane of the line, which the σ reflects.
        """
        nz, ny, nx = domain.core.n_core()
        (oz, oy, ox), (dz, dy, dx) = domain.core_grid()

        zc = oz + dz * np.arange(nz)
        yc = oy + dy * np.arange(ny)
        xc = ox + dx * np.arange(nx)
        gz, gy, gx = np.meshgrid(zc, yc, xc, indexing="ij")
        flat_z = gz.reshape(-1)
        flat_xy = np.column_stack([gx.reshape(-1), gy.reshape(-1)])

        # CORE cells → (s, z) in the section frame.
        s_core = self._along_line(elec_xy, flat_xy)
        # PyGIMLi paraDomain cell centres: column 0 = along-line s, column 1 = elevation z.
        cell_s = cell_centers[:, 0]
        cell_z = cell_centers[:, 1]

        # Section span — guard the off-section / out-of-depth fallback.
        s_lo, s_hi = float(cell_s.min()), float(cell_s.max())
        z_lo, z_hi = float(cell_z.min()), float(cell_z.max())
        margin_s = 0.05 * max(s_hi - s_lo, 1.0)
        margin_z = 0.05 * max(z_hi - z_lo, 1.0)

        background = float(params["background_resistivity"])
        rel_sigma = float(params["rel_sigma"])

        # Vectorised nearest-cell match in (s, z). Tiny meshes ⇒ a dense distance matrix is
        # cheap and dependency-free (no scipy needed).
        ds = s_core[:, None] - cell_s[None, :]
        dzz = flat_z[:, None] - cell_z[None, :]
        dist2 = ds * ds + dzz * dzz
        nearest = np.argmin(dist2, axis=1)

        values = model[nearest].astype(float)
        cov = coverage[nearest].astype(float)

        in_section = (
            (s_core >= s_lo - margin_s)
            & (s_core <= s_hi + margin_s)
            & (flat_z >= z_lo - margin_z)
            & (flat_z <= z_hi + margin_z)
        )
        values = np.where(in_section, values, background)

        # σ model (doc 10 §2.3): base relative σ, inflated where PyGIMLi coverage is low and
        # blown up off-section. Normalise coverage to [0, 1] over the section.
        cov_lo, cov_hi = float(np.min(coverage)), float(np.max(coverage))
        if cov_hi > cov_lo:
            cov_norm = (cov - cov_lo) / (cov_hi - cov_lo)
        else:
            cov_norm = np.ones_like(cov)
        cov_norm = np.clip(cov_norm, 0.0, 1.0)
        # poorly covered ⇒ up to 4× the base σ; well covered ⇒ ~1×.
        cov_inflation = 1.0 + 3.0 * (1.0 - cov_norm)
        sigma = rel_sigma * np.abs(values) * cov_inflation
        # off-section cells are essentially unconstrained: inflate hard but keep finite.
        sigma = np.where(in_section, sigma, np.maximum(rel_sigma, 0.5) * np.abs(values) * 5.0)
        # never zero (an inversion with no uncertainty is invalid, doc 10 §2.3).
        sigma = np.maximum(sigma, 1e-3 * np.abs(values) + 1e-6)

        return values.reshape((nz, ny, nx)), sigma.reshape((nz, ny, nx))

    # ──────────────────────────── convergence record (doc 10 §3) ────────────────────────────

    @staticmethod
    def _convergence(mgr: Any) -> tuple[int, float | None]:
        """Pull (iterations, χ²) from the PyGIMLi inversion framework (doc 10 §3).

        PyGIMLi exposes the data misfit as χ² (≈ φ_d normalised by the data errors); we surface
        it as ``final_phi_d``. ``iterations`` comes from the inversion's iteration counter, with
        a robust fallback across the 1.x API surface.
        """
        inv = getattr(mgr, "inv", None)
        chi2: float | None = None
        iterations = 0
        if inv is not None:
            try:
                chi2 = float(inv.chi2())
            except Exception:  # pragma: no cover
                chi2 = None
            for attr in ("iter", "iterCount"):
                val = getattr(inv, attr, None)
                if val is None:
                    continue
                try:
                    iterations = int(val() if callable(val) else val)
                    break
                except Exception:  # pragma: no cover
                    continue
        return iterations, chi2


def _pygimli_version(pg: Any) -> str:
    """Best-effort PyGIMLi version string for provenance (doc 10 §7)."""
    for getter in ("versionStr", "__version__"):
        attr = getattr(pg, getter, None)
        try:
            return str(attr() if callable(attr) else attr) if attr is not None else "unknown"
        except Exception:  # pragma: no cover
            continue
    return "unknown"


# Self-register on the process-wide plugin registry at import time (doc 08 §4f), exactly like
# the in-framework mock engine. Importing this module is enough to add ``pygimli.ert`` to the
# engine palette served at ``GET /inversion-engines`` (doc 10 §2).
register_inversion_engine(PygimliERTInversion())
