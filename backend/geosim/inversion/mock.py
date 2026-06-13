"""A trivial in-framework MOCK inversion engine (doc 10 §2, test harness fixture).

This engine carries NO heavy solver — it exists so the harness, provenance, persistence,
and fused-resample pipeline can be exercised without SimPEG/PyGIMLi (doc 10 §0). It is a
*linear toy*: it builds a smooth recovered model from a single scalar "target value"
param plus a Gaussian blob centred on the observation locations, runs a handful of fake
Gauss-Newton iterations (reporting decreasing φ_d / φ_m and honouring cooperative cancel),
and returns the recovered CORE model with a native sensitivity-style uncertainty.

It demonstrates the contract every real engine follows (doc 10 §2, §8): it consumes the
engine-agnostic :class:`~geosim.inversion.engine.InversionContext`, builds whatever it
needs *internally* from NumPy + the :class:`~geosim.inversion.domain.ModelDomain`, and
emits an :class:`~geosim.inversion.engine.InversionResult` with a MANDATORY uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .engine import (
    InversionContext,
    InversionEngineSpec,
    InversionResult,
)

__all__ = ["MockLinearEngine", "MOCK_SPEC"]


# Declarative spec (doc 10 §2). The ``params_schema`` is the JSON-Schema-subset the harness
# validates against BEFORE the engine runs (doc 10 §3).
MOCK_SPEC = InversionEngineSpec(
    id="mock.linear",
    kind="mock",
    library="mock",
    methods=["mt"],  # a canonical MethodKey (doc 02 §2) so the registry accepts it
    output_property="resistivity",
    mesh_types=("tensor",),
    coupling="standalone",
    compute="in_process",
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["target_value"],
        "properties": {
            "target_value": {"type": "number", "minimum": 0.0},
            "background_value": {"type": "number", "minimum": 0.0, "default": 100.0},
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 50, "default": 4},
            "blob_radius": {"type": "number", "minimum": 0.0, "default": 50.0},
        },
    },
)


class MockLinearEngine:
    """A solver-free linear toy engine for harness/pipeline tests (doc 10 §2)."""

    spec = MOCK_SPEC

    def run(self, ctx: InversionContext) -> InversionResult:
        """Build a smooth recovered model + native σ, faking Gauss-Newton iterations."""
        params = ctx.params
        target = float(params["target_value"])
        background = float(params.get("background_value", 100.0))
        max_it = int(params.get("max_iterations", 4))
        radius = float(params.get("blob_radius", 50.0))

        domain = ctx.domain
        nz, ny, nx = domain.core.n_core()
        (oz, oy, ox), (dz, dy, dx) = domain.core_grid()

        # Cell-centre coordinates of the CORE block (Engineering m, Z-up).
        zc = oz + dz * np.arange(nz)
        yc = oy + dy * np.arange(ny)
        xc = ox + dx * np.arange(nx)
        gz, gy, gx = np.meshgrid(zc, yc, xc, indexing="ij")

        # Anomaly centre = centroid of observation coordinates (fallback: core centre).
        centre = self._obs_centroid(ctx.observations, default=(
            float(zc.mean()), float(yc.mean()), float(xc.mean())
        ))
        r2 = (gz - centre[0]) ** 2 + (gy - centre[1]) ** 2 + (gx - centre[2]) ** 2
        blob = np.exp(-r2 / (2.0 * max(radius, 1e-6) ** 2))

        # Fake iterative recovery: each iteration sharpens the blob toward the target,
        # reporting decreasing φ_d / φ_m (doc 10 §3) and honouring cooperative cancel.
        recovered = np.full((nz, ny, nx), background, dtype=float)
        phi_d = phi_m = None
        iterations = 0
        for it in range(1, max_it + 1):
            if ctx.is_cancelled():
                # Cooperative cancel mid-run: the harness records a cancelled job (doc 10 §3).
                from geosim.jobs import Cancelled

                raise Cancelled
            frac_done = it / max_it
            recovered = background + (target - background) * blob * frac_done
            phi_d = float(1.0 / it)  # monotonically decreasing data misfit
            phi_m = float(np.mean(blob) * frac_done)  # growing model norm
            iterations = it
            # progress runs 0.1..0.85 across the iterations (harness owns 0/0.9/1.0).
            ctx.progress(
                0.1 + 0.75 * frac_done, "gauss-newton",
                iteration=it, phi_d=phi_d, phi_m=phi_m,
            )

        # Native uncertainty: larger where the recovered model departs from background
        # (a crude sensitivity proxy) — still MANDATORY (doc 10 §2.3).
        departure = np.abs(recovered - background)
        sigma = 0.1 * np.abs(recovered) + 0.25 * departure

        return InversionResult(
            values=recovered.astype(np.float32),
            sigma=sigma.astype(np.float32),
            iterations=iterations,
            final_phi_d=phi_d,
            final_phi_m=phi_m,
            metrics={"engine": "mock.linear", "centre": list(centre)},
        )

    @staticmethod
    def _obs_centroid(
        observations: list[dict[str, Any]], default: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Centroid of observation cell coords ``(z, y, x)`` (Engineering m), or default."""
        pts: list[list[float]] = []
        for obs in observations:
            coords = obs.get("coords") or []
            for c in coords:
                if len(c) >= 3:
                    pts.append([float(c[0]), float(c[1]), float(c[2])])
        if not pts:
            return default
        arr = np.asarray(pts, dtype=float)
        return (float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()))
