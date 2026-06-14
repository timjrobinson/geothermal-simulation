"""empymod 1-D layered TEM inversion engine (doc 10 §8, §9) — stitched conductivity volume.

A *quasi-3-D* transient-electromagnetic (TEM) inversion: every TEM **sounding** carries a
``TIME``/``VOLTAGE`` decay (a central-loop WalkTEM transient, doc 03 §2 em row); the engine
inverts each sounding INDEPENDENTLY to a fixed-depth **layered conductivity-depth** column
(log-σ Occam/Tikhonov), places that recovered column at the sounding's Engineering ``XY``,
and interpolates the columns laterally onto the CORE block of the
:class:`~geosim.inversion.domain.ModelDomain` to yield a single **conductivity**
``PropertyModel`` (S/m) with a mandatory uncertainty + provenance (doc 10 §0, §2.3, §7).

Design (doc 10 §8 — solver containers stay local):

1. **Forward** — :mod:`empymod` 1-D layered response of a central-loop TEM sounding. The
   transmitter is the finite square current loop (modeled as 4 electric wire segments via
   :func:`empymod.bipole`) and the receiver is the vertical magnetic coil at the loop
   centre; the measured quantity is ``dBz/dt`` (the WalkTEM voltage decay, ``signal=0``).
   A zero-offset *point* magnetic dipole would be insensitive to horizontal layering — the
   finite loop is what gives the sounding its conductivity-depth resolution.
2. **Inversion** — a regularised, FIXED depth-layer model in **log-conductivity** solved by
   a damped Gauss-Newton / Levenberg-Marquardt loop (:func:`scipy.optimize.least_squares`)
   that fits the log-voltage decay with a 1st-difference **smoothness** (Occam) term — one
   small, well-posed least-squares problem per sounding.
3. **Stitch** — each recovered σ(z) column is dropped at its sounding ``(x, y)`` and the
   columns are interpolated laterally (nearest / IDW) onto the CORE grid, sweeping depth
   layer-by-layer. Cells far from any sounding fall back to the background σ with an
   inflated uncertainty (doc 10 §4.4, §2.3).

GPU: the per-sounding 1-D forwards are independent and BATCHABLE. When a CUDA Torch device
is present (the user's RTX 4090) :mod:`geosim.compute` is used to batch the per-layer/
per-sounding lateral-interpolation stencils on the GPU; the empymod kernels themselves run
on host NumPy (empymod is CPU/SciPy), so this container's NumPy path is the one tested here.

``executionMode`` declares ``worker_process`` (doc 08 §2.1): a stitched multi-sounding TEM
inversion is a heavy, long-running native solve and must not block the request thread.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from geosim.compute import backend_name, gpu_available

from ..engine import (
    InversionContext,
    InversionEngineSpec,
    InversionResult,
    register_inversion_engine,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..domain import ModelDomain

__all__ = ["Tem1DInversion", "TEM1D_SPEC", "forward_tem_sounding", "invert_sounding"]

_log = logging.getLogger(__name__)

PROCESS_VERSION = "1.0.0"

# Air half-space resistivity for the empymod layered model (doc 10 §8). 2e14 ohm·m is the
# empymod-conventional "insulating air" used by every TEM example.
_AIR_RES = 2.0e14


# ───────────────────────────── declarative spec (doc 10 §2) ─────────────────────────────

TEM1D_SPEC = InversionEngineSpec(
    id="empymod.tem1d",
    kind="em",
    library="empymod",
    methods=["em"],  # canonical MethodKey (doc 02 §2)
    output_property="conductivity",  # recovers σ (S/m) on the core (doc 01 §5)
    mesh_types=("tensor",),
    coupling="standalone",
    compute="worker_process",  # stitched multi-sounding solve — off the request thread
    params_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            # Fixed layer bottom depths below surface (m); the last layer is the half-space.
            # Default: a log-spaced 6-layer column to ~300 m, a sensible WalkTEM DOI.
            "layer_depths": {"type": "array", "default": [20.0, 45.0, 80.0, 130.0, 200.0]},
            # Starting / reference conductivity for every layer (S/m) — also the lateral
            # background for CORE cells with no nearby sounding (doc 10 §4.4).
            "background_conductivity": {
                "type": "number", "minimum": 1e-6, "maximum": 1e3, "default": 0.02,
            },
            # log-σ bounds (S/m) clamping the recovered model to physical ground.
            "sigma_min": {"type": "number", "minimum": 1e-9, "default": 1e-4},
            "sigma_max": {"type": "number", "minimum": 1e-6, "default": 10.0},
            # Occam 1st-difference smoothness weight (λ): larger ⇒ smoother σ(z) (doc 10 §3).
            "smoothness": {"type": "number", "minimum": 0.0, "default": 1.0},
            "max_iterations": {"type": "integer", "minimum": 1, "maximum": 100, "default": 30},
            # Transmitter square-loop half-side (m); FORGE WalkTEM is a 40×40 m loop ⇒ 20 m.
            "loop_half_side": {"type": "number", "minimum": 0.5, "default": 20.0},
            # Relative noise floor on the (log) voltage decay when a sounding carries no σ.
            "rel_noise": {"type": "number", "minimum": 1e-4, "maximum": 1.0, "default": 0.05},
            # Shape-only fit: absorb an unknown per-sounding absolute amplitude calibration
            # (real WalkTEM carries instrument/loop-moment gains the raw .usf omits) so only
            # the conductivity-bearing decay SHAPE is fit (doc 10 §8). Synthetic data: false.
            "fit_gain": {"type": "boolean", "default": False},
            # Usable decay time window (s); gates outside are dropped (early ramp-affected +
            # late noise-floor gates, doc 03 §6). 0 ⇒ no bound on that side.
            "time_min": {"type": "number", "minimum": 0.0, "default": 0.0},
            "time_max": {"type": "number", "minimum": 0.0, "default": 0.0},
            # Lateral stitching: IDW power + search radius (m); 0 radius ⇒ unbounded NN.
            "idw_power": {"type": "number", "minimum": 0.0, "default": 2.0},
            "search_radius": {"type": "number", "minimum": 0.0, "default": 0.0},
            # GPU batching of the lateral-stitch stencils via geosim.compute (doc 10 §8):
            # "auto" → Torch/CUDA when present, else NumPy; "off" forces NumPy; "on" errors
            # without a GPU. The empymod forward kernels always run on host NumPy.
            "use_gpu": {"type": "string", "enum": ["auto", "on", "off"], "default": "auto"},
        },
    },
)


# ──────────────────────────────── forward model (doc 10 §8) ────────────────────────────────


def forward_tem_sounding(
    times: np.ndarray,
    layer_cond: np.ndarray,
    layer_depths: np.ndarray,
    *,
    loop_half_side: float = 20.0,
    srcpts: int = 1,
) -> np.ndarray:
    """empymod 1-D layered central-loop TEM forward: ``dBz/dt`` at the loop centre.

    The transmitter is a finite square current loop of half-side ``loop_half_side`` (the
    FORGE WalkTEM 40×40 m loop ⇒ 20 m), modeled as four electric wire segments summed via
    :func:`empymod.bipole`; the receiver is the vertical magnetic coil at the loop centre.
    A point magnetic dipole at zero offset is *insensitive* to horizontal layering, so the
    finite loop is essential to give the sounding its conductivity-depth resolution (doc 10
    §8).

    ``layer_cond`` are the per-layer conductivities (S/m), top→bottom, the last being the
    half-space; ``layer_depths`` are the cumulative bottom depths of every layer EXCEPT the
    half-space (length ``len(layer_cond) - 1``). Returns the ``dBz/dt`` decay at ``times``
    (the WalkTEM impulse-of-the-step response, ``signal=0``) as host NumPy.
    """
    import empymod

    times = np.asarray(times, dtype=float)
    layer_cond = np.clip(np.asarray(layer_cond, dtype=float), 1e-12, None)
    # empymod wants resistivities, air half-space prepended (doc 10 §8).
    res = np.concatenate([[_AIR_RES], 1.0 / layer_cond])
    depth = np.concatenate([[0.0], np.asarray(layer_depths, dtype=float)])  # incl. surface

    hl = float(loop_half_side)
    # Four electric wire segments forming the closed square loop (z=0 surface).
    segments = (
        [-hl, hl, -hl, -hl, 0.0, 0.0],  # bottom edge (+x)
        [hl, hl, -hl, hl, 0.0, 0.0],    # right edge (+y)
        [hl, -hl, hl, hl, 0.0, 0.0],    # top edge (−x)
        [-hl, -hl, hl, -hl, 0.0, 0.0],  # left edge (−y)
    )
    rec = [0.0, 0.0, 0.0, 0.0, 90.0]  # vertical magnetic dipole at the loop centre

    total = np.zeros(times.size, dtype=float)
    for seg in segments:
        out = empymod.bipole(
            src=seg, rec=rec, depth=depth, res=res, freqtime=times,
            signal=0, mrec=True, srcpts=int(srcpts), verb=0,
        )
        total = total + np.asarray(out, dtype=float)
    return total


# ──────────────────────────── per-sounding 1-D inversion (doc 10 §3) ────────────────────────────


def invert_sounding(
    times: np.ndarray,
    voltage: np.ndarray,
    layer_depths: np.ndarray,
    *,
    background_conductivity: float = 0.02,
    sigma_min: float = 1e-4,
    sigma_max: float = 10.0,
    smoothness: float = 1.0,
    max_iterations: int = 30,
    loop_half_side: float = 20.0,
    rel_noise: float = 0.05,
    fit_gain: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Invert ONE TEM decay → a fixed-depth layered σ(z) column (log-σ Occam, doc 10 §3).

    Solves a small, regularised least-squares problem in **log-conductivity** with
    :func:`scipy.optimize.least_squares` (Trust-Region-Reflective, finite-difference
    Jacobian — only a handful of layers). The data residual fits the **log-amplitude** of
    the ``dBz/dt`` decay (TEM voltages span many decades, so a log fit is the well-posed
    one); a 1st-difference **smoothness** block (weight ``smoothness``) Occam-regularises
    σ(z). Returns ``(sigma_layers S/m top→bottom, info)`` where ``info`` carries the final
    misfit + iteration count for the convergence record (doc 10 §3).

    ``fit_gain`` makes the inversion **shape-only**: each residual analytically removes the
    optimal mean-log amplitude shift between prediction and data, so an unknown per-sounding
    absolute calibration (real WalkTEM voltages carry instrument/loop-moment gains the raw
    ``.usf`` does not record) is absorbed and only the decay *shape* — which carries the
    conductivity-depth structure — is fit (doc 10 §8). Synthetic data needs no gain.
    """
    from scipy.optimize import least_squares

    times = np.asarray(times, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    layer_depths = np.asarray(layer_depths, dtype=float)
    n_layers = layer_depths.size + 1  # + half-space

    # Fit the LOG of |dBz/dt| (TEM decays span decades). Keep finite, positive-amplitude gates.
    amp = np.abs(voltage)
    good = np.isfinite(amp) & (amp > 0) & np.isfinite(times) & (times > 0)
    times = times[good]
    log_d = np.log(amp[good])
    if times.size < 2:
        raise ValueError("tem1d: sounding has < 2 usable decay gates to invert (doc 03 §6)")

    # Per-gate weight: a relative noise floor on the log-decay (doc 10 §3). Late-time gates
    # are noisier; a flat relative floor keeps the problem well-posed without per-gate σ.
    w = 1.0 / max(rel_noise, 1e-4)

    log_min, log_max = np.log(sigma_min), np.log(sigma_max)
    m0 = np.full(n_layers, np.log(np.clip(background_conductivity, sigma_min, sigma_max)))

    lam = float(np.sqrt(max(smoothness, 0.0)))

    def residuals(m: np.ndarray) -> np.ndarray:
        sigma = np.exp(m)
        pred = forward_tem_sounding(
            times, sigma, layer_depths, loop_half_side=loop_half_side,
        )
        pred_amp = np.abs(pred)
        # guard log of any (rare) sign-crossing / zero prediction.
        log_p = np.log(np.clip(pred_amp, 1e-300, None))
        misfit = log_p - log_d
        if fit_gain:
            # remove the analytic least-squares log-amplitude shift → shape-only fit.
            misfit = misfit - float(np.mean(misfit))
        data_res = w * misfit
        if n_layers > 1 and lam > 0:
            # 1st-difference smoothness on log-σ (Occam): penalise rough columns.
            smooth_res = lam * np.diff(m)
            return np.concatenate([data_res, smooth_res])
        return data_res

    result = least_squares(
        residuals, m0, method="trf",
        bounds=(np.full(n_layers, log_min), np.full(n_layers, log_max)),
        max_nfev=int(max_iterations) * (n_layers + 1),
        xtol=1e-8, ftol=1e-8,
    )
    sigma_layers = np.exp(result.x)
    # data-only misfit (drop the smoothness block) for the convergence record.
    res_full = residuals(result.x)
    n_data = times.size
    phi_d = float(np.sum(res_full[:n_data] ** 2))
    info = {
        "phi_d": phi_d,
        "phi_m": float(np.sum(res_full[n_data:] ** 2)) if res_full.size > n_data else 0.0,
        "nfev": int(result.nfev),
        "n_gates": int(n_data),
        "success": bool(result.success),
    }
    return sigma_layers, info


# ──────────────────────────────── the engine (doc 10 §8) ────────────────────────────────


class Tem1DInversion:
    """empymod 1-D layered TEM inversion → stitched conductivity volume (doc 10 §8, §9)."""

    spec = TEM1D_SPEC

    def run(self, ctx: InversionContext) -> InversionResult:
        """Invert every sounding to a σ(z) column, stitch onto the CORE (doc 10 §4.4, §8).

        Heavy :mod:`empymod` / :mod:`scipy` work happens here (not at import) and no solver
        type crosses the boundary (doc 10 §8).
        """
        params = ctx.params
        domain: ModelDomain = ctx.domain

        layer_depths = np.asarray(params["layer_depths"], dtype=float)
        background = float(params["background_conductivity"])
        use_gpu = self._resolve_gpu(str(params.get("use_gpu", "auto")))

        ctx.progress(0.03, "collecting TEM soundings")
        soundings = self._collect_soundings(
            ctx.observations,
            time_min=float(params["time_min"]),
            time_max=float(params["time_max"]),
        )
        if not soundings:
            raise ValueError(
                "empymod.tem1d: no TEM soundings found in observations — the engine needs "
                "per-sounding TIME/VOLTAGE transients (meta.transient) to invert (doc 03 §2)"
            )

        # ── 1) per-sounding 1-D inversions (independent, batchable; doc 10 §8) ──
        n = len(soundings)
        columns: list[np.ndarray] = []  # σ(z) per sounding, top→bottom
        col_xy: list[tuple[float, float]] = []
        phi_d_total = 0.0
        phi_m_total = 0.0
        nfev_total = 0
        for i, snd in enumerate(soundings):
            if ctx.is_cancelled():
                from geosim.jobs import Cancelled

                raise Cancelled
            sigma_layers, info = invert_sounding(
                snd["times"], snd["voltage"], layer_depths,
                background_conductivity=background,
                sigma_min=float(params["sigma_min"]),
                sigma_max=float(params["sigma_max"]),
                smoothness=float(params["smoothness"]),
                max_iterations=int(params["max_iterations"]),
                loop_half_side=float(params["loop_half_side"]),
                rel_noise=float(params["rel_noise"]),
                fit_gain=bool(params["fit_gain"]),
            )
            columns.append(sigma_layers)
            col_xy.append((snd["x"], snd["y"]))
            phi_d_total += info["phi_d"]
            phi_m_total += info["phi_m"]
            nfev_total += info["nfev"]
            ctx.progress(
                0.05 + 0.75 * (i + 1) / n,
                f"inverted sounding {i + 1}/{n}",
                iteration=i + 1, phi_d=info["phi_d"], phi_m=info["phi_m"],
            )

        col_cond = np.asarray(columns, dtype=float)  # (n_soundings, n_layers)
        col_xy_arr = np.asarray(col_xy, dtype=float)  # (n_soundings, 2) Engineering (x, y)

        # ── 2) stitch the columns laterally onto the CORE grid (doc 10 §4.4) ──
        ctx.progress(0.85, "stitching σ(z) columns onto the core grid")
        values, sigma = self._stitch_to_core(
            domain, col_xy_arr, col_cond, layer_depths, params, use_gpu=use_gpu,
        )

        iterations = int(np.ceil(nfev_total / max(n, 1)))
        metrics = {
            "engine": "empymod.tem1d",
            "processVersion": PROCESS_VERSION,
            "empymodVersion": _empymod_version(),
            "nSoundings": int(n),
            "nLayers": int(layer_depths.size + 1),
            "layerDepths": [float(d) for d in layer_depths],
            "backend": backend_name(),
            "gpu": bool(use_gpu),
            "conductivityRange": [float(np.min(values)), float(np.max(values))],
            "meanPhiD": float(phi_d_total / max(n, 1)),
        }
        ctx.progress(0.95, "tem1d inversion converged", iteration=iterations,
                     phi_d=phi_d_total / max(n, 1), phi_m=phi_m_total / max(n, 1))

        return InversionResult(
            values=values.astype(np.float32),
            sigma=sigma.astype(np.float32),
            iterations=iterations,
            final_phi_d=float(phi_d_total / max(n, 1)),
            final_phi_m=float(phi_m_total / max(n, 1)),
            metrics=metrics,
        )

    # ──────────────────────────── observation harvesting (doc 03 §2) ────────────────────────────

    @staticmethod
    def _collect_soundings(
        observations: list[dict[str, Any]],
        *,
        time_min: float = 0.0,
        time_max: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Gather per-sounding ``(x, y, times, voltage)`` from the EM Observations.

        Reads the engine-agnostic Observation dicts produced by the EM ``.usf`` adapter
        (doc 03 §2): each ``soundings`` Observation carries the UTM→Engineering ``(z, y, x)``
        site in ``coords[0]`` and the raw transient in ``meta.transient`` (``time_s`` +
        ``voltage`` gates). The stacked transient is collapsed to one clean decay
        (:func:`_stack_transient`) and clipped to the usable ``[time_min, time_max]`` window;
        soundings with no transient or no location are skipped.
        """
        out: list[dict[str, Any]] = []
        for obs in observations:
            meta = obs.get("meta") or {}
            transient = meta.get("transient") or {}
            times = transient.get("time_s")
            volts = transient.get("voltage")
            coords = obs.get("coords") or []
            if times is None or volts is None or not coords:
                continue
            c = coords[0]
            if len(c) < 3:
                continue
            # Engineering coords are (z, y, x) Z-up (doc 02 §10.2).
            z, y, x = float(c[0]), float(c[1]), float(c[2])
            t = np.asarray(times, dtype=float)
            v = np.asarray(volts, dtype=float)
            t, v = _stack_transient(t, v)
            if time_min > 0.0:
                keep = t >= time_min
                t, v = t[keep], v[keep]
            if time_max > 0.0:
                keep = t <= time_max
                t, v = t[keep], v[keep]
            if t.size < 2 or v.size < 2:
                continue
            out.append({
                "x": x, "y": y, "z": z,
                "times": t, "voltage": v,
                "name": meta.get("sounding_name"),
            })
        return out

    # ──────────────────────────── lateral stitch → core (doc 10 §4.4) ────────────────────────────

    def _stitch_to_core(
        self,
        domain: ModelDomain,
        col_xy: np.ndarray,
        col_cond: np.ndarray,
        layer_depths: np.ndarray,
        params: dict[str, Any],
        *,
        use_gpu: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate the per-sounding σ(z) columns onto the regular CORE block (doc 10 §4.4).

        Each CORE cell ``(z, y, x)`` is assigned: (a) the layer index its depth-below-surface
        falls into (the column is piecewise-constant in z over ``layer_depths``); (b) an
        inverse-distance-weighted blend of the soundings' conductivity in that layer using the
        cell's ``(x, y)`` distance to each sounding. Cells beyond ``search_radius`` of every
        sounding fall back to the background σ with an inflated uncertainty (doc 10 §2.3). The
        σ field grows with distance-to-nearest-sounding and with depth (DOI proxy).

        The lateral IDW stencil is the natural GPU-batchable kernel (one dense
        cell×sounding distance matrix per layer); :mod:`geosim.compute` runs it on Torch/CUDA
        when present (the user's RTX 4090) and on host NumPy here.
        """
        nz, ny, nx = domain.core.n_core()
        (oz, oy, ox), (dz, dy, dx) = domain.core_grid()
        background = float(params["background_conductivity"])
        idw_power = float(params["idw_power"])
        radius = float(params["search_radius"])

        # CORE cell centres (Engineering m, Z-up). Surface elevation = max core z (top).
        zc = oz + dz * np.arange(nz)
        yc = oy + dy * np.arange(ny)
        xc = ox + dx * np.arange(nx)
        surface_z = float(zc.max())
        # depth-below-surface of each z-row → layer index per row.
        depth_below = surface_z - zc  # (nz,), >= 0 going down
        layer_of_z = np.searchsorted(layer_depths, depth_below, side="right")  # 0..n_layers-1
        layer_of_z = np.clip(layer_of_z, 0, col_cond.shape[1] - 1)

        gy, gx = np.meshgrid(yc, xc, indexing="ij")  # (ny, nx)
        cell_xy = np.column_stack([gx.reshape(-1), gy.reshape(-1)])  # (ny*nx, 2)

        # IDW weights from each CORE (x, y) to each sounding (x, y). Batched on the active
        # backend (Torch/CUDA on the GPU box, NumPy here) — doc 10 §8.
        weights, dist_min = self._idw_weights(
            cell_xy, col_xy, power=idw_power, radius=radius, use_gpu=use_gpu,
        )  # weights: (ncell, n_snd); dist_min: (ncell,)

        # Per-(x,y) IDW-blended conductivity for EVERY layer, then pick the layer per z-row.
        # blended[:, l] = Σ_s w[:, s] * col_cond[s, l].
        blended = weights @ col_cond  # (ncell, n_layers)
        in_range = np.isfinite(dist_min) & (
            (radius <= 0.0) | (dist_min <= radius)
        )

        values = np.empty((nz, ny, nx), dtype=float)
        # base relative σ from the registry (S/m), inflated by depth + distance (doc 10 §2.3).
        rel = self._base_rel_sigma()
        sigma = np.empty((nz, ny, nx), dtype=float)

        # normalise distance for a [1..] inflation factor (well-sampled ⇒ 1, far ⇒ large).
        dmax = float(np.nanmax(dist_min)) if np.any(np.isfinite(dist_min)) else 1.0
        dmax = max(dmax, 1e-6)
        dist_norm = np.where(np.isfinite(dist_min), dist_min / dmax, 1.0)

        for k in range(nz):
            li = int(layer_of_z[k])
            layer_vals = blended[:, li].copy()
            layer_vals = np.where(in_range, layer_vals, background)
            values[k] = layer_vals.reshape(ny, nx)
            # depth (DOI) inflation: deeper rows (larger depth_below) less constrained.
            doi = 1.0 + 1.5 * (depth_below[k] / max(float(depth_below.max()), 1e-6))
            dist_infl = 1.0 + 3.0 * dist_norm
            lay_sigma = rel * np.abs(layer_vals) * doi * dist_infl
            # off-sample cells essentially unconstrained → inflate hard but keep finite.
            lay_sigma = np.where(in_range, lay_sigma, max(rel, 0.5) * np.abs(layer_vals) * 5.0)
            sigma[k] = np.maximum(lay_sigma, 1e-4 * np.abs(layer_vals) + 1e-9).reshape(ny, nx)

        return values, sigma

    @staticmethod
    def _idw_weights(
        cell_xy: np.ndarray,
        col_xy: np.ndarray,
        *,
        power: float,
        radius: float,
        use_gpu: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse-distance weights from each CORE (x, y) to each sounding (doc 10 §4.4).

        Returns ``(weights (ncell, n_snd) row-normalised, dist_min (ncell,))``. A cell
        sitting on a sounding gets all its weight there. When a GPU is active the dense
        cell×sounding distance matrix is built on the device via :mod:`geosim.compute`
        (Torch/CUDA), else on host NumPy — the result is always returned as host NumPy so it
        never crosses the boundary device-resident (doc 10 §8).
        """
        if use_gpu:
            ns = _try_torch_namespace()
            if ns is not None:
                torch, device = ns
                c = torch.as_tensor(cell_xy, dtype=torch.float64, device=device)
                s = torch.as_tensor(col_xy, dtype=torch.float64, device=device)
                d2 = ((c[:, None, :] - s[None, :, :]) ** 2).sum(-1)
                dist = torch.sqrt(d2)
                dmin = dist.min(dim=1).values
                w = 1.0 / torch.clamp(dist, min=1e-9) ** power
                on_node = dist < 1e-6
                w = torch.where(on_node, torch.full_like(w, 1e30), w)
                w = w / w.sum(dim=1, keepdim=True)
                # back to host NumPy so device tensors never cross the boundary (doc 10 §8).
                return (
                    np.asarray(w.detach().cpu().numpy()),
                    np.asarray(dmin.detach().cpu().numpy()),
                )

        # Host NumPy path (this container).
        diff = cell_xy[:, None, :] - col_xy[None, :, :]  # (ncell, n_snd, 2)
        dist = np.sqrt((diff ** 2).sum(-1))  # (ncell, n_snd)
        dist_min = dist.min(axis=1)
        with np.errstate(divide="ignore"):
            w = 1.0 / np.clip(dist, 1e-9, None) ** float(power)
        on_node = dist < 1e-6
        w = np.where(on_node, 1e30, w)
        w = w / w.sum(axis=1, keepdims=True)
        return w, dist_min

    @staticmethod
    def _base_rel_sigma() -> float:
        from geosim.spatial import REGISTRY

        try:
            return float(REGISTRY.get("conductivity").default_rel_sigma)
        except Exception:  # pragma: no cover - registry always has conductivity
            return 0.15

    @staticmethod
    def _resolve_gpu(mode: str) -> bool:
        """Resolve the ``use_gpu`` param against the actual backend (doc 10 §8).

        ``"auto"`` ⇒ GPU iff a CUDA Torch device is present; ``"off"`` ⇒ never; ``"on"`` ⇒
        require a GPU (raise if absent, so a misconfigured run fails loudly).
        """
        mode = (mode or "auto").lower()
        has_gpu = _torch_cuda_available() or gpu_available()
        if mode == "off":
            return False
        if mode == "on":
            if not has_gpu:
                raise ValueError(
                    "empymod.tem1d: use_gpu='on' but no CUDA device is available "
                    "(install torch+CUDA or set use_gpu='auto'/'off')"
                )
            return True
        return has_gpu


# ──────────────────────────── small helpers ────────────────────────────


def _stack_transient(
    times: np.ndarray, voltage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a STACKED raw transient into one clean decay (doc 03 §2, §6).

    A real WalkTEM ``.usf`` sounding stores every sweep/channel concatenated, so each gate
    time appears many times (FORGE: 112 stacks). We bin-average the voltage per **unique
    time** to recover the single ``dBz/dt`` decay, drop non-finite / non-positive gates
    (late-time noise floor where the signal has decayed below zero crossing), and sort by
    time. The result is the monotone-ish decay the 1-D inversion fits (doc 10 §3). A
    transient that is already one clean decay passes through unchanged.
    """
    times = np.asarray(times, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    finite = np.isfinite(times) & np.isfinite(voltage)
    times, voltage = times[finite], voltage[finite]
    if times.size == 0:
        return times, voltage
    uniq = np.unique(times)
    if uniq.size < times.size:
        # average the stacks at each gate time.
        avg = np.array([voltage[times == u].mean() for u in uniq])
        times, voltage = uniq, avg
    else:
        order = np.argsort(times)
        times, voltage = times[order], voltage[order]
    # keep positive-amplitude gates (drop the late-time sign-crossing noise floor).
    keep = voltage > 0
    return times[keep], voltage[keep]


def _try_torch_namespace() -> tuple[Any, str] | None:
    """Return ``(torch, 'cuda')`` iff Torch + CUDA are present, else ``None`` (doc 10 §8)."""
    from geosim.compute import torch_device, try_torch

    torch = try_torch()
    if torch is None:
        return None
    device = torch_device()
    if device != "cuda":
        return None
    return torch, device


def _torch_cuda_available() -> bool:
    from geosim.compute import torch_device

    return torch_device() == "cuda"


def _empymod_version() -> str:
    try:
        import empymod

        return str(getattr(empymod, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "unknown"


# Self-register on the process-wide plugin registry at import time (doc 08 §4f), exactly like
# the in-framework mock + SimPEG/PyGIMLi engines. Importing this module adds ``empymod.tem1d``
# to the engine palette served at ``GET /inversion-engines`` (doc 10 §2).
register_inversion_engine(Tem1DInversion())
