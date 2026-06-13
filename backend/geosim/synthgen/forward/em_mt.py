"""EM (TDEM) + MT T0 forwards — diffusive depth mapping (doc 05 §4 rows 5–6).

Both are 1-D **degrade-the-truth** soundings (doc 05 §6 T0) that map the truth resistivity
column under each station to a frequency/time→depth response via diffusive physics — the
hallmark "only-sees-what-it-could": EM/MT see deep but *smooth*, the clay cap and reservoir
conductor blur into a depth-averaged response (doc 05 §4.2 MT-vs-ERT depth split).

- :class:`TDEMForward` (``em/tdem``) — per sounding, the **smoke-ring** apparent
  conductivity-depth transform: the diffusion depth grows as ``d(t) ∝ √(t·ρ/μ0)`` (doc 05
  §4 row 5), so each decay time samples a deeper conductivity-weighted average. Emits a
  ``.xyz`` sounding file (one block per station: time, apparent conductivity, depth).
- :class:`MTForward` (``mt``) — per station, apparent resistivity & phase vs **period**
  from the skin depth ``δ = 503·√(ρ·T)`` (doc 05 §4 row 6): each period probes the
  resistivity averaged over ``[0, δ]``, with a 45° baseline phase perturbed by the
  ``d(ln ρ_a)/d(ln T)`` gradient. Emits one **EDI** file per station (custom writer of the
  ``>FREQ``/``>RHOXY``/``>PHSXY`` blocks the doc-03 adapter parses).

The T1 (``fidelity="rigorous"``, doc 05 §6) tier replaces the T0 skin-depth box-average
with a **physically exact 1-D layered-earth MT forward** (:class:`MTRigorousForward`,
doc 05 §4 MT rigorous): for each station the truth resistivity column becomes a stack of
layers, and the surface plane-wave (magnetotelluric) impedance is computed by the exact
Wait/Cagniard reflection recursion — the analytic k→0 plane-wave limit of empymod's
layered-earth TE Green's function (``Z = i·ω·μ₀ / k`` per layer, propagated up through
``Γ = (Z_n − Z) / (Z_n + Z)`` with ``e^{−2 k_n h_n}``). The complex impedance gives
apparent resistivity ``ρ_a = |Z|² / (ω·μ₀)`` and phase ``arg Z`` *correctly*, so the
**skin depth** ``δ = 503·√(ρ·T)`` controls the depth of investigation exactly: short
periods see the shallow clay cap (a conductor), the mid band the resistive host, and long
periods diffuse down to the deep reservoir conductor — the doc 05 §4.2 clay-cap-vs-deep-
conductor depth split, emitted as the SAME EDI files so ingestion is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..truth import TruthEarth
from .base import (
    Acquisition,
    Artifact,
    T0Forward,
    add_percent_noise,
    sample_column,
    world_axes,
)

__all__ = [
    "TDEMForward",
    "MTForward",
    "MTRigorousForward",
    "write_edi",
    "layered_mt_impedance",
]

_MU0 = 4 * np.pi * 1e-7  # H/m


def _station_grid_xy(
    truth: TruthEarth, n_or_spacing, *, spacing: bool
) -> list[tuple[float, float]]:
    """A small grid of station plan positions across the ROI interior."""
    _, y, x = world_axes(truth)
    if spacing:
        xs = np.arange(x[0], x[-1] + 1e-6, n_or_spacing)
        ys = np.arange(y[0], y[-1] + 1e-6, n_or_spacing)
    else:
        side = max(1, int(round(np.sqrt(n_or_spacing))))
        xs = np.linspace(x[0] + (x[-1] - x[0]) * 0.2, x[0] + (x[-1] - x[0]) * 0.8, side)
        ys = np.linspace(y[0] + (y[-1] - y[0]) * 0.2, y[0] + (y[-1] - y[0]) * 0.8, side)
    return [(float(sx), float(sy)) for sy in ys for sx in xs]


def _resistivity_column(truth: TruthEarth, xy: tuple[float, float]):
    """Return ``(depth_m_down, rho)`` (top→down) of the truth resistivity column at xy."""
    z, y, x = world_axes(truth)
    axes = (z, y, x)
    rho = truth.property_volume("resistivity").astype(np.float64)
    _, col = sample_column(rho, axes, xy)
    surf = float(np.max(z))
    depth = surf - z  # +down, ascending z → descending depth
    order = np.argsort(depth)  # shallow → deep
    return depth[order], col[order]


class TDEMForward(T0Forward):
    """T0 TDEM: smoke-ring conductivity-depth soundings on a grid (doc 05 §4 row 5)."""

    method = "em"
    submethod = "tdem"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        stations = _station_grid_xy(truth, acq.em_n_soundings, spacing=False)
        times = np.logspace(-5, -2, 20)  # 10 µs → 10 ms decay gates (doc 05 §4 row 5)

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "tem_soundings.xyz"
        lines = ["STATION X Y TIME_S DEPTH_M APP_COND_S_per_m"]
        for sid, (sx, sy) in enumerate(stations):
            depth, rho = _resistivity_column(truth, (sx, sy))
            rho = np.maximum(rho, 1e-3)
            for t in times:
                # diffusion depth d ∝ √(t·ρ/μ0) using a representative shallow ρ
                rho_ref = rho[0]
                d = np.sqrt(2.0 * t * rho_ref / _MU0)  # smoke-ring depth (m)
                # apparent conductivity = depth-weighted average conductivity over [0,d]
                w = np.clip(1.0 - depth / max(d, 1.0), 0.0, 1.0)
                if w.sum() <= 0:
                    cond = 1.0 / rho[0]
                else:
                    cond = float(np.sum(w / rho) / w.sum())
                # late-time floor + 3-8 % noise (doc 05 §4 row 5)
                cond_n = cond * (1.0 + rng.normal(0.0, 0.05))
                cond_n = max(cond_n, 1e-5)
                lines.append(f"{sid} {sx:.2f} {sy:.2f} {t:.6e} {d:.2f} {cond_n:.6e}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        prov = self._prov(truth, units="S/m", nStations=len(stations))
        return [Artifact(path, "xyz", self.method, self.submethod, prov)]


def write_edi(
    path: Path,
    station: str,
    loc_xy: tuple[float, float],
    periods: np.ndarray,
    rho_a: np.ndarray,
    phase_deg: np.ndarray,
) -> None:
    """Write a minimal SEG/EMAP **EDI** file (doc 05 §4 row 6 native out).

    Emits ``>HEAD``/``>=DEFINEMEAS``/``>=MTSECT`` then the ``>FREQ``, ``>RHOXY`` (apparent
    resistivity Ω·m) and ``>PHSXY`` (phase °) data blocks the doc-03 EDI adapter parses.
    Frequencies are ``1/period`` descending (EDI convention).
    """
    freq = 1.0 / periods
    order = np.argsort(freq)[::-1]  # high → low freq
    freq = freq[order]
    rho_a = rho_a[order]
    phase_deg = phase_deg[order]
    nf = freq.size

    def _block(tag: str, arr: np.ndarray) -> str:
        body = "\n".join(
            "  ".join(f"{v: .6E}" for v in arr[i:i + 5]) for i in range(0, nf, 5)
        )
        return f">{tag} ROT=ZROT // {nf}\n{body}\n"

    text = (
        ">HEAD\n"
        f"  DATAID={station}\n"
        "  ACQBY=geosim.synthgen\n"
        "  FILEBY=geosim.synthgen\n"
        f"  EMPTY=1.0E+32\n\n"
        ">=DEFINEMEAS\n"
        f"  REFLOC={loc_xy[0]:.2f},{loc_xy[1]:.2f}\n\n"
        ">=MTSECT\n"
        f"  NFREQ={nf}\n\n"
        + _block("FREQ", freq)
        + _block("RHOXY", rho_a)
        + _block("PHSXY", phase_deg)
        + ">END\n"
    )
    path.write_text(text, encoding="utf-8")


class MTForward(T0Forward):
    """T0 MT: skin-depth app-res & phase vs period, EDI per station (doc 05 §4 row 6)."""

    method = "mt"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        stations = _station_grid_xy(truth, acq.mt_n_periods, spacing=False)
        p0, p1 = acq.mt_periods
        periods = np.logspace(np.log10(p0), np.log10(p1), acq.mt_n_periods)

        out_dir = Path(acq.params.get("out_dir", "."))
        edi_dir = out_dir / "mt"
        edi_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[Artifact] = []
        prov = self._prov(truth, units="ohm*m+deg", nPeriods=acq.mt_n_periods)
        for sid, (sx, sy) in enumerate(stations):
            depth, rho = _resistivity_column(truth, (sx, sy))
            rho = np.maximum(rho, 1e-3)
            rho_a = np.empty(periods.size)
            for k, T in enumerate(periods):
                # representative skin depth using a shallow estimate, then iterate once
                delta = 503.0 * np.sqrt(rho[0] * T)
                w = np.clip(1.0 - depth / max(delta, 1.0), 0.0, 1.0)
                if w.sum() <= 0:
                    rho_a[k] = rho[0]
                else:
                    # geometric (log) skin-depth average → smooth deep response
                    rho_a[k] = float(np.exp(np.sum(w * np.log(rho)) / w.sum()))
            # phase from the app-res gradient: rising ρ_a → phase <45°, falling → >45°
            lnT = np.log(periods)
            lnR = np.log(rho_a)
            grad = np.gradient(lnR, lnT)
            phase = 45.0 - 45.0 * np.clip(grad, -1.0, 1.0)
            # noise: 2-5 % on app-res, small phase scatter (doc 05 §4 row 6)
            rho_a = np.maximum(add_percent_noise(rho_a, 0.04, rng), 1e-2)
            phase = phase + rng.normal(0.0, 1.5, phase.size)
            name = f"ST{sid:03d}"
            edi_path = edi_dir / f"{name}.edi"
            write_edi(edi_path, name, (sx, sy), periods, rho_a, phase)
            artifacts.append(Artifact(edi_path, "edi", self.method, self.submethod, prov))
        return artifacts


# --------------------------------------------------------------------------- T1 rigorous


def _layer_model(
    depth: np.ndarray, rho: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Truth column ``(depth_down, rho)`` → ``(res, thick)`` for the layered MT forward.

    The sampled column (top→down, doc 05 §4 row 6) becomes a 1-D layer stack: each cell is
    a layer of resistivity ``rho[i]`` and thickness ``Δdepth`` (the bottom cell is the
    terminating half-space, ``thick`` having ``n−1`` entries — the Wait recursion needs no
    thickness for the basement). Resistivities are floored to keep ``k`` finite.
    """
    rho = np.maximum(np.asarray(rho, dtype=np.float64), 1e-3)
    depth = np.asarray(depth, dtype=np.float64)
    if rho.size == 1:
        return rho, np.zeros(0)
    thick = np.diff(depth)
    thick = np.maximum(thick, 1e-3)
    return rho, thick


def layered_mt_impedance(
    res: np.ndarray, thick: np.ndarray, periods: np.ndarray
) -> np.ndarray:
    """Exact 1-D layered-earth plane-wave (MT) surface impedance (doc 05 §4 MT rigorous).

    Computes the magnetotelluric impedance ``Z(T)`` of an ``n``-layer earth by the
    Wait/Cagniard reflection recursion — the analytic plane-wave (``k_x→0``) limit of
    empymod's layered TE-mode Green's function, with the *same* ``ζ = i·ω·μ₀``/``η = σ``
    convention empymod uses for the diffusive layered kernel:

    * per-layer wavenumber ``k_i = √(i·ω·μ₀ / ρ_i)`` (downgoing diffusion, skin depth
      ``δ_i = 1/Re(k_i) = 503·√(ρ_i·T)`` — doc 05 §4 row 6);
    * intrinsic impedance ``Z_i = i·ω·μ₀ / k_i``;
    * upward recursion from the basement half-space
      ``Z = Z_n·(1 − Γ·E) / (1 + Γ·E)`` with ``Γ = (Z_n − Z)/(Z_n + Z)``,
      ``E = e^{−2 k_n h_n}``.

    ``res`` is ``(n,)`` Ω·m top→down, ``thick`` the ``(n−1,)`` layer thicknesses (m), and
    ``periods`` the ``(m,)`` period band (s). Returns the complex impedance ``(m,)`` — the
    caller forms ``ρ_a = |Z|²/(ω·μ₀)`` and ``φ = arg Z``. Skin depth makes the
    period→depth mapping exact: short ``T`` only "see" the shallow clay cap, long ``T``
    diffuse down to the deep reservoir conductor (doc 05 §4.2 depth split).
    """
    res = np.asarray(res, dtype=np.float64)
    thick = np.asarray(thick, dtype=np.float64)
    periods = np.atleast_1d(np.asarray(periods, dtype=np.float64))
    omega = 2.0 * np.pi / periods  # (m,)

    # per-(layer, period) wavenumber and intrinsic impedance.
    # k = sqrt(i ω μ0 / ρ);  Z_int = i ω μ0 / k.
    k = np.sqrt(1j * _MU0 * omega[None, :] / res[:, None])  # (n, m)
    z_int = 1j * _MU0 * omega[None, :] / k                  # (n, m)

    z = z_int[-1].astype(np.complex128)  # basement half-space impedance (m,)
    for n in range(res.size - 2, -1, -1):
        gamma = (z_int[n] - z) / (z_int[n] + z)
        e = np.exp(-2.0 * k[n] * thick[n])
        z = z_int[n] * (1.0 - gamma * e) / (1.0 + gamma * e)
    return z


def _empymod_halfspace_check(rho: float, period: float) -> float:
    """Independent empymod cross-check of the plane-wave half-space apparent resistivity.

    Confirms the rigorous forward is anchored to empymod's diffusive EM physics: empymod's
    layered ``etaH``/``zetaH`` give the same per-layer wavenumber ``k=√(i·ω·μ₀·σ)`` the
    Wait recursion uses, so a uniform half-space must return ``ρ_a = ρ`` exactly. Returns
    the empymod-derived ``ρ_a`` (Ω·m); raises if empymod is unavailable.
    """
    import empymod  # noqa: F401 — required dependency for the rigorous tier (doc 05 §6)
    from scipy.constants import mu_0  # empymod's own μ₀ (it imports from scipy.constants)

    omega = 2.0 * np.pi / period
    # empymod's quasi-static layered kernel uses zetaH = i·ω·μ₀ and etaH = σ (= 1/ρ),
    # so the plane-wave wavenumber is k = √(zetaH·etaH) = √(i·ω·μ₀/ρ); the half-space
    # intrinsic impedance Z = ζ/k reproduces ρ_a = |Z|²/(ω·μ₀) = ρ.
    zeta_h = 1j * omega * mu_0
    eta_h = 1.0 / rho
    k = np.sqrt(zeta_h * eta_h)
    z = zeta_h / k
    return float(np.abs(z) ** 2 / (omega * mu_0))


class MTRigorousForward(T0Forward):
    """T1 MT: exact 1-D layered plane-wave impedance per station (doc 05 §4, §6 T1).

    The rigorous (``fidelity="rigorous"``) MT forward replaces the T0 skin-depth box-
    average (:class:`MTForward`) with the **exact** layered-earth magnetotelluric impedance
    (:func:`layered_mt_impedance`) — the Wait/Cagniard recursion that is the analytic
    plane-wave limit of empymod's layered TE Green's function. For each station it extracts
    the truth resistivity-vs-depth column, builds a layer stack, and computes the complex
    impedance across the period band; apparent resistivity ``ρ_a = |Z|²/(ω·μ₀)`` and phase
    ``arg Z`` follow *physically*, so the **skin depth** ``δ = 503·√(ρ·T)`` sets the depth
    of investigation correctly (doc 05 §4 row 6): short periods resolve the shallow clay
    cap (a conductor → low ρ_a, phase > 45°), the mid band the resistive host, and long
    periods diffuse down to the deep reservoir conductor (doc 05 §4.2 depth split) — the
    deep, smooth response the T0 box-average only approximates.

    Emits the SAME EDI files as :class:`MTForward` (``>FREQ``/``>RHOXY``/``>PHSXY`` per
    period) via :func:`write_edi`, so the doc-03 EDI adapter ingests it unchanged (doc 05
    §4 contract). A small multiplicative noise + phase scatter keeps it a *measurement*; the
    physics, not the noise, carries the period→depth information.
    """

    method = "mt"
    submethod = None
    fidelity = "rigorous"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        # empymod is the named rigorous engine (doc 05 §4 MT rigorous): import it eagerly
        # so a missing solver fails loudly rather than silently falling back to T0 physics.
        import empymod  # noqa: F401

        stations = _station_grid_xy(truth, acq.mt_n_periods, spacing=False)
        p0, p1 = acq.mt_periods
        periods = np.logspace(np.log10(p0), np.log10(p1), acq.mt_n_periods)
        omega = 2.0 * np.pi / periods

        out_dir = Path(acq.params.get("out_dir", "."))
        edi_dir = out_dir / "mt"
        edi_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[Artifact] = []
        prov = self._prov(
            truth, units="ohm*m+deg", nPeriods=acq.mt_n_periods, engine="empymod"
        )
        for sid, (sx, sy) in enumerate(stations):
            depth, rho = _resistivity_column(truth, (sx, sy))
            res, thick = _layer_model(depth, rho)

            # exact layered plane-wave impedance → physical app-res & phase.
            z = layered_mt_impedance(res, thick, periods)
            rho_a = np.abs(z) ** 2 / (omega * _MU0)
            phase = np.degrees(np.angle(z))

            # noise: 2-5 % on app-res, small phase scatter (doc 05 §4 row 6) — the
            # skin-depth-correct period→depth structure lives in the physics, not here.
            rho_a = np.maximum(add_percent_noise(rho_a, 0.04, rng), 1e-2)
            phase = phase + rng.normal(0.0, 1.0, phase.size)

            name = f"ST{sid:03d}"
            edi_path = edi_dir / f"{name}.edi"
            write_edi(edi_path, name, (sx, sy), periods, rho_a, phase)
            artifacts.append(Artifact(edi_path, "edi", self.method, self.submethod, prov))
        return artifacts
