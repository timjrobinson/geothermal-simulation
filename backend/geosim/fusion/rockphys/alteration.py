"""Alteration-target rock-physics transforms (doc 07 §4.2).

Hydrothermal alteration (a smectite/clay cap) is a key geothermal indicator: alteration
clays are strongly **conductive** (low resistivity) and tend to concentrate in structurally
favourable zones (doc 07 §4.2 "Alteration").

- ``alteration_index`` — a heuristic index combining a **low-resistivity** membership with
  an optional **structure proxy** (here clay volume) — the classic clay-cap signature
  (smectite ⇒ low ρ).
- ``gmm_alteration_posterior`` — a **data-driven** wrapper: fit a 2-component Gaussian
  mixture (the same engine :func:`geosim.fusion.cluster_fused` uses) to the resistivity
  population on the grid and return the **posterior probability of the low-resistivity
  (altered) class** — clustering-as-a-transform (doc 07 §3.3, §4.2 "Alteration
  (data-driven)").

Both output the **alteration** index (dimensionless 0..1). Uncalibrated ⇒ proxy/likelihood.
"""

from __future__ import annotations

import numpy as np

from geosim.fusion.transform import (
    InputSpec,
    OutputSpec,
    Param,
    Transform,
    TransformContext,
)
from geosim.plugins import register

__all__ = ["AlterationIndex", "GmmAlterationPosterior"]


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AlterationIndex(Transform):
    """Heuristic clay-cap / alteration index from low resistivity (+ structure proxy).

    A smooth low-resistivity membership ``L = σ((log10 ρ_threshold − log10 ρ)/width)`` (high
    where ρ is below the clay-cap threshold) optionally fused with a structure proxy (clay
    volume) by a fuzzy conjunction (geometric blend). Output ∈ [0, 1] (doc 07 §4.2).
    """

    id = "rp.alteration_index"
    version = "1.0.0"
    title = "Resistivity (+structure) → Alteration Index (clay cap)"
    target = "alteration"

    assumptions = [
        "conductive (smectite) clay cap ⇒ low resistivity is the dominant alteration proxy",
        "structure proxy (clay volume) reinforces but does not solely determine alteration",
        "threshold/width/weights are heuristic params, calibratable to geochem/XRD",
    ]
    calibration_status = "uncalibrated"

    inputs = [
        InputSpec("resistivity", unit="ohm*m", required=True),
        InputSpec("clay_volume", unit="dimensionless", required=False),
    ]
    output = OutputSpec(
        "alteration", unit="dimensionless", valid_range=(0.0, 1.0),
        colormap="magma", proxy_when_uncalibrated=True,
    )

    params = [
        Param("rho_threshold_ohm_m", float, default=20.0, range=(1.0, 200.0)),
        Param("log_width", float, default=0.3, range=(0.05, 1.5)),
        Param("structure_weight", float, default=0.4, range=(0.0, 1.0)),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        resistivity,
        clay_volume=None,
        *,
        rho_threshold_ohm_m,
        log_width,
        structure_weight,
    ):
        """Low-resistivity membership, optionally fused with a structure (clay) proxy."""
        rho = np.maximum(np.asarray(resistivity, dtype=float), 1e-6)
        low_rho = _logistic((np.log10(rho_threshold_ohm_m) - np.log10(rho)) / log_width)
        if clay_volume is None:
            return ctx.as_output(low_rho)
        struct = np.clip(np.asarray(clay_volume, dtype=float), 0.0, 1.0)
        # Fuzzy blend: weighted geometric mean of the two memberships (conjunction-like).
        w = float(structure_weight)
        idx = (low_rho ** (1.0 - w)) * (struct**w)
        return ctx.as_output(idx)


class GmmAlterationPosterior(Transform):
    """Data-driven alteration likelihood = GMM posterior of the low-resistivity class.

    Fits a 2-component :class:`sklearn.mixture.GaussianMixture` to ``log10(resistivity)``
    over the valid cells (the same algorithm :func:`geosim.fusion.cluster_fused` uses for
    GMM clustering, doc 07 §3.3) and returns, per cell, the **posterior probability of the
    lower-resistivity component** — i.e. a data-driven "altered" class membership rather
    than a hand-set threshold (doc 07 §4.2 "Alteration (data-driven)"). Clustering-as-a-
    transform.
    """

    id = "rp.gmm_alteration_posterior"
    version = "1.0.0"
    title = "Resistivity → Alteration Posterior (data-driven GMM)"
    target = "alteration"

    assumptions = [
        "resistivity population separates into ~2 modes; the low-ρ mode = altered/clay",
        "GMM fit on log10(resistivity) over the present cells (no per-cell calibration)",
        "data-driven: the 'altered' threshold emerges from the data, not a fixed cutoff",
    ]
    calibration_status = "uncalibrated"

    inputs = [InputSpec("resistivity", unit="ohm*m", required=True)]
    output = OutputSpec(
        "alteration", unit="dimensionless", valid_range=(0.0, 1.0),
        colormap="magma", proxy_when_uncalibrated=True,
    )

    params = [
        Param("n_components", int, default=2, range=(2, 4)),
        Param("random_state", int, default=0),
    ]

    def apply(  # noqa: D401
        self,
        ctx: TransformContext,
        resistivity,
        *,
        n_components,
        random_state,
    ):
        """Fit a GMM to log10(ρ) and return the posterior of the lowest-ρ component."""
        from sklearn.mixture import GaussianMixture

        rho = np.maximum(np.asarray(resistivity, dtype=float).reshape(-1), 1e-6)
        x = np.log10(rho).reshape(-1, 1)
        k = int(n_components)
        if x.shape[0] < k:  # too few cells to fit — fall back to a flat membership
            return ctx.as_output(np.full(rho.shape, 0.5))
        gmm = GaussianMixture(n_components=k, random_state=int(random_state))
        gmm.fit(x)
        proba = gmm.predict_proba(x)  # (n, k)
        altered_class = int(np.argmin(gmm.means_.reshape(-1)))  # lowest log10(ρ) component
        return ctx.as_output(proba[:, altered_class])


register.transform(AlterationIndex())
register.transform(GmmAlterationPosterior())
