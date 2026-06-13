"""Synthetic data generator — minimal M1 slice (doc 05).

The full doc-05 generator authors a shared lithology/state field and *derives* every
property through rock-physics (doc 05 §2–§3). M1 needs only a single, deterministic,
seedable **resistivity volume** as the geothermal-anomaly stand-in: a layered halfspace
(plausible background 100–500 Ω·m) with a **conductive blob** embedded (low resistivity,
5–20 Ω·m — the hot/saline/altered signature, doc 05 §2.2 table, scene ``unit-cube-v1``),
plus a co-registered 1σ array (doc 02 §6).

Everything is reproducible from ``(shape, spacing, origin, seed)`` (doc 05 §1 invariant
"deterministic + seedable"): the noise realization is a seeded ``numpy`` sub-stream so
re-runs are byte-identical.
"""

from .resistivity import VolumeResult, build_resistivity_volume

__all__ = ["VolumeResult", "build_resistivity_volume"]
