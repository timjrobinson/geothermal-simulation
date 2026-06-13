"""Rock-physics transform library — the FULL doc 07 §4.2 starter table.

Each module in this package implements one **target family** of the starter taxonomy
(doc 07 §4.2), as one or more :class:`~geosim.fusion.transform.Transform` subclasses that
**self-register** via :func:`geosim.plugins.register.transform` (doc 08 §4c). Importing
this package registers the whole library (zero-config discovery, doc 08 §3.1).

Families (doc 07 §4.2 — one module per family):

- :mod:`.temperature` — ``resistivity_to_temperature`` (Archie + Arps fluid-conductivity
  vs T, kelvin out; uncalibrated ⇒ "temperature likelihood", §4.1 worked example).
- :mod:`.fluid` — ``archie_saturation`` (Archie Sw); ``dual_water`` / ``waxman_smits``
  (clay-surface-conduction-corrected Sw, doc 07 §4.2 row "Fluid / clay-conduction").
- :mod:`.porosity` — ``velocity_to_porosity`` (Wyllie time-average / Raymer-Hunt-Gardner);
  ``density_to_porosity`` (matrix/fluid density mixing).
- :mod:`.alteration` — ``alteration_index`` (low-ρ ∧ structure proxy → clay/alteration cap);
  ``gmm_alteration_posterior`` (data-driven, wraps :func:`geosim.fusion.cluster_fused`).
- :mod:`.fracture` — ``microseismic_density`` (KDE of an event cloud → smoothed fracture
  density volume → permeability proxy); ``vp_vs_fracture_proxy`` (Vp/Vs anomaly → index).
- :mod:`.permeability` — ``fracture_to_permeability`` (heuristic relative-perm index,
  flagged low-confidence / proxy).

**Honesty (doc 07 §4.1, §4.2).** Every transform declares ``assumptions`` and
``calibration_status='uncalibrated'`` (these are site-calibratable approximations, not
universal truths); every param (porosity, cementation exponent, salinity, matrix density,
fluid velocity …) is **first-class and user-tunable**; canonical units are honoured end to
end — **temperature is KELVIN** (doc 01 §5), permeability is m² (SI). Until calibrated each
output is a likelihood/proxy field (the harness retitles + stamps ``tier='proxy'``).
"""

from __future__ import annotations

from geosim.plugins import register
from geosim.spatial import REGISTRY as _REGISTRY
from geosim.spatial import PropertyType

# ──────────────────────────────────────────────────────────────────────────
# Output / input property types these transforms need that the doc 01 §5 core
# registry does not yet seed (doc 08 §4b — a transform family declares its
# properties once). Registered idempotently; a clashing re-register is
# quarantined by the registry, never raised (doc 08 §8).
# ──────────────────────────────────────────────────────────────────────────

_DERIVED_PROPERTY_TYPES = [
    # clay / shale volume (input to clay-corrected saturation + alteration), fraction 0..1.
    PropertyType(
        "clay_volume", "dimensionless", "YlOrBr", "linear", (0.0, 1.0), "linear",
        description="clay / shale volume fraction (Vsh)",
    ),
    # conductive-alteration / clay-cap index (smectite ⇒ low ρ), proxy fraction 0..1.
    PropertyType(
        "alteration", "dimensionless", "magma", "linear", (0.0, 1.0), "linear",
        description="hydrothermal-alteration (clay-cap) likelihood index",
    ),
    # intrinsic permeability — SI m² (1 mD ≈ 9.869e-16 m²). Spans many orders → log space.
    PropertyType(
        "permeability", "m**2", "viridis", "log", (1e-18, 1e-11), "log10",
        default_rel_sigma=0.5,
        description="intrinsic permeability proxy (SI m²; ~1 mD = 9.869e-16 m²)",
    ),
    # microseismic event count per fused cell (input to the KDE fracture-density transform).
    PropertyType(
        "microseismic", "dimensionless", "hot", "linear", (0.0, 100.0), "linear",
        description="microseismic event count per cell (binned event cloud)",
    ),
]

for _pt in _DERIVED_PROPERTY_TYPES:
    if _pt.key not in _REGISTRY:
        register.property_type(_pt)

# ──────────────────────────────────────────────────────────────────────────
# Import each family module: importing it runs its ``register.transform(...)``
# calls at module scope, so the whole §4.2 library is registered on import.
# ──────────────────────────────────────────────────────────────────────────

from . import alteration, fluid, fracture, permeability, porosity, temperature  # noqa: E402
from .alteration import AlterationIndex, GmmAlterationPosterior  # noqa: E402
from .fluid import ArchieSaturation, DualWaterSaturation, WaxmanSmitsSaturation  # noqa: E402
from .fracture import MicroseismicDensity, VpVsFractureProxy  # noqa: E402
from .permeability import FractureToPermeability  # noqa: E402
from .porosity import DensityToPorosity, VelocityToPorosity  # noqa: E402
from .temperature import ResistivityToTemperature  # noqa: E402

__all__ = [
    # families (doc 07 §4.2)
    "temperature",
    "fluid",
    "porosity",
    "alteration",
    "fracture",
    "permeability",
    # temperature
    "ResistivityToTemperature",
    # fluid
    "ArchieSaturation",
    "DualWaterSaturation",
    "WaxmanSmitsSaturation",
    # porosity
    "VelocityToPorosity",
    "DensityToPorosity",
    # alteration
    "AlterationIndex",
    "GmmAlterationPosterior",
    # fracture
    "MicroseismicDensity",
    "VpVsFractureProxy",
    # permeability
    "FractureToPermeability",
]
