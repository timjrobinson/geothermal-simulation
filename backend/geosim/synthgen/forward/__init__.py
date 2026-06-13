"""Per-method T0 forward models (doc 05 §4 + §6 T0 tier, doc 08 §4d contract).

Every survey method has a **T0 ("plausible") forward** (doc 05 §6) implementing the
uniform :class:`ForwardModel` protocol (doc 08 §4d): a canonical ``(method, submethod)``
pair (doc 02 §2), ``fidelity="plausible"``, and ``simulate(truth, acquisition, rng) ->
list[Artifact]`` emitting native-format file(s) the *same* method's doc-03 adapter could
ingest — closing the OVERVIEW §8 round-trip. The T0 recipe is **degrade-the-truth**
(doc 05 §6): the three universal degradations (acquisition geometry, resolution/DOI,
noise — doc 05 §4) are applied in every model, with analytic potential-field voxel sums
for gravity/magnetics.

Native outputs per method (doc 05 §4 table, §5 ``measured/``):

==================  =========================================================
method/submethod    native files (format key)
==================  =========================================================
gravity             CSV stations + GeoTIFF Bouguer grid
magnetics           ``.xyz`` lines + GeoTIFF RTP grid
ert/dc_resistivity  AGI-style ``.stg`` apparent-resistivity pseudosection
ip/ip_time          co-located ``.stg`` chargeability pseudosection
em/tdem             ``.xyz`` conductivity-depth soundings (smoke-ring DOI)
mt                  one EDI per station (app-res & phase vs period)
seismic/reflection  SEG-Y section + horizons GeoJSON
microseismic        QuakeML + CSV catalog (Gutenberg-Richter)
insar               GeoTIFF LOS deformation time-series
welllog             LAS along a deviation survey + deviation CSV
heatflow            CSV temperature points (kelvin canonical, °C display)
geology             GeoJSON surface contacts + fault traces
==================  =========================================================

:data:`FORWARD_MODELS` maps the canonical ``(method, submethod)`` pair → a ready
forward instance; :func:`get_forward` looks one up; :func:`all_forwards` lists them.
"""

from __future__ import annotations

from .base import (
    Acquisition,
    Artifact,
    ForwardModel,
    Provenance,
    T0Forward,
)
from .borehole import HeatFlowForward, WellLogForward
from .electrical import ERTForward, IPForward
from .em_mt import MTForward, TDEMForward
from .potential_field import GravityForward, MagneticsForward
from .seismic import MicroseismicForward, SeismicReflectionForward
from .surface import GeologyMapForward, InSARForward

__all__ = [
    # contract + DTOs (doc 08 §4d, doc 05 §4)
    "ForwardModel",
    "T0Forward",
    "Acquisition",
    "Artifact",
    "Provenance",
    # per-method T0 forwards (doc 05 §4 table)
    "GravityForward",
    "MagneticsForward",
    "ERTForward",
    "IPForward",
    "TDEMForward",
    "MTForward",
    "SeismicReflectionForward",
    "MicroseismicForward",
    "InSARForward",
    "WellLogForward",
    "HeatFlowForward",
    "GeologyMapForward",
    # registry
    "FORWARD_MODELS",
    "get_forward",
    "all_forwards",
]


def _build_registry() -> dict[tuple[str, str | None], T0Forward]:
    models: list[T0Forward] = [
        GravityForward(),
        MagneticsForward(),
        ERTForward(),
        IPForward(),
        TDEMForward(),
        MTForward(),
        SeismicReflectionForward(),
        MicroseismicForward(),
        InSARForward(),
        WellLogForward(),
        HeatFlowForward(),
        GeologyMapForward(),
    ]
    return {(m.method, m.submethod): m for m in models}


#: Canonical ``(method, submethod)`` → T0 forward instance (doc 02 §2 keys, doc 05 §4).
FORWARD_MODELS: dict[tuple[str, str | None], T0Forward] = _build_registry()


def get_forward(method: str, submethod: str | None = None) -> T0Forward:
    """Return the T0 forward for a canonical ``(method, submethod)`` pair (doc 05 §4)."""
    try:
        return FORWARD_MODELS[(method, submethod)]
    except KeyError as e:
        raise KeyError(
            f"no T0 forward for ({method!r}, {submethod!r}); "
            f"known: {sorted(FORWARD_MODELS)}"
        ) from e


def all_forwards() -> list[T0Forward]:
    """All registered T0 forward instances (doc 05 §6: T0 for every method)."""
    return list(FORWARD_MODELS.values())
