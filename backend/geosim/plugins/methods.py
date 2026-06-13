"""Canonical (method, submethod) registry (doc 02 §2, bound by doc 08 §1).

Plugins do **not** mint method strings. They register ``(method, submethod)`` pairs
drawn from the canonical ``MethodKey`` + ``submethod`` registry owned by doc 02 §2 —
never invented variants like ``"seismic_reflection"``. Subtypes live in the optional
``submethod`` field, not in new top-level keys.

Load-time validation (doc 08 §8) quarantines any contribution whose ``(method,
submethod)`` is not a canonical pair from this module.
"""

from __future__ import annotations

__all__ = ["METHOD_KEYS", "SUBMETHODS", "is_canonical_method", "is_canonical_pair"]

# CANONICAL MethodKey set (doc 02 §2, lines 114-116). Every doc keys on THESE.
METHOD_KEYS: frozenset[str] = frozenset(
    {
        "gravity", "magnetics", "ert", "ip", "em", "mt", "seismic", "microseismic",
        "insar", "welllog", "heatflow", "geology", "geochem",
        "derived", "fused", "synthetic",
    }
)

# Canonical submethod values per MethodKey (doc 02 §2, lines 118-122). A method absent
# here admits only ``submethod=None``; a method present here admits None or a listed value.
SUBMETHODS: dict[str, frozenset[str]] = {
    "seismic": frozenset({"reflection", "refraction", "ambient_noise", "tomography"}),
    "em": frozenset({"tdem", "fdem", "aem"}),
    "ert": frozenset({"dc_resistivity", "ip_time", "ip_freq"}),
    "ip": frozenset({"dc_resistivity", "ip_time", "ip_freq"}),
}


def is_canonical_method(method: str) -> bool:
    """True iff ``method`` is a canonical ``MethodKey`` (doc 02 §2)."""
    return method in METHOD_KEYS


def is_canonical_pair(method: str, submethod: str | None) -> bool:
    """True iff ``(method, submethod)`` is a canonical pair (doc 02 §2).

    ``submethod=None`` is always valid for a canonical method. A non-null submethod
    must appear in that method's canonical ``SUBMETHODS`` set.
    """
    if method not in METHOD_KEYS:
        return False
    if submethod is None:
        return True
    return submethod in SUBMETHODS.get(method, frozenset())
