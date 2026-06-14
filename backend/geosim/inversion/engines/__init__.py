"""Concrete inversion engines (doc 10 §8, §9) — self-registering plugin contributions.

Each module here is an :class:`~geosim.inversion.engine.InversionEngine` implementation that
builds its solver containers (SimPEG ``Survey``/``Data``, PyGIMLi managers) **internally**
from the engine-agnostic :class:`~geosim.inversion.engine.InversionContext` and never leaks
those types across the plugin boundary (doc 10 §8). Importing a module registers its engine
on the process-wide :class:`~geosim.plugins.PluginRegistry` (doc 08 §4f) via the
module-level :func:`~geosim.inversion.engine.register_inversion_engine` call, exactly like
the in-framework :class:`~geosim.inversion.mock.MockLinearEngine`.

This ``__init__`` **auto-imports every sibling module** (``pkgutil.iter_modules`` over the
package dir) so importing :mod:`geosim.inversion.engines` is enough to register the whole
palette — a new engine self-registers by simply *dropping a module in this package*, with
**no shared-file edit** (the doc-10 §9 / doc-03 §9 "one file adds an engine" contract). This
keeps parallel contributors decoupled: each lands an independent self-registering module.

- :mod:`.gravity_simpeg` — :class:`SimpegGravityInversion`: SimPEG linear gravity → density
  (doc 10 §8, §9), the plumbing-proof engine.
- :mod:`.ert_pygimli` — :class:`PygimliERTInversion`: PyGIMLi ERT (dipole-dipole
  apparent-resistivity pseudosection → resistivity), the first geothermally-meaningful,
  local-feasible engine (doc 10 §9). Heavy SimPEG/PyGIMLi imports live *inside* each module
  so importing this package stays cheap when no inversion is requested.
"""

from __future__ import annotations

import importlib
import pkgutil

__all__: list[str] = []


def _autoimport_siblings() -> None:
    """Import every sibling engine module so its ``register_inversion_engine`` call runs."""
    package = __name__
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{mod.name}")


_autoimport_siblings()
