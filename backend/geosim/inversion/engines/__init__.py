"""Concrete inversion engines (doc 10 §8, §9) — self-registering plugin contributions.

Each module here is an :class:`~geosim.inversion.engine.InversionEngine` implementation that
builds its solver containers (SimPEG ``Survey``/``Data``, PyGIMLi managers) **internally**
from the engine-agnostic :class:`~geosim.inversion.engine.InversionContext` and never leaks
those types across the plugin boundary (doc 10 §8). Importing a module registers its engine
on the process-wide :class:`~geosim.plugins.PluginRegistry` (doc 08 §4f) via the
module-level :func:`~geosim.inversion.engine.register_inversion_engine` call, exactly like
the in-framework :class:`~geosim.inversion.mock.MockLinearEngine`.

- :mod:`.ert_pygimli` — :class:`PygimliERTInversion`: PyGIMLi ERT (dipole-dipole
  apparent-resistivity pseudosection → resistivity), the first geothermally-meaningful,
  local-feasible engine (doc 10 §9). Heavy PyGIMLi imports live *inside* the module so
  importing this package stays cheap when no ERT inversion is requested.
"""

from __future__ import annotations

__all__: list[str] = []
