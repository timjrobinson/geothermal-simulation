"""Shippable scenarios + the ``build_scenario`` driver (doc 05 §5 outputs, §7 scenarios).

A *scenario* is a named, self-contained synthetic survey play: a declarative
:class:`~geosim.synthgen.scene.SceneSpec` (the earth, doc 05 §2.3) plus an
:class:`~geosim.synthgen.forward.Acquisition` (what gets collected, doc 05 §4.3). This
package ships two canonical scenarios (doc 05 §7):

- :data:`unit_cube_v1` — a single conductive cube in a layered halfspace, the CI smoke
  test (doc 05 §7 row 1); coarse + tiny so a build runs in seconds.
- :data:`great_basin_v1` — the **flagship** Basin-&-Range extensional hydrothermal play
  (doc 05 §7.1): alluvium / volcanics / carbonate / granite basement, a 60° range-front
  normal fault (~700 m throw) acting as the fluid conduit, and a fault-controlled
  hydrothermal upflow (~220 °C) with a shallow clay-cap conductor over a deep
  propylitically-altered fractured reservoir.

:func:`build_scenario` compiles the earth (:func:`~geosim.synthgen.compiler.compile_scene`),
runs **every** registered T0 forward (:data:`~geosim.synthgen.forward.FORWARD_MODELS`) onto
it, and writes the doc 05 §5 self-contained scenario folder::

    scenarios/<id>/
      scene.jsonc            # the authored spec (provenance input)
      acquisition.jsonc      # the survey plan (doc 05 §4.3)
      frame.json             # the scenario SpatialFrame (doc 01 §2)
      measured/              # native-format files — the ONLY thing ingestion reads
      truth/                 # ground-truth zarr + features — NEVER ingested (scoring)
      manifest.json          # seed, versions, per-file checksums + provenance

Every measured file carries synthetic provenance (``source="synthgen"``, scene id, seed —
doc 05 §5) via the forward's :class:`~geosim.synthgen.forward.Provenance` stamp, recorded
in ``manifest.json`` so a measured file is never mistaken for real instrument data.
"""

from __future__ import annotations

# Importing the scenario modules registers them in SCENARIOS (doc 05 §7).
from . import great_basin_v1 as _great_basin_v1  # noqa: F401  (registration side effect)
from . import unit_cube_v1 as _unit_cube_v1  # noqa: F401  (registration side effect)
from .builder import BuildResult, build_scenario
from .registry import (
    SCENARIOS,
    ScenarioSpec,
    get_scenario,
    list_scenarios,
)

__all__ = [
    "ScenarioSpec",
    "SCENARIOS",
    "get_scenario",
    "list_scenarios",
    "build_scenario",
    "BuildResult",
]
