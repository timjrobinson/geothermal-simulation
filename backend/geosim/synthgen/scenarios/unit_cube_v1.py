"""``unit-cube-v1`` — the CI smoke scenario (doc 05 §7 row 1).

A single conductive cube (a compact ``young_intrusive`` stock + a co-located
hydrothermal-plume state perturbation, decision #2) in a two-layer alluvium/granite
halfspace. Small + coarse so a full earth-compile + all-forward build runs in seconds,
giving the round-trip + scoring smoke asserts of doc 05 §7. The earth and survey plan are
authored in the sibling ``unit-cube-v1/scene.jsonc`` + ``acquisition.jsonc`` (doc 05 §2.3,
§4.3); this module loads them and registers the :class:`ScenarioSpec`.
"""

from __future__ import annotations

from pathlib import Path

from ..scene import load_scene
from .acquisition_io import load_acquisition
from .registry import ScenarioSpec, register_scenario

__all__ = ["SCENARIO", "DIR"]

#: directory holding the authored ``scene.jsonc`` + ``acquisition.jsonc`` (doc 05 §5).
DIR = Path(__file__).parent / "unit-cube-v1"

SCENARIO: ScenarioSpec = register_scenario(
    ScenarioSpec(
        scene=load_scene(DIR / "scene.jsonc"),
        acquisition=load_acquisition(DIR / "acquisition.jsonc"),
        title="Unit Cube v1 (CI smoke)",
        description=(
            "Single conductive cube in a layered halfspace — the doc 05 §7 round-trip "
            "and scoring smoke test; small + coarse so it builds in seconds."
        ),
    )
)
