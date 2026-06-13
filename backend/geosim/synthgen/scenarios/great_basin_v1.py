"""``great-basin-v1`` — the FLAGSHIP Basin-&-Range hydrothermal play (doc 05 §7.1).

An alluvium-filled valley over volcanics / carbonate over granite basement, cut by a 60°
range-front normal fault (~700 m throw) that is both the master structure and the fluid
conduit. The geothermal target is a fault-controlled hydrothermal upflow (~220 °C) — a
*state* perturbation (hot + saline + altered + fractured, decision #2) with a shallow
clay-cap conductor over a deep propylitically-altered fractured reservoir — yielding the
textbook joint signature (MT/EM conductor + magnetic low + gravity/seismic structure + hot
well, doc 05 §7.1, §4.2). The earth + survey plan are authored in the sibling
``great-basin-v1/scene.jsonc`` + ``acquisition.jsonc``; this module loads + registers them.
"""

from __future__ import annotations

from pathlib import Path

from ..scene import load_scene
from .acquisition_io import load_acquisition
from .registry import ScenarioSpec, register_scenario

__all__ = ["SCENARIO", "DIR"]

#: directory holding the authored ``scene.jsonc`` + ``acquisition.jsonc`` (doc 05 §5).
DIR = Path(__file__).parent / "great-basin-v1"

SCENARIO: ScenarioSpec = register_scenario(
    ScenarioSpec(
        scene=load_scene(DIR / "scene.jsonc"),
        acquisition=load_acquisition(DIR / "acquisition.jsonc"),
        title="Great Basin v1 (flagship)",
        description=(
            "Basin-&-Range extensional hydrothermal play: alluvium/volcanics/carbonate/"
            "granite basement, a 60 deg range-front normal fault (~700 m throw) acting as "
            "the fluid conduit, and a fault-controlled hydrothermal upflow (~220 C) with a "
            "shallow clay-cap conductor over a deep propylitic fractured reservoir "
            "(doc 05 §7.1)."
        ),
    )
)
