"""Scenario registry — name → (:class:`SceneSpec`, :class:`Acquisition`) (doc 05 §7).

A :class:`ScenarioSpec` binds an authored earth (the declarative
:class:`~geosim.synthgen.scene.SceneSpec`, doc 05 §2.3) to its survey plan (the
:class:`~geosim.synthgen.forward.Acquisition`, doc 05 §4.3) under a stable id. Scenario
modules call :func:`register_scenario` at import time; :func:`get_scenario` /
:func:`list_scenarios` look them up for the CLI / :func:`build_scenario`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..forward import Acquisition
from ..scene import SceneSpec

__all__ = [
    "ScenarioSpec",
    "SCENARIOS",
    "register_scenario",
    "get_scenario",
    "list_scenarios",
]


@dataclass(frozen=True)
class ScenarioSpec:
    """A named scenario: its earth + acquisition plan (doc 05 §5, §7).

    ``scene`` is the declarative earth; ``acquisition`` is what gets collected over it
    (doc 05 §4.3). ``title``/``description`` are human-readable provenance for the
    manifest. The scenario ``id`` is ``scene.id`` (doc 05 §5 ``scenarios/<id>/``).
    """

    scene: SceneSpec
    acquisition: Acquisition
    title: str = ""
    description: str = ""

    @property
    def id(self) -> str:
        return self.scene.id


#: id → :class:`ScenarioSpec` (populated by the scenario modules' registration, doc 05 §7).
SCENARIOS: dict[str, ScenarioSpec] = {}


def register_scenario(spec: ScenarioSpec) -> ScenarioSpec:
    """Register ``spec`` under ``spec.id`` in :data:`SCENARIOS` (doc 05 §7)."""
    SCENARIOS[spec.id] = spec
    return spec


def get_scenario(scenario_id: str) -> ScenarioSpec:
    """Return the registered :class:`ScenarioSpec` for ``scenario_id`` (doc 05 §7)."""
    try:
        return SCENARIOS[scenario_id]
    except KeyError as e:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; known: {sorted(SCENARIOS)}"
        ) from e


def list_scenarios() -> list[str]:
    """Sorted ids of all registered scenarios (doc 05 §7)."""
    return sorted(SCENARIOS)
