"""Synthetic data generator (doc 05).

The doc-05 generator authors **one geology** — a shared lithology field ``L`` and state
field ``S`` (doc 05 §2.1) — from a declarative :class:`SceneSpec` (doc 05 §2.3), then
*derives* every geophysical property through a named rock-physics ruleset (doc 05 §3),
guaranteeing cross-method consistency: a hot + saline + altered + porous voxel is
simultaneously more conductive, less dense, magnetically suppressed, and seismically
slower (doc 05 §1 decision #1). :func:`compile_scene` returns a :class:`TruthEarth`
holding all co-located property volumes plus ``L``/``S`` on the Engineering-coordinate
truth grid; :func:`write_truth_bundle` writes the ``truth/*.zarr`` + ``features.geojson``
scoring oracle (doc 05 §5). Everything is deterministic from ``(spec, seed)`` via
``numpy.random.SeedSequence`` sub-streams (doc 05 §1 invariant).

:func:`build_resistivity_volume` is the original minimal M1 slice — a single conductive
blob in a layered halfspace (scene ``unit-cube-v1``) — kept working for M1 callers.
"""

from .compiler import CompiledFields, compile_scene
from .resistivity import VolumeResult, build_resistivity_volume
from .rockphysics import (
    DEFAULT_UNIT_LIBRARY,
    RockPhysicsResult,
    RuleSet,
    default_v1,
    get_ruleset,
)
from .scene import (
    AnomalySpec,
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    IntrusionSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    UnitProps,
    load_scene,
    strip_jsonc,
)
from .truth import StateField, TruthEarth, write_truth_bundle

__all__ = [
    # M1 resistivity helper (kept working)
    "VolumeResult",
    "build_resistivity_volume",
    # scene spec (doc 05 §2.3)
    "SceneSpec",
    "FrameSpec",
    "SurfaceSpec",
    "LayerSpec",
    "IntrusionSpec",
    "FaultSpec",
    "GeothermSpec",
    "AnomalySpec",
    "UnitProps",
    "load_scene",
    "strip_jsonc",
    # rock physics (doc 05 §3)
    "RuleSet",
    "RockPhysicsResult",
    "DEFAULT_UNIT_LIBRARY",
    "get_ruleset",
    "default_v1",
    # compiler + truth (doc 05 §2, §5)
    "compile_scene",
    "CompiledFields",
    "TruthEarth",
    "StateField",
    "write_truth_bundle",
]
