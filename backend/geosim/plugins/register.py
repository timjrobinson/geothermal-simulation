"""The ``register`` decorator namespace + ``manifest`` helper (doc 08 §3.1, §5).

First-party plugins use these decorators (zero-config discovery, doc 08 §3.1); the
registration call is identical to the entry-point path — only *how the module gets
imported* differs. Each decorator routes to the process-wide ``PluginRegistry`` singleton
and returns its argument unchanged so it can decorate a class/instance in place:

    from geosim.plugins import register, PropertyType, IngestionAdapter

    register.property_type(PropertyType(key="density", canonical_unit="kg/m**3", ...))

    @register.adapter
    class GravityCSVAdapter:           # IngestionAdapter
        method = "gravity"; submethod = None; formats = ["csv"]
        def sniff(self, raw): ...
        def parse(self, raw, ctx): ...

Bad contributions are **quarantined** by the registry (doc 08 §8), never raised — so an
``@register`` decorator never crashes import of a plugin module.
"""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    ForwardModel,
    IngestionAdapter,
    InversionEngine,
    PropertyType,
    RendererSpec,
    Transform,
)
from .manifest import PluginManifest
from .registry import get_registry

__all__ = ["register", "manifest"]


class _Register:
    """The ``register.*`` namespace (doc 08 §3.1). One method per extension point."""

    def property_type(self, pt: PropertyType) -> PropertyType:
        """Register a property type (doc 08 §4b) — REUSES the doc 01 §5 registry."""
        return get_registry().register_property_type(pt)

    def adapter(self, adapter: IngestionAdapter) -> IngestionAdapter:
        """Register an ingestion adapter (doc 08 §4a). Usable as ``@register.adapter``."""
        return get_registry().register_adapter(adapter)

    def transform(self, transform: Transform) -> Transform:
        """Register a rock-physics transform (doc 08 §4c)."""
        return get_registry().register_transform(transform)

    def forward_model(self, model: ForwardModel) -> ForwardModel:
        """Register a forward model (doc 08 §4d)."""
        return get_registry().register_forward_model(model)

    def inversion_engine(self, engine: InversionEngine) -> InversionEngine:
        """Register an inversion engine (doc 08 §4f, Phase 6)."""
        return get_registry().register_inversion_engine(engine)

    def renderer(self, spec: RendererSpec) -> RendererSpec:
        """Register a renderer spec (doc 08 §4e)."""
        return get_registry().register_renderer(spec)


register = _Register()  # the singleton namespace plugins import (doc 08 §9 stable surface)


def manifest(source: str | Path | dict | PluginManifest) -> PluginManifest | None:
    """Load + validate + register a plugin manifest (doc 08 §5.1, §5.2).

    Accepts a ``plugin.json`` path, a dict, or a pre-built :class:`PluginManifest`. A
    schema/canonical/executionMode failure is quarantined by the registry (doc 08 §8) and
    returns ``None`` — it never crashes the importing plugin module.
    """
    reg = get_registry()
    try:
        if isinstance(source, PluginManifest):
            man = source
        elif isinstance(source, dict):
            man = PluginManifest.from_dict(source)
        else:
            p = Path(source)
            man = PluginManifest.from_file(p)
    except Exception as e:  # ManifestError — quarantine instead of crashing the plugin
        from .registry import QuarantineRecord

        ident = source if isinstance(source, (str, Path)) else "<inline>"
        reg._quarantine.append(QuarantineRecord(f"manifest:{ident}", str(e)))
        return None
    return reg.register_manifest(man)
