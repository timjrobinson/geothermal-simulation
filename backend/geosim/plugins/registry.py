"""The ``PluginRegistry`` singleton — the stable core surface (doc 08 §3.2, §8, §7).

Two discovery channels converge here (doc 08 §3.1): first-party decorators (zero-config)
and ``importlib.metadata`` entry points under group ``geosim.plugins`` (third-party). Both
end at one registry. Core code talks only to this interface; it never imports a concrete
plugin (doc 08 §3.2).

Load-time validation **quarantines** bad contributions (doc 08 §8): a failing manifest,
non-canonical ``(method, submethod)``, bad ``executionMode``, property-type integrity
clash, or interface non-conformance is logged + excluded — it never crashes the app.

``capabilities()`` produces the ``/api/capabilities`` document shape (doc 08 §7.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any

from geosim.spatial.property_types import REGISTRY as PROPERTY_REGISTRY
from geosim.spatial.property_types import PropertyType
from geosim.spatial.units import CANONICAL_UNITS

from .contracts import (
    ForwardModel,
    IngestionAdapter,
    InversionEngine,
    RendererSpec,
    Transform,
)
from .manifest import API_VERSION, PluginManifest
from .methods import is_canonical_pair

__all__ = [
    "PluginRegistry", "QuarantineRecord", "ENTRY_POINT_GROUP", "REGISTRY", "get_registry",
]

# importlib.metadata group for 3rd-party plugins (doc 08 §3.1)
ENTRY_POINT_GROUP = "geosim.plugins"

_log = logging.getLogger("geosim.plugins")


@dataclass(frozen=True)
class QuarantineRecord:
    """A contribution/plugin excluded at load time (doc 08 §8). Logged, not fatal."""

    contribution: str   # e.g. "adapter:gravity.csv" | "manifest:..." | "property_type:foo"
    reason: str
    plugin_id: str | None = None


class PluginRegistry:
    """In-process registry; the only surface core code touches (doc 08 §3.2)."""

    def __init__(self) -> None:
        self._adapters: dict[str, IngestionAdapter] = {}          # adapter key → adapter
        self._adapter_methods: dict[str, tuple[str, str | None]] = {}  # key → (method, submethod)
        self._formats: dict[str, str] = {}                        # format → adapter key
        self._transforms: dict[str, Transform] = {}
        self._forward_models: dict[str, ForwardModel] = {}        # method → forward model
        self._inversion_engines: dict[str, InversionEngine] = {}
        self._renderers: dict[str, RendererSpec] = {}
        self._manifests: dict[str, PluginManifest] = {}
        self._quarantine: list[QuarantineRecord] = []

    # ----------------------------------------------------------------- registration

    def register_property_type(self, pt: PropertyType) -> PropertyType:
        """Register a property type (doc 08 §4b). REUSES the doc 01 §5 registry.

        Property-type integrity (doc 08 §8): the canonical unit must exist in the doc 01
        ``pint`` registry, and a re-registration must not clash with an existing spec.
        On failure the contribution is quarantined (not raised).
        """
        contrib = f"property_type:{pt.key}"
        try:
            if not pt.categorical:
                # canonical_unit must be a known pint unit (doc 08 §8 property-type integrity).
                from geosim.spatial.units import _norm, ureg

                try:
                    ureg.Unit(_norm(pt.canonical_unit))
                except Exception as e:  # pint UndefinedUnitError etc.
                    raise ValueError(
                        f"canonical_unit {pt.canonical_unit!r} not in the doc 01 pint registry"
                    ) from e
                # If the unit registry already pins this key, it must agree (doc 08 §8 clash).
                pinned = CANONICAL_UNITS.get(pt.key)
                if pinned is not None and _norm(pinned) != _norm(pt.canonical_unit):
                    raise ValueError(
                        f"property {pt.key!r} unit {pt.canonical_unit!r} clashes with "
                        f"canonical {pinned!r}"
                    )
            PROPERTY_REGISTRY.register(pt)  # raises on key/spec clash with a different spec
        except (ValueError, KeyError) as e:
            self._quarantine.append(QuarantineRecord(contrib, str(e)))
            _log.error("quarantined %s: %s", contrib, e)
            return pt
        return pt

    def register_adapter(self, adapter: IngestionAdapter) -> IngestionAdapter:
        """Register an ingestion adapter (doc 08 §4a). Quarantines on validation failure."""
        method = getattr(adapter, "method", None)
        submethod = getattr(adapter, "submethod", None)
        formats = getattr(adapter, "formats", None)
        key = f"{method}.{formats[0]}" if formats else f"{method}.adapter"
        contrib = f"adapter:{key}"
        # Interface conformance (doc 08 §8): required attrs + Protocol methods.
        if method is None or formats is None or not callable(getattr(adapter, "parse", None)) \
                or not callable(getattr(adapter, "sniff", None)):
            self._quarantine.append(
                QuarantineRecord(contrib, "does not conform to IngestionAdapter")
            )
            _log.error("quarantined %s: interface non-conformance", contrib)
            return adapter
        # Canonical (method, submethod) — no invented variants (doc 02 §2 / doc 08 §8).
        if not is_canonical_pair(method, submethod):
            self._quarantine.append(
                QuarantineRecord(
                    contrib, f"non-canonical (method, submethod)=({method!r},{submethod!r})"
                )
            )
            _log.error("quarantined %s: non-canonical method/submethod", contrib)
            return adapter
        self._adapters[key] = adapter
        self._adapter_methods[key] = (method, submethod)
        for fmt in formats:
            # Key uniqueness (doc 08 §8): deterministic — first registrant keeps the format.
            self._formats.setdefault(fmt, key)
        return adapter

    def register_transform(self, transform: Transform) -> Transform:
        """Register a rock-physics transform (doc 08 §4c)."""
        key = getattr(transform, "key", None)
        contrib = f"transform:{key}"
        if key is None or not callable(getattr(transform, "apply", None)):
            self._quarantine.append(QuarantineRecord(contrib, "does not conform to Transform"))
            _log.error("quarantined %s: interface non-conformance", contrib)
            return transform
        self._transforms[key] = transform
        return transform

    def register_forward_model(self, model: ForwardModel) -> ForwardModel:
        """Register a forward model (doc 08 §4d). Keyed by canonical method (doc 02 §2)."""
        method = getattr(model, "method", None)
        submethod = getattr(model, "submethod", None)
        contrib = f"forward:{method}"
        if method is None or not callable(getattr(model, "simulate", None)):
            self._quarantine.append(QuarantineRecord(contrib, "does not conform to ForwardModel"))
            _log.error("quarantined %s: interface non-conformance", contrib)
            return model
        if not is_canonical_pair(method, submethod):
            self._quarantine.append(
                QuarantineRecord(
                    contrib, f"non-canonical (method, submethod)=({method!r},{submethod!r})"
                )
            )
            _log.error("quarantined %s: non-canonical method/submethod", contrib)
            return model
        self._forward_models[method] = model
        return model

    def register_inversion_engine(self, engine: InversionEngine) -> InversionEngine:
        """Register an inversion engine (doc 08 §4f, Phase 6)."""
        key = getattr(engine, "key", None)
        contrib = f"inversion:{key}"
        if key is None or not callable(getattr(engine, "invert", None)):
            self._quarantine.append(
                QuarantineRecord(contrib, "does not conform to InversionEngine")
            )
            _log.error("quarantined %s: interface non-conformance", contrib)
            return engine
        self._inversion_engines[key] = engine
        return engine

    def register_renderer(self, spec: RendererSpec) -> RendererSpec:
        """Register a renderer spec (doc 08 §4e). Declarative; impl is frontend (doc 08 §7.2)."""
        contrib = f"renderer:{getattr(spec, 'key', None)}"
        if not isinstance(spec, RendererSpec):
            self._quarantine.append(QuarantineRecord(contrib, "not a RendererSpec"))
            _log.error("quarantined %s: not a RendererSpec", contrib)
            return spec
        self._renderers[spec.key] = spec
        return spec

    def register_manifest(self, manifest: PluginManifest) -> PluginManifest | None:
        """Register a validated manifest (doc 08 §5.1). Quarantines an invalid one.

        Accepts a pre-built (validated) :class:`PluginManifest`. ``from_dict``/``from_file``
        already validate; if a manifest somehow re-fails ``validate`` here it is quarantined
        rather than raised (doc 08 §8).
        """
        try:
            manifest.validate()
        except Exception as e:  # ManifestError
            self._quarantine.append(
                QuarantineRecord(f"manifest:{getattr(manifest, 'id', '?')}", str(e),
                                 plugin_id=getattr(manifest, "id", None))
            )
            _log.error("quarantined manifest %s: %s", getattr(manifest, "id", "?"), e)
            return None
        self._manifests[manifest.id] = manifest
        return manifest

    # --------------------------------------------------------------- core query API (§3.2)

    def adapters(self) -> dict[str, IngestionAdapter]:
        return dict(self._adapters)

    def adapter_for_format(self, fmt: str) -> IngestionAdapter | None:
        key = self._formats.get(fmt)
        return self._adapters.get(key) if key else None

    def property_type(self, key: str) -> PropertyType:
        return PROPERTY_REGISTRY.get(key)

    def transforms(self) -> list[Transform]:
        return list(self._transforms.values())

    def forward_model(self, method: str) -> ForwardModel | None:
        return self._forward_models.get(method)

    def inversion_engines(self) -> list[InversionEngine]:
        return list(self._inversion_engines.values())

    def renderer_specs(self) -> list[RendererSpec]:
        return list(self._renderers.values())

    def manifest(self, plugin_id: str) -> PluginManifest:
        return self._manifests[plugin_id]

    def manifests(self) -> list[PluginManifest]:
        return list(self._manifests.values())

    def quarantined(self) -> list[QuarantineRecord]:
        """Contributions excluded at load (doc 08 §8) — surfaced at /api/plugins."""
        return list(self._quarantine)

    # ------------------------------------------------------------- capabilities (§7.1)

    def capabilities(self) -> dict[str, Any]:
        """The ``/api/capabilities`` document (doc 08 §7.1).

        Keys: ``api_version``, ``property_types``, ``methods``, ``renderers``,
        ``transforms``, ``plugins`` — the single backend→frontend contract that makes the
        UI method-agnostic. Property types flow straight from the doc 01 §5 registry.
        """
        property_types = [
            {
                "key": pt.key,
                "unit": pt.canonical_unit,
                "colormap": pt.default_colormap,
                "scaling": pt.default_scaling,
                "display_range": list(pt.display_range) if pt.display_range is not None else None,
            }
            for pt in PROPERTY_REGISTRY.all()
        ]
        # Methods: one entry per registered adapter's (method) — formats + produces + fwd flag.
        methods: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key, adapter in self._adapters.items():
            method, _submethod = self._adapter_methods[key]
            if method in seen:
                continue
            seen.add(method)
            man = next((m for m in self._manifests.values() if m.method == method), None)
            produces = list(man.provides.get("property_types", [])) if man else []
            methods.append(
                {
                    "id": method,
                    "name": man.name if man else method,
                    "formats": list(getattr(adapter, "formats", [])),
                    "produces": produces,
                    "has_forward_model": method in self._forward_models,
                }
            )
        renderers = [spec.to_dict() for spec in self._renderers.values()]
        transforms = [
            {
                "key": t.key,
                "inputs": list(getattr(t, "inputs", [])),
                "outputs": list(getattr(t, "outputs", [])),
            }
            for t in self._transforms.values()
        ]
        plugins = [{"id": m.id, "version": m.version} for m in self._manifests.values()]
        return {
            "api_version": API_VERSION,
            "property_types": property_types,
            "methods": methods,
            "renderers": renderers,
            "transforms": transforms,
            "plugins": plugins,
        }

    # ----------------------------------------------------- third-party discovery (§3.1)

    def discover_entry_points(self, group: str = ENTRY_POINT_GROUP) -> None:
        """Enumerate + import 3rd-party plugins from ``importlib.metadata`` entry points.

        Each advertised module is imported (running its ``@register`` decorators). An
        import failure quarantines that entry point (doc 08 §8) rather than crashing.
        """
        try:
            eps = metadata.entry_points(group=group)
        except TypeError:  # pragma: no cover — very old importlib.metadata API
            eps = metadata.entry_points().get(group, [])  # type: ignore[attr-defined]
        for ep in eps:
            try:
                ep.load() if hasattr(ep, "load") else import_module(ep.value)
            except Exception as e:  # any import-time failure
                self._quarantine.append(
                    QuarantineRecord(f"entry_point:{getattr(ep, 'name', '?')}", str(e))
                )
                _log.error("quarantined entry point %s: %s", getattr(ep, "name", "?"), e)


REGISTRY = PluginRegistry()  # the process-wide singleton (doc 08 §3.1)


def get_registry() -> PluginRegistry:
    """Return the process-wide :class:`PluginRegistry` singleton (doc 08 §3.1)."""
    return REGISTRY
