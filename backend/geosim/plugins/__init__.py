"""Plugin architecture & extensibility framework (doc 08).

The *single* extensibility mechanism for the whole stack: one ``PluginRegistry`` and six
extension points (adapter, property type, transform, forward model, renderer, inversion
engine — doc 08 §1). A new survey method is one plugin package + a manifest, with **no
core changes** (doc 08 decision #1).

This package is the **only stable surface a plugin may import** (doc 08 §9): the six
Protocols/specs, ``PropertyType`` (REUSED from ``geosim.spatial`` — doc 01 §5),
``RendererSpec``/``TransferFunction``/``Transform``, the ``register``/``manifest`` helpers,
and ``PluginManifest``/``ExecutionMode``. Everything else in core is private.

Discovery is hybrid (doc 08 §3.1): first-party decorators + ``importlib.metadata`` entry
points (group ``geosim.plugins``), both converging on the one registry. Load-time
validation quarantines bad contributions without crashing (doc 08 §8).
"""

from __future__ import annotations

from .contracts import (
    ExecutionMode,
    ForwardModel,
    IngestionAdapter,
    InversionEngine,
    PropertyType,
    RendererSpec,
    TransferFunction,
    Transform,
)
from .manifest import (
    API_VERSION,
    SUPPORTED_API_RANGE,
    ManifestError,
    PluginManifest,
    api_version_compatible,
)
from .methods import (
    METHOD_KEYS,
    SUBMETHODS,
    is_canonical_method,
    is_canonical_pair,
)
from .register import manifest, register
from .registry import (
    ENTRY_POINT_GROUP,
    REGISTRY,
    PluginRegistry,
    QuarantineRecord,
    get_registry,
)

__all__ = [
    # extension-point contracts (doc 08 §4)
    "IngestionAdapter", "PropertyType", "Transform", "ForwardModel",
    "RendererSpec", "TransferFunction", "InversionEngine",
    "ExecutionMode",
    # registration surface (doc 08 §3.1)
    "register", "manifest",
    # manifest (doc 08 §5.1, §8, §9)
    "PluginManifest", "ManifestError",
    "API_VERSION", "SUPPORTED_API_RANGE", "api_version_compatible",
    # registry (doc 08 §3.2, §7, §8)
    "PluginRegistry", "REGISTRY", "get_registry", "QuarantineRecord", "ENTRY_POINT_GROUP",
    # canonical method registry (doc 02 §2)
    "METHOD_KEYS", "SUBMETHODS", "is_canonical_method", "is_canonical_pair",
]
