"""PluginManifest dataclass + load-time schema validation (doc 08 §5.1, §8, §9).

A **method bundle** (doc 08 §5) declares a manifest — in code or as ``plugin.json`` —
validated at load. Validation catches *bugs* (not attacks; trust is global, doc 08 §2):
manifest schema, API-version compatibility, canonical ``(method, submethod)`` (doc 02 §2),
and per-contribution ``executionMode`` validity (doc 08 §2.1). A failing manifest is
**quarantined** by the registry (doc 08 §8) — logged + excluded, never crashes the app.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import ExecutionMode
from .methods import is_canonical_pair

__all__ = [
    "API_VERSION",
    "SUPPORTED_API_RANGE",
    "PluginManifest",
    "ManifestError",
    "api_version_compatible",
]

# Core plugin-API contract version this build implements (doc 08 §9). Plugins target a
# range like "1.x"; a "1.*" plugin runs on any core "1.*".
API_VERSION = "1.0"
SUPPORTED_API_RANGE = "1.x"  # core supports the whole 1.* major (doc 08 §9 compat policy)

_KINDS = {"method-bundle", "single-contribution"}
_PROVIDES_GROUPS = (
    "adapters", "property_types", "transforms",
    "forward_models", "renderers", "inversion_engines",
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # reverse-DNS-ish id (doc 08 §5.1)


class ManifestError(ValueError):
    """A manifest failed schema / canonical / executionMode validation (doc 08 §8)."""


def _major(api_version: str) -> str:
    """Major component of an ``api_version`` like ``"1.x"`` / ``"1.0"`` / ``"1.2.3"``."""
    return api_version.split(".", 1)[0].strip()


def api_version_compatible(manifest_api: str, supported: str = SUPPORTED_API_RANGE) -> bool:
    """True iff a plugin targeting ``manifest_api`` runs on core ``supported`` (doc 08 §9).

    Same-major compatibility: a ``1.x`` plugin runs on any core ``1.*``.
    """
    try:
        return _major(manifest_api) == _major(supported) and _major(manifest_api).isdigit()
    except (AttributeError, IndexError):
        return False


@dataclass(frozen=True)
class PluginManifest:
    """A method-bundle / single-contribution manifest (doc 08 §5.1).

    Construct via :meth:`from_dict` / :meth:`from_file` so the JSON-schema-ish
    validation in :meth:`validate` runs and raises :class:`ManifestError` on a bad
    manifest — which the registry turns into a quarantine (doc 08 §8).
    """

    id: str                              # globally unique, reverse-DNS-ish (doc 08 §5.1)
    name: str
    version: str                         # semver of THIS plugin (provenance, doc 08 §6)
    api_version: str                     # core contract it targets (doc 08 §9)
    kind: str                            # "method-bundle" | "single-contribution"
    method: str                          # canonical MethodKey (doc 02 §2)
    submethod: str | None = None         # canonical submethod (doc 02 §2) or None
    provides: dict[str, list[str]] = field(default_factory=dict)
    # per-contribution executionMode (doc 08 §2.1); default in_process when omitted.
    execution_modes: dict[str, str] = field(default_factory=dict)
    requires_property_types: list[str] = field(default_factory=list)
    python_requires: str = ">=3.11"
    dependencies: list[str] = field(default_factory=list)

    # ----------------------------------------------------------------- constructors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        """Build + validate a manifest from a dict. Raises :class:`ManifestError`."""
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        required = ("id", "name", "version", "api_version", "kind", "method")
        missing = [k for k in required if k not in data or data[k] in (None, "")]
        if missing:
            raise ManifestError(f"manifest missing required field(s): {missing}")
        try:
            man = cls(
                id=str(data["id"]),
                name=str(data["name"]),
                version=str(data["version"]),
                api_version=str(data["api_version"]),
                kind=str(data["kind"]),
                method=str(data["method"]),
                submethod=data.get("submethod"),
                provides=dict(data.get("provides") or {}),
                execution_modes=dict(data.get("execution_modes") or {}),
                requires_property_types=list(data.get("requires_property_types") or []),
                python_requires=str(data.get("python_requires", ">=3.11")),
                dependencies=list(data.get("dependencies") or []),
            )
        except (TypeError, ValueError) as e:
            raise ManifestError(f"manifest field has wrong type: {e}") from e
        man.validate()
        return man

    @classmethod
    def from_file(cls, path: str | Path) -> PluginManifest:
        """Load + validate a ``plugin.json`` manifest (doc 08 §5.2)."""
        p = Path(path)
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ManifestError(f"cannot read/parse manifest {p}: {e}") from e
        return cls.from_dict(data)

    # ------------------------------------------------------------------ validation

    def validate(self) -> None:
        """JSON-schema-ish validation (doc 08 §8). Raises :class:`ManifestError`.

        Checks: id shape, kind enum, API-version compatibility, canonical
        ``(method, submethod)`` (doc 02 §2), ``provides`` shape, and every declared
        ``executionMode`` ∈ the canonical set (doc 08 §2.1).
        """
        if not _ID_RE.match(self.id):
            raise ManifestError(f"invalid plugin id {self.id!r} (reverse-DNS-ish required)")
        if self.kind not in _KINDS:
            raise ManifestError(f"invalid kind {self.kind!r}; expected one of {sorted(_KINDS)}")
        if not api_version_compatible(self.api_version):
            raise ManifestError(
                f"api_version {self.api_version!r} incompatible with core {SUPPORTED_API_RANGE!r}"
            )
        # Canonical (method, submethod) — no invented variants (doc 02 §2 / doc 08 §8).
        if not is_canonical_pair(self.method, self.submethod):
            raise ManifestError(
                f"non-canonical (method, submethod) = ({self.method!r}, {self.submethod!r}); "
                "must be a pair from the doc 02 §2 registry"
            )
        # provides: each present group must be a list of strings.
        for group, vals in self.provides.items():
            if group not in _PROVIDES_GROUPS:
                raise ManifestError(f"unknown provides group {group!r}")
            if not isinstance(vals, list) or any(not isinstance(v, str) for v in vals):
                raise ManifestError(f"provides[{group!r}] must be a list of strings")
        # executionMode validity (doc 08 §2.1) — coerce raises ValueError on a bad mode.
        for contrib, mode in self.execution_modes.items():
            try:
                ExecutionMode.coerce(mode)
            except ValueError as e:
                raise ManifestError(
                    f"invalid executionMode {mode!r} for {contrib!r}; "
                    f"expected one of {[m.value for m in ExecutionMode]}"
                ) from e

    def execution_mode(self, contribution_key: str) -> ExecutionMode:
        """ExecutionMode for a contribution key (e.g. ``"adapter:mt.edi"``).

        Defaults to ``IN_PROCESS`` when not declared (doc 08 §2.1).
        """
        return ExecutionMode.coerce(self.execution_modes.get(contribution_key))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "kind": self.kind,
            "method": self.method,
            "submethod": self.submethod,
            "provides": {g: list(v) for g, v in self.provides.items()},
            "execution_modes": dict(self.execution_modes),
            "requires_property_types": list(self.requires_property_types),
            "python_requires": self.python_requires,
            "dependencies": list(self.dependencies),
        }
