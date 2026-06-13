"""Ingestion adapter registry — integrated with ``geosim.plugins`` (doc 03 §1, doc 08).

Doc 03 §1 specifies a ``@register`` decorator + entry-point discovery + ``detect()``
(highest ``sniff()`` score wins). Doc 08 makes the *one* registry the unified
``geosim.plugins`` ``PluginRegistry`` under the ``geosim.plugins`` entry-point group —
there is no separate adapter registry. This module is the ingestion-side surface over
that shared registry:

- :func:`adapter` — a decorator (``@adapter``) that registers a doc-03
  :class:`~geosim.ingestion.base.IngestionAdapter` into the plugins registry. Adapters
  here use the richer ``sniff(sample, filename)`` / ``parse(source)`` signatures (doc 03
  §1); the plugins ``IngestionAdapter`` Protocol uses ``sniff(raw)`` / ``parse(raw, ctx)``
  (doc 08 §4a). We register the adapter *as-is* (it conforms — it has ``method``,
  ``formats``-or-``extensions``, ``sniff``, ``parse``) and bridge the call sites here.
- :func:`adapters` / :func:`adapters_for` — lookup helpers (doc 03 §1).
- :func:`detect` — ``sniff()``-based format detection (doc 03 §7 step 3): highest score
  wins; a user ``method_hint`` overrides; ties / all-zero scores raise so the pipeline can
  surface a "choose adapter" prompt (doc 03 §7 step 3).
- :func:`discover_entry_points` — third-party plugin discovery via the shared
  ``geosim.plugins`` entry-point group (doc 03 §1, doc 08 §3.1).

The plugins registry expects ``formats: list[str]`` on an adapter (doc 08 §4a); doc-03
adapters declare ``extensions`` (``[".stg"]``). We synthesize ``formats`` from
``extensions`` (dropping the leading dot) when absent, so a doc-03 adapter registers
cleanly without the author duplicating the list.
"""

from __future__ import annotations

from geosim.plugins import get_registry

from .base import IngestionAdapter, RawSource

__all__ = [
    "adapter",
    "register_adapter",
    "adapters",
    "adapters_for",
    "adapter_named",
    "detect",
    "discover_entry_points",
    "DetectionError",
]


class DetectionError(RuntimeError):
    """No adapter (or an ambiguous tie) for a file (doc 03 §7 step 3).

    The pipeline turns this into an ``IngestReport(status=failed)`` ("choose adapter"),
    never a crash.
    """


def _ensure_formats(adapter_obj: IngestionAdapter) -> None:
    """Synthesize ``formats`` from ``extensions`` if the adapter omits it (doc 08 §4a)."""
    if getattr(adapter_obj, "formats", None):
        return
    exts = getattr(adapter_obj, "extensions", None) or []
    formats = [e[1:] if e.startswith(".") else e for e in exts]
    # set on the instance so the plugins registry (which reads .formats) sees it
    try:
        adapter_obj.formats = formats  # type: ignore[attr-defined]
    except Exception:  # frozen/slots adapter — fall back to a class-level default
        pass


def register_adapter(adapter_obj: IngestionAdapter) -> IngestionAdapter:
    """Register a doc-03 ingestion adapter into the shared plugins registry (doc 08 §4a).

    Accepts a class (instantiated) or an instance. Defaults ``submethod``/``version`` so
    a minimal author need only set ``method`` / ``name`` / ``extensions``. Canonical
    ``(method, submethod)`` + interface conformance are validated by the plugins registry,
    which **quarantines** (never raises) a bad contribution (doc 08 §8).
    """
    inst = adapter_obj() if isinstance(adapter_obj, type) else adapter_obj
    if getattr(inst, "submethod", "__missing__") == "__missing__":
        inst.submethod = None  # type: ignore[attr-defined]
    if not getattr(inst, "version", None):
        inst.version = "v1"  # type: ignore[attr-defined]
    _ensure_formats(inst)
    get_registry().register_adapter(inst)
    return adapter_obj


def adapter(adapter_obj: IngestionAdapter) -> IngestionAdapter:
    """Decorator form: ``@adapter`` on a doc-03 :class:`IngestionAdapter` class/instance.

    Mirrors ``geosim.plugins.register.adapter`` (doc 08 §3.1) but accepts the richer
    doc-03 protocol and returns its argument unchanged so it decorates in place.
    """
    register_adapter(adapter_obj)
    return adapter_obj


def adapters() -> dict[str, IngestionAdapter]:
    """All registered ingestion adapters, keyed by the plugins-registry key (doc 08 §3.2)."""
    return get_registry().adapters()  # type: ignore[return-value]


def adapters_for(method: str | None = None, ext: str | None = None) -> list[IngestionAdapter]:
    """Adapters filtered by canonical ``method`` and/or file ``ext`` (doc 03 §1).

    ``ext`` matches against each adapter's ``extensions`` (leading-dot-insensitive).
    """
    out: list[IngestionAdapter] = []
    want_ext = None if ext is None else ("." + ext.lstrip("."))
    for a in adapters().values():
        if method is not None and getattr(a, "method", None) != method:
            continue
        if want_ext is not None:
            exts = {("." + e.lstrip(".")).lower() for e in getattr(a, "extensions", [])}
            if want_ext.lower() not in exts:
                continue
        out.append(a)
    return out


def adapter_named(name: str) -> IngestionAdapter | None:
    """Look up a registered adapter by its doc-03 ``name`` (e.g. ``"ert-stg-v1"``)."""
    for a in adapters().values():
        if getattr(a, "name", None) == name:
            return a
    return None


def detect(source: RawSource) -> IngestionAdapter:
    """Pick the adapter for ``source`` by ``sniff()`` score (doc 03 §7 step 3).

    A user ``source.method_hint`` overrides detection (doc 03 §7 step 1): among adapters
    of that method, the highest sniff score wins; if none recognise it, the first adapter
    of the hinted method is used (the hint is authoritative). Otherwise the global highest
    ``sniff()`` score wins. A zero-confidence field or a *tie* at the top raises
    :class:`DetectionError` so the pipeline can prompt for an explicit adapter.
    """
    sample = source.sample()
    filename = source.filename
    candidates = list(adapters().values())
    if source.method_hint:
        hinted = [a for a in candidates if getattr(a, "method", None) == source.method_hint]
        if hinted:
            candidates = hinted

    scored: list[tuple[float, IngestionAdapter]] = []
    for a in candidates:
        try:
            score = float(a.sniff(sample, filename))
        except Exception:
            score = 0.0
        scored.append((score, a))

    scored.sort(key=lambda t: t[0], reverse=True)
    if not scored:
        raise DetectionError(f"no ingestion adapters registered for {filename!r}")

    top_score, top = scored[0]
    if top_score <= 0.0:
        # Hint is authoritative even when nothing sniffs (doc 03 §7 step 1).
        if source.method_hint and candidates:
            return candidates[0]
        raise DetectionError(
            f"no adapter recognised {filename!r} (all sniff scores 0) — choose an adapter"
        )
    if len(scored) > 1 and scored[1][0] == top_score:
        tied = [getattr(a, "name", "?") for s, a in scored if s == top_score]
        raise DetectionError(
            f"ambiguous format for {filename!r}: tie at {top_score} between {tied} — "
            "choose an adapter (doc 03 §7)"
        )
    return top


def discover_entry_points() -> None:
    """Import third-party adapter plugins via the ``geosim.plugins`` group (doc 08 §3.1).

    Delegates to the shared registry's discovery (which quarantines a failing entry point
    rather than crashing, doc 08 §8). First-party adapters self-register on import of the
    :mod:`geosim.ingestion.adapters` package (which auto-imports its siblings).
    """
    get_registry().discover_entry_points()
