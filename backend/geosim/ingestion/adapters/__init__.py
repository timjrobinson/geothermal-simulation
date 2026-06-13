"""First-party ingestion adapters — auto-imported, self-registering (doc 03 §1, §9).

Drop a new adapter module in this package and it registers itself on import: this
``__init__`` walks every sibling ``.py`` module and imports it, so each module's
``@adapter`` decorator runs (doc 08 §3.1 first-party discovery) with **no shared-file
edit** — the doc-03 §9 "a new survey method is added by one file" contract.

Third-party (out-of-tree) adapters register instead via the ``geosim.plugins``
setuptools entry-point group (doc 03 §1, doc 08 §3.1), discovered by
:func:`geosim.ingestion.registry.discover_entry_points`.
"""

from __future__ import annotations

import importlib
import pkgutil

__all__: list[str] = []


def _autoimport_siblings() -> None:
    """Import every sibling module so its ``@adapter`` decorator self-registers."""
    package = __name__
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        importlib.import_module(f"{package}.{mod.name}")


_autoimport_siblings()
