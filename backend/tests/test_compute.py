"""Tests for the GPU compute shim + engine auto-registration (doc 10 §8).

This container has **no GPU** (no cupy / no torch CUDA), so these tests pin the *NumPy/CPU
fallback path* of :mod:`geosim.compute`: ``xp()`` is NumPy, ``gpu_available()`` is False,
and the host-conversion helpers (``asnumpy`` / ``to_device`` / ``array_namespace``) are
no-ops that keep arrays on the CPU. They also assert that importing
:mod:`geosim.inversion.engines` auto-registers the first-party engines (and any new
self-registering sibling modules) on the process-wide plugin registry (doc 08 §4f).
"""

from __future__ import annotations

import numpy as np

from geosim import compute

# ─────────────────────────────── compute backend (CPU path) ───────────────────────────────


def test_gpu_unavailable_on_this_container() -> None:
    assert compute.gpu_available() is False
    assert compute.backend_name() == "numpy"


def test_xp_is_numpy() -> None:
    assert compute.xp() is np


def test_asnumpy_passthrough_for_numpy() -> None:
    a = np.arange(6.0).reshape(2, 3)
    out = compute.asnumpy(a)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, a)


def test_asnumpy_coerces_non_arrays() -> None:
    out = compute.asnumpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_to_device_is_noop_on_cpu() -> None:
    a = np.ones((3, 3))
    out = compute.to_device(a)
    assert isinstance(out, np.ndarray)
    # No GPU: same host buffer comes straight back.
    assert out is a
    # And a non-array input is coerced to NumPy (not a cupy array).
    coerced = compute.to_device([1.0, 2.0])
    assert isinstance(coerced, np.ndarray)


def test_array_namespace_picks_numpy() -> None:
    a = np.zeros(4)
    b = np.ones(4)
    assert compute.array_namespace(a, b) is np
    assert compute.array_namespace() is np


def test_torch_helpers_degrade_to_cpu() -> None:
    # No torch installed here → try_torch is None and the device falls back to cpu.
    assert compute.try_torch() is None
    assert compute.torch_device() == "cpu"


def test_backend_summary_present() -> None:
    assert "geosim.compute backend:" in compute.BACKEND_SUMMARY
    assert "array=numpy" in compute.BACKEND_SUMMARY
    assert "gpu=no" in compute.BACKEND_SUMMARY


# ─────────────────────────────── engine auto-registration ───────────────────────────────


def test_importing_engines_registers_first_party() -> None:
    """Importing the engines package self-registers the first-party engines (doc 08 §4f)."""
    import geosim.inversion.engines  # noqa: F401  (import triggers auto-registration)
    from geosim.plugins import get_registry

    keys = {getattr(e, "key", None) for e in get_registry().inversion_engines()}
    # Both shipped engines must be present...
    assert "simpeg.gravity" in keys
    assert "pygimli.ert" in keys


def test_auto_import_covers_all_sibling_modules() -> None:
    """Every sibling engine module is imported (so a dropped-in engine self-registers)."""
    import pkgutil

    import geosim.inversion.engines as engines_pkg
    from geosim.plugins import get_registry

    sibling_modules = [
        m.name for m in pkgutil.iter_modules(engines_pkg.__path__) if not m.name.startswith("_")
    ]
    # The package shipped at least the two known engines as modules.
    assert "gravity_simpeg" in sibling_modules
    assert "ert_pygimli" in sibling_modules

    # Each known engine module imported cleanly into sys.modules via the auto-import.
    import sys

    for name in sibling_modules:
        assert f"geosim.inversion.engines.{name}" in sys.modules

    # And the registry holds at least as many engines as there are engine modules
    # (every sibling module is expected to contribute exactly one self-registering engine).
    registered = {getattr(e, "key", None) for e in get_registry().inversion_engines()}
    assert len(registered) >= len(sibling_modules)
