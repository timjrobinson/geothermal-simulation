"""Array-backend abstraction: CuPy/Torch on GPU, NumPy/CPU everywhere else (doc 10 §8).

A *thin* compute shim so heavy engines (SimPEG sensitivity products, fusion stencils, MT
1D kernels) can be written **once** against the array-API surface and run on an RTX-class
GPU when one is present, while degrading transparently to NumPy on a CPU-only box (this
container, and any CI). The contract:

- :func:`xp` — the active array module: CuPy iff it is importable **and** a CUDA device is
  actually present, else NumPy. This is the module engines should reach for to allocate
  work arrays (``xp().zeros(...)``) so the allocation lands on the right device.
- :func:`gpu_available` — ``True`` iff CuPy + a real CUDA device are available.
- :func:`asnumpy` — bring an array back to host NumPy (``cupy.ndarray`` → ``np.ndarray``);
  a passthrough for arrays that are already NumPy, so it is always safe to call before
  handing data across the plugin boundary or into SciPy/discretize.
- :func:`to_device` — move a host array onto the GPU when one is active (NumPy → CuPy),
  else return it unchanged.
- :func:`array_namespace` — pick the module that owns a set of arrays (CuPy if **any** is
  a CuPy array, else NumPy) so a routine can stay device-agnostic.
- :func:`torch_device` / :func:`try_torch` — the same fallback story for the Torch-based
  paths (e.g. learned priors): CUDA device + Torch module when available, else CPU/``None``.

CuPy and Torch are imported **lazily inside the functions** — never at module top — so
importing :mod:`geosim.compute` (and anything that transitively imports it) stays cheap and
never fails on a box without those wheels. The active backend is summarised once at import
in :data:`BACKEND_SUMMARY` for logging / capabilities (doc 08 §7).
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import Any

import numpy as np

__all__ = [
    "xp",
    "gpu_available",
    "asnumpy",
    "to_device",
    "array_namespace",
    "torch_device",
    "try_torch",
    "backend_name",
    "BACKEND_SUMMARY",
]


# ─────────────────────────────── CuPy / NumPy backend ───────────────────────────────


@functools.lru_cache(maxsize=1)
def _try_cupy() -> ModuleType | None:
    """Return the imported ``cupy`` module iff it is importable **and** sees a CUDA device.

    Cached: the import + device probe runs at most once per process. A box with the CuPy
    wheel but no usable GPU (driver missing, ``CUDA_VISIBLE_DEVICES=``) reports *no* GPU —
    importability alone is not enough (doc 10 §8 "degrade to NumPy when CUDA is absent").
    """
    try:
        import cupy  # type: ignore[import-not-found]
    except Exception:  # ImportError, or a broken/partial CUDA install
        return None
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
    except Exception:  # cupy present but the CUDA runtime is unusable
        return None
    return cupy


def gpu_available() -> bool:
    """``True`` iff a CuPy-backed CUDA device is available (False on this container)."""
    return _try_cupy() is not None


def xp() -> ModuleType:
    """The active array module — ``cupy`` on a GPU box, else ``numpy``.

    Allocate device-resident work arrays through this (``geosim.compute.xp().empty(...)``)
    so the same engine code lands on the GPU when present and on host NumPy otherwise.
    """
    cupy = _try_cupy()
    return cupy if cupy is not None else np


def backend_name() -> str:
    """Short label for the active array backend: ``"cupy"`` or ``"numpy"`` (doc 08 §7)."""
    return "cupy" if gpu_available() else "numpy"


def asnumpy(a: Any) -> np.ndarray:
    """Return ``a`` as a host :class:`numpy.ndarray` (CuPy → NumPy; NumPy passthrough).

    Always safe to call before crossing the plugin boundary or handing arrays to SciPy /
    ``discretize`` / PyGIMLi, which only understand host NumPy (doc 10 §8). Non-array
    inputs are coerced with :func:`numpy.asarray`.
    """
    cupy = _try_cupy()
    if cupy is not None and isinstance(a, cupy.ndarray):
        return cupy.asnumpy(a)
    return np.asarray(a)


def to_device(a: Any) -> Any:
    """Move a host array onto the active GPU (NumPy → CuPy); unchanged when CPU-only.

    The inverse of :func:`asnumpy`: use it to stage inputs on the device before a hot loop.
    Already-on-device CuPy arrays are returned as-is; on a CPU box this is a no-op coercion
    to NumPy via :func:`numpy.asarray`.
    """
    cupy = _try_cupy()
    if cupy is None:
        return np.asarray(a) if not isinstance(a, np.ndarray) else a
    if isinstance(a, cupy.ndarray):
        return a
    return cupy.asarray(a)


def array_namespace(*arrays: Any) -> ModuleType:
    """Pick the module that owns ``arrays`` — CuPy if **any** is a CuPy array, else NumPy.

    Lets a routine stay device-agnostic: ``ns = array_namespace(a, b); ns.dot(a, b)``
    dispatches on whichever device the inputs already live on, with no eager CuPy import on
    a CPU-only box (the import only happens when a CuPy array is actually passed in).
    """
    cupy = _try_cupy()
    if cupy is not None and any(isinstance(a, cupy.ndarray) for a in arrays):
        return cupy
    return np


# ──────────────────────────────── Torch backend ────────────────────────────────


def try_torch() -> ModuleType | None:
    """Return the imported ``torch`` module, or ``None`` when it is not installed.

    Lazy by design (never imported at module top) so :mod:`geosim.compute` imports without
    Torch present (this container). Returns the module regardless of CUDA so a caller can
    still use CPU Torch; pair with :func:`torch_device` to place tensors.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    return torch


def torch_device(*, prefer_gpu: bool = True) -> str:
    """The Torch device string: ``"cuda"`` iff Torch + CUDA are present, else ``"cpu"``.

    Returns ``"cpu"`` when Torch is absent (this container) or CUDA is unavailable, so it is
    safe to pass straight into ``tensor.to(torch_device())`` on any box.
    """
    torch = try_torch()
    if torch is None or not prefer_gpu:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch present but CUDA probe failed
        return "cpu"
    return "cpu"


# A one-line, import-time summary of the active backend for logs / capabilities (doc 08 §7).
BACKEND_SUMMARY: str = (
    f"geosim.compute backend: array={backend_name()} "
    f"(gpu={'yes' if gpu_available() else 'no'}), torch_device={torch_device()}"
)
