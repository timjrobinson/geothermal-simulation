"""Multiresolution pyramid downsampling (doc 02 §10.3, doc 04 §5).

Levels are a power-of-two pyramid: ``0`` = full resolution; each coarser level
halves every spatial axis (×⅛ voxels). Value arrays downsample by **mean** (block
average — preserves field values); ``_sigma`` arrays downsample **variance-correct**
so confidence survives LOD (doc 02 §10.3). The pyramid is built until the coarsest
level fits in ~1 chunk (a ~64³ thumbnail of the whole volume; doc 04 §5).

NaN-aware throughout: masked/outside-coverage cells are NaN (doc 02 §10.2) and are
ignored in the block reduction; an all-NaN block stays NaN.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "pyramid_level_count",
    "downsample_mean",
    "downsample_sigma",
    "build_value_pyramid",
    "build_sigma_pyramid",
]


def pyramid_level_count(shape: tuple[int, ...], chunk: int = 64) -> int:
    """Number of levels until the coarsest fits in ~1 chunk per spatial axis.

    Level 0 is full resolution; each step halves every spatial axis. We stop when
    every spatial dimension is ``<= chunk`` (a ~chunk³ thumbnail, doc 04 §5).
    Returns a count of **>= 1** (level 0 always exists).
    """
    spatial = shape[-3:] if len(shape) >= 3 else shape
    levels = 1
    cur = list(spatial)
    while any(d > chunk for d in cur):
        cur = [max(1, (d + 1) // 2) for d in cur]
        levels += 1
    return levels


def _block_pad(arr: np.ndarray) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Pad odd spatial dims by one (with NaN) so each 2×2×2 block is complete."""
    pad = [(0, 0)] * arr.ndim
    spatial_axes = range(arr.ndim - 3, arr.ndim)
    for ax in spatial_axes:
        if arr.shape[ax] % 2 == 1:
            pad[ax] = (0, 1)
    if all(p == (0, 0) for p in pad):
        return arr, tuple(slice(None) for _ in range(arr.ndim))
    padded = np.pad(arr.astype(np.float64), pad, mode="constant", constant_values=np.nan)
    return padded, tuple(slice(None) for _ in range(arr.ndim))


def _reshape_blocks(arr: np.ndarray) -> np.ndarray:
    """Reshape spatial axes into (n, 2) pairs, returning a view ending in the 2×2×2 block.

    Result shape: leading non-spatial dims, then (z//2, 2, y//2, 2, x//2, 2).
    """
    lead = arr.shape[: arr.ndim - 3]
    z, y, x = arr.shape[-3:]
    return arr.reshape(*lead, z // 2, 2, y // 2, 2, x // 2, 2)


def downsample_mean(arr: np.ndarray) -> np.ndarray:
    """Halve each spatial axis by NaN-aware block mean (doc 02 §10.3, doc 04 §5).

    Operates on the trailing three axes ``(z, y, x)``; any leading axes (e.g. ``t``,
    ``class``) are preserved (time is not downsampled — doc 04 §5).
    """
    a = arr.astype(np.float64, copy=False)
    a, _ = _block_pad(a)
    blocks = _reshape_blocks(a)
    block_axes = (-5, -3, -1)  # the three "2" axes
    with warnings.catch_warnings():
        # an all-NaN block intentionally yields NaN (masked cells stay masked)
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(blocks, axis=block_axes)
    return out.astype(np.float32)


def downsample_sigma(sigma: np.ndarray) -> np.ndarray:
    """Variance-correct downsample of a 1σ array (doc 02 §10.3).

    Averaging N independent cells reduces the mean's variance by 1/N: the coarse
    cell's variance is the **mean of the fine variances divided by the count**, so
    ``sigma_coarse = sqrt(mean(sigma_fine**2) / n_valid)``. NaN cells are excluded
    and ``n_valid`` counts only the contributing (non-NaN) fine cells, so confidence
    is correctly tightened by the averaging.
    """
    a = sigma.astype(np.float64, copy=False)
    a, _ = _block_pad(a)
    var = a * a
    blocks = _reshape_blocks(var)
    block_axes = (-5, -3, -1)
    n_valid = np.sum(~np.isnan(blocks), axis=block_axes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_var = np.nanmean(blocks, axis=block_axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        coarse_var = np.where(n_valid > 0, mean_var / n_valid, np.nan)
    return np.sqrt(coarse_var).astype(np.float32)


def build_value_pyramid(level0: np.ndarray, chunk: int = 64) -> list[np.ndarray]:
    """Build the full mean-downsampled value pyramid (level 0 first; doc 04 §5)."""
    levels = [np.asarray(level0, dtype=np.float32)]
    n = pyramid_level_count(level0.shape, chunk)
    for _ in range(1, n):
        levels.append(downsample_mean(levels[-1]))
    return levels


def build_sigma_pyramid(sigma0: np.ndarray, chunk: int = 64) -> list[np.ndarray]:
    """Build the variance-correct ``_sigma`` pyramid (level 0 first; doc 02 §10.3)."""
    levels = [np.asarray(sigma0, dtype=np.float32)]
    n = pyramid_level_count(sigma0.shape, chunk)
    for _ in range(1, n):
        levels.append(downsample_sigma(levels[-1]))
    return levels
