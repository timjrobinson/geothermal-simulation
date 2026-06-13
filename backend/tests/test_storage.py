"""Tests for the storage layer (doc 02 §10, doc 04 §3–§5).

Everything runs against local temp dirs (``tmp_path``) — no Docker/Postgres/Redis.
"""

import math

import numpy as np
import pytest

from geosim.storage import (
    BULK_STORES,
    GridSpec,
    RawStore,
    build_sigma_pyramid,
    downsample_mean,
    downsample_sigma,
    ensure_project_layout,
    open_property_model,
    pyramid_level_count,
    sha256_bytes,
    write_property_model,
)


# --------------------------------------------------------------- project layout


def test_project_layout_creates_all_bulk_stores(tmp_path):
    # doc 04 §3: arrays/ grids/ meshes/ vectors/ points/ raw/ cache/
    layout = ensure_project_layout(tmp_path, "proj01")
    assert layout.root == tmp_path / "proj01"
    for name in BULK_STORES:
        assert (layout.root / name).is_dir(), f"missing bulk store {name}"
    # cache subdirs (doc 04 §8)
    for sub in ("slices", "isosurfaces", "tiles"):
        assert (layout.cache / sub).is_dir()


def test_project_layout_idempotent(tmp_path):
    ensure_project_layout(tmp_path, "p")
    layout = ensure_project_layout(tmp_path, "p")  # second call must not error
    assert layout.zarr_path("pm_abc").name == "pm_abc.zarr"
    assert layout.zarr_path("pm_abc").parent == layout.arrays


# -------------------------------------------------------------- pyramid math


def test_pyramid_level_count_to_thumbnail():
    # halve until every spatial axis <= 64 (doc 04 §5).
    # (256,256,256) -> 256,128,64 => 3 levels
    assert pyramid_level_count((256, 256, 256), chunk=64) == 3
    # already a thumbnail
    assert pyramid_level_count((32, 16, 16), chunk=64) == 1
    # (128,64,64) -> 64,32,32 => 2 levels
    assert pyramid_level_count((128, 64, 64), chunk=64) == 2


def test_mean_downsample_value():
    a = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    out = downsample_mean(a)
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == pytest.approx(np.mean(a))


def test_mean_downsample_nan_aware():
    a = np.full((2, 2, 2), np.nan, dtype=np.float32)
    a[0, 0, 0] = 4.0
    out = downsample_mean(a)
    # one valid cell in the block -> coarse value is that cell, not NaN
    assert out[0, 0, 0] == pytest.approx(4.0)
    # an all-NaN block stays NaN
    b = np.full((2, 2, 2), np.nan, dtype=np.float32)
    assert np.isnan(downsample_mean(b)[0, 0, 0])


def test_sigma_downsample_is_variance_correct():
    # doc 02 §10.3: averaging N=8 cells of equal sigma s -> coarse sigma = s/sqrt(8).
    s = 2.0
    sigma = np.full((2, 2, 2), s, dtype=np.float32)
    out = downsample_sigma(sigma)
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == pytest.approx(s / math.sqrt(8), rel=1e-5)


def test_sigma_pyramid_variance_correct_across_levels():
    s = 3.0
    sigma0 = np.full((4, 4, 4), s, dtype=np.float32)
    levels = build_sigma_pyramid(sigma0, chunk=2)
    # 4->2->... level1 averages 8 cells: s/sqrt(8)
    assert levels[1][0, 0, 0] == pytest.approx(s / math.sqrt(8), rel=1e-5)


# -------------------------------------------------- property model (zarr v3)


def _resistivity_volume():
    rng = np.random.default_rng(0)
    vol = rng.uniform(10.0, 1000.0, size=(32, 16, 16)).astype(np.float32)
    # punch a NaN hole (outside-DOI cells are NaN, never 0; doc 02 §10.2)
    vol[0, 0, 0] = np.nan
    sigma = np.full_like(vol, 5.0)
    return vol, sigma


def test_write_read_level0_roundtrip(tmp_path):
    vol, sigma = _resistivity_volume()
    grid = GridSpec(origin=(-100.0, 50.0, 25.0), spacing=(10.0, 5.0, 5.0), cell_ref="center")
    path = write_property_model(
        tmp_path / "pm_r.zarr", "resistivity", vol, grid=grid, sigma=sigma
    )

    r = open_property_model(path)
    assert r.properties == ["resistivity"]
    assert r.has_sigma("resistivity")

    lvl0 = r.read_level("resistivity", 0)
    # shape + axis order [z,y,x] (doc 02 §10.2)
    assert lvl0.shape == (32, 16, 16)
    # NaN fill survives roundtrip
    assert np.isnan(lvl0[0, 0, 0])
    # values survive
    np.testing.assert_allclose(
        np.nan_to_num(lvl0), np.nan_to_num(vol), rtol=1e-5
    )

    attrs = r.attrs("resistivity", 0)
    assert attrs["propertyType"] == "resistivity"
    assert attrs["canonicalUnit"] == "ohm*m"
    assert attrs["scaling"] == "log"
    assert attrs["colormap"] == "turbo"
    assert attrs["displayRange"] == [1, 10000]
    assert attrs["origin"] == [-100.0, 50.0, 25.0]
    assert attrs["spacing"] == [10.0, 5.0, 5.0]
    assert attrs["cellRef"] == "center"
    assert attrs["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]


def test_cubic_chunks_64(tmp_path):
    vol, _ = _resistivity_volume()
    path = write_property_model(tmp_path / "pm.zarr", "resistivity", vol)
    r = open_property_model(path)
    # 32x16x16 volume clamps cubic 64^3 chunk to the array extent
    assert r.group["resistivity/0"].chunks == (32, 16, 16)
    # a bigger volume keeps the 64^3 cubic chunk
    big = np.ones((128, 128, 128), dtype=np.float32)
    path2 = write_property_model(tmp_path / "pm_big.zarr", "resistivity", big)
    r2 = open_property_model(path2)
    assert r2.group["resistivity/0"].chunks == (64, 64, 64)


def test_pyramid_levels_and_coarse_read(tmp_path):
    vol, sigma = _resistivity_volume()
    path = write_property_model(tmp_path / "pm.zarr", "resistivity", vol, sigma=sigma)
    r = open_property_model(path)

    # (32,16,16) is already <=64 on every axis -> 1 level
    assert r.level_count("resistivity") == 1

    # a volume needing multiple levels
    big = np.ones((256, 128, 64), dtype=np.float32)
    big_sigma = np.full_like(big, 4.0)
    p2 = write_property_model(
        tmp_path / "pm_big.zarr", "resistivity", big, sigma=big_sigma
    )
    r2 = open_property_model(p2)
    n = pyramid_level_count((256, 128, 64))
    assert r2.level_count("resistivity") == n
    assert r2.level_count("resistivity_sigma") == n

    # read a coarser level back: half the spatial extent per axis per level
    coarse = r2.read_level("resistivity", 1)
    assert coarse.shape == (128, 64, 32)
    # mean of all-ones stays one
    np.testing.assert_allclose(coarse, 1.0, rtol=1e-5)


def test_sigma_downsample_variance_correct_in_store(tmp_path):
    big = np.ones((256, 128, 64), dtype=np.float32)
    s = 6.0
    big_sigma = np.full_like(big, s)
    p = write_property_model(
        tmp_path / "pm.zarr", "resistivity", big, sigma=big_sigma
    )
    r = open_property_model(p)
    coarse_sigma = r.read_sigma_level("resistivity", 1)
    # interior cell: 8 fine cells averaged -> s/sqrt(8) (doc 02 §10.3)
    assert coarse_sigma[10, 10, 10] == pytest.approx(s / math.sqrt(8), rel=1e-4)


def test_multiscales_block_present(tmp_path):
    big = np.ones((256, 128, 64), dtype=np.float32)
    p = write_property_model(tmp_path / "pm.zarr", "resistivity", big)
    r = open_property_model(p)
    ms = r.multiscales("resistivity")
    assert ms[0]["version"] == "0.4"
    assert [ax["name"] for ax in ms[0]["axes"]] == ["z", "y", "x"]
    n = pyramid_level_count((256, 128, 64))
    assert len(ms[0]["datasets"]) == n
    # level-1 scale is double the level-0 spacing (doc 04 §5)
    sc0 = ms[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    sc1 = ms[0]["datasets"][1]["coordinateTransformations"][0]["scale"]
    assert sc1 == [2 * sc0[0], 2 * sc0[1], 2 * sc0[2]]


def test_z_up_axis_order_documented(tmp_path):
    vol, _ = _resistivity_volume()
    p = write_property_model(tmp_path / "pm.zarr", "resistivity", vol)
    r = open_property_model(p)
    assert r.group.attrs["geosim"]["axisOrder"] == ["z", "y", "x"]
    assert r.attrs("resistivity")["_ARRAY_DIMENSIONS"] == ["z", "y", "x"]


# --------------------------------------------------- content-addressed raw store


def test_raw_store_writes_sha256_path(tmp_path):
    layout = ensure_project_layout(tmp_path, "p")
    store = RawStore(layout.raw)
    data = b"verbatim source bytes"
    ref = store.put_bytes("survey.csv", data)
    assert ref.sha256 == sha256_bytes(data)
    # raw/<sha256>/<name> (doc 04 §3, §8.1)
    assert ref.path == layout.raw / ref.sha256 / "survey.csv"
    assert ref.path.read_bytes() == data
    assert store.get_bytes(ref.sha256, "survey.csv") == data


def test_raw_store_dedupes_identical_content(tmp_path):
    store = RawStore(tmp_path / "raw")
    data = b"identical content"
    ref1 = store.put_bytes("a.bin", data)
    ref2 = store.put_bytes("a.bin", data)
    assert ref1.sha256 == ref2.sha256
    # only one sha256 directory exists for identical content (doc 04 §8.1)
    sha_dirs = [d for d in (tmp_path / "raw").iterdir() if d.is_dir()]
    assert len(sha_dirs) == 1
    # different content -> different address
    ref3 = store.put_bytes("b.bin", b"other content")
    assert ref3.sha256 != ref1.sha256
    sha_dirs = [d for d in (tmp_path / "raw").iterdir() if d.is_dir()]
    assert len(sha_dirs) == 2
