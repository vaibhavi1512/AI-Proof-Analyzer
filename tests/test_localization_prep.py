"""Tests for localization dataset preparation (synthetic trees only)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.localization.prepare_dataset import (
    LocalizationPrepConfig,
    LocalizationPrepError,
    apply_letterbox,
    assign_folder_splits,
    binarize_mask_array,
    compute_letterbox_geometry,
    discover_pairs,
    mask_to_binary_l,
    prepare_localization_dataset,
)


def _write_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG", quality=95)


def _write_mask(path: Path, size: tuple[int, int], rect: tuple[int, int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    x0, y0, x1, y1 = rect
    arr[y0:y1, x0:x1] = 255
    Image.fromarray(arr, mode="L").convert("RGB").save(path, format="PNG")


def _imd_tree(root: Path) -> Path:
    src = root / "IMD2020"
    # IMD-style folder
    a = src / "1a41rr"
    _write_rgb(a / "1a41rr_orig.jpg", (40, 30), (10, 20, 30))
    _write_rgb(a / "c8ty2bs_0.jpg", (40, 20), (200, 10, 10))
    _write_mask(a / "c8ty2bs_0_mask.png", (40, 20), (2, 2, 10, 12))
    _write_rgb(a / "c8tzqf0_0.jpg", (40, 20), (10, 200, 10))
    _write_mask(a / "c8tzqf0_0_mask.png", (40, 20), (20, 4, 30, 16))
    # z* naming
    z = src / "z10"
    _write_rgb(z / "00021_orig.jpg", (32, 32), (1, 2, 3))
    _write_rgb(z / "00021_fake.jpg", (32, 24), (9, 9, 9))
    _write_mask(z / "00021_fake_mask.png", (32, 24), (0, 0, 8, 8))
    return src


def test_pairing_imd_and_z_conventions(tmp_path: Path) -> None:
    src = _imd_tree(tmp_path)
    pairs, issues = discover_pairs(src)
    stems = sorted(p.fake_path.stem for p in pairs)
    assert stems == ["00021_fake", "c8ty2bs_0", "c8tzqf0_0"]
    assert all(p.mask_path.stem == p.fake_path.stem + "_mask" for p in pairs)
    assert issues == []


def test_letterbox_preserves_aspect_and_is_shared() -> None:
    geo = compute_letterbox_geometry(100, 50, 256)
    assert geo.resized_width == 256
    assert geo.resized_height == 128
    assert geo.pad_left == 0
    assert geo.pad_top == 64
    rgb = Image.new("RGB", (100, 50), color=(255, 0, 0))
    mask = Image.new("L", (100, 50), color=0)
    out_i = apply_letterbox(rgb, geo, 256, resample=Image.Resampling.BILINEAR, fill=(0, 0, 0))
    out_m = apply_letterbox(mask, geo, 256, resample=Image.Resampling.NEAREST, fill=0)
    assert out_i.size == (256, 256)
    assert out_m.size == (256, 256)


def test_binarize_threshold_and_0_1() -> None:
    hi = binarize_mask_array(np.array([[0, 127, 128, 255]], dtype=np.uint8), 127)
    np.testing.assert_array_equal(hi, [[0, 0, 255, 255]])
    low = binarize_mask_array(np.array([[0, 1]], dtype=np.uint8), 127)
    np.testing.assert_array_equal(low, [[0, 255]])


def test_non_standard_soft_mask_is_kept() -> None:
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[2:6, 2:6] = 200
    arr[0, 0] = 50
    img = Image.fromarray(arr, mode="L").convert("RGB")
    binary, mask_type, unique = mask_to_binary_l(img, 127)
    assert mask_type == "non_standard"
    out = np.asarray(binary)
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert out[0, 0] == 0
    assert out[3, 3] == 255
    assert 50 in unique and 200 in unique


def test_folder_split_no_overlap() -> None:
    folders = [f"f{i:02d}" for i in range(20)]
    assigned = assign_folder_splits(
        folders, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    by = {s: {k for k, v in assigned.items() if v == s} for s in ("train", "val", "test")}
    assert not (by["train"] & by["val"])
    assert not (by["train"] & by["test"])
    assert not (by["val"] & by["test"])
    assert by["train"] | by["val"] | by["test"] == set(folders)
    again = assign_folder_splits(
        folders, seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    assert assigned == again


def test_prepare_writes_256_binary_and_refuses_overwrite(tmp_path: Path) -> None:
    src = _imd_tree(tmp_path)
    # Several extra folders so 70/15/15 is meaningful
    for i in range(8):
        folder = src / f"case{i}"
        _write_rgb(folder / f"case{i}_orig.jpg", (16, 16), (0, 0, 0))
        _write_rgb(folder / f"fx{i}_0.jpg", (16, 12), (40, 40, 40))
        _write_mask(folder / f"fx{i}_0_mask.png", (16, 12), (1, 1, 6, 6))

    out = tmp_path / "derived"
    cfg = LocalizationPrepConfig(
        project_root=tmp_path,
        source_dir=src,
        output_dir=out,
        image_size=256,
        seed=42,
        overwrite=False,
    )
    report = prepare_localization_dataset(cfg)
    assert report.total_pairs == 3 + 8
    assert report.failures == 0
    assert report.parent_folder_overlap is False
    assert (out / "manifest.csv").is_file()
    assert (out / "README.md").is_file()

    with (out / "manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report.total_pairs
    folders = {}
    for row in rows:
        img = tmp_path / row["output_image_path"]
        msk = tmp_path / row["output_mask_path"]
        with Image.open(img) as im:
            assert im.size == (256, 256)
            assert im.mode == "RGB"
        with Image.open(msk) as mm:
            assert mm.size == (256, 256)
            assert mm.mode == "L"
            assert set(np.unique(np.asarray(mm)).tolist()) <= {0, 255}
        folders.setdefault(row["source_folder"], set()).add(row["split"])
    assert all(len(v) == 1 for v in folders.values())

    cfg_again = LocalizationPrepConfig(
        project_root=tmp_path,
        source_dir=src,
        output_dir=out,
        overwrite=False,
    )
    with pytest.raises(LocalizationPrepError, match="already exists"):
        prepare_localization_dataset(cfg_again)

    cfg_force = LocalizationPrepConfig(
        project_root=tmp_path,
        source_dir=src,
        output_dir=out,
        overwrite=True,
        seed=7,
    )
    report2 = prepare_localization_dataset(cfg_force)
    assert report2.total_pairs == report.total_pairs


def test_refuses_write_into_source_tree(tmp_path: Path) -> None:
    src = _imd_tree(tmp_path)
    cfg = LocalizationPrepConfig(
        project_root=tmp_path,
        source_dir=src,
        output_dir=src / "derived",
    )
    with pytest.raises(LocalizationPrepError, match="inside the source tree"):
        prepare_localization_dataset(cfg)
