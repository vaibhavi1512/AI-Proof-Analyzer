"""Prepare a derived 256×256 fake/mask corpus for localization training.

Source ``dataset 16.09/IMD2020`` is read-only. Outputs go to
``artifacts/localization_dataset/`` (or a caller-supplied path).
"""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("maya.ai.localization.prepare_dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = [
    "split",
    "source_folder",
    "source_fake_path",
    "source_mask_path",
    "output_image_path",
    "output_mask_path",
    "original_width",
    "original_height",
    "output_width",
    "output_height",
    "mask_type",
    "mask_threshold",
    "unique_values_before",
    "foreground_pixels",
    "pad_left",
    "pad_top",
    "resized_width",
    "resized_height",
]


class LocalizationPrepError(RuntimeError):
    """Raised for configuration or output-directory safety failures."""


@dataclass
class LocalizationPrepConfig:
    """Tunables for the localization dataset preparation stage."""

    project_root: Path
    source_dir: Path
    output_dir: Path
    image_size: int = 256
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    mask_threshold: int = 127
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.source_dir = Path(self.source_dir)
        if not self.source_dir.is_absolute():
            self.source_dir = (self.project_root / self.source_dir).resolve()
        else:
            self.source_dir = self.source_dir.resolve()
        self.output_dir = Path(self.output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = (self.project_root / self.output_dir).resolve()
        else:
            self.output_dir = self.output_dir.resolve()
        if self.image_size < 1:
            raise LocalizationPrepError("image_size must be >= 1")
        if self.mask_threshold < 0 or self.mask_threshold > 255:
            raise LocalizationPrepError("mask_threshold must be in [0, 255]")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise LocalizationPrepError(
                f"Split ratios must sum to 1.0, got {total:.6f}"
            )


@dataclass
class FakeMaskPair:
    folder: Path
    folder_id: str
    fake_path: Path
    mask_path: Path


@dataclass
class Geometry:
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


@dataclass
class PrepReport:
    total_pairs: int
    train_pairs: int
    val_pairs: int
    test_pairs: int
    train_folders: int
    val_folders: int
    test_folders: int
    min_fakes_per_folder: int
    max_fakes_per_folder: int
    mean_fakes_per_folder: float
    non_standard_masks: int
    non_standard_normalized: int
    failures: int
    foreground_pixels: int
    background_pixels: int
    parent_folder_overlap: bool
    output_dir: str
    mask_threshold: int
    seed: int
    non_standard_details: list[dict[str, str]] = field(default_factory=list)
    failure_details: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_config(project_root: Path | None = None) -> LocalizationPrepConfig:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    return LocalizationPrepConfig(
        project_root=root,
        source_dir=root / "dataset 16.09" / "IMD2020",
        output_dir=root / "artifacts" / "localization_dataset",
    )


def discover_pairs(source_dir: Path) -> tuple[list[FakeMaskPair], list[dict[str, str]]]:
    """Find fake images and matching ``{stem}_mask`` files (read-only walk)."""

    if not source_dir.is_dir():
        raise LocalizationPrepError(f"Source directory not found: {source_dir}")

    by_folder: dict[Path, list[Path]] = defaultdict(list)
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        by_folder[path.parent].append(path)

    pairs: list[FakeMaskPair] = []
    issues: list[dict[str, str]] = []
    for folder, files in sorted(by_folder.items(), key=lambda item: str(item[0]).lower()):
        masks: dict[str, Path] = {}
        fakes: list[Path] = []
        for path in files:
            stem = path.stem.lower()
            if stem.endswith("_orig"):
                continue
            if stem.endswith("_mask"):
                key = stem
                if key in masks:
                    issues.append(
                        {
                            "kind": "duplicate_mask_stem",
                            "path": str(path),
                        }
                    )
                    continue
                masks[key] = path
            else:
                fakes.append(path)

        for fake in sorted(fakes, key=lambda p: p.name.lower()):
            expected = fake.stem.lower() + "_mask"
            mask = masks.get(expected)
            if mask is None:
                issues.append(
                    {
                        "kind": "fake_without_mask",
                        "path": str(fake),
                    }
                )
                continue
            pairs.append(
                FakeMaskPair(
                    folder=folder,
                    folder_id=folder.name,
                    fake_path=fake,
                    mask_path=mask,
                )
            )
            del masks[expected]

        for leftover in masks.values():
            issues.append(
                {
                    "kind": "mask_without_fake",
                    "path": str(leftover),
                }
            )

    return pairs, issues


def classify_mask_values(unique: list[int]) -> str:
    values = set(int(v) for v in unique)
    if values <= {0, 255} or values <= {0} or values <= {255}:
        return "binary_0_255"
    if values <= {0, 1}:
        return "binary_0_1"
    if len(values) <= 4 and 0 in values and 255 in values:
        return "near_binary"
    return "non_standard"


def binarize_mask_array(arr: np.ndarray, threshold: int) -> np.ndarray:
    """Map a single-channel array to {0, 255}.

    If the source looks like a 0/1 mask (max <= 1), any positive pixel is
    foreground. Otherwise pixels **greater than** ``threshold`` are foreground.
    Default threshold 127 leaves {0, 255} masks unchanged.
    """

    flat = arr.astype(np.float32, copy=False)
    if float(np.nanmax(flat)) <= 1.0:
        binary = (flat > 0).astype(np.uint8) * 255
    else:
        binary = (flat > float(threshold)).astype(np.uint8) * 255
    return binary


def mask_to_binary_l(
    image: Image.Image,
    threshold: int,
) -> tuple[Image.Image, str, list[int]]:
    """Convert any mask mode to a binary L image without changing geometry."""

    if image.mode == "RGBA":
        rgba = image
        rgb = Image.new("RGB", rgba.size, (0, 0, 0))
        rgb.paste(rgba, mask=rgba.split()[-1])
        gray = rgb.convert("L")
    elif image.mode == "P":
        gray = image.convert("L")
    elif image.mode == "L":
        gray = image
    elif image.mode == "1":
        gray = image.convert("L")
    else:
        gray = image.convert("L")

    arr = np.asarray(gray, dtype=np.uint8)
    unique = sorted(int(v) for v in np.unique(arr).tolist())
    mask_type = classify_mask_values(unique)
    binary = binarize_mask_array(arr, threshold)
    out = Image.fromarray(binary, mode="L")
    return out, mask_type, unique


def compute_letterbox_geometry(width: int, height: int, size: int) -> Geometry:
    """Fit ``(width, height)`` inside ``size×size``; pad remaining borders."""

    if width < 1 or height < 1:
        raise LocalizationPrepError(f"Invalid source size {width}x{height}")
    scale = min(size / width, size / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized_w = min(size, resized_w)
    resized_h = min(size, resized_h)
    pad_left = (size - resized_w) // 2
    pad_top = (size - resized_h) // 2
    pad_right = size - resized_w - pad_left
    pad_bottom = size - resized_h - pad_top
    return Geometry(
        resized_width=resized_w,
        resized_height=resized_h,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )


def apply_letterbox(
    image: Image.Image,
    geometry: Geometry,
    size: int,
    *,
    resample: Image.Resampling,
    fill: int | tuple[int, int, int],
) -> Image.Image:
    resized = image.resize(
        (geometry.resized_width, geometry.resized_height),
        resample=resample,
    )
    if image.mode == "L":
        canvas = Image.new("L", (size, size), color=int(fill) if isinstance(fill, int) else 0)
    else:
        color = fill if isinstance(fill, tuple) else (fill, fill, fill)
        canvas = Image.new("RGB", (size, size), color=color)
    canvas.paste(resized, (geometry.pad_left, geometry.pad_top))
    return canvas


def assign_folder_splits(
    folder_ids: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    """Map each parent folder to exactly one split (deterministic)."""

    unique = sorted(set(folder_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    if n == 0:
        return {}
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val = max(0, n_val + n_test)
        n_test = n - n_train - n_val
    if n_train + n_val + n_test != n:
        n_test = n - n_train - n_val
    # Guarantee every requested split gets a folder when n is large enough.
    if n >= 3:
        if n_train == 0:
            n_train, n_test = 1, n_test - 1
        if n_val == 0:
            n_val, n_train = 1, n_train - 1
        if n_test == 0:
            n_test, n_train = 1, n_train - 1
    assignment: dict[str, str] = {}
    index = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for folder_id in unique[index : index + count]:
            assignment[folder_id] = split
        index += count
    return assignment


def _assert_output_safe(config: LocalizationPrepConfig) -> None:
    source = config.source_dir
    output = config.output_dir
    try:
        output.relative_to(source)
        raise LocalizationPrepError(
            f"Refusing to write derived data inside the source tree: {output}"
        )
    except ValueError:
        pass
    forbidden = config.project_root / "dataset"
    try:
        output.resolve().relative_to(forbidden.resolve())
        raise LocalizationPrepError(
            f"Refusing to write into the existing dataset/ directory: {output}"
        )
    except ValueError:
        pass
    named = config.project_root / "dataset 16.09"
    try:
        output.resolve().relative_to(named.resolve())
        raise LocalizationPrepError(
            f"Refusing to write into dataset 16.09: {output}"
        )
    except ValueError:
        pass


def _prepare_output_dir(config: LocalizationPrepConfig) -> None:
    _assert_output_safe(config)
    out = config.output_dir
    if out.exists():
        nonempty = any(out.iterdir())
        if nonempty and not config.overwrite:
            raise LocalizationPrepError(
                f"Output directory already exists and is not empty: {out}. "
                "Pass overwrite=True / --overwrite to replace THIS derived "
                "directory only (source data is never deleted)."
            )
        if nonempty and config.overwrite:
            logger.warning("Removing existing derived localization dataset at %s", out)
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def process_pair(
    pair: FakeMaskPair,
    config: LocalizationPrepConfig,
    split: str,
) -> dict[str, object]:
    size = config.image_size
    with Image.open(pair.fake_path) as fake_im:
        fake_im = ImageOps.exif_transpose(fake_im)
        rgb = fake_im.convert("RGB")
        orig_w, orig_h = rgb.size
        geometry = compute_letterbox_geometry(orig_w, orig_h, size)
        out_rgb = apply_letterbox(
            rgb,
            geometry,
            size,
            resample=Image.Resampling.BILINEAR,
            fill=(0, 0, 0),
        )

    with Image.open(pair.mask_path) as mask_im:
        if mask_im.size != (orig_w, orig_h):
            raise LocalizationPrepError(
                f"Mask size {mask_im.size} != fake size {(orig_w, orig_h)} "
                f"for {pair.fake_path}"
            )
        binary, mask_type, unique_before = mask_to_binary_l(
            mask_im, config.mask_threshold
        )
        out_mask = apply_letterbox(
            binary,
            geometry,
            size,
            resample=Image.Resampling.NEAREST,
            fill=0,
        )

    stem = f"{pair.folder_id}__{pair.fake_path.stem}"
    image_name = f"{stem}.png"
    mask_name = f"{stem}.png"
    image_path = config.output_dir / split / "images" / image_name
    mask_path = config.output_dir / split / "masks" / mask_name
    if image_path.exists() or mask_path.exists():
        raise LocalizationPrepError(f"Refusing to overwrite existing pair: {stem}")
    out_rgb.save(image_path, format="PNG")
    out_mask.save(mask_path, format="PNG")

    mask_arr = np.asarray(out_mask, dtype=np.uint8)
    foreground = int((mask_arr == 255).sum())
    return {
        "split": split,
        "source_folder": pair.folder_id,
        "source_fake_path": _rel(pair.fake_path, config.project_root),
        "source_mask_path": _rel(pair.mask_path, config.project_root),
        "output_image_path": _rel(image_path, config.project_root),
        "output_mask_path": _rel(mask_path, config.project_root),
        "original_width": orig_w,
        "original_height": orig_h,
        "output_width": out_rgb.size[0],
        "output_height": out_rgb.size[1],
        "mask_type": mask_type,
        "mask_threshold": config.mask_threshold,
        "unique_values_before": ",".join(str(v) for v in unique_before[:32])
        + ("..." if len(unique_before) > 32 else ""),
        "unique_count_before": len(unique_before),
        "foreground_pixels": foreground,
        "pad_left": geometry.pad_left,
        "pad_top": geometry.pad_top,
        "resized_width": geometry.resized_width,
        "resized_height": geometry.resized_height,
        "non_standard": mask_type not in {"binary_0_255", "binary_0_1"},
    }


def validate_derived_dataset(
    config: LocalizationPrepConfig,
    rows: list[dict[str, object]],
    folder_to_split: dict[str, str],
) -> None:
    """Fail if the derived corpus violates the localization contract."""

    if not rows:
        raise LocalizationPrepError("No pairs were written.")

    seen_outputs: set[str] = set()
    folders_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        img = config.project_root / str(row["output_image_path"])
        msk = config.project_root / str(row["output_mask_path"])
        if not img.is_file() or not msk.is_file():
            raise LocalizationPrepError(f"Missing output files for {row}")
        key = str(row["output_image_path"])
        if key in seen_outputs:
            raise LocalizationPrepError(f"Duplicate generated pair: {key}")
        seen_outputs.add(key)
        with Image.open(img) as im:
            if im.size != (config.image_size, config.image_size):
                raise LocalizationPrepError(f"Image not {config.image_size}: {img} {im.size}")
            if im.mode != "RGB":
                raise LocalizationPrepError(f"Image mode {im.mode} != RGB: {img}")
        with Image.open(msk) as mm:
            if mm.size != (config.image_size, config.image_size):
                raise LocalizationPrepError(f"Mask not {config.image_size}: {msk} {mm.size}")
            if mm.mode != "L":
                raise LocalizationPrepError(f"Mask mode {mm.mode} != L: {msk}")
            arr = np.asarray(mm)
            extra = set(np.unique(arr).tolist()) - {0, 255}
            if extra:
                raise LocalizationPrepError(f"Mask not binary {sorted(extra)[:8]}: {msk}")
        folders_by_split[str(row["split"])].add(str(row["source_folder"]))

    overlap = (
        (folders_by_split["train"] & folders_by_split["val"])
        | (folders_by_split["train"] & folders_by_split["test"])
        | (folders_by_split["val"] & folders_by_split["test"])
    )
    if overlap:
        raise LocalizationPrepError(f"Parent-folder overlap between splits: {sorted(overlap)}")

    for folder_id, split in folder_to_split.items():
        observed = [s for s, folders in folders_by_split.items() if folder_id in folders]
        if observed and set(observed) != {split}:
            raise LocalizationPrepError(
                f"Folder {folder_id} assigned {split} but found in {observed}"
            )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def _write_readme(path: Path, config: LocalizationPrepConfig, report: PrepReport) -> None:
    text = f"""# MAYA localization derived dataset

Read-only source: `{_rel(config.source_dir, config.project_root)}`

This directory is a **derived** pixel-level localization corpus. The source
tree `dataset 16.09` and the existing `dataset/` classifier corpus were not
modified.

## Geometry

- Model input size: **{config.image_size}×{config.image_size}**
- Aspect ratio is preserved
- The pair is scaled with `scale = min(size/w, size/h)` so it fits inside the square
- Remaining borders are **center-padded with 0** (black / background)
- Images: bilinear resize, then pad, saved as RGB PNG
- Masks: binarized at original resolution, **nearest-neighbor** resize, then the **same** pad

## Mask binarization

- Converted to single-channel `L`
- RGBA masks are composited onto black using the alpha channel, then converted to `L`
- If `max(pixel) <= 1`, treat as a 0/1 mask (`> 0` → 255)
- Otherwise foreground is `pixel > {config.mask_threshold}` → 255, else 0
- Default threshold **{config.mask_threshold}** leaves `{0, 255}` masks unchanged
- The five non-standard IMD2020 masks are **kept** and thresholded, not dropped

## Split

- Unit of split: **parent sample folder** (all fakes from one folder stay together)
- Ratios: train {config.train_ratio:.0%} / val {config.val_ratio:.0%} / test {config.test_ratio:.0%}
- Seed: **{config.seed}** (`random.Random`, folders sorted then shuffled)

## Counts

- Total pairs: {report.total_pairs}
- Train pairs: {report.train_pairs} ({report.train_folders} folders)
- Val pairs: {report.val_pairs} ({report.val_folders} folders)
- Test pairs: {report.test_pairs} ({report.test_folders} folders)
- Non-standard masks normalized: {report.non_standard_normalized} / {report.non_standard_masks}
- Failures: {report.failures}
- Parent-folder overlap: **{"YES — INVALID" if report.parent_folder_overlap else "ZERO"}**
"""
    path.write_text(text, encoding="utf-8")


def _write_audit(path: Path, report: PrepReport) -> None:
    lines = [
        "# Localization dataset preparation audit",
        "",
        f"- Total pairs: `{report.total_pairs}`",
        f"- Train pairs: `{report.train_pairs}`",
        f"- Validation pairs: `{report.val_pairs}`",
        f"- Test pairs: `{report.test_pairs}`",
        f"- Train source folders: `{report.train_folders}`",
        f"- Validation source folders: `{report.val_folders}`",
        f"- Test source folders: `{report.test_folders}`",
        f"- Min fakes / source folder: `{report.min_fakes_per_folder}`",
        f"- Max fakes / source folder: `{report.max_fakes_per_folder}`",
        f"- Mean fakes / source folder: `{report.mean_fakes_per_folder:.4f}`",
        f"- Non-standard masks: `{report.non_standard_masks}`",
        f"- Successfully normalized: `{report.non_standard_normalized}`",
        f"- Failures: `{report.failures}`",
        f"- Foreground (255) pixels: `{report.foreground_pixels}`",
        f"- Background (0) pixels: `{report.background_pixels}`",
        f"- Parent-folder overlap: `{report.parent_folder_overlap}`",
        f"- Mask threshold: `{report.mask_threshold}`",
        f"- Seed: `{report.seed}`",
        f"- Output: `{report.output_dir}`",
        "",
        "## Non-standard masks",
        "",
    ]
    if not report.non_standard_details:
        lines.append("None.")
    else:
        for item in report.non_standard_details:
            lines.append(
                f"- `{item.get('source_mask_path')}` type=`{item.get('mask_type')}` "
                f"unique_before=`{item.get('unique_values_before')}` "
                f"→ binary via threshold `{report.mask_threshold}`"
            )
    if report.failure_details:
        lines.extend(["", "## Failures", ""])
        for item in report.failure_details:
            lines.append(f"- `{item}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_localization_dataset(config: LocalizationPrepConfig) -> PrepReport:
    """Discover pairs, split by parent folder, write 256×256 derived files."""

    _prepare_output_dir(config)
    pairs, discover_issues = discover_pairs(config.source_dir)
    if not pairs:
        raise LocalizationPrepError(f"No fake/mask pairs found under {config.source_dir}")

    folder_counts: dict[str, int] = Counter(p.folder_id for p in pairs)
    folder_to_split = assign_folder_splits(
        list(folder_counts),
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    non_standard: list[dict[str, str]] = []
    non_standard_ok = 0

    for pair in pairs:
        split = folder_to_split[pair.folder_id]
        try:
            row = process_pair(pair, config, split)
        except (UnidentifiedImageError, OSError, ValueError, LocalizationPrepError) as exc:
            failures.append(
                {
                    "fake": str(pair.fake_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            logger.warning("Failed pair %s: %s", pair.fake_path, exc)
            continue
        rows.append(row)
        if row["non_standard"]:
            non_standard.append(
                {
                    "source_mask_path": str(row["source_mask_path"]),
                    "mask_type": str(row["mask_type"]),
                    "unique_values_before": str(row["unique_values_before"]),
                }
            )
            non_standard_ok += 1

    validate_derived_dataset(config, rows, folder_to_split)

    counts = Counter(str(r["split"]) for r in rows)
    folders_by_split = {
        split: {str(r["source_folder"]) for r in rows if r["split"] == split}
        for split in SPLITS
    }
    overlap = bool(
        (folders_by_split["train"] & folders_by_split["val"])
        | (folders_by_split["train"] & folders_by_split["test"])
        | (folders_by_split["val"] & folders_by_split["test"])
    )
    per_folder = list(folder_counts.values())
    fg = sum(int(r["foreground_pixels"]) for r in rows)
    total_px = len(rows) * config.image_size * config.image_size
    report = PrepReport(
        total_pairs=len(rows),
        train_pairs=counts["train"],
        val_pairs=counts["val"],
        test_pairs=counts["test"],
        train_folders=len(folders_by_split["train"]),
        val_folders=len(folders_by_split["val"]),
        test_folders=len(folders_by_split["test"]),
        min_fakes_per_folder=min(per_folder),
        max_fakes_per_folder=max(per_folder),
        mean_fakes_per_folder=float(statistics.mean(per_folder)),
        non_standard_masks=len(non_standard),
        non_standard_normalized=non_standard_ok,
        failures=len(failures),
        foreground_pixels=fg,
        background_pixels=total_px - fg,
        parent_folder_overlap=overlap,
        output_dir=str(config.output_dir),
        mask_threshold=config.mask_threshold,
        seed=config.seed,
        non_standard_details=non_standard,
        failure_details=failures + discover_issues,
    )

    _write_manifest(config.output_dir / "manifest.csv", rows)
    _write_readme(config.output_dir / "README.md", config, report)
    _write_audit(config.output_dir / "AUDIT.md", report)
    (config.output_dir / "prep_report.json").write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    return report


def format_audit_report(report: PrepReport) -> str:
    overlap = "ZERO (confirmed)" if not report.parent_folder_overlap else "OVERLAP DETECTED"
    fg_frac = 0.0
    total_px = report.foreground_pixels + report.background_pixels
    if total_px:
        fg_frac = 100.0 * report.foreground_pixels / total_px
    return "\n".join(
        [
            "Localization dataset preparation complete.",
            f"  total pairs              : {report.total_pairs}",
            f"  train pairs              : {report.train_pairs}",
            f"  validation pairs         : {report.val_pairs}",
            f"  test pairs               : {report.test_pairs}",
            f"  train source folders     : {report.train_folders}",
            f"  validation source folders: {report.val_folders}",
            f"  test source folders      : {report.test_folders}",
            f"  min fakes / folder       : {report.min_fakes_per_folder}",
            f"  max fakes / folder       : {report.max_fakes_per_folder}",
            f"  mean fakes / folder      : {report.mean_fakes_per_folder:.4f}",
            f"  non-standard masks       : {report.non_standard_masks}",
            f"  successfully normalized  : {report.non_standard_normalized}",
            f"  failures                 : {report.failures}",
            f"  mask pixels 255 (fg)     : {report.foreground_pixels} ({fg_frac:.4f}%)",
            f"  mask pixels 0 (bg)       : {report.background_pixels}",
            f"  parent-folder overlap    : {overlap}",
            f"  threshold                : {report.mask_threshold}",
            f"  seed                     : {report.seed}",
            f"  output                   : {report.output_dir}",
        ]
    )
