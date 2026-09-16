"""Configurable PyTorch Dataset for MAYA processed splits.

Purpose:
    Lazy-load processed images for training/evaluation without hardcoding paths.
"""

from __future__ import annotations

import logging
import csv
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch.utils.data import Dataset

from ai.datasets.dataset_config import (
    CLASS_ALIASES,
    ClassName,
    DatasetConfig,
    SplitName,
    get_dataset_config,
)
from ai.datasets.utils.fs import is_supported_image
from ai.datasets.utils.images import open_rgb

logger = logging.getLogger("maya.datasets.torch_dataset")

TransformType = Callable[[Image.Image], Any]


def _label_from_folder(name: str) -> int:
    """Map folder name to numeric label (REAL=0, FAKE=1)."""

    cls = CLASS_ALIASES.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown class folder: {name}")
    return 0 if cls == ClassName.REAL else 1


class MayaImageDataset(Dataset):
    """Folder-based image dataset for a single processed split.

    Expected layout::

        root/REAL/*.jpg
        root/FAKE/*.jpg

    Images are opened on demand (lazy) to keep RAM usage low.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        transform: TransformType | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or get_dataset_config().supported_extensions
        self.samples: list[tuple[Path, int]] = []
        self.sample_group_ids: list[str] = []
        self.class_to_idx = {ClassName.REAL.value: 0, ClassName.FAKE.value: 1}
        self._index_samples()

    def _index_samples(self) -> None:
        if not self.root.exists():
            logger.error("Dataset root missing: %s", self.root)
            return

        try:
            class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
        except OSError as exc:
            logger.error("Cannot list dataset root %s: %s", self.root, exc)
            return

        manifest_groups = self._manifest_groups()
        for class_dir in class_dirs:
            try:
                label = _label_from_folder(class_dir.name)
            except ValueError:
                logger.warning("Skipping non-class folder: %s", class_dir)
                continue

            try:
                children = sorted(class_dir.iterdir())
            except OSError as exc:
                logger.warning("Cannot list %s: %s", class_dir, exc)
                continue

            for path in children:
                if is_supported_image(path, self.extensions):
                    self.samples.append((path, label))
                    group_id = manifest_groups.get(str(path.resolve()))
                    self.sample_group_ids.append(group_id or str(path.resolve()))

        logger.info("Indexed %s samples from %s", len(self.samples), self.root)

    def _manifest_groups(self) -> dict[str, str]:
        manifest = self.root.parent / "manifest.csv"
        if not manifest.is_file():
            logger.warning(
                "Dataset manifest missing; using per-file groups for %s: %s",
                self.root,
                manifest,
            )
            return {}
        groups: dict[str, str] = {}
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                output_path = row.get("output_path", "")
                if not output_path:
                    continue
                normalized = output_path.replace("\\", os.sep).replace("/", os.sep)
                groups[str((self.root.parents[2] / normalized).resolve())] = row["case_id"]
        return groups

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        path, label = self.samples[index]
        try:
            image = open_rgb(path)
        except (OSError, ValueError) as exc:
            logger.error("Failed to read %s: %s", path, exc)
            raise
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_split_dataset(
    split: SplitName | str,
    *,
    config: DatasetConfig | None = None,
    transform: TransformType | None = None,
) -> MayaImageDataset:
    """Construct a dataset for train/validation/test using config paths."""

    cfg = config or get_dataset_config()
    split_name = SplitName(split) if not isinstance(split, SplitName) else split
    root = cfg.split_dir(split_name)
    return MayaImageDataset(root, transform=transform, extensions=cfg.supported_extensions)
