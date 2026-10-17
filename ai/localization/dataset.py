"""Paired loader for the prepared localization image and binary mask splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "val", "test")


class LocalizationDataset(Dataset):
    """Load filename-matched RGB images and L-mode binary masks lazily."""
    def __init__(self, root: str | Path, split: str):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.images_dir = self.root / split / "images"
        self.masks_dir = self.root / split / "masks"
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Missing image/mask directory for split {split}: {self.root}")
        images = {p.name: p for p in self.images_dir.iterdir() if p.is_file()}
        masks = {p.name: p for p in self.masks_dir.iterdir() if p.is_file()}
        if not images:
            raise ValueError(f"No localization images found in {self.images_dir}")
        if images.keys() != masks.keys():
            missing_masks = sorted(images.keys() - masks.keys())[:5]
            missing_images = sorted(masks.keys() - images.keys())[:5]
            raise ValueError(f"Unpaired files in {split}: missing masks={missing_masks}, missing images={missing_images}")
        self.pairs = [(images[name], masks[name]) for name in sorted(images)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            with Image.open(mask_path) as mm:
                mask = mm.convert("L")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read localization pair {image_path.name}") from exc
        if image.size != (256, 256) or mask.size != (256, 256):
            raise ValueError(f"Expected prepared 256x256 pair, got {image.size} and {mask.size}")
        img = np.asarray(image, dtype=np.float32).copy() / 255.0
        msk = np.asarray(mask, dtype=np.uint8).copy()
        values = set(np.unique(msk).tolist())
        if values - {0, 255}:
            raise ValueError(f"Mask {mask_path} is not binary: {sorted(values)[:8]}")
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy((msk > 0).astype(np.float32))[None]


def create_localization_loader(root: str | Path, split: str, *, batch_size: int = 4, num_workers: int = 0, shuffle: bool | None = None) -> DataLoader:
    dataset = LocalizationDataset(root, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train" if shuffle is None else shuffle), num_workers=num_workers)
