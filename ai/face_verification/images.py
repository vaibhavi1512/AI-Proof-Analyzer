"""Image loading helpers for face verification (Pillow + numpy)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from ai.face_verification.types import REASON_CORRUPTED_IMAGE, REASON_UNSUPPORTED_IMAGE

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceImageError(ValueError):
    def __init__(self, message: str, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def load_rgb_image(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise FaceImageError(
            f"Unsupported image type '{suffix}'",
            REASON_UNSUPPORTED_IMAGE,
        )
    try:
        with Image.open(path) as img:
            img.load()
            rgb = img.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise FaceImageError("Corrupted or unreadable image", REASON_CORRUPTED_IMAGE) from exc
    except OSError as exc:
        raise FaceImageError("Corrupted or unreadable image", REASON_CORRUPTED_IMAGE) from exc
    if array.size == 0 or array.ndim != 3:
        raise FaceImageError("Corrupted or unreadable image", REASON_CORRUPTED_IMAGE)
    return array


def crop_with_padding(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: float,
) -> np.ndarray:
    h, w = image.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    xa = max(0, x1 - pad_x)
    ya = max(0, y1 - pad_y)
    xb = min(w, x2 + pad_x)
    yb = min(h, y2 + pad_y)
    if xb <= xa or yb <= ya:
        return image[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)].copy()
    return image[ya:yb, xa:xb].copy()
