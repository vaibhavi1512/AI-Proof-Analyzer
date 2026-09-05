"""Face crop quality checks (size, sharpness, brightness)."""

from __future__ import annotations

import numpy as np

from ai.face_verification.config import FaceVerificationConfig
from ai.face_verification.types import FaceQuality


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    # RGB
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)


def laplacian_variance(gray: np.ndarray) -> float:
    """Approximate sharpness without requiring a particular OpenCV API."""

    arr = np.asarray(gray, dtype=np.float64)
    if arr.size < 9:
        return 0.0
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = np.pad(arr, 1, mode="edge")
    acc = np.zeros_like(arr)
    for i in range(3):
        for j in range(3):
            acc += kernel[i, j] * padded[i : i + arr.shape[0], j : j + arr.shape[1]]
    return float(acc.var())


def assess_face_crop(crop: np.ndarray, config: FaceVerificationConfig) -> FaceQuality:
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return FaceQuality(
            usable=False,
            sharpness=0.0,
            brightness=0.0,
            width=0,
            height=0,
            reasons=["empty_crop"],
        )

    height, width = int(crop.shape[0]), int(crop.shape[1])
    gray = _to_gray(crop)
    sharpness = laplacian_variance(gray)
    brightness = float(np.mean(gray))
    reasons: list[str] = []
    if min(width, height) < config.min_face_size:
        reasons.append("face_too_small")
    if sharpness < config.min_sharpness:
        reasons.append("low_sharpness")
    if brightness < config.min_brightness:
        reasons.append("too_dark")
    if brightness > config.max_brightness:
        reasons.append("too_bright")
    return FaceQuality(
        usable=not reasons,
        sharpness=sharpness,
        brightness=brightness,
        width=width,
        height=height,
        reasons=reasons,
    )
