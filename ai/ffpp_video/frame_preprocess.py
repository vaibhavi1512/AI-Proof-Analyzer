"""Prepare one OpenCV frame for a later PyTorch / EfficientNet model."""

from __future__ import annotations

import cv2
import numpy as np

from .constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def preprocess_frame(frame: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Convert one BGR frame to a normalized RGB tensor array.

    The result has shape ``(3, image_size, image_size)`` and dtype ``float32``.
    Values are scaled to ``[0, 1]`` and then normalized with ImageNet mean and
    standard deviation. Channel order is RGB, so a later pipeline can wrap the
    array with ``torch.from_numpy`` without rearranging channels.
    """
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise TypeError("image_size must be an int")
    if image_size < 1:
        raise ValueError("image_size must be at least 1")
    if not isinstance(frame, np.ndarray):
        raise TypeError("preprocess_frame expected a numpy image array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            "preprocess_frame expected an OpenCV BGR image with shape (height, width, 3), "
            f"got shape {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(
            "preprocess_frame expected a uint8 BGR frame from OpenCV, "
            f"got dtype {frame.dtype}"
        )
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("preprocess_frame received an empty frame")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    pixels = resized.astype(np.float32) / np.float32(255.0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    normalized = (pixels - mean) / std
    channels_first = np.transpose(normalized, (2, 0, 1))
    return np.ascontiguousarray(channels_first, dtype=np.float32)
