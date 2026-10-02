"""Crop one detected face with context, clipped to the source frame."""

from __future__ import annotations

import numpy as np

from .exceptions import FacePreprocessingError
from .face_detector import DetectedFace


def crop_face(frame: np.ndarray, face: DetectedFace, margin: float) -> np.ndarray | None:
    """Return a uint8 BGR crop, or None when the padded box is empty."""
    if margin < 0:
        raise FacePreprocessingError(f"Face crop margin must be non-negative, got {margin}.")
    height, width = frame.shape[:2]
    pad_x = int(round(face.width * margin))
    pad_y = int(round(face.height * margin))
    x1 = max(0, face.x - pad_x)
    y1 = max(0, face.y - pad_y)
    x2 = min(width, face.x + face.width + pad_x)
    y2 = min(height, face.y + face.height + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()
