"""CPU face detection. The default detector is OpenCV's bundled Haar cascade."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import FacePreprocessConfig
from .exceptions import FacePreprocessingError


@dataclass(frozen=True)
class DetectedFace:
    """One face box in pixel coordinates of the original BGR frame."""

    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None

    @property
    def area(self) -> int:
        return self.width * self.height


def select_primary_face(faces: list[DetectedFace]) -> DetectedFace:
    """Choose the largest face. Ties break toward higher confidence, then upper-left."""
    if not faces:
        raise FacePreprocessingError("Cannot select a face from an empty detection list.")

    def sort_key(face: DetectedFace) -> tuple[int, float, int, int]:
        confidence = face.confidence if face.confidence is not None else float("-inf")
        return (face.area, confidence, -face.x, -face.y)

    return max(faces, key=sort_key)


class HaarFaceDetector:
    """Frontal-face Haar cascade. It does not require a GPU or an extra model file."""

    def __init__(self, config: FacePreprocessConfig | None = None) -> None:
        self.config = config or FacePreprocessConfig()
        cascade_path = cv2.data.haarcascades + self.config.cascade_name
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise FacePreprocessingError(f"Could not load the OpenCV face cascade: {cascade_path}")

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Return useful faces in a deterministic order. An empty list means no usable face."""
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise FacePreprocessingError(
                f"Face detection expected a uint8 BGR frame, got shape {getattr(frame, 'shape', None)}."
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles, _reject_levels, level_weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.config.scale_factor,
            minNeighbors=self.config.min_neighbors,
            minSize=(self.config.min_face_size, self.config.min_face_size),
            outputRejectLevels=True,
        )
        faces: list[DetectedFace] = []
        weights = list(level_weights) if level_weights is not None else []
        for index, rectangle in enumerate(rectangles):
            x, y, width, height = (int(value) for value in rectangle)
            confidence = float(weights[index]) if index < len(weights) else None
            if confidence is not None and confidence < self.config.min_confidence:
                continue
            if width < self.config.min_face_size or height < self.config.min_face_size:
                continue
            faces.append(DetectedFace(x=x, y=y, width=width, height=height, confidence=confidence))
        faces.sort(key=lambda face: (-face.area, -(face.confidence or float("-inf")), face.x, face.y))
        return faces
