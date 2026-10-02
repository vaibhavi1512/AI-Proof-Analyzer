"""Deterministic face selection that prefers the face continuing from the previous frame.

The first detection is chosen with face size and image position. Later frames
keep a candidate that overlaps or stays near the previous box. A slightly
larger face in another part of the picture does not take over the track.
"""

from __future__ import annotations

import math

from .face_detector import DetectedFace

# Size still matters, but a slightly larger face far from the image center loses.
INITIAL_POSITION_WEIGHT = 0.5
# A candidate continues the track when either signal is strong.
MIN_CONTINUITY_IOU = 0.10
MAX_CENTER_SHIFT_RATIO = 1.5

METHOD_INITIAL = "initial"
METHOD_TEMPORAL = "temporal_continuity"
METHOD_REINITIALIZE = "reinitialize"
METHOD_FULL_FRAME = "full_frame_fallback"


def _center(face: DetectedFace) -> tuple[float, float]:
    return (face.x + face.width / 2.0, face.y + face.height / 2.0)


def box_iou(first: DetectedFace, second: DetectedFace) -> float:
    """Intersection over union of two face boxes."""
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first.area + second.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_distance(first: DetectedFace, second: DetectedFace) -> float:
    x1, y1 = _center(first)
    x2, y2 = _center(second)
    return math.hypot(x1 - x2, y1 - y2)


def _confidence(face: DetectedFace) -> float:
    if face.confidence is None:
        return float("-inf")
    return float(face.confidence)


def select_initial_face(
    faces: list[DetectedFace],
    image_width: int,
    image_height: int,
) -> DetectedFace:
    """Pick an opening face from size and how close it is to the image center."""
    if not faces:
        raise ValueError("Initial selection requires at least one face.")
    max_area = max(face.area for face in faces) or 1
    diagonal = math.hypot(image_width, image_height) or 1.0
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    def sort_key(face: DetectedFace) -> tuple[float, int, float, int, int]:
        center_x, center_y = _center(face)
        distance = math.hypot(center_x - image_center_x, center_y - image_center_y) / diagonal
        score = (face.area / max_area) - INITIAL_POSITION_WEIGHT * distance
        return (score, face.area, _confidence(face), -face.x, -face.y)

    return max(faces, key=sort_key)


def _continues_track(candidate: DetectedFace, previous: DetectedFace) -> bool:
    overlap = box_iou(candidate, previous)
    if overlap >= MIN_CONTINUITY_IOU:
        return True
    limit = MAX_CENTER_SHIFT_RATIO * max(previous.width, previous.height, 1)
    return center_distance(candidate, previous) <= limit


def choose_face(
    faces: list[DetectedFace],
    previous: DetectedFace | None,
    image_width: int,
    image_height: int,
) -> tuple[DetectedFace | None, str]:
    """Select one face, or none when the frame has no usable detection."""
    if not faces:
        return None, METHOD_FULL_FRAME
    if previous is None:
        return select_initial_face(faces, image_width, image_height), METHOD_INITIAL
    continuing = [face for face in faces if _continues_track(face, previous)]
    if not continuing:
        return select_initial_face(faces, image_width, image_height), METHOD_REINITIALIZE

    def sort_key(face: DetectedFace) -> tuple[float, float, int, float, int, int]:
        return (
            box_iou(face, previous),
            -center_distance(face, previous),
            face.area,
            _confidence(face),
            -face.x,
            -face.y,
        )

    return max(continuing, key=sort_key), METHOD_TEMPORAL
