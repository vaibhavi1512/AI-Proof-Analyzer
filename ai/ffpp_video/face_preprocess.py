"""Turn each sampled frame into one normalized model input.

A detected face is cropped and then passed through the Phase 2 preprocessor.
If no usable face is found, the same preprocessor runs on the full frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import FacePreprocessConfig
from .exceptions import FacePreprocessingError
from .face_crop import crop_face
from .face_detector import DetectedFace, HaarFaceDetector
from .face_selection import METHOD_FULL_FRAME, choose_face
from .frame_preprocess import preprocess_frame
from .sampling import SampledFrame, SampledVideo


@dataclass(frozen=True)
class FrameMetadata:
    """How one sampled frame became its model input."""

    frame_index: int
    timestamp_seconds: float
    face_detected: bool
    number_of_faces: int
    selected_face_bbox: tuple[int, int, int, int] | None
    selected_face_confidence: float | None
    used_face_crop: bool
    used_full_frame_fallback: bool
    detected_face_count: int
    selection_method: str

    def as_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "face_detected": self.face_detected,
            "number_of_faces": self.number_of_faces,
            "detected_face_count": self.detected_face_count,
            "selected_face_bbox": None if self.selected_face_bbox is None else list(self.selected_face_bbox),
            "selected_face_confidence": self.selected_face_confidence,
            "used_face_crop": self.used_face_crop,
            "used_full_frame_fallback": self.used_full_frame_fallback,
            "selection_method": self.selection_method,
        }


@dataclass(frozen=True)
class ProcessedFrame:
    """One float32 model input and the detection record for that frame."""

    array: np.ndarray
    metadata: FrameMetadata
    selected_face: DetectedFace | None


@dataclass(frozen=True)
class ProcessedVideo:
    """Sixteen model inputs in the same order as the sampled frames."""

    arrays: np.ndarray
    frames: tuple[ProcessedFrame, ...]

    @property
    def face_crop_count(self) -> int:
        return sum(1 for frame in self.frames if frame.metadata.used_face_crop)

    @property
    def fallback_count(self) -> int:
        return sum(1 for frame in self.frames if frame.metadata.used_full_frame_fallback)

    @property
    def multi_face_count(self) -> int:
        return sum(1 for frame in self.frames if frame.metadata.number_of_faces > 1)


def _bbox(face: DetectedFace) -> tuple[int, int, int, int]:
    return (face.x, face.y, face.width, face.height)


def process_sampled_frame(
    sampled: SampledFrame,
    detector: HaarFaceDetector | object,
    config: FacePreprocessConfig | None = None,
    previous_face: DetectedFace | None = None,
) -> ProcessedFrame:
    """Detect a face or fall back to the full frame. Always returns one input."""
    settings = config or FacePreprocessConfig()
    frame = sampled.frame
    faces = list(detector.detect(frame))
    image_height, image_width = frame.shape[:2]
    selected, selection_method = choose_face(faces, previous_face, image_width, image_height)
    cropped = None
    if selected is not None:
        cropped = crop_face(frame, selected, settings.margin)
    use_crop = cropped is not None
    if not use_crop:
        selection_method = METHOD_FULL_FRAME
    source = cropped if use_crop else frame
    array = preprocess_frame(source, image_size=settings.image_size)
    if array.dtype != np.float32 or array.shape != (3, settings.image_size, settings.image_size):
        raise FacePreprocessingError(
            f"Frame {sampled.frame_index} produced shape {array.shape} {array.dtype}."
        )
    if not np.isfinite(array).all():
        raise FacePreprocessingError(f"Frame {sampled.frame_index} produced non-finite values.")
    metadata = FrameMetadata(
        frame_index=int(sampled.frame_index),
        timestamp_seconds=float(sampled.timestamp_seconds),
        face_detected=bool(faces),
        number_of_faces=len(faces),
        selected_face_bbox=_bbox(selected) if use_crop and selected is not None else None,
        selected_face_confidence=selected.confidence if use_crop and selected is not None else None,
        used_face_crop=use_crop,
        used_full_frame_fallback=not use_crop,
        detected_face_count=len(faces),
        selection_method=selection_method,
    )
    if metadata.used_face_crop == metadata.used_full_frame_fallback:
        raise FacePreprocessingError(f"Frame {sampled.frame_index} did not choose exactly one input path.")
    kept_face = selected if use_crop else None
    return ProcessedFrame(array=array, metadata=metadata, selected_face=kept_face)


def process_sampled_video(
    sampled_video: SampledVideo,
    detector: HaarFaceDetector | object | None = None,
    config: FacePreprocessConfig | None = None,
) -> ProcessedVideo:
    """Process every sampled frame in order. Missing faces do not drop a frame."""
    settings = config or FacePreprocessConfig()
    if len(sampled_video.frames) != settings.num_frames:
        raise FacePreprocessingError(
            f"Expected {settings.num_frames} sampled frames, got {len(sampled_video.frames)}."
        )
    active_detector = detector if detector is not None else HaarFaceDetector(settings)
    processed_frames: list[ProcessedFrame] = []
    previous_face: DetectedFace | None = None
    for frame in sampled_video.frames:
        processed_frame = process_sampled_frame(frame, active_detector, settings, previous_face)
        processed_frames.append(processed_frame)
        if processed_frame.selected_face is not None:
            previous_face = processed_frame.selected_face
    processed = tuple(processed_frames)
    if len(processed) != settings.num_frames:
        raise FacePreprocessingError(
            f"Expected {settings.num_frames} processed frames, got {len(processed)}."
        )
    indices = [frame.metadata.frame_index for frame in processed]
    if indices != [frame.frame_index for frame in sampled_video.frames]:
        raise FacePreprocessingError("Processed frames are not in sampled order.")
    arrays = np.stack([frame.array for frame in processed], axis=0)
    expected = (settings.num_frames, 3, settings.image_size, settings.image_size)
    if arrays.shape != expected or arrays.dtype != np.float32 or not np.isfinite(arrays).all():
        raise FacePreprocessingError(f"Video tensor is {arrays.shape} {arrays.dtype}, expected {expected} float32.")
    return ProcessedVideo(arrays=arrays, frames=processed)
