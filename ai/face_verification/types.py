"""Dataclasses for face detection, quality, and verification outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DECISION_MATCH = "MATCH"
DECISION_NO_MATCH = "NO_MATCH"
DECISION_INCONCLUSIVE = "INCONCLUSIVE"

STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_PROCESSING = "PROCESSING"

REASON_OK = "OK"
REASON_NO_FACE_REFERENCE = "NO_FACE_REFERENCE"
REASON_NO_FACE_EVIDENCE = "NO_FACE_EVIDENCE"
REASON_MULTIPLE_FACES_REFERENCE = "MULTIPLE_FACES_REFERENCE"
REASON_MULTIPLE_FACES_EVIDENCE = "MULTIPLE_FACES_EVIDENCE"
REASON_LOW_QUALITY_REFERENCE = "LOW_QUALITY_REFERENCE"
REASON_LOW_QUALITY_EVIDENCE = "LOW_QUALITY_EVIDENCE"
REASON_UNUSABLE_REFERENCE = "UNUSABLE_REFERENCE"
REASON_UNUSABLE_EVIDENCE = "UNUSABLE_EVIDENCE"
REASON_EMBEDDING_FAILURE = "EMBEDDING_FAILURE"
REASON_UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE"
REASON_CORRUPTED_IMAGE = "CORRUPTED_IMAGE"
REASON_INCONCLUSIVE_SCORE = "INCONCLUSIVE_SCORE"


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "score": self.score,
        }


@dataclass
class FaceQuality:
    usable: bool
    sharpness: float
    brightness: float
    width: int
    height: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "sharpness": self.sharpness,
            "brightness": self.brightness,
            "width": self.width,
            "height": self.height,
            "reasons": list(self.reasons),
        }


@dataclass
class DetectedFace:
    bbox: BoundingBox
    crop: np.ndarray
    quality: FaceQuality
    landmarks: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bbox": self.bbox.to_dict(),
            "quality": self.quality.to_dict(),
        }
        if self.landmarks:
            payload["landmarks"] = self.landmarks
        return payload


@dataclass
class FaceVerificationResult:
    verification_status: str
    decision: str
    similarity_score: float | None
    distance_score: float | None
    threshold: float
    no_match_threshold: float
    model_name: str
    model_version: str
    engine_name: str
    reference_face_count: int
    evidence_face_count: int
    reason_code: str
    reference_metadata: dict[str, Any] = field(default_factory=dict)
    evidence_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    reference_crop: np.ndarray | None = None
    evidence_crop: np.ndarray | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "verification_status": self.verification_status,
            "decision": self.decision,
            "similarity_score": self.similarity_score,
            "distance_score": self.distance_score,
            "threshold": self.threshold,
            "no_match_threshold": self.no_match_threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "engine_name": self.engine_name,
            "reference_face_count": self.reference_face_count,
            "evidence_face_count": self.evidence_face_count,
            "reason_code": self.reason_code,
            "reference_face": self.reference_metadata,
            "evidence_face": self.evidence_metadata,
            "error": self.error,
        }
