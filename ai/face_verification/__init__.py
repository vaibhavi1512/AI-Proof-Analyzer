"""MAYA Phase 5 face reference verification."""

from ai.face_verification.config import FaceVerificationConfig, get_face_verification_config
from ai.face_verification.pipeline import run_face_verification
from ai.face_verification.types import FaceVerificationResult

__all__ = [
    "FaceVerificationConfig",
    "FaceVerificationResult",
    "get_face_verification_config",
    "run_face_verification",
]
