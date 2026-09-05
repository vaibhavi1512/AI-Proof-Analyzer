"""Face reference vs evidence verification pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ai.face_verification.compare import cosine_similarity, decide_verification, euclidean_distance
from ai.face_verification.config import FaceVerificationConfig
from ai.face_verification.engine import FaceEngine, get_face_engine
from ai.face_verification.images import FaceImageError, load_rgb_image
from ai.face_verification.types import (
    DECISION_INCONCLUSIVE,
    DetectedFace,
    FaceVerificationResult,
    REASON_EMBEDDING_FAILURE,
    REASON_LOW_QUALITY_EVIDENCE,
    REASON_LOW_QUALITY_REFERENCE,
    REASON_MULTIPLE_FACES_EVIDENCE,
    REASON_MULTIPLE_FACES_REFERENCE,
    REASON_NO_FACE_EVIDENCE,
    REASON_NO_FACE_REFERENCE,
    REASON_UNUSABLE_EVIDENCE,
    REASON_UNUSABLE_REFERENCE,
    STATUS_COMPLETED,
    STATUS_FAILED,
)

logger = logging.getLogger("maya.ai.face_verification.pipeline")


def _inconclusive(
    *,
    config: FaceVerificationConfig,
    engine: FaceEngine,
    reason: str,
    ref_count: int,
    evd_count: int,
    ref_meta: dict | None = None,
    evd_meta: dict | None = None,
    ref_crop: np.ndarray | None = None,
    evd_crop: np.ndarray | None = None,
) -> FaceVerificationResult:
    return FaceVerificationResult(
        verification_status=STATUS_COMPLETED,
        decision=DECISION_INCONCLUSIVE,
        similarity_score=None,
        distance_score=None,
        threshold=config.match_threshold,
        no_match_threshold=config.no_match_threshold,
        model_name=engine.model_name,
        model_version=engine.model_version,
        engine_name=engine.name,
        reference_face_count=ref_count,
        evidence_face_count=evd_count,
        reason_code=reason,
        reference_metadata=ref_meta or {},
        evidence_metadata=evd_meta or {},
        reference_crop=ref_crop,
        evidence_crop=evd_crop,
    )


def _select_single_face(
    faces: list[DetectedFace],
    *,
    none_reason: str,
    multi_reason: str,
    low_quality_reason: str,
    unusable_reason: str,
) -> tuple[DetectedFace | None, str | None]:
    if not faces:
        return None, none_reason
    if len(faces) > 1:
        return None, multi_reason
    face = faces[0]
    if face.crop is None or face.crop.size == 0:
        return None, unusable_reason
    if not face.quality.usable:
        return None, low_quality_reason
    return face, None


def run_face_verification(
    reference_path: Path,
    evidence_path: Path,
    *,
    config: FaceVerificationConfig | None = None,
    engine: FaceEngine | None = None,
) -> FaceVerificationResult:
    """Compare a reference face image to an evidence image.

    Does not assume either image contains exactly one face. Unreliable
    conditions yield INCONCLUSIVE rather than a forced MATCH / NO_MATCH.
    """

    cfg = config or FaceVerificationConfig()
    face_engine = engine or get_face_engine(cfg)

    try:
        reference_image = load_rgb_image(Path(reference_path))
    except FaceImageError as exc:
        logger.info("Unusable reference image: %s", exc.reason_code)
        return FaceVerificationResult(
            verification_status=STATUS_COMPLETED,
            decision=DECISION_INCONCLUSIVE,
            similarity_score=None,
            distance_score=None,
            threshold=cfg.match_threshold,
            no_match_threshold=cfg.no_match_threshold,
            model_name=face_engine.model_name,
            model_version=face_engine.model_version,
            engine_name=face_engine.name,
            reference_face_count=0,
            evidence_face_count=0,
            reason_code=exc.reason_code,
            error=str(exc),
        )

    try:
        evidence_image = load_rgb_image(Path(evidence_path))
    except FaceImageError as exc:
        logger.info("Unusable evidence image: %s", exc.reason_code)
        return FaceVerificationResult(
            verification_status=STATUS_COMPLETED,
            decision=DECISION_INCONCLUSIVE,
            similarity_score=None,
            distance_score=None,
            threshold=cfg.match_threshold,
            no_match_threshold=cfg.no_match_threshold,
            model_name=face_engine.model_name,
            model_version=face_engine.model_version,
            engine_name=face_engine.name,
            reference_face_count=0,
            evidence_face_count=0,
            reason_code=exc.reason_code,
            error=str(exc),
        )

    try:
        reference_faces = face_engine.detect(reference_image)
        evidence_faces = face_engine.detect(evidence_image)
    except Exception:
        logger.exception("Face detection failed")
        return FaceVerificationResult(
            verification_status=STATUS_FAILED,
            decision=DECISION_INCONCLUSIVE,
            similarity_score=None,
            distance_score=None,
            threshold=cfg.match_threshold,
            no_match_threshold=cfg.no_match_threshold,
            model_name=face_engine.model_name,
            model_version=face_engine.model_version,
            engine_name=face_engine.name,
            reference_face_count=0,
            evidence_face_count=0,
            reason_code=REASON_EMBEDDING_FAILURE,
            error="Face detection failed",
        )

    ref_face, ref_reason = _select_single_face(
        reference_faces,
        none_reason=REASON_NO_FACE_REFERENCE,
        multi_reason=REASON_MULTIPLE_FACES_REFERENCE,
        low_quality_reason=REASON_LOW_QUALITY_REFERENCE,
        unusable_reason=REASON_UNUSABLE_REFERENCE,
    )
    evd_face, evd_reason = _select_single_face(
        evidence_faces,
        none_reason=REASON_NO_FACE_EVIDENCE,
        multi_reason=REASON_MULTIPLE_FACES_EVIDENCE,
        low_quality_reason=REASON_LOW_QUALITY_EVIDENCE,
        unusable_reason=REASON_UNUSABLE_EVIDENCE,
    )

    ref_meta = {"face_count": len(reference_faces)}
    evd_meta = {"face_count": len(evidence_faces)}
    if ref_face:
        ref_meta.update(ref_face.metadata())
    elif reference_faces:
        ref_meta["faces"] = [f.metadata() for f in reference_faces]
    if evd_face:
        evd_meta.update(evd_face.metadata())
    elif evidence_faces:
        evd_meta["faces"] = [f.metadata() for f in evidence_faces]

    if ref_reason:
        return _inconclusive(
            config=cfg,
            engine=face_engine,
            reason=ref_reason,
            ref_count=len(reference_faces),
            evd_count=len(evidence_faces),
            ref_meta=ref_meta,
            evd_meta=evd_meta,
            ref_crop=ref_face.crop if ref_face else None,
            evd_crop=evd_face.crop if evd_face else None,
        )
    if evd_reason:
        return _inconclusive(
            config=cfg,
            engine=face_engine,
            reason=evd_reason,
            ref_count=len(reference_faces),
            evd_count=len(evidence_faces),
            ref_meta=ref_meta,
            evd_meta=evd_meta,
            ref_crop=ref_face.crop if ref_face else None,
            evd_crop=evd_face.crop if evd_face else None,
        )

    assert ref_face is not None and evd_face is not None
    try:
        ref_emb = face_engine.embed(ref_face.crop)
        evd_emb = face_engine.embed(evd_face.crop)
        similarity = cosine_similarity(ref_emb, evd_emb)
        distance = euclidean_distance(ref_emb, evd_emb)
    except Exception:
        logger.exception("Face embedding failed")
        return FaceVerificationResult(
            verification_status=STATUS_FAILED,
            decision=DECISION_INCONCLUSIVE,
            similarity_score=None,
            distance_score=None,
            threshold=cfg.match_threshold,
            no_match_threshold=cfg.no_match_threshold,
            model_name=face_engine.model_name,
            model_version=face_engine.model_version,
            engine_name=face_engine.name,
            reference_face_count=len(reference_faces),
            evidence_face_count=len(evidence_faces),
            reason_code=REASON_EMBEDDING_FAILURE,
            reference_metadata=ref_meta,
            evidence_metadata=evd_meta,
            error="Face embedding failed",
            reference_crop=ref_face.crop,
            evidence_crop=evd_face.crop,
        )

    decision, reason = decide_verification(
        similarity,
        match_threshold=cfg.match_threshold,
        no_match_threshold=cfg.no_match_threshold,
    )
    return FaceVerificationResult(
        verification_status=STATUS_COMPLETED,
        decision=decision,
        similarity_score=similarity,
        distance_score=distance,
        threshold=cfg.match_threshold,
        no_match_threshold=cfg.no_match_threshold,
        model_name=face_engine.model_name,
        model_version=face_engine.model_version,
        engine_name=face_engine.name,
        reference_face_count=len(reference_faces),
        evidence_face_count=len(evidence_faces),
        reason_code=reason,
        reference_metadata=ref_meta,
        evidence_metadata=evd_meta,
        reference_crop=ref_face.crop,
        evidence_crop=evd_face.crop,
    )
