"""Associate extracted video frames with an Evidence record.

Phase 2 extracts ~1 FPS stills. Phase 3 classifies those stills with the
existing image EfficientNet pipeline (no video-level verdict).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import current_app

from ai.video.exceptions import VideoPathError, VideoUnreadableError
from ai.video.frame_extractor import (
    METADATA_FILENAME,
    ExtractedFrame,
    VideoExtractionResult,
    extract_frames,
)
from ai.video.metadata import VideoMetadata
from ai.video.video_config import VideoConfig, get_video_config
from backend.app.exceptions import ValidationError
from backend.app.models.entities import Evidence, User
from backend.app.services.evidence_service import absolute_evidence_path, get_evidence
from backend.app.utils.paths import is_within, resolve_within

logger = logging.getLogger("maya.backend.video")

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def is_video_evidence(evidence: Evidence) -> bool:
    """True when the row is tagged video or the stored file is a video container."""

    if str(evidence.media_type or "").strip().lower() == "video":
        return True
    for name in (
        evidence.stored_filename,
        evidence.storage_path,
        evidence.original_filename,
    ):
        if name and Path(str(name)).suffix.lower() in _VIDEO_SUFFIXES:
            return True
    return False


def _finite_probability(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number == number and 0.0 <= number <= 1.0


def _extraction_from_sidecar(payload: dict) -> VideoExtractionResult | None:
    raw_meta = payload.get("metadata")
    raw_frames = payload.get("frames")
    if not isinstance(raw_meta, dict) or not isinstance(raw_frames, list):
        return None
    try:
        metadata = VideoMetadata(
            duration_seconds=float(raw_meta.get("duration_seconds") or 0.0),
            fps=float(raw_meta.get("fps") or 0.0),
            frame_count=int(raw_meta.get("frame_count") or 0),
            width=int(raw_meta.get("width") or 0),
            height=int(raw_meta.get("height") or 0),
            extracted_frame_count=int(
                raw_meta.get("extracted_frame_count") or len(raw_frames)
            ),
            sampling_rate_fps=float(raw_meta.get("sampling_rate_fps") or 1.0),
            sample_interval_frames=int(raw_meta.get("sample_interval_frames") or 1),
        )
        frames = [
            ExtractedFrame(
                frame_number=int(item.get("frame_number", 0)),
                timestamp_seconds=float(item.get("timestamp_seconds") or 0.0),
                frame_path=str(item.get("frame_path") or ""),
                prediction=item.get("prediction"),
                confidence=item.get("confidence"),
                real_probability=item.get("real_probability"),
                fake_probability=item.get("fake_probability"),
            )
            for item in raw_frames
            if isinstance(item, dict)
        ]
    except (TypeError, ValueError):
        return None
    return VideoExtractionResult(
        metadata=metadata,
        frames=frames,
        output_dir=str(payload.get("output_dir") or ""),
        metadata_path=str(payload.get("metadata_path") or ""),
    )


def _sidecar_predictions_are_valid(
    result: VideoExtractionResult, upload_root: Path
) -> bool:
    if not result.frames:
        return False
    for frame in result.frames:
        label = str(frame.prediction or "").strip().upper()
        if label not in {"REAL", "FAKE"}:
            return False
        if not _finite_probability(frame.real_probability):
            return False
        if not _finite_probability(frame.fake_probability):
            return False
        stored = Path(frame.frame_path)
        image_path = stored if stored.is_absolute() else upload_root / stored
        if image_path.suffix.lower() in _VIDEO_SUFFIXES:
            return False
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            return False
    return True


def load_valid_classified_video(
    user: User, evidence_id: int
) -> VideoExtractionResult | None:
    """Return sidecar frames when every JPEG already has a valid Phase 3 score."""

    evidence = get_evidence(user, evidence_id)
    if not is_video_evidence(evidence):
        raise ValidationError("Evidence is not a video")
    upload_root = Path(current_app.config["UPLOAD_DIR"]).resolve()
    sidecar = video_frames_dir(evidence, upload_root) / METADATA_FILENAME
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    result = _extraction_from_sidecar(payload)
    if result is None or not _sidecar_predictions_are_valid(result, upload_root):
        return None
    result.output_dir = _to_relative(upload_root, video_frames_dir(evidence, upload_root))
    result.metadata_path = _to_relative(upload_root, sidecar)
    logger.info(
        "Reusing classified video sidecar evidence=%s frames=%s",
        evidence_id,
        len(result.frames),
    )
    return result


def get_classified_video_frames(
    user: User,
    evidence_id: int,
    *,
    config: VideoConfig | None = None,
) -> VideoExtractionResult:
    """Reuse a valid Phase 3 sidecar, otherwise extract and classify."""

    existing = load_valid_classified_video(user, evidence_id)
    if existing is not None:
        return existing
    return analyze_video_evidence(user, evidence_id, config=config)


def video_frames_dir(evidence: Evidence, upload_root: Path) -> Path:
    """Return ``uploads/cases/{case_id}/frames/{evidence_id}/`` inside ``upload_root``."""

    relative = Path("cases") / str(int(evidence.case_id)) / "frames" / str(int(evidence.id))
    dest = resolve_within(upload_root, relative)
    if dest is None:
        raise ValidationError("Invalid storage path")
    return dest


def _to_relative(upload_root: Path, path: Path | str) -> str:
    resolved = Path(path).resolve()
    if not is_within(upload_root, resolved):
        raise ValidationError("Invalid storage path")
    return resolved.relative_to(upload_root.resolve()).as_posix()


def _relocate_result(
    result: VideoExtractionResult, upload_root: Path
) -> VideoExtractionResult:
    result.output_dir = _to_relative(upload_root, result.output_dir)
    if result.metadata_path:
        result.metadata_path = _to_relative(upload_root, result.metadata_path)
    for frame in result.frames:
        frame.frame_path = _to_relative(upload_root, frame.frame_path)
    return result


def _clear_frames_dir(frames_dir: Path, upload_root: Path) -> None:
    if not frames_dir.exists():
        return
    if not is_within(upload_root, frames_dir):
        raise ValidationError("Invalid storage path")
    for child in frames_dir.iterdir():
        if child.is_file():
            child.unlink()


def prepare_video_evidence(
    user: User,
    evidence_id: int,
    *,
    config: VideoConfig | None = None,
) -> VideoExtractionResult:
    """Read metadata and extract ~1 FPS frames for an uploaded video Evidence row."""

    evidence = get_evidence(user, evidence_id)
    if not is_video_evidence(evidence):
        raise ValidationError("Evidence is not a video")

    upload_root = Path(current_app.config["UPLOAD_DIR"]).resolve()
    video_path = absolute_evidence_path(evidence)
    if not is_within(upload_root, video_path):
        raise ValidationError("Invalid storage path")

    frames_dir = video_frames_dir(evidence, upload_root)
    _clear_frames_dir(frames_dir, upload_root)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not is_within(upload_root, frames_dir):
        raise ValidationError("Invalid storage path")

    try:
        result = extract_frames(
            video_path,
            frames_dir,
            config=config or get_video_config(),
        )
    except VideoUnreadableError as exc:
        _clear_frames_dir(frames_dir, upload_root)
        raise ValidationError("Invalid or unreadable video") from exc
    except VideoPathError as exc:
        _clear_frames_dir(frames_dir, upload_root)
        raise ValidationError("Invalid storage path") from exc

    result = _relocate_result(result, upload_root)
    sidecar = frames_dir / METADATA_FILENAME
    sidecar.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    result.metadata_path = _to_relative(upload_root, sidecar)
    logger.info(
        "Prepared video evidence=%s frames=%s dir=%s",
        evidence.id,
        result.metadata.extracted_frame_count,
        result.output_dir,
    )
    return result


def classify_prepared_video(
    user: User,
    evidence_id: int,
    result: VideoExtractionResult,
) -> VideoExtractionResult:
    """Run the product image model on already extracted frames and rewrite sidecar."""

    evidence = get_evidence(user, evidence_id)
    if not is_video_evidence(evidence):
        raise ValidationError("Evidence is not a video")
    upload_root = Path(current_app.config["UPLOAD_DIR"]).resolve()
    from ai.video.frame_analyzer import analyze_extracted_frames
    from backend.app.integrations import get_inference_pipeline

    analyze_extracted_frames(
        result,
        path_root=upload_root,
        pipeline=get_inference_pipeline(),
    )
    frames_dir = video_frames_dir(evidence, upload_root)
    metadata_file = frames_dir / METADATA_FILENAME
    metadata_file.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    result.metadata_path = _to_relative(upload_root, metadata_file)
    logger.info(
        "Classified video evidence=%s frames=%s",
        evidence_id,
        result.metadata.extracted_frame_count,
    )
    return result


def analyze_video_evidence(
    user: User,
    evidence_id: int,
    *,
    config: VideoConfig | None = None,
) -> VideoExtractionResult:
    """Extract frames then classify each JPEG with the product image model."""

    result = prepare_video_evidence(user, evidence_id, config=config)
    return classify_prepared_video(user, evidence_id, result)
