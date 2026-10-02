"""MAYA entry point for frozen 16-frame video inference.

Checkpoint locations come from application configuration, which reads
``VIDEO_VISUAL_CHECKPOINT`` and ``VIDEO_LSTM_CHECKPOINT``.
"""

from __future__ import annotations

from pathlib import Path

from flask import current_app

from ai.ffpp_video.analyze import analyze_video as run_video_model
from ai.ffpp_video.analyze import reset_video_runtime

_PUBLIC_FRAME_KEYS = (
    "frame_index",
    "timestamp_seconds",
    "face_detected",
    "used_face_crop",
    "used_full_frame_fallback",
    "temporal_importance",
)


def analyze_video(video_path: str | Path, generate_xai: bool = False, *, artifact_dir: str | Path | None = None) -> dict:
    """Run the production video model. Paths stay inside the service result filter."""

    return run_video_model(
        video_path,
        generate_xai,
        artifact_dir=artifact_dir,
        visual_checkpoint=current_app.config.get("VIDEO_VISUAL_CHECKPOINT"),
        lstm_checkpoint=current_app.config.get("VIDEO_LSTM_CHECKPOINT"),
    )


def public_video_analysis(result: dict) -> dict:
    """API view of a video result. Filesystem paths are omitted."""

    payload = {
        "prediction": result.get("prediction"),
        "p_fake": result.get("p_fake"),
        "threshold": result.get("threshold"),
        "frames_analyzed": result.get("frames_analyzed"),
        "face_crop_frames": result.get("face_crop_frames"),
        "fallback_frames": result.get("fallback_frames"),
        "xai_available": bool(result.get("xai_available")),
        "model_name": result.get("model_name"),
        "model_version": result.get("model_version"),
        "p_fake_meaning": result.get("p_fake_meaning"),
        "frames": [_public_frame(item) for item in result.get("frames") or []],
    }
    if result.get("raw_fake_logit") is not None:
        payload["raw_fake_logit"] = result.get("raw_fake_logit")
    xai = result.get("xai")
    if isinstance(xai, dict):
        payload["xai"] = {
            "spatial_wording": xai.get("spatial_wording"),
            "temporal_wording": xai.get("temporal_wording"),
            "frames": [_public_frame(item) for item in xai.get("frames") or []],
        }
    return payload


def _public_frame(item: dict) -> dict:
    return {key: item.get(key) for key in _PUBLIC_FRAME_KEYS if key in item}


__all__ = ["analyze_video", "public_video_analysis", "reset_video_runtime"]
