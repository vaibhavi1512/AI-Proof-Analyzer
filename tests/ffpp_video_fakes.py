"""Shared stand-in for the frozen 16-frame video model."""

from __future__ import annotations


def lstm_result(
    *,
    prediction: str = "REAL",
    p_fake: float = 0.09,
    fallback_frames: int = 0,
    with_xai: bool = False,
) -> dict:
    face_crop_frames = 16 - fallback_frames
    frames = []
    for index in range(16):
        fallback = index < fallback_frames
        frames.append(
            {
                "frame_index": index * 5,
                "timestamp_seconds": round(index * 0.25, 3),
                "face_detected": not fallback,
                "used_face_crop": not fallback,
                "used_full_frame_fallback": fallback,
            }
        )
    result = {
        "prediction": prediction,
        "p_fake": p_fake,
        "raw_fake_logit": -2.31 if prediction == "REAL" else 1.2,
        "threshold": 0.5,
        "frames_analyzed": 16,
        "face_crop_frames": face_crop_frames,
        "full_frame_fallback_frames": fallback_frames,
        "fallback_frames": fallback_frames,
        "model_name": "ffpp_video_lstm_v2",
        "model_version": "v2-16frame",
        "visual_model_name": "ffpp_visual_efficientnet_b0",
        "xai_available": with_xai,
        "p_fake_meaning": "model-predicted probability for the FAKE class",
        "frames": frames,
        "xai_artifact_names": ["explanation.json"] if with_xai else [],
    }
    if with_xai:
        explained = []
        for frame in frames:
            explained.append({**frame, "temporal_importance": 1.0 / 16})
        result["xai"] = {
            "spatial_wording": "regions contributing to the model prediction",
            "temporal_wording": "frames with higher relative contribution",
            "frames": explained,
        }
    return result
