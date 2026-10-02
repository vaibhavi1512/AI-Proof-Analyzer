"""JSON response helpers and lightweight schema builders."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from flask import jsonify
from werkzeug.wrappers import Response

from backend.app.models.entities import AnalysisRun, Case, Evidence, FaceVerification, User

_FRAME_DIR_RE = re.compile(r"frame_(\d+)")


def api_success(data: Any = None, *, status: int = 200, message: str | None = None) -> Response:
    body: dict[str, Any] = {"ok": True}
    if message is not None:
        body["message"] = message
    if data is not None:
        body["data"] = data
    return jsonify(body), status


def api_error(
    message: str,
    *,
    status: int = 400,
    error_code: str = "error",
) -> Response:
    return jsonify({"ok": False, "error": error_code, "message": message}), status


def user_to_dict(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    }


def case_to_dict(case: Case) -> dict[str, Any]:
    return {
        "id": case.id,
        "case_id": case.id,
        "case_number": case.case_number,
        "title": case.title,
        "description": case.description,
        "status": case.status,
        "priority": case.priority,
        "created_by": case.created_by_user_id,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "updated_at": case.updated_at.isoformat() if case.updated_at else None,
        "closed_at": case.closed_at.isoformat() if case.closed_at else None,
    }


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.id,
        "case_id": evidence.case_id,
        "original_filename": evidence.original_filename,
        "stored_filename": evidence.stored_filename,
        "media_type": evidence.media_type,
        "mime_type": evidence.mime_type,
        "file_size": evidence.file_size_bytes,
        "sha256": evidence.sha256_hash,
        "status": evidence.status,
        "analysis_status": evidence.analysis_status,
        "uploaded_by": evidence.uploaded_by_user_id,
        "created_at": evidence.uploaded_at.isoformat() if evidence.uploaded_at else None,
        "storage_path": evidence.storage_path,
        "notes": evidence.notes,
    }


def _frame_number_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = _FRAME_DIR_RE.search(Path(str(path)).as_posix())
    return int(match.group(1)) if match else None


def _explained_frame_numbers(run: AnalysisRun, parsed: dict[str, Any] | None) -> tuple[int | None, list[int]]:
    """List frames that already have Grad-CAM dirs. Does not generate new XAI."""

    found: set[int] = set()
    primary = _frame_number_from_path(run.heatmap_path)
    if primary is not None:
        found.add(primary)
    if isinstance(parsed, dict):
        explanation = parsed.get("explanation") or {}
        if isinstance(explanation, dict):
            raw_primary = explanation.get("frame_number")
            if isinstance(raw_primary, int):
                found.add(raw_primary)
                if primary is None:
                    primary = raw_primary
            extra = explanation.get("frame_numbers")
            if isinstance(extra, list):
                for item in extra:
                    if isinstance(item, int):
                        found.add(item)
    if run.artifact_dir:
        gradcam = Path(run.artifact_dir) / "xai" / "gradcam"
        try:
            if gradcam.is_dir():
                for child in gradcam.iterdir():
                    if not child.is_dir():
                        continue
                    match = re.fullmatch(r"frame_(\d+)", child.name)
                    if match and (child / "heatmap.png").is_file():
                        found.add(int(match.group(1)))
        except OSError:
            pass
    return primary, sorted(found)


def analysis_to_dict(run: AnalysisRun) -> dict[str, Any]:
    explanation = None
    parsed: dict[str, Any] | None = None
    if run.generate_explanation or run.overlay_path or run.heatmap_path:
        explanation = {
            "explainer": run.explainer_name,
            "heatmap": run.heatmap_path,
            "overlay": run.overlay_path,
            "explanation_json": run.explanation_json_path,
        }
    advanced_xai_results = None
    real_probability = None
    fake_probability = None
    frames = None
    aggregation = None
    suspicious_frames = None
    temporal_analysis = None
    video_analysis = None
    if run.raw_result_json:
        try:
            import json as _json
            parsed = _json.loads(run.raw_result_json)
            advanced_xai_results = parsed.get("advanced_xai_results")
            investigation = parsed.get("investigation") or {}
            real_probability = investigation.get("real_probability")
            fake_probability = investigation.get("fake_probability")
            if real_probability is None:
                real_probability = parsed.get("real_probability")
            if fake_probability is None:
                fake_probability = parsed.get("fake_probability")
            frames = parsed.get("frames")
            aggregation = parsed.get("aggregation")
            suspicious_frames = parsed.get("suspicious_frames")
            temporal_analysis = parsed.get("temporal_analysis")
            video_analysis = parsed.get("video_analysis")
            if not isinstance(video_analysis, dict):
                video_analysis = None
        except Exception:
            advanced_xai_results = None
            parsed = None
            video_analysis = None
    if explanation is not None:
        primary, explained = _explained_frame_numbers(run, parsed)
        if primary is not None:
            explanation["frame_number"] = primary
        if explained:
            explanation["explained_frames"] = explained
    payload = {
        "analysis_id": run.id,
        "investigation_id": run.investigation_id,
        "evidence_id": run.evidence_id,
        "case_id": run.case_id,
        "prediction": run.prediction,
        "confidence": run.confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "analysis_status": run.status,
        "model_name": run.model_name,
        "model_version": run.model_version,
        "dataset_version": run.dataset_version,
        "artifact_dir": run.artifact_dir,
        "trust_score": run.trust_score,
        "quality_score": run.quality_score,
        "advanced_xai_results": advanced_xai_results,
        "error_message": run.error_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "explanation": explanation,
    }
    if frames is not None:
        payload["frames"] = frames
    if aggregation is not None:
        payload["aggregation"] = aggregation
    if suspicious_frames is not None:
        payload["suspicious_frames"] = suspicious_frames
    if temporal_analysis is not None:
        payload["temporal_analysis"] = temporal_analysis
    if video_analysis is not None:
        public_video = _public_video_analysis(video_analysis)
        names = parsed.get("xai_artifact_names") if isinstance(parsed, dict) else None
        _attach_video_xai_flags(public_video, names if isinstance(names, list) else [])
        payload["video_analysis"] = public_video
    return payload


def _attach_video_xai_flags(public_video: dict[str, Any], names: list) -> None:
    """Record which Grad-CAM files exist without exposing filesystem paths."""

    indices: list[int] = []
    for name in names:
        match = re.fullmatch(r"gradcam_frame_(\d+)\.png", str(name))
        if match:
            indices.append(int(match.group(1)))
    public_video["gradcam_contact_sheet"] = "gradcam_contact_sheet.png" in {str(name) for name in names}
    public_video["gradcam_frame_indices"] = indices


def _public_video_analysis(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the video block free of filesystem paths."""

    frame_keys = (
        "frame_index",
        "timestamp_seconds",
        "face_detected",
        "used_face_crop",
        "used_full_frame_fallback",
        "temporal_importance",
    )

    def frames(items: object) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cleaned.append({key: item[key] for key in frame_keys if key in item})
        return cleaned

    payload: dict[str, Any] = {
        "prediction": raw.get("prediction"),
        "p_fake": raw.get("p_fake"),
        "threshold": raw.get("threshold"),
        "frames_analyzed": raw.get("frames_analyzed"),
        "face_crop_frames": raw.get("face_crop_frames"),
        "fallback_frames": raw.get("fallback_frames"),
        "xai_available": bool(raw.get("xai_available")),
    }
    for key in ("raw_fake_logit", "model_name", "model_version", "p_fake_meaning"):
        if key in raw:
            payload[key] = raw.get(key)
    if raw.get("frames"):
        payload["frames"] = frames(raw.get("frames"))
    xai = raw.get("xai")
    if isinstance(xai, dict):
        payload["xai"] = {
            "spatial_wording": xai.get("spatial_wording"),
            "temporal_wording": xai.get("temporal_wording"),
            "frames": frames(xai.get("frames")),
        }
    return payload


def face_verification_to_dict(row: FaceVerification) -> dict[str, Any]:
    return {
        "verification_id": row.id,
        "case_id": row.case_id,
        "evidence_id": row.evidence_id,
        "analysis_id": row.analysis_id,
        "investigation_id": row.investigation_id,
        "verification_status": row.verification_status,
        "decision": row.decision,
        "similarity_score": row.similarity_score,
        "distance_score": row.distance_score,
        "threshold": row.threshold,
        "no_match_threshold": row.no_match_threshold,
        "reason_code": row.reason_code,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "engine_name": row.engine_name,
        "reference_face_count": row.reference_face_count,
        "evidence_face_count": row.evidence_face_count,
        "reference_filename": row.reference_filename,
        "artifact_dir": row.artifact_dir,
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }
