"""Analysis orchestration service — no model math inside."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app

from backend.app.audit import record_audit
from backend.app.exceptions import AnalysisNotFoundError, ValidationError
from backend.app.extensions import db
from backend.app.integrations import run_advanced_xai, run_explanation, run_inference
from backend.app.models.entities import AnalysisRun, Evidence, User
from backend.app.models.enums import (
    AnalysisStatus,
    AuditEventType,
    EvidenceStatus,
)
from backend.app.services.evidence_service import (
    absolute_evidence_path,
    get_evidence,
    verify_integrity,
)

logger = logging.getLogger("maya.backend.analysis")

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def analyze_evidence(
    user: User,
    evidence_id: int,
    *,
    generate_explanation: bool = True,
    explainer: str = "gradcam",
    verify_before_analyze: bool = True,
    advanced_xai: dict | None = None,
) -> AnalysisRun:
    evidence = get_evidence(user, evidence_id)

    if verify_before_analyze:
        integrity = verify_integrity(user, evidence_id)
        status = integrity.get("integrity_status")
        # evidence_service emits VALID / MODIFIED / MISSING / ERROR (TAMPERED kept for compat)
        if status in {"TAMPERED", "MODIFIED", "MISSING", "ERROR"}:
            raise ValidationError(
                f"Evidence integrity check failed ({status})"
            )
        # re-load evidence after verify commit
        evidence = get_evidence(user, evidence_id)

    # Phase 3+4: classify extracted stills, then mean-aggregate a video-level
    # assessment. Do not send the video file through the still-image pipeline.
    from backend.app.services.video_service import is_video_evidence

    if is_video_evidence(evidence):
        return _analyze_video_evidence(
            user,
            evidence,
            generate_explanation=generate_explanation,
            explainer=explainer,
        )

    path = absolute_evidence_path(evidence)
    if path.suffix.lower() in _VIDEO_SUFFIXES:
        logger.warning(
            "Evidence %s looks like a video container; using frame analysis",
            evidence.id,
        )
        return _analyze_video_evidence(
            user,
            evidence,
            generate_explanation=generate_explanation,
            explainer=explainer,
        )

    run = AnalysisRun(
        evidence_id=evidence.id,
        case_id=evidence.case_id,
        status=AnalysisStatus.PROCESSING.value,
        generate_explanation=bool(generate_explanation),
        explainer_name=explainer if generate_explanation else None,
        created_by_user_id=user.id,
        started_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    evidence.status = EvidenceStatus.PROCESSING.value
    evidence.analysis_status = AnalysisStatus.PROCESSING.value
    db.session.add(run)
    db.session.flush()
    record_audit(
        AuditEventType.ANALYSIS_STARTED,
        user_id=user.id,
        case_id=evidence.case_id,
        evidence_id=evidence.id,
        analysis_id=run.id,
        details={"explainer": explainer, "generate_explanation": generate_explanation},
    )
    db.session.commit()

    from backend.app.exceptions import MayaProductError
    from ai.inference.artifacts import write_inference_artifacts

    try:
        investigation = run_inference(path)
        run.investigation_id = investigation.investigation_id
        run.prediction = investigation.prediction
        run.confidence = float(investigation.confidence)
        run.model_name = investigation.model_name
        run.model_version = investigation.model_version
        run.dataset_version = investigation.dataset_version

        explanation_meta: dict | None = None
        artifact_root = (
            Path(current_app.config["ROOT_DIR"])
            / "artifacts"
            / "investigations"
            / investigation.investigation_id
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        # Re-persist inference artefacts into the per-investigation directory so
        # every forensic bundle is self-contained under artifacts/investigations/.
        write_inference_artifacts(investigation, artifact_root)
        run.artifact_dir = str(artifact_root)

        # Explainability is an optional overlay on the authenticity verdict.
        # A Grad-CAM / SHAP failure degrades the investigation, it does not
        # invalidate an inference that already succeeded.
        xai_errors: dict[str, str] = {}

        if generate_explanation:
            try:
                xai_dir = artifact_root / "xai" / (explainer or "gradcam")
                xai_dir.mkdir(parents=True, exist_ok=True)
                explanation_meta = run_explanation(
                    path,
                    investigation_id=investigation.investigation_id,
                    explainer=explainer or "gradcam",
                    artifact_dir=xai_dir,
                )
                run.heatmap_path = explanation_meta.get("heatmap")
                run.overlay_path = explanation_meta.get("overlay")
                run.explanation_json_path = explanation_meta.get("explanation_json")
                run.explainer_name = explanation_meta.get("explainer", explainer)
                record_audit(
                    AuditEventType.XAI_GENERATED,
                    user_id=user.id,
                    case_id=evidence.case_id,
                    evidence_id=evidence.id,
                    analysis_id=run.id,
                    details={
                        "explainer": explainer,
                        "investigation_id": investigation.investigation_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - optional stage
                logger.exception(
                    "XAI generation failed analysis=%s explainer=%s", run.id, explainer
                )
                explanation_meta = None
                xai_errors["explanation"] = type(exc).__name__
                record_audit(
                    AuditEventType.XAI_FAILED,
                    user_id=user.id,
                    case_id=evidence.case_id,
                    evidence_id=evidence.id,
                    analysis_id=run.id,
                    details={
                        "explainer": explainer,
                        "error_type": type(exc).__name__,
                        "investigation_id": investigation.investigation_id,
                    },
                )

        advanced_xai_results: dict | None = None
        if advanced_xai:
            try:
                adv_xai_dir = artifact_root / "xai" / "advanced"
                adv_xai_dir.mkdir(parents=True, exist_ok=True)
                xai_cfg = dict(advanced_xai)
                xai_cfg.setdefault("explainer", explainer or "gradcam")
                advanced_xai_results = run_advanced_xai(
                    path,
                    investigation_id=investigation.investigation_id,
                    artifact_dir=adv_xai_dir,
                    xai_config=xai_cfg,
                )
                if advanced_xai_results.get("trust_score") is not None:
                    run.trust_score = float(advanced_xai_results["trust_score"])
                if advanced_xai_results.get("quality_score") is not None:
                    run.quality_score = float(advanced_xai_results["quality_score"])
                record_audit(
                    AuditEventType.XAI_GENERATED,
                    user_id=user.id,
                    case_id=evidence.case_id,
                    evidence_id=evidence.id,
                    analysis_id=run.id,
                    details={
                        "advanced": True,
                        "methods_run": advanced_xai_results.get("methods_run", []),
                        "investigation_id": investigation.investigation_id,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - optional stage
                logger.exception("Advanced XAI failed analysis=%s", run.id)
                advanced_xai_results = None
                xai_errors["advanced_xai"] = type(exc).__name__
                record_audit(
                    AuditEventType.XAI_FAILED,
                    user_id=user.id,
                    case_id=evidence.case_id,
                    evidence_id=evidence.id,
                    analysis_id=run.id,
                    details={
                        "advanced": True,
                        "error_type": type(exc).__name__,
                        "investigation_id": investigation.investigation_id,
                    },
                )

        payload = {
            "investigation": investigation.to_dict(),
            "explanation": explanation_meta,
            "advanced_xai_results": advanced_xai_results,
            "xai_errors": xai_errors or None,
        }
        run.raw_result_json = json.dumps(payload, default=str)
        run.status = AnalysisStatus.COMPLETED.value
        run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        evidence.status = EvidenceStatus.ANALYZED.value
        evidence.analysis_status = AnalysisStatus.COMPLETED.value

        record_audit(
            AuditEventType.ANALYSIS_COMPLETED,
            user_id=user.id,
            case_id=evidence.case_id,
            evidence_id=evidence.id,
            analysis_id=run.id,
            details={
                "investigation_id": run.investigation_id,
                "prediction": run.prediction,
                "confidence": run.confidence,
            },
        )
        db.session.commit()
        logger.info(
            "Analysis %s completed inv=%s pred=%s",
            run.id,
            run.investigation_id,
            run.prediction,
        )
        return run
    except MayaProductError:
        db.session.rollback()
        raise
    except Exception as exc:
        logger.exception("Analysis failed evidence=%s", evidence_id)
        run_id, evidence_pk, case_pk = run.id, evidence.id, evidence.case_id
        # Discard the partial writes from the failed attempt (and recover the
        # session if the failure came from the DB layer) before recording the
        # terminal state, so no run is left stuck in PROCESSING.
        db.session.rollback()
        try:
            failed_run = db.session.get(AnalysisRun, run_id)
            failed_evidence = db.session.get(Evidence, evidence_pk)
            if failed_run is not None:
                failed_run.status = AnalysisStatus.FAILED.value
                failed_run.error_message = str(exc)[:2000]
                failed_run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if failed_evidence is not None:
                failed_evidence.status = EvidenceStatus.FAILED.value
                failed_evidence.analysis_status = AnalysisStatus.FAILED.value
            record_audit(
                AuditEventType.ANALYSIS_FAILED,
                user_id=user.id,
                case_id=case_pk,
                evidence_id=evidence_pk,
                analysis_id=run_id,
                details={"error_type": type(exc).__name__},
            )
            db.session.commit()
        except Exception:  # noqa: BLE001 - never mask the original failure
            db.session.rollback()
            logger.exception("Failed to persist FAILED state for analysis=%s", run_id)
        from backend.app.exceptions import AnalysisProcessingError

        raise AnalysisProcessingError("Analysis processing failed. See logs for details.") from exc


def _analyze_video_evidence(
    user: User,
    evidence: Evidence,
    *,
    generate_explanation: bool = False,
    explainer: str = "gradcam",
) -> AnalysisRun:
    """Score a video with the frozen 16-frame visual encoder and LSTM.

    Image evidence never reaches this function. Integrity, evidence identity,
    and chain-of-custody audit events stay on the existing analysis record.
    """

    from ai.ffpp_video.exceptions import (
        CheckpointNotFoundError,
        InsufficientFramesError,
        VideoModelError,
        VideoOpenError,
    )
    from ai.inference.investigation_id import InvestigationIDGenerator
    from backend.app.exceptions import AnalysisProcessingError, MayaProductError
    from backend.app.services.video_model_service import analyze_video, public_video_analysis

    path = absolute_evidence_path(evidence)
    try:
        root = Path(current_app.config["ROOT_DIR"])
        investigation_id = InvestigationIDGenerator(
            root / "artifacts" / "investigations" / "investigation_id_state.json"
        ).next_id()
        artifact_root = root / "artifacts" / "investigations" / investigation_id
        xai_dir = artifact_root / "xai" / "video" if generate_explanation else None
        model_result = analyze_video(
            path,
            generate_xai=bool(generate_explanation),
            artifact_dir=xai_dir,
        )
    except (VideoOpenError, InsufficientFramesError) as exc:
        raise ValidationError("Invalid or unreadable video") from exc
    except CheckpointNotFoundError as exc:
        raise AnalysisProcessingError(
            "Video model checkpoints are not configured."
        ) from exc
    except VideoModelError as exc:
        raise AnalysisProcessingError("Video analysis failed. See logs for details.") from exc

    video_analysis = public_video_analysis(model_result)
    run = AnalysisRun(
        evidence_id=evidence.id,
        case_id=evidence.case_id,
        investigation_id=investigation_id,
        status=AnalysisStatus.PROCESSING.value,
        generate_explanation=bool(generate_explanation),
        explainer_name="video_gradcam" if generate_explanation else None,
        created_by_user_id=user.id,
        started_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    evidence.status = EvidenceStatus.PROCESSING.value
    evidence.analysis_status = AnalysisStatus.PROCESSING.value
    db.session.add(run)
    db.session.flush()
    record_audit(
        AuditEventType.ANALYSIS_STARTED,
        user_id=user.id,
        case_id=evidence.case_id,
        evidence_id=evidence.id,
        analysis_id=run.id,
        details={
            "media_type": "video",
            "model": video_analysis.get("model_name"),
            "frames_analyzed": video_analysis.get("frames_analyzed"),
            "requested_explainer": explainer,
        },
    )
    db.session.commit()

    try:
        if generate_explanation and video_analysis.get("xai_available"):
            run.artifact_dir = str(artifact_root)
            run.explainer_name = "video_gradcam"
            record_audit(
                AuditEventType.XAI_GENERATED,
                user_id=user.id,
                case_id=evidence.case_id,
                evidence_id=evidence.id,
                analysis_id=run.id,
                details={
                    "media_type": "video",
                    "explainer": "video_gradcam",
                    "investigation_id": investigation_id,
                    "frame_count": video_analysis.get("frames_analyzed"),
                },
            )
        elif generate_explanation and model_result.get("xai_error"):
            record_audit(
                AuditEventType.XAI_FAILED,
                user_id=user.id,
                case_id=evidence.case_id,
                evidence_id=evidence.id,
                analysis_id=run.id,
                details={
                    "media_type": "video",
                    "error_type": str(model_result.get("xai_error")),
                    "investigation_id": investigation_id,
                },
            )

        p_fake = float(video_analysis["p_fake"])
        payload = {
            "media_type": "video",
            "prediction": video_analysis["prediction"],
            "fake_probability": p_fake,
            "real_probability": float(1.0 - p_fake),
            "threshold": video_analysis["threshold"],
            "video_analysis": video_analysis,
            "xai_artifact_names": list(model_result.get("xai_artifact_names") or []),
            "model_name": video_analysis.get("model_name"),
            "model_version": video_analysis.get("model_version"),
            "p_fake_meaning": video_analysis.get("p_fake_meaning"),
        }
        run.prediction = str(video_analysis["prediction"])
        run.model_name = video_analysis.get("model_name")
        run.model_version = video_analysis.get("model_version")
        run.raw_result_json = json.dumps(payload, default=str)
        run.status = AnalysisStatus.COMPLETED.value
        run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        evidence.status = EvidenceStatus.ANALYZED.value
        evidence.analysis_status = AnalysisStatus.COMPLETED.value
        record_audit(
            AuditEventType.ANALYSIS_COMPLETED,
            user_id=user.id,
            case_id=evidence.case_id,
            evidence_id=evidence.id,
            analysis_id=run.id,
            details={
                "media_type": "video",
                "prediction": run.prediction,
                "p_fake": p_fake,
                "frames_analyzed": video_analysis.get("frames_analyzed"),
                "fallback_frames": video_analysis.get("fallback_frames"),
                "model": run.model_name,
            },
        )
        db.session.commit()
        logger.info(
            "Video analysis %s completed evidence=%s pred=%s frames=%s",
            run.id,
            evidence.id,
            run.prediction,
            video_analysis.get("frames_analyzed"),
        )
        return run
    except MayaProductError:
        db.session.rollback()
        raise
    except Exception as exc:
        logger.exception("Video analysis failed evidence=%s", evidence.id)
        run_id, evidence_pk, case_pk = run.id, evidence.id, evidence.case_id
        db.session.rollback()
        try:
            failed_run = db.session.get(AnalysisRun, run_id)
            failed_evidence = db.session.get(Evidence, evidence_pk)
            if failed_run is not None:
                failed_run.status = AnalysisStatus.FAILED.value
                failed_run.error_message = str(exc)[:2000]
                failed_run.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            if failed_evidence is not None:
                failed_evidence.status = EvidenceStatus.FAILED.value
                failed_evidence.analysis_status = AnalysisStatus.FAILED.value
            record_audit(
                AuditEventType.ANALYSIS_FAILED,
                user_id=user.id,
                case_id=case_pk,
                evidence_id=evidence_pk,
                analysis_id=run_id,
                details={"error_type": type(exc).__name__, "media_type": "video"},
            )
            db.session.commit()
        except Exception:  # noqa: BLE001 - never mask the original failure
            db.session.rollback()
            logger.exception("Failed to persist FAILED state for analysis=%s", run_id)
        raise AnalysisProcessingError("Analysis processing failed. See logs for details.") from exc



def _explain_video_frames(
    *,
    user: User,
    evidence: Evidence,
    run: AnalysisRun,
    aggregation,
    frames,
    upload_root: Path,
    explainer: str,
    xai_errors: dict[str, str],
) -> dict | None:
    """Reuse still-image Grad-CAM on a few extracted JPEGs. Never fails the verdict."""

    from ai.inference.investigation_id import InvestigationIDGenerator
    from ai.video.frame_xai import explain_video_frames, select_video_xai_frames

    try:
        selected = select_video_xai_frames(
            frames,
            video_prediction=aggregation.prediction,
            path_root=upload_root,
        )
        if not selected:
            logger.info("No usable JPEG frames for video XAI evidence=%s", evidence.id)
            return None

        root = Path(current_app.config["ROOT_DIR"])
        if not run.investigation_id:
            state_path = root / "artifacts" / "investigations" / "investigation_id_state.json"
            run.investigation_id = InvestigationIDGenerator(state_path).next_id()
        artifact_root = root / "artifacts" / "investigations" / run.investigation_id
        artifact_root.mkdir(parents=True, exist_ok=True)
        run.artifact_dir = str(artifact_root)

        explained = explain_video_frames(
            selected,
            investigation_id=run.investigation_id,
            explainer=explainer,
            artifact_root=artifact_root,
            path_root=upload_root,
            explain_fn=run_explanation,
        )
        if not explained:
            return None
        primary = explained[0]
        run.heatmap_path = primary.get("heatmap")
        run.overlay_path = primary.get("overlay")
        run.explanation_json_path = primary.get("explanation_json")
        run.explainer_name = primary.get("explainer", explainer)
        record_audit(
            AuditEventType.XAI_GENERATED,
            user_id=user.id,
            case_id=evidence.case_id,
            evidence_id=evidence.id,
            analysis_id=run.id,
            details={
                "media_type": "video",
                "explainer": explainer,
                "investigation_id": run.investigation_id,
                "frame_count": len(explained),
                "frame_numbers": [item.get("frame_number") for item in explained],
            },
        )
        return primary
    except Exception as exc:  # noqa: BLE001 - optional forensic overlay
        logger.exception(
            "Video XAI generation failed analysis=%s explainer=%s", run.id, explainer
        )
        xai_errors["explanation"] = type(exc).__name__
        record_audit(
            AuditEventType.XAI_FAILED,
            user_id=user.id,
            case_id=evidence.case_id,
            evidence_id=evidence.id,
            analysis_id=run.id,
            details={
                "media_type": "video",
                "explainer": explainer,
                "error_type": type(exc).__name__,
                "investigation_id": run.investigation_id,
            },
        )
        return None


def get_analysis(user: User, analysis_id: int) -> AnalysisRun:
    run = db.session.get(AnalysisRun, analysis_id)
    if run is None:
        raise AnalysisNotFoundError(f"Analysis {analysis_id} not found")
    get_evidence(user, run.evidence_id)  # ownership check
    return run
