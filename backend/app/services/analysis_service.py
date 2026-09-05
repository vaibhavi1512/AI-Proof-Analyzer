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

    path = absolute_evidence_path(evidence)
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


def get_analysis(user: User, analysis_id: int) -> AnalysisRun:
    run = db.session.get(AnalysisRun, analysis_id)
    if run is None:
        raise AnalysisNotFoundError(f"Analysis {analysis_id} not found")
    get_evidence(user, run.evidence_id)  # ownership check
    return run
