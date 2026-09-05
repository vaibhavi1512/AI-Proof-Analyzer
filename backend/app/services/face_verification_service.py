"""Face reference verification orchestration (authz, artifacts, audit, persistence)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app
from werkzeug.datastructures import FileStorage

from ai.datasets.utils.checksums import hash_file
from ai.face_verification.artifacts import write_face_verification_artifacts
from ai.face_verification.config import config_from_mapping
from ai.face_verification.engine import get_face_engine
from ai.face_verification.pipeline import run_face_verification
from ai.inference.investigation_id import InvestigationIDGenerator
from backend.app.audit import record_audit
from backend.app.exceptions import (
    FaceVerificationNotFoundError,
    FaceVerificationProcessingError,
    InvalidEvidenceError,
    MayaProductError,
    ValidationError,
)
from backend.app.extensions import db
from backend.app.models.entities import AnalysisRun, FaceVerification, User
from backend.app.models.enums import AuditEventType, FaceVerificationStatus
from backend.app.services.evidence_service import absolute_evidence_path, get_evidence
from backend.app.storage.evidence_storage import extension_of, validate_upload_file
from backend.app.utils.paths import is_within

logger = logging.getLogger("maya.backend.face_verification")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _relative_artifact_dir(abs_dir: Path) -> str:
    root = Path(current_app.config["ROOT_DIR"]).resolve()
    resolved = abs_dir.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        artifacts = (root / "artifacts").resolve()
        try:
            return ("artifacts/" + resolved.relative_to(artifacts).as_posix()).replace(
                "\\", "/"
            )
        except ValueError:
            return abs_dir.name


def _parse_optional_float(raw: str | None, field: str) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid {field}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"{field} must be between 0 and 1")
    return value


def _resolve_investigation(
    user: User,
    evidence,
    provided: str | None,
) -> tuple[str, int | None]:
    provided = (provided or "").strip() or None
    if provided:
        run = (
            AnalysisRun.query.filter_by(investigation_id=provided, evidence_id=evidence.id)
            .order_by(AnalysisRun.id.desc())
            .first()
        )
        if run is None:
            run = (
                AnalysisRun.query.filter_by(
                    investigation_id=provided, case_id=evidence.case_id
                )
                .order_by(AnalysisRun.id.desc())
                .first()
            )
        if run is None:
            raise ValidationError("investigation_id does not belong to this evidence or case")
        get_evidence(user, run.evidence_id)
        return provided, run.id

    latest = (
        AnalysisRun.query.filter_by(evidence_id=evidence.id)
        .filter(AnalysisRun.investigation_id.isnot(None))
        .order_by(AnalysisRun.id.desc())
        .first()
    )
    if latest and latest.investigation_id:
        return str(latest.investigation_id), latest.id

    root = Path(current_app.config["ROOT_DIR"])
    state_path = root / "artifacts" / "investigations" / "investigation_id_state.json"
    investigation_id = InvestigationIDGenerator(state_path).next_id()
    return investigation_id, None


def _save_reference_temp(file: FileStorage, *, max_bytes: int) -> tuple[Path, str, str, int, str]:
    display_name, mime = validate_upload_file(file, max_bytes=max_bytes)
    ext = extension_of(display_name) or ".png"
    tmp_root = Path(current_app.config["UPLOAD_DIR"]) / "_tmp_face_ref"
    tmp_root.mkdir(parents=True, exist_ok=True)
    upload_root = Path(current_app.config["UPLOAD_DIR"]).resolve()
    if not is_within(upload_root, tmp_root):
        raise InvalidEvidenceError("Invalid temporary storage path")

    temp_path = (tmp_root / f"{uuid.uuid4().hex}{ext}").resolve()
    if not is_within(upload_root, temp_path):
        raise InvalidEvidenceError("Invalid temporary storage path")
    try:
        file.save(temp_path)
        size = temp_path.stat().st_size
        if size <= 0:
            raise InvalidEvidenceError("Empty file rejected")
        if size > max_bytes:
            raise InvalidEvidenceError("File exceeds maximum upload size")
        digest = hash_file(temp_path, algorithm="sha256")
        return temp_path, display_name, mime, size, digest
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def create_face_verification(
    user: User,
    evidence_id: int,
    reference_file: FileStorage,
    *,
    threshold: str | float | None = None,
    no_match_threshold: str | float | None = None,
    investigation_id: str | None = None,
    engine: object | None = None,
) -> FaceVerification:
    evidence = get_evidence(user, evidence_id)
    max_bytes = int(current_app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))
    cfg = config_from_mapping(current_app.config)

    match_override = _parse_optional_float(
        None if threshold is None else str(threshold), "threshold"
    )
    no_match_override = _parse_optional_float(
        None if no_match_threshold is None else str(no_match_threshold),
        "no_match_threshold",
    )
    if match_override is not None:
        cfg.match_threshold = match_override
        if no_match_override is None:
            cfg.no_match_threshold = min(
                cfg.no_match_threshold, max(0.0, match_override - cfg.inconclusive_margin)
            )
    if no_match_override is not None:
        cfg.no_match_threshold = no_match_override
    if cfg.no_match_threshold > cfg.match_threshold:
        raise ValidationError("no_match_threshold must be <= threshold")

    inv_id, analysis_id = _resolve_investigation(user, evidence, investigation_id)
    evidence_path = absolute_evidence_path(evidence)
    if not evidence_path.is_file():
        raise InvalidEvidenceError("Evidence file is missing")

    temp_path, display_name, mime, size, digest = _save_reference_temp(
        reference_file, max_bytes=max_bytes
    )

    row = FaceVerification(
        case_id=evidence.case_id,
        evidence_id=evidence.id,
        analysis_id=analysis_id,
        investigation_id=inv_id,
        verification_status=FaceVerificationStatus.PROCESSING.value,
        reference_filename=display_name,
        reference_sha256=digest,
        reference_mime_type=mime,
        reference_file_size_bytes=size,
        created_by_user_id=user.id,
        threshold=cfg.match_threshold,
        no_match_threshold=cfg.no_match_threshold,
    )
    db.session.add(row)
    db.session.flush()
    record_audit(
        AuditEventType.FACE_VERIFICATION_STARTED,
        user_id=user.id,
        case_id=evidence.case_id,
        evidence_id=evidence.id,
        analysis_id=analysis_id,
        details={
            "verification_id": row.id,
            "investigation_id": inv_id,
            "reference_filename": display_name,
        },
    )
    db.session.commit()

    try:
        face_engine = engine or get_face_engine(cfg)
        result = run_face_verification(
            temp_path,
            evidence_path,
            config=cfg,
            engine=face_engine,
        )
        token = uuid.uuid4().hex
        abs_dir = (
            Path(current_app.config["ROOT_DIR"])
            / "artifacts"
            / "investigations"
            / inv_id
            / "face_verification"
            / token
        )
        write_face_verification_artifacts(
            result,
            abs_dir,
            investigation_id=inv_id,
            extra={"verification_id": row.id, "evidence_id": evidence.id},
        )
        rel_dir = _relative_artifact_dir(abs_dir)
        row.verification_status = result.verification_status
        row.decision = result.decision
        row.similarity_score = result.similarity_score
        row.distance_score = result.distance_score
        row.threshold = result.threshold
        row.no_match_threshold = result.no_match_threshold
        row.reason_code = result.reason_code
        row.reference_face_count = result.reference_face_count
        row.evidence_face_count = result.evidence_face_count
        row.reference_metadata_json = json.dumps(result.reference_metadata, default=str)
        row.evidence_metadata_json = json.dumps(result.evidence_metadata, default=str)
        row.model_name = result.model_name
        row.model_version = result.model_version
        row.engine_name = result.engine_name
        row.artifact_dir = rel_dir
        row.result_json_path = f"{rel_dir}/comparison_result.json"
        row.completed_at = _utcnow()
        row.error_message = result.error
        if result.verification_status == FaceVerificationStatus.FAILED.value:
            record_audit(
                AuditEventType.FACE_VERIFICATION_FAILED,
                user_id=user.id,
                case_id=evidence.case_id,
                evidence_id=evidence.id,
                analysis_id=analysis_id,
                details={
                    "verification_id": row.id,
                    "reason_code": result.reason_code,
                    "error_type": "pipeline_failed",
                },
            )
        else:
            record_audit(
                AuditEventType.FACE_VERIFICATION_COMPLETED,
                user_id=user.id,
                case_id=evidence.case_id,
                evidence_id=evidence.id,
                analysis_id=analysis_id,
                details={
                    "verification_id": row.id,
                    "decision": result.decision,
                    "reason_code": result.reason_code,
                    "investigation_id": inv_id,
                },
            )
        db.session.commit()
        logger.info(
            "Face verification %s status=%s decision=%s",
            row.id,
            row.verification_status,
            row.decision,
        )
        return row
    except Exception as exc:
        logger.exception("Face verification failed evidence=%s", evidence_id)
        # The PROCESSING row was already committed, so it must be closed out as
        # FAILED on every error path — including MayaProductError — otherwise a
        # rollback leaves it stuck in PROCESSING forever.
        _mark_verification_failed(
            row.id,
            user_id=user.id,
            case_id=evidence.case_id,
            evidence_id=evidence.id,
            analysis_id=analysis_id,
            error_type=type(exc).__name__,
        )
        if isinstance(exc, MayaProductError):
            raise
        raise FaceVerificationProcessingError(
            "Face verification failed. See logs for details."
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def _mark_verification_failed(
    verification_id: int,
    *,
    user_id: int,
    case_id: int,
    evidence_id: int,
    analysis_id: int | None,
    error_type: str,
) -> None:
    """Persist the terminal FAILED state after discarding partial writes."""

    db.session.rollback()
    try:
        row = db.session.get(FaceVerification, verification_id)
        if row is not None:
            row.verification_status = FaceVerificationStatus.FAILED.value
            row.error_message = error_type
            row.completed_at = _utcnow()
        record_audit(
            AuditEventType.FACE_VERIFICATION_FAILED,
            user_id=user_id,
            case_id=case_id,
            evidence_id=evidence_id,
            analysis_id=analysis_id,
            details={"verification_id": verification_id, "error_type": error_type},
        )
        db.session.commit()
    except Exception:  # noqa: BLE001 - never mask the original failure
        db.session.rollback()
        logger.exception(
            "Failed to persist FAILED state for face verification=%s", verification_id
        )


def get_face_verification(user: User, verification_id: int) -> FaceVerification:
    row = db.session.get(FaceVerification, verification_id)
    if row is None:
        raise FaceVerificationNotFoundError(f"Face verification {verification_id} not found")
    get_evidence(user, row.evidence_id)
    return row
