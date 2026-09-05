"""Evidence management + integrity verification."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import current_app
from werkzeug.datastructures import FileStorage

from ai.datasets.utils.checksums import hash_file
from backend.app.audit import record_audit
from backend.app.exceptions import (
    EvidenceNotFoundError,
    IntegrityCheckError,
    InvalidEvidenceError,
)
from backend.app.extensions import db
from backend.app.models.entities import Evidence, User
from backend.app.models.enums import (
    AnalysisStatus,
    AuditEventType,
    EvidenceStatus,
    IntegrityStatus,
)
from backend.app.services.case_service import require_case_access
from backend.app.storage import store_evidence_file
from backend.app.utils.paths import resolve_within

logger = logging.getLogger("maya.backend.evidence")


def upload_evidence(
    user: User,
    case_id: int,
    file: FileStorage,
    *,
    notes: str | None = None,
) -> Evidence:
    case = require_case_access(user, case_id)
    upload_root = Path(current_app.config["UPLOAD_DIR"])
    max_bytes = int(current_app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    display_name, stored, abs_path, size = store_evidence_file(
        file,
        upload_root=upload_root,
        case_id=case.id,
        max_bytes=max_bytes,
    )
    try:
        digest = hash_file(abs_path, algorithm="sha256")
    except OSError as exc:
        abs_path.unlink(missing_ok=True)
        raise InvalidEvidenceError(f"Failed to hash evidence: {exc}") from exc

    rel_path = abs_path.relative_to(upload_root).as_posix()
    evidence = Evidence(
        case_id=case.id,
        original_filename=display_name,
        stored_filename=stored,
        storage_path=rel_path,
        media_type="image",
        mime_type=file.mimetype or "image/jpeg",
        file_size_bytes=size,
        sha256_hash=digest,
        status=EvidenceStatus.UPLOADED.value,
        analysis_status=AnalysisStatus.NONE.value,
        uploaded_by_user_id=user.id,
        notes=notes,
    )
    db.session.add(evidence)
    db.session.flush()
    record_audit(
        AuditEventType.EVIDENCE_UPLOADED,
        user_id=user.id,
        case_id=case.id,
        evidence_id=evidence.id,
        details={"sha256": digest, "filename": display_name, "size": size},
    )
    db.session.commit()
    logger.info("Evidence %s uploaded to case %s", evidence.id, case.id)
    return evidence


def get_evidence(user: User, evidence_id: int) -> Evidence:
    evidence = db.session.get(Evidence, evidence_id)
    if evidence is None:
        raise EvidenceNotFoundError(f"Evidence {evidence_id} not found")
    require_case_access(user, evidence.case_id)
    return evidence


def list_case_evidence(user: User, case_id: int) -> list[Evidence]:
    require_case_access(user, case_id)
    return (
        Evidence.query.filter_by(case_id=case_id)
        .order_by(Evidence.uploaded_at.desc())
        .all()
    )


def absolute_evidence_path(evidence: Evidence) -> Path:
    root = Path(current_app.config["UPLOAD_DIR"])
    path = resolve_within(root, evidence.storage_path)
    if path is None:
        raise IntegrityCheckError("Evidence path failed safety check")
    return path


def verify_integrity(user: User, evidence_id: int) -> dict:
    evidence = get_evidence(user, evidence_id)
    path = absolute_evidence_path(evidence)
    error_msg = ""
    try:
        if not path.exists():
            status = IntegrityStatus.MISSING.value
            current = ""
        elif not path.is_file():
            status = IntegrityStatus.MISSING.value
            current = ""
        else:
            current = hash_file(path, algorithm="sha256")
            if current == evidence.sha256_hash:
                status = IntegrityStatus.VALID.value
            else:
                status = IntegrityStatus.MODIFIED.value
    except OSError as exc:
        logger.exception("Integrity check failed evidence=%s", evidence_id)
        status = IntegrityStatus.ERROR.value
        current = ""
        error_msg = str(exc)

    audit_details = {"integrity_status": status}
    if error_msg:
        audit_details["error"] = error_msg
    record_audit(
        AuditEventType.EVIDENCE_VERIFIED,
        user_id=user.id,
        case_id=evidence.case_id,
        evidence_id=evidence.id,
        details=audit_details,
    )
    db.session.commit()
    return {
        "evidence_id": evidence.id,
        "stored_sha256": evidence.sha256_hash,
        "current_sha256": current,
        "integrity_status": status,
    }
