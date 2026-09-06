"""Case management service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.app.audit import record_audit
from backend.app.exceptions import AuthorizationError, CaseNotFoundError, ValidationError
from backend.app.extensions import db
from backend.app.models.entities import (
    AnalysisRun,
    Case,
    Evidence,
    FaceVerification,
    InvestigationReport,
    User,
)
from backend.app.models.enums import AuditEventType, CasePriority, CaseStatus

logger = logging.getLogger("maya.backend.cases")


def _allocate_case_number() -> str:
    year = datetime.now(timezone.utc).year
    prefix = f"CASE-{year}-"
    latest = (
        Case.query.filter(Case.case_number.like(f"{prefix}%"))
        .order_by(Case.case_number.desc())
        .first()
    )
    seq = 1
    if latest and latest.case_number.startswith(prefix):
        try:
            seq = int(latest.case_number.split("-")[-1]) + 1
        except ValueError:
            seq = 1
    return f"{prefix}{seq:06d}"


def _can_access(user: User, case: Case) -> bool:
    return user.is_admin or case.created_by_user_id == user.id


def require_case_access(user: User, case_id: int) -> Case:
    case = db.session.get(Case, case_id)
    if case is None:
        raise CaseNotFoundError(f"Case {case_id} not found")
    if not _can_access(user, case):
        raise AuthorizationError("Not authorized to access this case")
    return case


def create_case(
    user: User,
    *,
    title: str,
    description: str = "",
    priority: str = CasePriority.MEDIUM.value,
) -> Case:
    title = (title or "").strip()
    if not title:
        raise ValidationError("Case title is required")
    try:
        priority_enum = CasePriority(priority.upper())
    except ValueError as exc:
        raise ValidationError("Invalid priority") from exc

    case = Case(
        case_number=_allocate_case_number(),
        title=title,
        description=(description or "").strip(),
        status=CaseStatus.OPEN.value,
        priority=priority_enum.value,
        created_by_user_id=user.id,
    )
    db.session.add(case)
    db.session.flush()
    record_audit(
        AuditEventType.CASE_CREATED,
        user_id=user.id,
        case_id=case.id,
        details={"case_number": case.case_number, "title": title},
    )
    db.session.commit()
    logger.info("Case created %s by user %s", case.case_number, user.id)
    return case


def list_cases(user: User) -> list[Case]:
    q = Case.query
    if not user.is_admin:
        q = q.filter_by(created_by_user_id=user.id)
    return q.order_by(Case.updated_at.desc()).all()


def get_case(user: User, case_id: int) -> Case:
    return require_case_access(user, case_id)


def update_case(
    user: User,
    case_id: int,
    *,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> Case:
    case = require_case_access(user, case_id)
    if case.status == CaseStatus.ARCHIVED.value:
        raise ValidationError("Archived cases cannot be updated")

    if title is not None:
        title = title.strip()
        if not title:
            raise ValidationError("Title cannot be empty")
        case.title = title
    if description is not None:
        case.description = description.strip()
    if priority is not None:
        try:
            case.priority = CasePriority(priority.upper()).value
        except ValueError as exc:
            raise ValidationError("Invalid priority") from exc
    if status is not None:
        try:
            new_status = CaseStatus(status.upper())
        except ValueError as exc:
            raise ValidationError("Invalid status") from exc
        case.status = new_status.value
        if new_status == CaseStatus.CLOSED:
            case.closed_at = datetime.now(timezone.utc).replace(tzinfo=None)

    case.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    record_audit(
        AuditEventType.CASE_UPDATED,
        user_id=user.id,
        case_id=case.id,
        details={"status": case.status, "priority": case.priority},
    )
    db.session.commit()
    return case


def delete_case(user: User, case_id: int) -> dict[str, int | str]:
    """Delete a case and its dependent records (owner or admin only).

    Audit rows are deliberately preserved: the custody trail must survive the
    record it describes. Stored evidence files on disk are left untouched so a
    UI action can never destroy the original bytes.
    """

    case = require_case_access(user, case_id)
    case_number = case.case_number

    # Delete children explicitly: only Case.evidence_items declares a cascade,
    # so reports / analyses / face verifications would otherwise be orphaned.
    report_count = InvestigationReport.query.filter_by(case_id=case.id).delete(
        synchronize_session=False
    )
    face_count = FaceVerification.query.filter_by(case_id=case.id).delete(
        synchronize_session=False
    )
    analysis_count = AnalysisRun.query.filter_by(case_id=case.id).delete(
        synchronize_session=False
    )
    evidence_count = Evidence.query.filter_by(case_id=case.id).delete(
        synchronize_session=False
    )

    db.session.delete(case)
    # case_id is intentionally omitted: the row it referenced no longer exists.
    record_audit(
        AuditEventType.CASE_DELETED,
        user_id=user.id,
        details={
            "case_id": case_id,
            "case_number": case_number,
            "deleted_evidence": int(evidence_count),
            "deleted_analyses": int(analysis_count),
            "deleted_face_verifications": int(face_count),
            "deleted_reports": int(report_count),
        },
    )
    db.session.commit()
    logger.info("Case %s deleted by user %s", case_number, user.id)
    return {
        "case_id": case_id,
        "case_number": case_number,
        "deleted_evidence": int(evidence_count),
        "deleted_analyses": int(analysis_count),
        "deleted_face_verifications": int(face_count),
        "deleted_reports": int(report_count),
    }


def close_case(user: User, case_id: int) -> Case:
    case = require_case_access(user, case_id)
    if case.status == CaseStatus.CLOSED.value:
        return case
    case.status = CaseStatus.CLOSED.value
    case.closed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    case.updated_at = case.closed_at
    record_audit(
        AuditEventType.CASE_CLOSED,
        user_id=user.id,
        case_id=case.id,
        details={"case_number": case.case_number},
    )
    db.session.commit()
    return case
