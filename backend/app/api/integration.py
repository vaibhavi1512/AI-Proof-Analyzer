"""Thin read-only endpoints required by the EVIDEX frontend.

Why this exists:
    The EVIDEX UI needs data that already exists in the database/filesystem but
    had no HTTP surface: XAI artifact images, the stored evidence image, the
    analyses belonging to an evidence item, a per-evidence custody trail, and
    dashboard aggregates. Every handler here is a thin read over existing models
    and reuses the existing authorization services (``get_analysis`` /
    ``get_evidence``) — no new business logic, no new tables, no duplicated
    subsystem.
"""

from __future__ import annotations

import json
import mimetypes
from datetime import datetime, timedelta, timezone
from pathlib import Path

from flask import Blueprint, current_app, request, send_from_directory
from flask_login import current_user
from sqlalchemy import func

from backend.app.exceptions import AuthorizationError, NotFoundError, ValidationError
from backend.app.extensions import db
from backend.app.models.entities import (
    AnalysisRun,
    AuditLog,
    Case,
    Evidence,
    FaceVerification,
    InvestigationReport,
    User,
)
from backend.app.models.enums import UserRole
from backend.app.schemas import (
    analysis_to_dict,
    api_success,
    evidence_to_dict,
    user_to_dict,
)
from backend.app.security import login_required_api
from backend.app.services import analysis_service
from backend.app.services.evidence_service import absolute_evidence_path, get_evidence
from backend.app.services.video_service import is_video_evidence, video_frames_dir
from backend.app.utils.paths import is_within, resolve_within

integration_bp = Blueprint("api_integration", __name__, url_prefix="/api")

# Files written by the existing Grad-CAM visualization bundle.
_ARTIFACT_FILES = {
    "heatmap": "heatmap.png",
    "overlay": "overlay.png",
    "original": "original.png",
}

# Analysis columns that hold a filesystem path to a generated XAI image.
_ARTIFACT_KINDS = {
    "heatmap": "heatmap_path",
    "overlay": "overlay_path",
}

_MAX_FRAME_NUMBER = 10_000_000
_FRAME_SUFFIXES = (".jpg", ".jpeg", ".png")


def _resolve_under(root: Path, stored: str | Path | None) -> Path | None:
    """Resolve ``stored`` only when it stays inside ``root``."""

    if not stored:
        return None
    stored_path = Path(stored)
    if stored_path.is_absolute():
        resolved = stored_path.resolve()
        return resolved if is_within(root, resolved) else None
    return resolve_within(root, stored_path)


def _parse_frame_number(raw: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError("frame must be an integer") from exc
    if value < 0 or value > _MAX_FRAME_NUMBER:
        raise ValidationError("frame is out of range")
    return value


def _send_file(path: Path, *, mimetype: str | None = None):
    guessed = mimetype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return send_from_directory(
        str(path.parent),
        path.name,
        mimetype=guessed,
        as_attachment=False,
    )


def _original_next_to_heatmap(run, artifacts_root: Path) -> Path:
    heatmap = _resolve_under(artifacts_root, run.heatmap_path)
    if heatmap is None:
        raise AuthorizationError("Artifact path failed safety check")
    original = (heatmap.parent / _ARTIFACT_FILES["original"]).resolve()
    if not is_within(artifacts_root, original):
        raise AuthorizationError("Artifact path failed safety check")
    if not original.is_file():
        raise NotFoundError(f"No original artifact for analysis {run.id}")
    return original


def _video_xai_file(run, artifacts_root: Path, filename: str) -> Path:
    """Serve one file from the video XAI directory. The name is not a client path."""

    artifact_dir = _resolve_under(artifacts_root, run.artifact_dir)
    if artifact_dir is None:
        raise NotFoundError(f"No video explanation artifact for analysis {run.id}")
    target = (artifact_dir / "xai" / "video" / filename).resolve()
    if not is_within(artifacts_root, target) or not is_within(artifact_dir, target):
        raise AuthorizationError("Artifact path failed safety check")
    if not target.is_file():
        raise NotFoundError(f"No video explanation artifact {filename} for analysis {run.id}")
    return target


def _frame_xai_file(run, artifacts_root: Path, kind: str, frame_number: int) -> Path:
    artifact_dir = _resolve_under(artifacts_root, run.artifact_dir)
    if artifact_dir is None:
        raise NotFoundError(f"No {kind} artifact for analysis {run.id} frame {frame_number}")
    target = (
        artifact_dir / "xai" / "gradcam" / f"frame_{frame_number:06d}" / _ARTIFACT_FILES[kind]
    ).resolve()
    if not is_within(artifacts_root, target):
        raise AuthorizationError("Artifact path failed safety check")
    if not target.is_file():
        raise NotFoundError(f"No {kind} artifact for analysis {run.id} frame {frame_number}")
    return target


def _is_admin() -> bool:
    return current_user.role == UserRole.ADMIN.value


def _owned_case_ids() -> list[int]:
    """Case ids the caller may see (all of them for ADMIN)."""

    q = Case.query.with_entities(Case.id)
    if not _is_admin():
        q = q.filter(Case.created_by_user_id == current_user.id)
    return [row[0] for row in q.all()]


@integration_bp.get("/analysis/<int:analysis_id>/artifact/<kind>")
@login_required_api
def get_analysis_artifact(analysis_id: int, kind: str):
    """Serve Grad-CAM PNGs produced by the existing XAI stage.

    ``original`` is the still next to the heatmap (extracted frame), never the
    stored MP4. Optional ``?frame=N`` serves that frame's existing CAM files.
    """

    if kind not in _ARTIFACT_FILES and kind not in {"video_contact_sheet", "video_gradcam"}:
        raise ValidationError(
            f"Unsupported artifact kind. Allowed: {sorted(_ARTIFACT_FILES)}"
        )
    run = analysis_service.get_analysis(current_user, analysis_id)  # ownership check
    artifacts_root = (Path(current_app.config["ROOT_DIR"]) / "artifacts").resolve()
    frame_raw = request.args.get("frame")
    if kind == "video_contact_sheet":
        return _send_file(_video_xai_file(run, artifacts_root, "gradcam_contact_sheet.png"))
    if kind == "video_gradcam":
        if frame_raw is None or str(frame_raw).strip() == "":
            raise ValidationError("frame is required")
        frame_index = _parse_frame_number(str(frame_raw))
        return _send_file(
            _video_xai_file(run, artifacts_root, f"gradcam_frame_{frame_index}.png")
        )
    if frame_raw is not None and str(frame_raw).strip() != "":
        full_path = _frame_xai_file(
            run, artifacts_root, kind, _parse_frame_number(str(frame_raw))
        )
    elif kind == "original":
        full_path = _original_next_to_heatmap(run, artifacts_root)
    else:
        stored = getattr(run, _ARTIFACT_KINDS[kind], None)
        if not stored:
            raise NotFoundError(f"No {kind} artifact for analysis {analysis_id}")
        full_path = _resolve_under(artifacts_root, stored)
        if full_path is None:
            raise AuthorizationError("Artifact path failed safety check")
        if not full_path.is_file():
            raise NotFoundError(f"Artifact file for analysis {analysis_id} is not available")

    return _send_file(full_path)


@integration_bp.get("/evidence/<int:evidence_id>/file")
@login_required_api
def get_evidence_file(evidence_id: int):
    """Serve the stored evidence image inline for side-by-side XAI comparison.

    Reuses ``get_evidence`` for ownership and ``absolute_evidence_path`` for the
    UPLOAD_DIR containment check, so no new path handling is introduced.
    """

    evidence = get_evidence(current_user, evidence_id)  # ownership check
    path = absolute_evidence_path(evidence)
    if not path.is_file():
        raise NotFoundError(f"Evidence {evidence_id} file is not available")
    return send_from_directory(
        str(path.parent),
        path.name,
        mimetype=evidence.mime_type or "application/octet-stream",
        as_attachment=False,
    )


@integration_bp.get("/evidence/<int:evidence_id>/frames/<int:frame_number>")
@login_required_api
def get_evidence_frame(evidence_id: int, frame_number: int):
    """Serve one extracted video JPEG from the evidence frames directory."""

    if frame_number < 0 or frame_number > _MAX_FRAME_NUMBER:
        raise ValidationError("frame is out of range")
    evidence = get_evidence(current_user, evidence_id)
    if not is_video_evidence(evidence):
        raise ValidationError("Frame files are only available for video evidence")
    upload_root = Path(current_app.config["UPLOAD_DIR"]).resolve()
    frames_dir = video_frames_dir(evidence, upload_root)
    if not is_within(upload_root, frames_dir):
        raise AuthorizationError("Frame path failed safety check")
    full_path = None
    for suffix in _FRAME_SUFFIXES:
        candidate = (frames_dir / f"frame_{frame_number:06d}{suffix}").resolve()
        if not is_within(upload_root, candidate):
            raise AuthorizationError("Frame path failed safety check")
        if candidate.is_file():
            full_path = candidate
            break
    if full_path is None:
        raise NotFoundError(
            f"Extracted frame {frame_number} is not available for evidence {evidence_id}"
        )
    return _send_file(full_path)



@integration_bp.get("/evidence/<int:evidence_id>/analyses")
@login_required_api
def list_evidence_analyses(evidence_id: int):
    """Analyses for one evidence item, newest first.

    ``GET /api/analysis/<id>`` needs an analysis id, and the evidence record
    only carries a status, so a reloaded page had no way back to an existing
    result. This exposes the existing ``Evidence.analysis_runs`` relationship.
    """

    evidence = get_evidence(current_user, evidence_id)  # ownership check
    rows = (
        AnalysisRun.query.filter_by(evidence_id=evidence.id)
        .order_by(AnalysisRun.id.desc())
        .all()
    )
    return api_success([analysis_to_dict(r) for r in rows])


@integration_bp.get("/evidence/<int:evidence_id>/custody")
@login_required_api
def get_evidence_custody(evidence_id: int):
    """Chain-of-custody timeline for one evidence item, from existing AuditLog rows."""

    evidence = get_evidence(current_user, evidence_id)  # ownership check
    rows = (
        AuditLog.query.filter_by(evidence_id=evidence.id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )
    events = [
        {
            "audit_id": r.id,
            "event_type": r.event_type,
            "user_id": r.user_id,
            "case_id": r.case_id,
            "analysis_id": r.analysis_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "details": json.loads(r.details_json or "{}"),
        }
        for r in rows
    ]
    case = evidence.case
    return api_success(
        {
            "evidence": evidence_to_dict(evidence),
            "case_number": case.case_number if case else None,
            "case_title": case.title if case else None,
            "events": events,
            "event_count": len(events),
        }
    )


@integration_bp.get("/admin/users")
@login_required_api
def list_admin_users():
    """Directory of real accounts for the admin console (ADMIN only).

    The user table had no HTTP surface, so the admin console had nothing real to
    render. This is a read-only projection of ``User`` via the existing
    ``user_to_dict`` schema plus per-user case/evidence counts.
    """

    if not _is_admin():
        raise AuthorizationError("Administrator role required")

    users = User.query.order_by(User.id.asc()).all()
    case_counts: dict[int, int] = {}
    for owner_id, count in (
        db.session.query(Case.created_by_user_id, func.count(Case.id))
        .group_by(Case.created_by_user_id)
        .all()
    ):
        case_counts[int(owner_id)] = int(count)
    evidence_counts: dict[int, int] = {}
    for uploader_id, count in (
        db.session.query(Evidence.uploaded_by_user_id, func.count(Evidence.id))
        .group_by(Evidence.uploaded_by_user_id)
        .all()
    ):
        evidence_counts[int(uploader_id)] = int(count)

    return api_success(
        [
            {
                **user_to_dict(u),
                "case_count": case_counts.get(u.id, 0),
                "evidence_count": evidence_counts.get(u.id, 0),
            }
            for u in users
        ]
    )


@integration_bp.get("/dashboard/stats")
@login_required_api
def get_dashboard_stats():
    """Ownership-scoped aggregates for the dashboard (plain SQL counts, no AI)."""

    case_ids = _owned_case_ids()

    cases = Case.query
    evidence = Evidence.query
    analyses = AnalysisRun.query
    faces = FaceVerification.query
    reports = InvestigationReport.query
    if not _is_admin():
        visible = case_ids or [-1]
        cases = cases.filter(Case.created_by_user_id == current_user.id)
        evidence = evidence.filter(Evidence.case_id.in_(visible))
        analyses = analyses.filter(AnalysisRun.case_id.in_(visible))
        faces = faces.filter(FaceVerification.case_id.in_(visible))
        reports = reports.filter(InvestigationReport.case_id.in_(visible))

    case_rows = cases.all()
    evidence_rows = evidence.all()
    analysis_rows = analyses.all()

    def _tally(values) -> dict[str, int]:
        out: dict[str, int] = {}
        for value in values:
            key = str(value or "UNKNOWN")
            out[key] = out.get(key, 0) + 1
        return out

    completed = [r for r in analysis_rows if r.status == "COMPLETED"]

    # Real 14-day authenticity trend and 6-week case activity, derived from
    # stored timestamps. No synthetic series.
    today = datetime.now(timezone.utc).replace(tzinfo=None).date()
    trend = []
    for offset in range(13, -1, -1):
        day = today - timedelta(days=offset)
        day_runs = [
            r for r in completed
            if r.completed_at is not None and r.completed_at.date() == day
        ]
        trend.append(
            {
                "date": day.isoformat(),
                "real": sum(1 for r in day_runs if r.prediction == "REAL"),
                "fake": sum(1 for r in day_runs if r.prediction == "FAKE"),
            }
        )

    case_activity = []
    for offset in range(5, -1, -1):
        end = today - timedelta(days=offset * 7)
        start = end - timedelta(days=6)
        case_activity.append(
            {
                "week_start": start.isoformat(),
                "opened": sum(
                    1 for c in case_rows
                    if c.created_at is not None and start <= c.created_at.date() <= end
                ),
            }
        )

    return api_success(
        {
            "scope": "all" if _is_admin() else "own",
            "counts": {
                "cases": len(case_rows),
                "evidence": len(evidence_rows),
                "analyses": len(analysis_rows),
                "analyses_completed": len(completed),
                "face_verifications": faces.count(),
                "reports": reports.count(),
            },
            "case_status": _tally(c.status for c in case_rows),
            "case_priority": _tally(c.priority for c in case_rows),
            "evidence_status": _tally(e.status for e in evidence_rows),
            "analysis_status": _tally(r.status for r in analysis_rows),
            # Authenticity distribution comes straight from stored predictions;
            # nothing here is inferred or recomputed.
            "prediction": _tally(r.prediction for r in completed),
            "trend": trend,
            "case_activity": case_activity,
            "recent_analyses": [
                {
                    "analysis_id": r.id,
                    "evidence_id": r.evidence_id,
                    "case_id": r.case_id,
                    "investigation_id": r.investigation_id,
                    "prediction": r.prediction,
                    "confidence": r.confidence,
                    "status": r.status,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in sorted(analysis_rows, key=lambda x: x.id, reverse=True)[:10]
            ],
        }
    )
