"""Analysis / investigation JSON API + report generation endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, request, send_from_directory
from flask_login import current_user
from pathlib import Path

from backend.app.exceptions import (
    AuthorizationError,
    ReportFileMissingError,
    ReportNotFoundError,
)
from backend.app.extensions import db
from backend.app.models import InvestigationReport
from backend.app.schemas import analysis_to_dict, api_success
from backend.app.security import login_required_api
from backend.app.services import analysis_service
from backend.app.services.report_service import (
    generate_investigation_report,
    report_to_dict,
)
from backend.app.utils.paths import resolve_within

analysis_bp = Blueprint("api_analysis", __name__, url_prefix="/api")


@analysis_bp.post("/evidence/<int:evidence_id>/analyze")
@login_required_api
def analyze_evidence(evidence_id: int):
    payload = request.get_json(silent=True) or {}
    generate = payload.get("generate_explanation", True)
    explainer = str(payload.get("explainer", "gradcam"))
    advanced_xai = payload.get("advanced_xai") or None
    explainers_list = payload.get("explainers")
    if explainers_list and not advanced_xai:
        advanced_xai = {}
    if explainers_list and advanced_xai is not None:
        if isinstance(explainers_list, list) and len(explainers_list) > 1:
            advanced_xai["multi_explainer"] = True
        if isinstance(explainers_list, list) and len(explainers_list) == 1:
            advanced_xai["explainer"] = explainers_list[0]
    run = analysis_service.analyze_evidence(
        current_user,
        evidence_id,
        generate_explanation=bool(generate),
        explainer=explainer,
        verify_before_analyze=bool(payload.get("verify_before_analyze", True)),
        advanced_xai=advanced_xai,
    )
    return api_success(analysis_to_dict(run), status=201)


@analysis_bp.get("/analysis/<int:analysis_id>")
@login_required_api
def get_analysis(analysis_id: int):
    run = analysis_service.get_analysis(current_user, analysis_id)
    return api_success(analysis_to_dict(run))


@analysis_bp.get("/investigations/<int:analysis_id>")
@login_required_api
def get_investigation(analysis_id: int):
    run = analysis_service.get_analysis(current_user, analysis_id)
    return api_success(analysis_to_dict(run))


@analysis_bp.post("/analysis/<int:analysis_id>/report")
@login_required_api
def generate_report(analysis_id: int):
    payload = request.get_json(silent=True) or {}
    report = generate_investigation_report(
        current_user,
        analysis_id,
        investigator_notes=payload.get("investigator_notes"),
        report_format=payload.get("format", "pdf"),
    )
    return api_success(report_to_dict(report), status=201)


@analysis_bp.get("/analysis/<int:analysis_id>/reports")
@login_required_api
def list_reports(analysis_id: int):
    analysis_service.get_analysis(current_user, analysis_id)  # ownership check
    rows = (
        InvestigationReport.query.filter_by(analysis_id=analysis_id)
        .order_by(InvestigationReport.generated_at.desc())
        .all()
    )
    return api_success([report_to_dict(r) for r in rows])


@analysis_bp.get("/reports/<int:report_id>")
@login_required_api
def get_report_metadata(report_id: int):
    report = db.session.get(InvestigationReport, report_id)
    if report is None:
        raise ReportNotFoundError(f"Report {report_id} not found")
    analysis_service.get_analysis(current_user, report.analysis_id)  # ownership check
    return api_success(report_to_dict(report))


@analysis_bp.get("/reports/<int:report_id>/download")
@login_required_api
def download_report(report_id: int):
    report = db.session.get(InvestigationReport, report_id)
    if report is None:
        raise ReportNotFoundError(f"Report {report_id} not found")
    analysis_service.get_analysis(current_user, report.analysis_id)  # ownership check

    # Reports may be recorded relative to REPORT_DIR or to ROOT_DIR, but the
    # served file must always land inside REPORT_DIR — ROOT_DIR is the whole
    # repository and would otherwise expose .env or the SQLite database.
    report_root = Path(current_app.config["REPORT_DIR"]).resolve()
    root = Path(current_app.config["ROOT_DIR"]).resolve()
    candidates = [
        resolve_within(report_root, report.storage_path),
        resolve_within(root, report.storage_path),
    ]
    full_path = next(
        (
            path
            for path in candidates
            if path is not None and path.is_file() and resolve_within(report_root, path)
        ),
        None,
    )
    if full_path is None:
        stored = (root / report.storage_path).resolve()
        if stored.is_file() or (report_root / report.storage_path).resolve().is_file():
            raise AuthorizationError("Report path failed safety check")
        raise ReportFileMissingError(f"Report {report_id} file is not available")
    return send_from_directory(
        str(full_path.parent),
        full_path.name,
        as_attachment=True,
        download_name=f"{report.report_number}.{report.report_format}",
    )
