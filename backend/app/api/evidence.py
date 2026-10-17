"""Evidence JSON API."""

from __future__ import annotations

from flask import Blueprint, request
from flask_login import current_user

from backend.app.exceptions import ValidationError
from backend.app.schemas import api_success, evidence_to_dict
from backend.app.security import login_required_api
from backend.app.services import evidence_service

evidence_bp = Blueprint("api_evidence", __name__, url_prefix="/api/evidence")


@evidence_bp.post("/cases/<int:case_id>")
@login_required_api
def upload_evidence(case_id: int):
    if "file" not in request.files:
        raise ValidationError("Multipart field 'file' is required")
    notes = request.form.get("notes")
    evidence = evidence_service.upload_evidence(
        current_user,
        case_id,
        request.files["file"],
        notes=notes,
    )
    return api_success(evidence_to_dict(evidence), status=201)


@evidence_bp.post("/compare-integrity")
@login_required_api
def compare_integrity():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValidationError("A JSON object with evidence_ids is required")
    result = evidence_service.compare_integrity(current_user, payload.get("evidence_ids"))
    return api_success(result)


@evidence_bp.get("/cases/<int:case_id>")
@login_required_api
def list_evidence(case_id: int):
    items = evidence_service.list_case_evidence(current_user, case_id)
    return api_success([evidence_to_dict(e) for e in items])


@evidence_bp.get("/<int:evidence_id>")
@login_required_api
def get_evidence(evidence_id: int):
    evidence = evidence_service.get_evidence(current_user, evidence_id)
    return api_success(evidence_to_dict(evidence))


@evidence_bp.post("/<int:evidence_id>/verify-integrity")
@login_required_api
def verify_integrity(evidence_id: int):
    result = evidence_service.verify_integrity(current_user, evidence_id)
    return api_success(result)
