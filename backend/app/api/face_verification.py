"""Face reference verification JSON API."""

from __future__ import annotations

from flask import Blueprint, request
from flask_login import current_user

from backend.app.exceptions import ValidationError
from backend.app.schemas import api_success, face_verification_to_dict
from backend.app.security import login_required_api
from backend.app.services import face_verification_service

face_verification_bp = Blueprint("api_face_verification", __name__, url_prefix="/api")


@face_verification_bp.post("/evidence/<int:evidence_id>/face-verification")
@login_required_api
def create_face_verification(evidence_id: int):
    if "file" not in request.files:
        raise ValidationError("Multipart field 'file' is required (reference image)")
    row = face_verification_service.create_face_verification(
        current_user,
        evidence_id,
        request.files["file"],
        threshold=request.form.get("threshold"),
        no_match_threshold=request.form.get("no_match_threshold"),
        investigation_id=request.form.get("investigation_id"),
    )
    return api_success(face_verification_to_dict(row), status=201)


@face_verification_bp.get("/face-verifications/<int:verification_id>")
@login_required_api
def get_face_verification(verification_id: int):
    row = face_verification_service.get_face_verification(current_user, verification_id)
    return api_success(face_verification_to_dict(row))
