"""Cases JSON API."""

from __future__ import annotations

from flask import Blueprint, request
from flask_login import current_user

from backend.app.schemas import api_success, case_to_dict
from backend.app.security import login_required_api
from backend.app.services import case_service

cases_bp = Blueprint("api_cases", __name__, url_prefix="/api/cases")


@cases_bp.post("")
@login_required_api
def create_case():
    payload = request.get_json(silent=True) or {}
    case = case_service.create_case(
        current_user,
        title=str(payload.get("title", "")),
        description=str(payload.get("description", "")),
        priority=str(payload.get("priority", "MEDIUM")),
    )
    return api_success(case_to_dict(case), status=201)


@cases_bp.get("")
@login_required_api
def list_cases():
    cases = case_service.list_cases(current_user)
    return api_success([case_to_dict(c) for c in cases])


@cases_bp.get("/<int:case_id>")
@login_required_api
def get_case(case_id: int):
    case = case_service.get_case(current_user, case_id)
    return api_success(case_to_dict(case))


@cases_bp.patch("/<int:case_id>")
@login_required_api
def patch_case(case_id: int):
    payload = request.get_json(silent=True) or {}
    case = case_service.update_case(
        current_user,
        case_id,
        title=payload.get("title"),
        description=payload.get("description"),
        status=payload.get("status"),
        priority=payload.get("priority"),
    )
    return api_success(case_to_dict(case))


@cases_bp.post("/<int:case_id>/close")
@login_required_api
def close_case(case_id: int):
    case = case_service.close_case(current_user, case_id)
    return api_success(case_to_dict(case), message="Case closed")


@cases_bp.delete("/<int:case_id>")
@login_required_api
def delete_case(case_id: int):
    result = case_service.delete_case(current_user, case_id)
    return api_success(result, message="Case deleted")
