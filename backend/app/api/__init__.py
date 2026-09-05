"""JSON API blueprint registration."""

from __future__ import annotations

from flask import Flask

from backend.app.api.analysis import analysis_bp
from backend.app.api.audit import audit_bp
from backend.app.api.auth import auth_bp
from backend.app.api.cases import cases_bp
from backend.app.api.evidence import evidence_bp
from backend.app.api.face_verification import face_verification_bp
from backend.app.exceptions import MayaProductError
from backend.app.schemas import api_error


def register_api_blueprints(app: Flask) -> None:
    app.register_blueprint(auth_bp)
    app.register_blueprint(cases_bp)
    app.register_blueprint(evidence_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(audit_bp)
    app.register_blueprint(face_verification_bp)

    @app.errorhandler(MayaProductError)
    def handle_product_error(exc: MayaProductError):
        return api_error(exc.message, status=exc.status_code, error_code=exc.error_code)
