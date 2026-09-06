"""Blueprint registration for the Application layer.

Why this exists:
    Keeps ``create_app`` thin by centralizing blueprint imports and
    registration in one place.
"""

from __future__ import annotations

from flask import Flask

from backend.app.routes.evidex import evidex_bp
from backend.app.routes.health import health_bp
from backend.app.routes.shell import shell_bp


def register_blueprints(app: Flask) -> None:
    """Attach HTTP blueprints to the Flask application."""

    app.register_blueprint(health_bp)
    # Presentation shell only — no business/domain routes in Phase 1.
    app.register_blueprint(shell_bp)
    # EVIDEX frontend, served same-origin so the session cookie stays first-party.
    app.register_blueprint(evidex_bp)
