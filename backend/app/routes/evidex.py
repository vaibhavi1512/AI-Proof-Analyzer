"""Serve the EVIDEX frontend (DIGITALEVIDENCE_FIXED/) from the Flask origin.

Why this exists:
    EVIDEX authenticates with the existing Flask-Login *session cookie*. Serving
    it from the same origin as /api keeps the cookie first-party, so no CORS
    configuration and no cross-origin credential relaxation is needed.

    This does not touch the separate ``frontend/`` health shell, which keeps
    owning ``/`` and ``/static``.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, send_from_directory

from backend.app.utils.paths import resolve_within

evidex_bp = Blueprint("evidex", __name__, url_prefix="/evidex")

# Only the three files the frontend actually consists of are servable.
_ALLOWED_ASSETS = {
    "index.html",
    "script.js",
    "api.js",
    "styles.css",
    "video_result.js",
    "password_policy.js",
}


def _evidex_dir() -> Path:
    return Path(current_app.config["ROOT_DIR"]) / "DIGITALEVIDENCE_FIXED"


@evidex_bp.get("/")
def evidex_index():
    return _unstored(send_from_directory(str(_evidex_dir()), "index.html", max_age=0))


def _unstored(response):
    """Keep EVIDEX assets from being reused after a file is added or replaced."""

    response.headers["Cache-Control"] = "no-store"
    return response


@evidex_bp.get("/<path:asset>")
def evidex_asset(asset: str):
    root = _evidex_dir()
    full = resolve_within(root, asset) if asset in _ALLOWED_ASSETS else None
    if full is None or not full.is_file():
        from flask import make_response

        return _unstored(make_response("Not found", 404))
    return _unstored(send_from_directory(str(root), full.name, max_age=0))
