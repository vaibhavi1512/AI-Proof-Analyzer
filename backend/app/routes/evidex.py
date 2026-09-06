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

from flask import Blueprint, abort, current_app, send_from_directory

from backend.app.utils.paths import resolve_within

evidex_bp = Blueprint("evidex", __name__, url_prefix="/evidex")

# Only the three files the frontend actually consists of are servable.
_ALLOWED_ASSETS = {"index.html", "script.js", "api.js", "styles.css"}


def _evidex_dir() -> Path:
    return Path(current_app.config["ROOT_DIR"]) / "DIGITALEVIDENCE_FIXED"


@evidex_bp.get("/")
def evidex_index():
    return send_from_directory(str(_evidex_dir()), "index.html")


@evidex_bp.get("/<path:asset>")
def evidex_asset(asset: str):
    if asset not in _ALLOWED_ASSETS:
        abort(404)
    root = _evidex_dir()
    full = resolve_within(root, asset)
    if full is None or not full.is_file():
        abort(404)
    return send_from_directory(str(root), full.name)
