"""Phase 8 regression tests — one test per hardening bug actually found.

Each test below failed against the pre-Phase-8 backend. They are not generic
coverage: they pin the specific behaviours that were wrong.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login  # noqa: E402
from tests.test_analysis_api import _fake_investigation  # noqa: E402


@pytest.fixture()
def app(tmp_path: Path):
    from backend.app import create_app
    from backend.app.extensions import db

    application = create_app("testing")
    application.config["UPLOAD_DIR"] = tmp_path / "uploads"
    application.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["ROOT_DIR"] = tmp_path
    application.config["REPORT_DIR"] = tmp_path / "reports"
    application.config["REPORT_DIR"].mkdir(parents=True, exist_ok=True)
    application.config["SECRET_KEY"] = "test-secret-key"
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def _png(size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(30, 90, 160)).save(buf, format="PNG")
    return buf.getvalue()


def _upload_evidence(client) -> tuple[int, int]:
    case_id = client.post("/api/cases", json={"title": "Hardening"}).get_json()["data"]["case_id"]
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "ev.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.get_json()
    return case_id, resp.get_json()["data"]["evidence_id"]


def _analyze(client, evidence_id: int, **overrides) -> "object":
    body = {"generate_explanation": False, "verify_before_analyze": False}
    body.update(overrides)
    return client.post(f"/api/evidence/{evidence_id}/analyze", json=body)


# --------------------------------------------------------------------------
# HTTP error envelope
# --------------------------------------------------------------------------


def test_wrong_http_method_returns_405_not_500(client) -> None:
    """The catch-all Exception handler used to swallow HTTPException -> 500."""
    resp = client.get("/api/auth/login")
    assert resp.status_code == 405
    body = resp.get_json()
    assert body["ok"] is False
    assert body["error"] == "method_not_allowed"


def test_oversized_upload_returns_413_not_500(client, app) -> None:
    """A payload over MAX_CONTENT_LENGTH must report the limit, not a 500."""
    register_and_login(client, username="bigupload", email="big@maya.test")
    case_id = client.post("/api/cases", json={"title": "Big"}).get_json()["data"]["case_id"]
    app.config["MAX_CONTENT_LENGTH"] = 2048

    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png((512, 512))), "big.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    assert resp.get_json()["error"] == "payload_too_large"


def test_missing_report_returns_404_not_400(client) -> None:
    """Metadata used to answer 400 validation_error while download answered 404."""
    register_and_login(client, username="rep404", email="rep404@maya.test")
    meta = client.get("/api/reports/999999")
    download = client.get("/api/reports/999999/download")
    assert meta.status_code == 404
    assert meta.get_json()["error"] == "report_not_found"
    assert download.status_code == 404
    assert download.get_json()["error"] == "report_not_found"


# --------------------------------------------------------------------------
# AI / XAI robustness
# --------------------------------------------------------------------------


def test_xai_failure_does_not_fail_the_investigation(client) -> None:
    """A Grad-CAM crash must not discard a successful inference."""
    register_and_login(client, username="xaifail", email="xaifail@maya.test")
    _, evidence_id = _upload_evidence(client)

    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ), patch(
        "backend.app.services.analysis_service.run_explanation",
        side_effect=RuntimeError("gradcam exploded"),
    ):
        resp = _analyze(client, evidence_id, generate_explanation=True)

    assert resp.status_code == 201, resp.get_json()
    data = resp.get_json()["data"]
    assert data["analysis_status"] == "COMPLETED"
    assert data["prediction"] in {"REAL", "FAKE"}

    events = [e["event_type"] for e in client.get("/api/audit").get_json()["data"]]
    assert "XAI_FAILED" in events
    assert "ANALYSIS_COMPLETED" in events


def test_advanced_xai_failure_does_not_fail_the_investigation(client) -> None:
    """Optional advanced XAI is best-effort; its failure must be non-fatal."""
    register_and_login(client, username="advxai", email="advxai@maya.test")
    _, evidence_id = _upload_evidence(client)

    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ), patch(
        "backend.app.services.analysis_service.run_advanced_xai",
        side_effect=RuntimeError("shap exploded"),
    ):
        resp = _analyze(client, evidence_id, advanced_xai={"shap": True})

    assert resp.status_code == 201, resp.get_json()
    assert resp.get_json()["data"]["analysis_status"] == "COMPLETED"


def test_inference_failure_leaves_no_processing_row(client, app) -> None:
    """A genuine inference failure must still reach a terminal FAILED state."""
    from backend.app.models import AnalysisRun

    register_and_login(client, username="inffail", email="inffail@maya.test")
    _, evidence_id = _upload_evidence(client)

    with patch(
        "backend.app.services.analysis_service.run_inference",
        side_effect=RuntimeError("model missing"),
    ):
        resp = _analyze(client, evidence_id)

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "analysis_processing_error"
    rows = AnalysisRun.query.filter_by(evidence_id=evidence_id).all()
    assert rows and all(r.status == "FAILED" for r in rows)


# --------------------------------------------------------------------------
# Report generation / download
# --------------------------------------------------------------------------


def _generate_report(client, evidence_id: int) -> tuple[int, int]:
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]
    gen = client.post(f"/api/analysis/{analysis_id}/report", json={})
    assert gen.status_code == 201, gen.get_json()
    return analysis_id, gen.get_json()["data"]["report_id"]


def test_report_download_returns_pdf(client) -> None:
    """Regression for the reported `get_analysis() takes 1 positional argument`."""
    register_and_login(client, username="downloader", email="dl@maya.test")
    _, evidence_id = _upload_evidence(client)
    _, report_id = _generate_report(client, evidence_id)

    resp = client.get(f"/api/reports/{report_id}/download")
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    assert resp.headers["Content-Type"] == "application/pdf"
    assert resp.get_data().startswith(b"%PDF")


def test_report_download_rejects_path_traversal(client, app, tmp_path: Path) -> None:
    """A poisoned storage_path must never serve a file outside REPORT_DIR."""
    from backend.app.extensions import db
    from backend.app.models import InvestigationReport

    secret = tmp_path.parent / "hardening_secret.txt"
    secret.write_text("TOPSECRET", encoding="utf-8")

    register_and_login(client, username="trav", email="trav@maya.test")
    _, evidence_id = _upload_evidence(client)
    _, report_id = _generate_report(client, evidence_id)

    row = db.session.get(InvestigationReport, report_id)
    for poisoned in ("../hardening_secret.txt", str(secret)):
        row.storage_path = poisoned
        db.session.commit()
        resp = client.get(f"/api/reports/{report_id}/download")
        assert resp.status_code == 403, poisoned
        assert b"TOPSECRET" not in resp.get_data()


def test_report_format_cannot_escape_report_dir(client) -> None:
    """`format` is interpolated into the output filename, so it is allowlisted."""
    register_and_login(client, username="fmt", email="fmt@maya.test")
    _, evidence_id = _upload_evidence(client)
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]

    resp = client.post(
        f"/api/analysis/{analysis_id}/report",
        json={"format": "../../../../escaped.pdf"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "validation_error"


# --------------------------------------------------------------------------
# Path containment helper
# --------------------------------------------------------------------------


def test_sibling_directory_is_not_inside_root(tmp_path: Path) -> None:
    """`startswith` treated /x/uploads-evil as living under /x/uploads."""
    from backend.app.utils.paths import is_within, resolve_within

    root = tmp_path / "uploads"
    root.mkdir()
    sibling = tmp_path / "uploads-evil"
    sibling.mkdir()

    assert is_within(root, root / "cases" / "1" / "a.png")
    assert not is_within(root, sibling / "a.png")
    assert resolve_within(root, "cases/1/a.png") is not None
    assert resolve_within(root, "../uploads-evil/a.png") is None
    assert resolve_within(root, str(sibling / "a.png")) is None


# --------------------------------------------------------------------------
# Production configuration safety
# --------------------------------------------------------------------------


def test_production_rejects_placeholder_secret_key() -> None:
    """A development SECRET_KEY must not silently protect production sessions."""
    from backend.app.config import (
        InsecureConfigurationError,
        ProductionConfig,
        validate_production_config,
    )

    class Placeholder(ProductionConfig):
        SECRET_KEY = "dev-only-change-me"

    class TooShort(ProductionConfig):
        SECRET_KEY = "abc123"

    class Strong(ProductionConfig):
        SECRET_KEY = "n4Xk_2sQeR7hTvB1yLpZ0aWmCdFgHjKl"

    for bad in (Placeholder, TooShort):
        with pytest.raises(InsecureConfigurationError):
            validate_production_config(bad)
    validate_production_config(Strong)


def test_production_config_marks_cookies_secure() -> None:
    from backend.app.config import ProductionConfig

    assert ProductionConfig.SESSION_COOKIE_SECURE is True
    assert ProductionConfig.SESSION_COOKIE_HTTPONLY is True
    assert ProductionConfig.DEBUG is False
