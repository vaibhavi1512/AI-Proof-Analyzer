"""Regression tests for the thin endpoints added for the EVIDEX frontend.

These cover the five routes introduced during frontend integration:
    GET /api/dashboard/stats
    GET /api/evidence/<id>/custody
    GET /api/evidence/<id>/analyses
    GET /api/evidence/<id>/file
    GET /api/analysis/<id>/artifact/<kind>

The emphasis is authorization and path safety: each endpoint exposes existing
data, so the risk is leaking it to the wrong caller or serving a file from
outside its permitted root.
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


def _upload_evidence(client, title: str = "Integration") -> tuple[int, int]:
    case_id = client.post("/api/cases", json={"title": title}).get_json()["data"]["case_id"]
    resp = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "ev.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201, resp.get_json()
    return case_id, resp.get_json()["data"]["evidence_id"]


def _analyze(client, evidence_id: int):
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        return client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )


# --------------------------------------------------------------------------
# Authentication: none of the new endpoints may answer an anonymous caller.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/api/dashboard/stats",
        "/api/evidence/1/custody",
        "/api/evidence/1/analyses",
        "/api/evidence/1/file",
        "/api/analysis/1/artifact/heatmap",
    ],
)
def test_new_endpoints_require_authentication(client, path):
    resp = client.get(path)
    assert resp.status_code == 401
    assert resp.get_json()["ok"] is False


# --------------------------------------------------------------------------
# Dashboard stats
# --------------------------------------------------------------------------


def test_dashboard_stats_reports_real_counts(client):
    register_and_login(client, username="dashuser")
    _case_id, evidence_id = _upload_evidence(client)
    assert _analyze(client, evidence_id).status_code == 201

    data = client.get("/api/dashboard/stats").get_json()["data"]
    assert data["scope"] == "own"
    assert data["counts"]["cases"] == 1
    assert data["counts"]["evidence"] == 1
    assert data["counts"]["analyses"] == 1
    assert data["counts"]["analyses_completed"] == 1
    # The fake investigation predicts REAL, so that is what must be tallied.
    assert data["prediction"]["REAL"] == 1
    assert len(data["trend"]) == 14
    assert len(data["case_activity"]) == 6
    assert sum(day["real"] for day in data["trend"]) == 1


def test_dashboard_stats_excludes_other_users_data(client, app):
    register_and_login(client, username="owneruser")
    _upload_evidence(client, title="Owned")
    client.post("/api/auth/logout")

    other = app.test_client()
    register_and_login(other, username="otheruser")
    data = other.get("/api/dashboard/stats").get_json()["data"]
    assert data["counts"]["cases"] == 0
    assert data["counts"]["evidence"] == 0


# --------------------------------------------------------------------------
# Custody trail
# --------------------------------------------------------------------------


def test_custody_returns_real_audit_events(client):
    register_and_login(client, username="custodyuser")
    _case_id, evidence_id = _upload_evidence(client)
    assert _analyze(client, evidence_id).status_code == 201

    data = client.get(f"/api/evidence/{evidence_id}/custody").get_json()["data"]
    assert data["evidence"]["evidence_id"] == evidence_id
    assert data["event_count"] == len(data["events"])
    kinds = {e["event_type"] for e in data["events"]}
    assert "EVIDENCE_UPLOADED" in kinds
    assert "ANALYSIS_COMPLETED" in kinds
    # Timestamps must be ascending so the UI timeline is chronological.
    stamps = [e["timestamp"] for e in data["events"]]
    assert stamps == sorted(stamps)


def test_custody_denied_for_other_users_evidence(client, app):
    register_and_login(client, username="custodyowner")
    _case_id, evidence_id = _upload_evidence(client)
    client.post("/api/auth/logout")

    other = app.test_client()
    register_and_login(other, username="custodyintruder")
    resp = other.get(f"/api/evidence/{evidence_id}/custody")
    assert resp.status_code in (403, 404)
    assert resp.get_json()["ok"] is False


# --------------------------------------------------------------------------
# Analyses for an evidence item
# --------------------------------------------------------------------------


def test_evidence_analyses_lists_runs_newest_first(client):
    register_and_login(client, username="analyseslist")
    _case_id, evidence_id = _upload_evidence(client)
    assert _analyze(client, evidence_id).status_code == 201
    assert _analyze(client, evidence_id).status_code == 201

    data = client.get(f"/api/evidence/{evidence_id}/analyses").get_json()["data"]
    assert len(data) == 2
    assert data[0]["analysis_id"] > data[1]["analysis_id"]
    assert data[0]["analysis_status"] == "COMPLETED"


def test_evidence_analyses_denied_for_other_user(client, app):
    register_and_login(client, username="analysesowner")
    _case_id, evidence_id = _upload_evidence(client)
    client.post("/api/auth/logout")

    other = app.test_client()
    register_and_login(other, username="analysesintruder")
    assert other.get(f"/api/evidence/{evidence_id}/analyses").status_code in (403, 404)


# --------------------------------------------------------------------------
# Evidence image
# --------------------------------------------------------------------------


def test_evidence_file_serves_the_stored_image(client):
    register_and_login(client, username="fileuser")
    _case_id, evidence_id = _upload_evidence(client)

    resp = client.get(f"/api/evidence/{evidence_id}/file")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"
    # Real bytes, not a JSON envelope.
    assert resp.data.startswith(b"\x89PNG")


def test_evidence_file_denied_for_other_user(client, app):
    register_and_login(client, username="fileowner")
    _case_id, evidence_id = _upload_evidence(client)
    client.post("/api/auth/logout")

    other = app.test_client()
    register_and_login(other, username="fileintruder")
    assert other.get(f"/api/evidence/{evidence_id}/file").status_code in (403, 404)


# --------------------------------------------------------------------------
# XAI artifact images
# --------------------------------------------------------------------------


def test_artifact_rejects_unknown_kind(client):
    register_and_login(client, username="artifactkind")
    _case_id, evidence_id = _upload_evidence(client)
    analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]

    resp = client.get(f"/api/analysis/{analysis_id}/artifact/secrets")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "validation_error"


def test_artifact_returns_404_when_no_explanation_was_generated(client):
    register_and_login(client, username="artifactmissing")
    _case_id, evidence_id = _upload_evidence(client)
    analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]

    resp = client.get(f"/api/analysis/{analysis_id}/artifact/heatmap")
    assert resp.status_code == 404
    assert resp.get_json()["ok"] is False


def test_artifact_serves_real_heatmap_file(client, app):
    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun

    register_and_login(client, username="artifactok")
    _case_id, evidence_id = _upload_evidence(client)
    analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]

    heatmap = Path(app.config["ROOT_DIR"]) / "artifacts" / "investigations" / "heat.png"
    heatmap.parent.mkdir(parents=True, exist_ok=True)
    heatmap.write_bytes(_png((32, 32)))
    run = db.session.get(AnalysisRun, analysis_id)
    run.heatmap_path = str(heatmap)
    db.session.commit()

    resp = client.get(f"/api/analysis/{analysis_id}/artifact/heatmap")
    assert resp.status_code == 200
    assert resp.data.startswith(b"\x89PNG")


def test_artifact_refuses_path_outside_artifacts_root(client, app):
    """A stored path pointing outside artifacts/ must not be served."""

    from backend.app.extensions import db
    from backend.app.models.entities import AnalysisRun

    register_and_login(client, username="artifactescape")
    _case_id, evidence_id = _upload_evidence(client)
    analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]

    secret = Path(app.config["ROOT_DIR"]) / "secret.env"
    secret.write_text("SECRET_KEY=leaked", encoding="utf-8")
    run = db.session.get(AnalysisRun, analysis_id)
    run.heatmap_path = str(secret)
    db.session.commit()

    resp = client.get(f"/api/analysis/{analysis_id}/artifact/heatmap")
    assert resp.status_code == 403
    assert b"leaked" not in resp.data


def test_artifact_denied_for_other_user(client, app):
    register_and_login(client, username="artifactowner")
    _case_id, evidence_id = _upload_evidence(client)
    analysis_id = _analyze(client, evidence_id).get_json()["data"]["analysis_id"]
    client.post("/api/auth/logout")

    other = app.test_client()
    register_and_login(other, username="artifactintruder")
    assert other.get(f"/api/analysis/{analysis_id}/artifact/heatmap").status_code in (403, 404)


# --------------------------------------------------------------------------
# Static hosting of the EVIDEX frontend
# --------------------------------------------------------------------------


def test_evidex_shell_is_served(client, app):
    """Served from the repo root so the session cookie stays first-party."""

    app.config["ROOT_DIR"] = ROOT  # real frontend lives in the repo, not tmp_path
    resp = client.get("/evidex/")
    assert resp.status_code == 200
    assert b"<div id=\"app\"></div>" in resp.data


def test_evidex_only_serves_allowlisted_assets(client, app):
    app.config["ROOT_DIR"] = ROOT
    assert client.get("/evidex/script.js").status_code == 200
    assert client.get("/evidex/api.js").status_code == 200
    # Anything not in the allowlist is refused, traversal included.
    assert client.get("/evidex/../.env").status_code == 404
    assert client.get("/evidex/secrets.txt").status_code == 404
