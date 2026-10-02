"""Automatic email of a newly generated forensic PDF to the signed-in user."""

from __future__ import annotations

import io
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest_product import register_and_login  # noqa: E402
from tests.test_analysis_api import _fake_investigation  # noqa: E402
from tests.test_video_report import (  # noqa: E402
    _analyze_image,
    _install_video_analysis,
    _upload_image,
    _video_payload,
)

MAIL_PASSWORD = "smtp-secret-not-for-logs"


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
    application.config["MAIL_ENABLED"] = True
    application.config["MAIL_HOST"] = "smtp.test.local"
    application.config["MAIL_PORT"] = 587
    application.config["MAIL_USERNAME"] = "mailer"
    application.config["MAIL_PASSWORD"] = MAIL_PASSWORD
    application.config["MAIL_FROM"] = "reports@maya.test"
    application.config["MAIL_USE_TLS"] = True
    application.config["REPORT_EMAIL_LIMIT"] = 5
    application.config["REPORT_EMAIL_WINDOW_SECONDS"] = 60
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


class _CaptureSMTP:
    sent: list[EmailMessage] = []
    fail = False
    disconnect = False
    quit_disconnect = False
    calls: list[object] = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout
        type(self).calls.append(("connect", timeout))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def ehlo(self, name=""):
        type(self).calls.append("ehlo")
        return None

    def starttls(self, *, context=None):
        type(self).calls.append("starttls")
        return None

    def login(self, username, password):
        type(self).calls.append(("login", username))
        return None

    def send_message(self, message):
        type(self).calls.append("send")
        if type(self).disconnect:
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
        if type(self).fail:
            raise smtplib.SMTPException("authentication failed for hidden-user")
        type(self).sent.append(message)

    def quit(self):
        type(self).calls.append("quit")
        if type(self).quit_disconnect:
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")
        return (221, b"bye")

    def close(self):
        type(self).calls.append("close")


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(20, 40, 80)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client) -> int:
    case_id = client.post("/api/cases", json={"title": "Email case"}).get_json()["data"]["case_id"]
    uploaded = client.post(
        f"/api/evidence/cases/{case_id}",
        data={"file": (io.BytesIO(_png()), "still.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201, uploaded.get_json()
    return uploaded.get_json()["data"]["evidence_id"]


def _completed_analysis(client, evidence_id: int) -> int:
    with patch(
        "backend.app.services.analysis_service.run_inference",
        return_value=_fake_investigation(),
    ):
        response = client.post(
            f"/api/evidence/{evidence_id}/analyze",
            json={"generate_explanation": False, "verify_before_analyze": False},
        )
    assert response.status_code == 201, response.get_json()
    return response.get_json()["data"]["analysis_id"]


def _enable_capture():
    _CaptureSMTP.sent = []
    _CaptureSMTP.fail = False
    _CaptureSMTP.disconnect = False
    _CaptureSMTP.quit_disconnect = False
    _CaptureSMTP.calls = []
    return patch("smtplib.SMTP", _CaptureSMTP)


def _attachment(message: EmailMessage) -> tuple[str, str, bytes]:
    parts = list(message.iter_attachments())
    assert len(parts) == 1
    part = parts[0]
    return part.get_filename(), part.get_content_type(), part.get_payload(decode=True)


def _registered_email(app, username: str) -> str:
    from backend.app.models import User

    with app.app_context():
        return User.query.filter_by(username=username).one().email


def _generate(client, analysis_id: int, **extra):
    body = {"investigator_notes": None}
    body.update(extra)
    return client.post(f"/api/analysis/{analysis_id}/report", json=body)


def test_image_report_is_emailed_to_the_registered_address(client, app) -> None:
    register_and_login(client, username="mailowner", email="owner.one@maya.test")
    analysis_id = _completed_analysis(client, _upload(client))
    registered = _registered_email(app, "mailowner")
    with _enable_capture():
        response = _generate(
            client,
            analysis_id,
            email="attacker@evil.test",
            path="C:/secret.pdf",
            pdf_path="/some/arbitrary/file.pdf",
        )
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    delivery = data["email_delivery"]
    assert delivery["status"] == "sent"
    assert delivery["message"] == "Report generated and emailed to your registered email address."
    assert delivery["masked_recipient"] == "o***@maya.test"
    assert registered not in response.get_data(as_text=True)
    assert "attacker@evil.test" not in response.get_data(as_text=True)
    assert MAIL_PASSWORD not in response.get_data(as_text=True)
    assert "smtp.test.local" not in response.get_data(as_text=True)
    message = _CaptureSMTP.sent[0]
    assert message["To"] == registered
    assert message["To"] != "attacker@evil.test"
    assert data["report_number"] in message["Subject"]
    filename, mime, payload = _attachment(message)
    assert filename == f"{data['report_number']}.pdf"
    assert mime == "application/pdf"
    assert payload.startswith(b"%PDF")
    body = message.get_body(preferencelist=("plain",)).get_content()
    assert "still.png" in body
    assert "REAL" in body
    assert "Your MAYA / EVIDEX forensic analysis report has been generated." in body
    from backend.app.extensions import db
    from backend.app.models import InvestigationReport
    from backend.app.services.report_service import resolve_stored_report_file

    with app.app_context():
        row = db.session.get(InvestigationReport, data["report_id"])
        assert payload == resolve_stored_report_file(row).read_bytes()
        assert row.email_status == "sent"
        assert row.emailed_at is not None


def test_video_report_is_emailed_to_the_registered_address(client, app) -> None:
    register_and_login(client, username="mailvideo", email="video.user@maya.test")
    _, evidence_id = _upload_image(client, "Video email")
    analysis_id = _analyze_image(client, evidence_id)
    _install_video_analysis(app, analysis_id, payload=_video_payload(), xai_frames=[90])
    registered = _registered_email(app, "mailvideo")
    with _enable_capture():
        response = _generate(client, analysis_id, email="someoneelse@evil.test")
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["email_delivery"]["status"] == "sent"
    assert data["email_delivery"]["masked_recipient"] == "v***@maya.test"
    message = _CaptureSMTP.sent[0]
    assert message["To"] == registered
    filename, mime, payload = _attachment(message)
    assert filename.endswith(".pdf")
    assert mime == "application/pdf"
    assert payload.startswith(b"%PDF")
    text = message.get_body(preferencelist=("plain",)).get_content()
    assert "clip.mp4" in text
    downloaded = client.get(f"/api/reports/{data['report_id']}/download")
    assert downloaded.status_code == 200
    assert downloaded.get_data() == payload
    assert len(_CaptureSMTP.sent) == 1


def test_refresh_and_download_do_not_send_again(client, app) -> None:
    register_and_login(client, username="mailonce")
    analysis_id = _completed_analysis(client, _upload(client))
    with _enable_capture():
        created = _generate(client, analysis_id)
        assert created.status_code == 201
        report_id = created.get_json()["data"]["report_id"]
        assert client.get(f"/api/analysis/{analysis_id}").status_code == 200
        assert client.get(f"/api/analysis/{analysis_id}/reports").status_code == 200
        assert client.get(f"/api/reports/{report_id}/download").status_code == 200
        from backend.app.extensions import db
        from backend.app.models import InvestigationReport, User
        from backend.app.services.email_service import deliver_generated_report

        with app.app_context():
            user = User.query.filter_by(username="mailonce").one()
            report = db.session.get(InvestigationReport, report_id)
            deliver_generated_report(user, report)
            deliver_generated_report(user, report)
    assert len(_CaptureSMTP.sent) == 1


def test_each_new_report_is_emailed_once(client) -> None:
    register_and_login(client, username="mailtwice")
    analysis_id = _completed_analysis(client, _upload(client))
    with _enable_capture():
        first = _generate(client, analysis_id)
        second = _generate(client, analysis_id)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.get_json()["data"]["report_id"] != second.get_json()["data"]["report_id"]
    assert len(_CaptureSMTP.sent) == 2
    assert _CaptureSMTP.sent[0]["To"] == _CaptureSMTP.sent[1]["To"]


def test_missing_mail_configuration_keeps_the_pdf(client) -> None:
    client.application.config["MAIL_ENABLED"] = False
    register_and_login(client, username="maildisabled")
    analysis_id = _completed_analysis(client, _upload(client))
    with _enable_capture():
        response = _generate(client, analysis_id)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["email_delivery"]["status"] == "unavailable"
    assert data["email_delivery"]["message"] == (
        "Report generated successfully, but email delivery is unavailable."
    )
    assert MAIL_PASSWORD not in response.get_data(as_text=True)
    assert _CaptureSMTP.sent == []
    downloaded = client.get(f"/api/reports/{data['report_id']}/download")
    assert downloaded.status_code == 200
    assert downloaded.get_data().startswith(b"%PDF")


def test_smtp_failure_keeps_the_pdf(client, caplog) -> None:
    register_and_login(client, username="mailfail")
    analysis_id = _completed_analysis(client, _upload(client))
    _CaptureSMTP.sent = []
    _CaptureSMTP.fail = True
    _CaptureSMTP.disconnect = False
    _CaptureSMTP.quit_disconnect = False
    _CaptureSMTP.calls = []
    with patch("smtplib.SMTP", _CaptureSMTP):
        response = _generate(client, analysis_id)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["email_delivery"]["status"] == "failed"
    assert "email delivery is unavailable" in data["email_delivery"]["message"]
    text = response.get_data(as_text=True) + caplog.text
    assert MAIL_PASSWORD not in text
    assert "hidden-user" not in text
    assert "authentication failed" not in data["email_delivery"]["message"]
    downloaded = client.get(f"/api/reports/{data['report_id']}/download")
    assert downloaded.status_code == 200
    assert downloaded.get_data().startswith(b"%PDF")


def test_old_recipient_endpoint_is_gone(client) -> None:
    register_and_login(client, username="mailgone")
    analysis_id = _completed_analysis(client, _upload(client))
    response = client.post(
        f"/api/analysis/{analysis_id}/report/email",
        json={"email": "attacker@evil.test"},
    )
    assert response.status_code == 404


def test_unauthenticated_report_generation_is_rejected(client) -> None:
    response = client.post("/api/analysis/1/report", json={})
    assert response.status_code == 401
    assert response.get_json()["error"] == "authentication_error"


def test_other_investigator_cannot_generate_or_email_the_report(client) -> None:
    register_and_login(client, username="mailowner2", email="owner.two@maya.test")
    analysis_id = _completed_analysis(client, _upload(client))
    client.post("/api/auth/logout")
    register_and_login(client, username="mailother", email="other.user@maya.test")
    with _enable_capture():
        response = _generate(client, analysis_id, email="other.user@maya.test")
    assert response.status_code == 403
    assert response.get_json()["error"] == "authorization_error"
    assert _CaptureSMTP.sent == []


def test_smtp_session_sends_then_quits(client) -> None:
    register_and_login(client, username="mailorder", email="order.user@maya.test")
    analysis_id = _completed_analysis(client, _upload(client))
    with _enable_capture():
        response = _generate(client, analysis_id)
    assert response.status_code == 201, response.get_json()
    assert response.get_json()["data"]["email_delivery"]["status"] == "sent"
    names = [item[0] if isinstance(item, tuple) else item for item in _CaptureSMTP.calls]
    assert names == ["connect", "ehlo", "starttls", "ehlo", "login", "send", "quit"]
    timeout = _CaptureSMTP.calls[0][1]
    assert isinstance(timeout, int) and timeout >= 30
    assert _CaptureSMTP.calls[-1] == "quit"


def test_disconnect_during_send_keeps_the_pdf(client) -> None:
    register_and_login(client, username="maildrop", email="drop.user@maya.test")
    analysis_id = _completed_analysis(client, _upload(client))
    with _enable_capture():
        _CaptureSMTP.disconnect = True
        response = _generate(client, analysis_id)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["email_delivery"]["status"] == "failed"
    assert data["email_delivery"]["status"] != "sent"
    assert "email delivery is unavailable" in data["email_delivery"]["message"]
    assert "SMTP" not in data["email_delivery"]["message"]
    downloaded = client.get(f"/api/reports/{data['report_id']}/download")
    assert downloaded.status_code == 200
    assert downloaded.get_data().startswith(b"%PDF")
    assert _CaptureSMTP.sent == []


def test_quit_disconnect_after_accept_still_marks_sent(client, app) -> None:
    register_and_login(client, username="mailquit", email="quit.user@maya.test")
    analysis_id = _completed_analysis(client, _upload(client))
    with _enable_capture():
        _CaptureSMTP.quit_disconnect = True
        response = _generate(client, analysis_id)
    assert response.status_code == 201, response.get_json()
    data = response.get_json()["data"]
    assert data["email_delivery"]["status"] == "sent"
    assert _CaptureSMTP.sent
    from backend.app.extensions import db
    from backend.app.models import InvestigationReport

    with app.app_context():
        row = db.session.get(InvestigationReport, data["report_id"])
        assert row.email_status == "sent"
        assert row.emailed_at is not None


def test_large_pdf_gets_a_longer_smtp_timeout() -> None:
    from backend.app.services.email_service import _smtp_timeout

    small = EmailMessage()
    small.set_content("hello")
    small.add_attachment(b"%PDF-1.4\n", maintype="application", subtype="pdf", filename="a.pdf")
    large = EmailMessage()
    large.set_content("hello")
    large.add_attachment(
        b"%PDF" + (b"0" * (3 * 1024 * 1024)),
        maintype="application",
        subtype="pdf",
        filename="big.pdf",
    )
    assert _smtp_timeout(small) >= 30
    assert _smtp_timeout(large) > _smtp_timeout(small)
