"""Send an existing forensic PDF by email.

SMTP settings stay in server configuration. This module never accepts a
filesystem path from the caller and never logs the mail password.
"""

from __future__ import annotations

import logging
import re
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from flask import current_app

from backend.app.audit import record_audit
from backend.app.exceptions import (
    AuthorizationError,
    MailDeliveryError,
    RateLimitError,
    ReportFileMissingError,
    ValidationError,
)
from backend.app.extensions import db
from backend.app.models.entities import InvestigationReport, User
from backend.app.models.enums import AuditEventType
from backend.app.security.rate_limit import enforce_report_email_rate_limit
from backend.app.services.report_service import resolve_stored_report_file

logger = logging.getLogger("maya.backend.email")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UNAVAILABLE = "Not recorded"
_SEND_FAILURE = (
    "Unable to send the report right now. Please check the email settings or try again later."
)


def mask_email(email: str) -> str:
    """Show only enough of an address to recognise it, such as a***@example.com."""

    local, separator, domain = str(email or "").partition("@")
    if not separator or not local or not domain:
        return "your registered email"
    return f"{local[:1]}***@{domain}"


def normalize_recipient(value: object) -> str:
    """Accept a single mailbox address. Reject blanks, newlines, and obvious junk."""

    email = str(value or "").strip()
    if any(char in email for char in "\r\n\x00"):
        raise ValidationError("A valid recipient email is required.")
    if not email or len(email) > 254 or not _EMAIL_RE.match(email):
        raise ValidationError("A valid recipient email is required.")
    return email


def _mail_settings() -> dict[str, object]:
    port = current_app.config.get("MAIL_PORT", 587)
    try:
        port = int(port)
    except (TypeError, ValueError):
        port = 587
    if port < 1 or port > 65535:
        port = 587
    return {
        "enabled": bool(current_app.config.get("MAIL_ENABLED")),
        "host": str(current_app.config.get("MAIL_HOST") or "").strip(),
        "port": port,
        "username": str(current_app.config.get("MAIL_USERNAME") or ""),
        "password": str(current_app.config.get("MAIL_PASSWORD") or ""),
        "sender": str(current_app.config.get("MAIL_FROM") or "").strip(),
        "use_tls": bool(current_app.config.get("MAIL_USE_TLS", True)),
    }


def _mail_is_configured(settings: dict[str, object]) -> bool:
    return bool(settings["enabled"] and settings["host"] and settings["sender"])


def _analysis_result_label(prediction: str | None) -> str:
    text = str(prediction or "").strip()
    if not text:
        return _UNAVAILABLE
    if text.upper() in {"REAL", "FAKE"}:
        return text.upper()
    return text


def _attachment_name(report: InvestigationReport) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in str(report.report_number or "report")
    )
    return f"{cleaned or 'report'}.pdf"


def _read_pdf(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReportFileMissingError("The report file could not be read.") from exc
    if not payload.startswith(b"%PDF"):
        raise ReportFileMissingError("The report file could not be read.")
    return payload


def _build_message(
    *,
    sender: str,
    recipient: str,
    report: InvestigationReport,
    pdf_bytes: bytes,
) -> EmailMessage:
    evidence_name = _UNAVAILABLE
    result = _UNAVAILABLE
    if report.evidence is not None and report.evidence.original_filename:
        evidence_name = " ".join(str(report.evidence.original_filename).split()) or _UNAVAILABLE
    if report.analysis is not None:
        result = " ".join(_analysis_result_label(report.analysis.prediction).split()) or _UNAVAILABLE
    subject_number = " ".join(str(report.report_number or "").split()) or "report"

    message = EmailMessage()
    message["Subject"] = f"MAYA / EVIDEX Forensic Analysis Report - {subject_number}"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        "Hello,\n\n"
        "Your MAYA / EVIDEX forensic analysis report has been generated.\n\n"
        f"Report: {subject_number}\n"
        f"Evidence: {evidence_name}\n"
        f"Analysis Result: {result}\n\n"
        "Please find the forensic analysis report attached.\n\n"
        "Regards,\n"
        "MAYA / EVIDEX\n"
    )
    message.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=_attachment_name(report),
    )
    return message


def _smtp_timeout(message: EmailMessage) -> int:
    """Seconds to keep the SMTP socket open.

    A short text message finishes well inside 30 seconds. A forensic PDF is
    several megabytes after base64 encoding, and a 20 second socket timeout
    expires while that payload is still in flight. smtplib then reports the
    timeout as SMTPServerDisconnected.
    """

    size = 0
    for part in message.walk():
        payload = part.get_payload(decode=True)
        if payload:
            size += len(payload)
    # One second per 50 KB of attachment, with the same 30 second floor as a
    # known-good session, and a cap so a huge file cannot hold the request open.
    return min(180, max(30, 30 + (size // (50 * 1024))))


def _close_smtp(smtp: smtplib.SMTP, *, accepted: bool) -> None:
    """End the session only after send_message has returned.

    A server that drops the connection on QUIT must not turn an accepted
    message into a delivery failure.
    """

    try:
        smtp.quit()
    except (OSError, smtplib.SMTPException) as exc:
        if accepted:
            logger.info("Report email accepted; SMTP close failed error=%s", type(exc).__name__)
        try:
            smtp.close()
        except OSError:
            pass


def _deliver(message: EmailMessage, settings: dict[str, object]) -> None:
    host = str(settings["host"])
    port = int(settings["port"])
    smtp: smtplib.SMTP | None = None
    accepted = False
    try:
        smtp = smtplib.SMTP(host, port, timeout=_smtp_timeout(message))
        smtp.ehlo()
        if settings["use_tls"]:
            smtp.starttls()
            smtp.ehlo()
        username = str(settings["username"] or "")
        if username:
            smtp.login(username, str(settings["password"] or ""))
        smtp.send_message(message)
        accepted = True
    except (OSError, smtplib.SMTPException) as exc:
        # Log the exception type only. SMTP messages can echo the username.
        logger.warning("Report email delivery failed error=%s", type(exc).__name__)
        raise MailDeliveryError(_SEND_FAILURE) from exc
    finally:
        if smtp is not None:
            _close_smtp(smtp, accepted=accepted)


def _mark(report: InvestigationReport, status: str) -> None:
    report.email_status = status
    db.session.commit()


def deliver_generated_report(user: User, report: InvestigationReport) -> None:
    """Email this report once to the account that generated it.

    The address is read from the user row. Callers cannot supply a different
    recipient. A report that was already accepted by the mail server is left
    alone, so a later download or page load cannot send it again. Failure
    updates the report status and does not undo the PDF.
    """

    if report.email_status == "sent":
        return

    try:
        address = normalize_recipient(user.email)
    except ValidationError:
        logger.info("Report email skipped; registered address is unusable report=%s", report.report_number)
        _mark(report, "unavailable")
        return

    try:
        pdf_bytes = _read_pdf(resolve_stored_report_file(report))
    except (AuthorizationError, ReportFileMissingError, OSError):
        logger.info("Report email skipped; PDF is not readable report=%s", report.report_number)
        _mark(report, "unavailable")
        return

    settings = _mail_settings()
    if not _mail_is_configured(settings):
        logger.info("Report email skipped; mail is not configured report=%s", report.report_number)
        _mark(report, "unavailable")
        return

    try:
        # One attempt per new report. This bucket is separate from login and analysis.
        enforce_report_email_rate_limit()
    except RateLimitError:
        logger.info("Report email skipped; send limit reached report=%s", report.report_number)
        _mark(report, "unavailable")
        return

    message = _build_message(
        sender=str(settings["sender"]),
        recipient=address,
        report=report,
        pdf_bytes=pdf_bytes,
    )
    try:
        _deliver(message, settings)
    except MailDeliveryError:
        _mark(report, "failed")
        return

    report.email_status = "sent"
    report.emailed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    record_audit(
        AuditEventType.REPORT_EMAILED,
        user_id=user.id,
        case_id=report.case_id,
        evidence_id=report.evidence_id,
        analysis_id=report.analysis_id,
        details={
            "report_number": report.report_number,
            "recipient_masked": mask_email(address),
        },
    )
    db.session.commit()
    logger.info("Report email accepted report=%s analysis=%s", report.report_number, report.analysis_id)
