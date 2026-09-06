"""Investigation PDF report generator (reportlab).

Presentation improvements (report V2) adjust title, authenticity interpretation,
integrity summary layout, XAI captions/disclaimers, audit-table wrapping, and
legal/readiness wording. AI inference, XAI, face verification, and API behaviour
are unchanged — only the generated PDF content and layout are modified.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ai.datasets.utils.checksums import hash_file
from backend.app.audit import record_audit
from backend.app.exceptions import ValidationError
from backend.app.extensions import db
from backend.app.models.entities import (
    AnalysisRun,
    AuditLog,
    Case,
    Evidence,
    FaceVerification,
    InvestigationReport,
    User,
)
from backend.app.models.enums import AuditEventType, IntegrityStatus
from backend.app.services.analysis_service import get_analysis
from backend.app.services.evidence_service import get_evidence

logger = logging.getLogger("maya.backend.reports")

SUPPORTED_REPORT_FORMATS = {"pdf"}

# Report-only branding. Does not rename the application, routes, or APIs.
REPORT_TITLE = (
    "AI-Based Digital Evidence Authenticity Verification System and Legal Admissibility"
)
REPORT_SUBTITLE = "Investigation Report"
SYSTEM_LABEL = "AI-Based Digital Evidence Authenticity Verification System"
# Shown wherever a value exists in the schema but was never produced by a run.
UNAVAILABLE = "Not available — not produced by this system"

# A4 usable content width with 1.8 cm side margins.
_PAGE_WIDTH, _PAGE_HEIGHT = A4
_SIDE_MARGIN = 1.8 * cm
_CONTENT_WIDTH = _PAGE_WIDTH - (2 * _SIDE_MARGIN)

_DISCLAIMER = (
    "This report is generated for investigative and academic use. "
    "AI authenticity predictions are probabilistic: they reflect the model's "
    "learned distribution over its training corpus and can fail on novel "
    "manipulation methods, out-of-distribution media, or low-quality inputs. "
    "Integrity verification (SHA-256) confirms that a stored file matches the "
    "hash recorded at ingest; it does not by itself establish authenticity, "
    "authorship, or legal admissibility. Grad-CAM and other explainability "
    "visualizations highlight regions that influenced the classifier and are "
    "explanatory — not independent proof of manipulation or authenticity. "
    "This system does not determine legal admissibility. Admissibility is decided "
    "under the applicable legal framework by the appropriate authority, court, "
    "or qualified legal/forensic professional. Independent verification may be "
    "required before any operational or legal decision."
)

_XAI_CAPTION = (
    "Grad-CAM highlights image regions that contributed more strongly to the "
    "model's classification. The visualization is explanatory and should not be "
    "interpreted as independent proof of manipulation or authenticity."
)

_READINESS_CLOSING = (
    "This system provides technical and investigative evidence-readiness information. "
    "These records do not by themselves establish authenticity, authorship, "
    "tampering, or legal admissibility. Legal admissibility is determined under "
    "the applicable legal framework by the appropriate authority or court."
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _allocate_report_number() -> str:
    year = _utcnow().year
    prefix = f"RPT-{year}-"
    latest = (
        InvestigationReport.query.filter(
            InvestigationReport.report_number.like(f"{prefix}%")
        )
        .order_by(InvestigationReport.report_number.desc())
        .first()
    )
    seq = 1
    if latest and latest.report_number.startswith(prefix):
        try:
            seq = int(latest.report_number.split("-")[-1]) + 1
        except ValueError:
            seq = 1
    return f"{prefix}{seq:06d}"


def _extract_raw_payload(run: AnalysisRun) -> dict[str, Any]:
    if not run.raw_result_json:
        return {}
    try:
        parsed = json.loads(run.raw_result_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _extract_advanced_xai(run: AnalysisRun) -> dict[str, Any]:
    return _extract_raw_payload(run).get("advanced_xai_results") or {}


def _extract_investigation(run: AnalysisRun) -> dict[str, Any]:
    inv = _extract_raw_payload(run).get("investigation") or {}
    return inv if isinstance(inv, dict) else {}


def _safe_image(path_str: str | None, max_w: float = 15 * cm, max_h: float = 10 * cm) -> Image | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        img = Image(str(p))
        w, h = img.drawWidth, img.drawHeight
        if w <= 0 or h <= 0:
            return None
        scale = min(max_w / w, max_h / h, 1.0)
        out = Image(str(p), width=w * scale, height=h * scale)
        out.hAlign = "CENTER"
        return out
    except Exception:
        return None


def _styles():
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    styles.add(
        ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#183B8A"),
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#1F2A44"),
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            "MetaCenter",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#4A5568"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodyJustified",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor("#1F2A44"),
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor("#4A4A4A"),
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=8.2,
            leading=10.5,
            textColor=colors.HexColor("#4A4A4A"),
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            "Cell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#1F2A44"),
        )
    )
    styles.add(
        ParagraphStyle(
            "CellHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            "KeyCell",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#1F2A44"),
        )
    )
    styles.add(
        ParagraphStyle(
            "ValueCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            textColor=colors.HexColor("#1F2A44"),
        )
    )
    return styles


def _is_rich_markup(value: str) -> bool:
    """True when the caller intentionally supplied a small HTML fragment."""

    return any(tag in value for tag in ("<b>", "<font", "<br/>", "<i>", "<br />"))


def _kv_table(rows: list[tuple[str, str]], styles) -> Table:
    """Key/value table with wrapped Paragraph cells (prevents text overflow)."""

    data = []
    for k, v in rows:
        key_p = Paragraph(escape(str(k)), styles["KeyCell"])
        vs = str(v)
        val_p = Paragraph(vs if _is_rich_markup(vs) else escape(vs), styles["ValueCell"])
        data.append([key_p, val_p])
    # Content width ≈ 17.4 cm with current margins.
    t = Table(data, hAlign="LEFT", colWidths=[5.0 * cm, _CONTENT_WIDTH - 5.0 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2FF")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1F2A44")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E5")),
            ]
        )
    )
    return t


def _section_title(text: str, styles) -> Paragraph:
    return Paragraph(
        escape(text),
        ParagraphStyle(
            "sec",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#183B8A"),
            spaceBefore=12,
            spaceAfter=6,
        ),
    )


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _fmt_pct(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return UNAVAILABLE
    return f"{float(value):.{digits}f}%"


def _fmt_prob(value: Any) -> str | None:
    """Return a display string for a stored class probability, or None if unusable."""

    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= p <= 1.0):
        return None
    return f"{p:.4f} ({p * 100.0:.2f}%)"


def _level_from_confidence(confidence_pct: float) -> str:
    """Mirror the inference confidence bands (fraction thresholds)."""

    frac = float(confidence_pct) / 100.0
    if frac >= 0.95:
        return "Very High"
    if frac >= 0.85:
        return "High"
    if frac >= 0.70:
        return "Medium"
    return "Low"


def _confidence_interpretation(confidence: float | None, stored_level: str | None) -> str:
    if confidence is None:
        return UNAVAILABLE
    conf = float(confidence)
    base = (stored_level or "").strip() or _level_from_confidence(conf)
    # Near the binary decision boundary (threshold 0.5 → ~50% confidence).
    if conf < 60.0:
        return f"{base.upper()} / NEAR DECISION BOUNDARY"
    return base.upper()


def _assessment_narrative(prediction: str, confidence: float | None) -> str:
    pred = prediction or "N/A"
    if confidence is None:
        return (
            f"Assessment: The model classified the submitted evidence as {pred}. "
            "Confidence was not recorded for this analysis. The result should not "
            "be treated as definitive proof of authenticity or manipulation."
        )
    conf = float(confidence)
    conf_disp = f"{conf:.2f}"
    if conf < 60.0:
        return (
            f"Assessment: The model classified the submitted evidence as {pred} with a "
            f"{conf_disp}% confidence score. This result is near the binary decision "
            f"boundary and therefore indicates substantial model uncertainty. "
            f"The model predicted {pred}, but the prediction is close to the binary "
            f"decision boundary and should be treated as uncertain rather than as "
            f"definitive proof of authenticity."
        )
    if conf < 70.0:
        return (
            f"Assessment: The model classified the submitted evidence as {pred} with a "
            f"{conf_disp}% confidence score. Confidence is moderate-to-low; the result "
            f"should be interpreted cautiously and not as definitive proof."
        )
    return (
        f"Assessment: The model classified the submitted evidence as {pred} with a "
        f"{conf_disp}% confidence score. Higher confidence indicates stronger agreement "
        f"with the model's learned patterns, but the result remains probabilistic and "
        f"should not be treated as definitive proof of authenticity or manipulation."
    )


def _wrap_monospace(value: str, chunk: int = 32) -> str:
    """Insert soft line breaks so long hashes wrap inside table cells."""

    text = str(value or "")
    if len(text) <= chunk:
        return escape(text)
    parts = [escape(text[i : i + chunk]) for i in range(0, len(text), chunk)]
    return "<br/>".join(parts)


def _uploader_label(evidence: Evidence) -> str:
    user = db.session.get(User, evidence.uploaded_by_user_id)
    if user is None:
        return f"user id {evidence.uploaded_by_user_id}"
    name = user.full_name or user.username
    return f"{name} (id={user.id}, username={user.username})"


def _read_integrity_status(evidence: Evidence) -> str:
    """Read-only SHA-256 comparison for the PDF summary (no audit side-effects)."""

    try:
        from backend.app.services.evidence_service import absolute_evidence_path

        path = absolute_evidence_path(evidence)
        if not path.exists() or not path.is_file():
            return IntegrityStatus.MISSING.value
        current = hash_file(path, algorithm="sha256")
        if current == evidence.sha256_hash:
            return IntegrityStatus.VALID.value
        return IntegrityStatus.MODIFIED.value
    except Exception:
        logger.exception("Read-only integrity check failed for evidence=%s", evidence.id)
        return IntegrityStatus.ERROR.value


def _format_audit_details(details_json: str | None) -> str:
    if not details_json:
        return "—"
    try:
        data = json.loads(details_json)
    except Exception:
        return escape(str(details_json))
    if isinstance(data, dict):
        if not data:
            return "—"
        lines = [f"<b>{escape(str(k))}</b>: {escape(str(v))}" for k, v in data.items()]
        return "<br/>".join(lines)
    return escape(str(data))


def _captioned_figure(img: Image, caption: str, styles) -> KeepTogether:
    return KeepTogether(
        [
            Paragraph(escape(caption), styles["Caption"]),
            img,
            Spacer(1, 3 * mm),
        ]
    )


def _append_face_verification_section(
    story: list[Any],
    styles,
    rows: list[FaceVerification],
) -> None:
    if not rows:
        return
    story.append(_section_title("Face Reference Verification", styles))
    latest = rows[0]
    fv_rows = [
        ("Reference face verification performed", "Yes"),
        ("Verification ID", str(latest.id)),
        ("Status", str(latest.verification_status or "")),
        ("Decision", str(latest.decision or "N/A")),
        ("Similarity score", _fmt_score(latest.similarity_score)),
        ("Distance score", _fmt_score(latest.distance_score)),
        ("Match threshold", _fmt_score(latest.threshold)),
        ("No-match threshold", _fmt_score(latest.no_match_threshold)),
        ("Model", str(latest.model_name or "")),
        ("Model version", str(latest.model_version or "")),
        ("Engine", str(latest.engine_name or "")),
        ("Reason code", str(latest.reason_code or "")),
        ("Artifact directory", str(latest.artifact_dir or "")),
        ("Result JSON", str(latest.result_json_path or "")),
    ]
    story.append(_kv_table(fv_rows, styles))

    from flask import current_app

    root = Path(current_app.config["ROOT_DIR"])
    if latest.artifact_dir:
        ref_img = _safe_image(
            str(root / latest.artifact_dir / "reference_face.png"),
            max_w=7 * cm,
            max_h=7 * cm,
        )
        evd_img = _safe_image(
            str(root / latest.artifact_dir / "evidence_face.png"),
            max_w=7 * cm,
            max_h=7 * cm,
        )
        if ref_img:
            story.append(_captioned_figure(ref_img, "Reference face crop", styles))
        if evd_img:
            story.append(_captioned_figure(evd_img, "Evidence face crop", styles))


def _build_audit_table(audit_events: list[AuditLog], styles) -> LongTable:
    """Multi-page audit table with wrapped Paragraph cells (no overflow)."""

    header = [
        Paragraph("Time (UTC)", styles["CellHeader"]),
        Paragraph("Event", styles["CellHeader"]),
        Paragraph("Case", styles["CellHeader"]),
        Paragraph("Evidence", styles["CellHeader"]),
        Paragraph("Details", styles["CellHeader"]),
    ]
    data: list[list[Any]] = [header]

    for ev in audit_events:
        stamp = ev.timestamp.isoformat(timespec="seconds") if ev.timestamp else "—"
        details = _format_audit_details(ev.details_json)
        # Append analysis id inside Details when present so the Event/Case/Evidence
        # columns stay narrow enough for the printable width.
        if ev.analysis_id is not None:
            details = f"<b>analysis_id</b>: {escape(str(ev.analysis_id))}<br/>{details}"
        data.append(
            [
                Paragraph(escape(stamp), styles["Cell"]),
                Paragraph(escape(str(ev.event_type or "—")), styles["Cell"]),
                Paragraph(escape(str(ev.case_id) if ev.case_id is not None else "—"), styles["Cell"]),
                Paragraph(
                    escape(str(ev.evidence_id) if ev.evidence_id is not None else "—"),
                    styles["Cell"],
                ),
                Paragraph(details, styles["Cell"]),
            ]
        )

    # Column widths must sum to printable content width.
    col_widths = [3.4 * cm, 3.6 * cm, 1.6 * cm, 2.0 * cm, _CONTENT_WIDTH - 10.6 * cm]
    table = LongTable(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E5")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#183B8A")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _build_pdf(
    output_path: Path,
    *,
    report_number: str,
    case: Case,
    evidence: Evidence,
    run: AnalysisRun,
    audit_events: list[AuditLog],
    generator: User,
    notes: str | None = None,
    face_verifications: list[FaceVerification] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=_SIDE_MARGIN,
        rightMargin=_SIDE_MARGIN,
        title=f"{REPORT_TITLE} — {run.investigation_id}",
        author=SYSTEM_LABEL,
        subject=f"Case {case.case_number}",
    )
    styles = _styles()
    story: list[Any] = []
    inv = _extract_investigation(run)

    # ------------------------------------------------------------------ header
    # The document title is printed once, below this bar; the bar only carries
    # the report identity so no branding line is duplicated.
    header_tbl = Table(
        [[
            Paragraph(
                f"<b><font size='11' color='#183B8A'>{escape(REPORT_SUBTITLE.upper())}</font></b><br/>"
                f"<font size='8'>{escape(str(report_number))}</font>",
                ParagraphStyle("brand", alignment=TA_LEFT, leading=14),
            ),
            Paragraph(
                f"<b>Case {escape(str(case.case_number))}</b><br/>"
                f"<font size='8'>Generated {_utcnow().isoformat(timespec='seconds')} UTC</font>",
                ParagraphStyle(
                    "hd",
                    alignment=TA_CENTER,
                    textColor=colors.HexColor("#1F2A44"),
                    leading=12,
                ),
            ),
        ]],
        colWidths=[8.5 * cm, _CONTENT_WIDTH - 8.5 * cm],
        hAlign="LEFT",
    )
    header_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7FF")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#183B8A")),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(header_tbl)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(escape(REPORT_TITLE), styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Technical investigation report — evidence-readiness information only. "
            "This system does not determine legal admissibility.",
            styles["MetaCenter"],
        )
    )
    story.append(Spacer(1, 4 * mm))

    # ----------------------------------------------------- investigation IDs
    story.append(_section_title("Investigation Identification", styles))
    ids = [
        ("Investigation ID", str(run.investigation_id or "")),
        ("Report Number", report_number),
        ("Case Number", str(case.case_number)),
        ("Case Title", str(case.title)),
        ("Case Status", str(case.status or "")),
        ("Evidence ID", f"EVD-{evidence.id}"),
        ("Analyst / Investigator", generator.full_name or generator.username),
        ("Generated At (UTC)", _utcnow().isoformat(timespec="seconds")),
    ]
    story.append(_kv_table(ids, styles))

    # ------------------------------------------------ authenticity assessment
    story.append(_section_title("Authenticity Assessment", styles))
    confidence_val = float(run.confidence) if run.confidence is not None else None
    # Prefer exact stored confidence; fall back to investigation payload only if
    # the AnalysisRun column is empty (never invent or rescale).
    if confidence_val is None and inv.get("confidence") is not None:
        try:
            confidence_val = float(inv["confidence"])
        except (TypeError, ValueError):
            confidence_val = None

    pred = str(run.prediction or inv.get("prediction") or "N/A")
    pred_color = colors.HexColor("#106F3A") if pred == "REAL" else colors.HexColor("#8A1430")
    stored_level = None
    if isinstance(inv.get("confidence_level"), str):
        stored_level = inv["confidence_level"]
    interpretation = _confidence_interpretation(confidence_val, stored_level)

    pred_rows: list[tuple[str, str]] = [
        (
            "Prediction",
            f"<b><font color='#{pred_color.hexval()[2:]}'>{escape(pred)}</font></b>",
        ),
        ("Model Confidence", _fmt_pct(confidence_val)),
        ("Confidence Interpretation", interpretation),
        ("Model", str(run.model_name or inv.get("model_name") or "")),
        ("Model Version", str(run.model_version or inv.get("model_version") or "")),
        ("Dataset Version", str(run.dataset_version or inv.get("dataset_version") or "")),
        ("Analysis Status", str(run.status or "")),
    ]

    # Class probabilities — only when the stored investigation payload provides them.
    real_disp = _fmt_prob(inv.get("real_probability"))
    fake_disp = _fmt_prob(inv.get("fake_probability"))
    if real_disp is not None:
        pred_rows.append(("REAL Class Probability", real_disp))
    if fake_disp is not None:
        pred_rows.append(("FAKE Class Probability", fake_disp))
    if inv.get("threshold") is not None:
        try:
            pred_rows.append(("Decision Threshold", f"{float(inv['threshold']):.4f}"))
        except (TypeError, ValueError):
            pass

    if run.trust_score is not None:
        pred_rows.append(("XAI Trust Score", f"{float(run.trust_score):.2f}"))
    if run.quality_score is not None:
        pred_rows.append(("XAI Quality Score", f"{float(run.quality_score):.2f}"))
    if run.started_at:
        pred_rows.append(("Started At", run.started_at.isoformat(timespec="seconds")))
    if run.completed_at:
        pred_rows.append(("Completed At", run.completed_at.isoformat(timespec="seconds")))

    story.append(_kv_table(pred_rows, styles))
    story.append(Paragraph(escape(_assessment_narrative(pred, confidence_val)), styles["BodyJustified"]))
    story.append(
        Paragraph(
            "Note: Model confidence is the probability assigned to the predicted class. "
            "It must not be read as a percentage of authenticity or as a legal certainty.",
            styles["Caption"],
        )
    )

    # ----------------------------------------------------- evidence integrity
    # Keep title + table together so the section header is never orphaned at a
    # page bottom with the body starting on the next page.
    integrity_status = _read_integrity_status(evidence)
    integrity_color = (
        "#106F3A" if integrity_status == IntegrityStatus.VALID.value else "#8A1430"
    )
    ev_rows = [
        (
            "Integrity Status",
            f"<b><font color='{integrity_color}'>{escape(integrity_status)}</font></b>",
        ),
        ("Evidence ID", f"EVD-{evidence.id}"),
        ("Original Filename", str(evidence.original_filename)),
        ("Stored Identifier", _wrap_monospace(str(evidence.stored_filename), 40)),
        ("SHA-256", _wrap_monospace(str(evidence.sha256_hash), 32)),
        ("MIME Type", str(evidence.mime_type)),
        ("File Size", f"{evidence.file_size_bytes:,} bytes"),
        ("Uploaded By", _uploader_label(evidence)),
        (
            "Uploaded At",
            evidence.uploaded_at.isoformat(timespec="seconds") if evidence.uploaded_at else "",
        ),
        ("Evidence Status", str(evidence.status)),
        ("Analysis Status (evidence)", str(evidence.analysis_status or "")),
    ]
    story.append(
        KeepTogether(
            [
                _section_title("Evidence Integrity", styles),
                _kv_table(ev_rows, styles),
                Paragraph(
                    "Integrity Status reflects a read-only comparison of the stored file's "
                    "SHA-256 digest against the hash recorded at ingest. A VALID result means "
                    "the bytes are unchanged since upload; it does not establish authenticity.",
                    styles["Caption"],
                ),
            ]
        )
    )

    # -------------------------------------------------------------- page 2 XAI
    story.append(PageBreak())
    story.append(_section_title("Explainability (XAI) — Grad-CAM", styles))
    adv = _extract_advanced_xai(run)
    xai_rows: list[tuple[str, str]] = [
        ("Primary Explainer", str(run.explainer_name or "Grad-CAM") if (run.explainer_name or run.overlay_path or run.heatmap_path) else UNAVAILABLE),
    ]
    methods_run = adv.get("methods_run") or []
    if methods_run:
        xai_rows.append(("Methods Executed", ", ".join(str(m) for m in methods_run)))
    trust = adv.get("trust")
    if isinstance(trust, dict) and trust.get("grade"):
        xai_rows.append(("Trust Grade", str(trust.get("grade"))))
    if not run.overlay_path and not run.heatmap_path and not run.explainer_name:
        xai_rows = [("Status", UNAVAILABLE)]
    story.append(_kv_table(xai_rows, styles))
    story.append(Paragraph(escape(_XAI_CAPTION), styles["BodyJustified"]))

    primary_overlay = _safe_image(run.overlay_path, max_w=14 * cm, max_h=9 * cm)
    primary_heatmap = _safe_image(run.heatmap_path, max_w=12 * cm, max_h=8 * cm)
    if primary_overlay:
        story.append(_section_title("Explainer Overlay", styles))
        story.append(
            _captioned_figure(
                primary_overlay,
                "Figure: Grad-CAM explainer overlay on the submitted evidence image.",
                styles,
            )
        )
    if primary_heatmap:
        story.append(_section_title("Raw Attention Heatmap", styles))
        story.append(
            _captioned_figure(
                primary_heatmap,
                "Figure: Raw Grad-CAM attention heatmap produced for this analysis.",
                styles,
            )
        )
    if not primary_overlay and not primary_heatmap:
        story.append(
            Paragraph(
                "No Grad-CAM visualization artifacts were available for this analysis run.",
                styles["BodyJustified"],
            )
        )

    _append_face_verification_section(story, styles, face_verifications or [])

    if adv:
        story.append(_section_title("Advanced XAI — Artifact References", styles))
        ref_rows: list[tuple[str, str]] = []
        artifact_paths = adv.get("artifact_paths") or {}
        if isinstance(artifact_paths, dict):
            for k, v in artifact_paths.items():
                ref_rows.append((str(k).replace("_", " ").title(), str(v)))
        for key, label in (
            ("shap", "SHAP"),
            ("faithfulness", "Faithfulness"),
            ("counterfactual", "Counterfactual"),
            ("fusion", "Explanation Fusion"),
        ):
            block = adv.get(key)
            if isinstance(block, dict) and block.get("enabled"):
                summary = ", ".join(f"{kk}={vv}" for kk, vv in block.items() if kk != "enabled")
                ref_rows.append((label, summary[:800]))
        trust_dict = adv.get("trust")
        if isinstance(trust_dict, dict) and trust_dict.get("enabled"):
            comps = trust_dict.get("components")
            comp_str = json.dumps(comps) if isinstance(comps, (dict, list)) else str(comps)
            ref_rows.append(("Trust Components", comp_str[:600]))
        if ref_rows:
            story.append(_kv_table(ref_rows, styles))
        else:
            story.append(
                Paragraph(
                    "Advanced XAI stages were not recorded for this analysis.",
                    styles["BodyJustified"],
                )
            )

    # --------------------------------------- page 3 readiness + audit + close
    story.append(PageBreak())
    story.append(_section_title("Evidence Readiness and Legal Considerations", styles))
    readiness_rows = [
        ("Evidence identified", f"Yes — EVD-{evidence.id}"),
        ("SHA-256 recorded at ingest", "Yes" if evidence.sha256_hash else "No"),
        ("Integrity status (read-only check)", integrity_status),
        (
            "Upload timestamp",
            evidence.uploaded_at.isoformat(timespec="seconds") if evidence.uploaded_at else UNAVAILABLE,
        ),
        ("Audit / custody history", f"{len(audit_events)} recorded event(s)"),
        ("AI analysis record", f"{run.status} — prediction {pred}"),
        (
            "Model / version traceability",
            f"{run.model_name or '—'} / {run.model_version or '—'}",
        ),
        (
            "XAI explanation available",
            "Yes" if (run.overlay_path or run.heatmap_path) else "No",
        ),
        (
            "Face reference verification",
            "Yes" if face_verifications else "Not performed for this investigation",
        ),
    ]
    story.append(_kv_table(readiness_rows, styles))
    story.append(Paragraph(escape(_READINESS_CLOSING), styles["BodyJustified"]))

    story.append(_section_title("Investigation Audit Timeline", styles))
    story.append(
        Paragraph(
            "The following timeline lists stored audit events related to this "
            "case or the generating investigator. Timestamps and event types are "
            "taken from the AuditLog without alteration.",
            styles["Caption"],
        )
    )
    if audit_events:
        story.append(_build_audit_table(audit_events, styles))
    else:
        story.append(
            Paragraph(
                "No audit events were available for this investigation.",
                styles["BodyJustified"],
            )
        )

    story.append(_section_title("Final Interpretation", styles))
    story.append(Paragraph(escape(_assessment_narrative(pred, confidence_val)), styles["BodyJustified"]))

    if notes:
        story.append(_section_title("Investigator Notes", styles))
        story.append(Paragraph(escape(str(notes)), styles["BodyJustified"]))

    story.append(_section_title("Limitations & Disclaimer", styles))
    story.append(Paragraph(escape(_DISCLAIMER), styles["Disclaimer"]))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    w, _h = A4
    canvas.setStrokeColor(colors.HexColor("#C9D2E5"))
    canvas.setLineWidth(0.4)
    canvas.line(_SIDE_MARGIN, 1.25 * cm, w - _SIDE_MARGIN, 1.25 * cm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(
        _SIDE_MARGIN,
        0.85 * cm,
        "Investigation report — confidential work product · Not a legal determination",
    )
    canvas.drawRightString(w - _SIDE_MARGIN, 0.85 * cm, f"Page {doc.page}")
    canvas.restoreState()


def generate_investigation_report(
    user: User,
    analysis_id: int,
    *,
    investigator_notes: str | None = None,
    report_format: str = "pdf",
) -> InvestigationReport:
    run = get_analysis(user, analysis_id)
    if run.status != "COMPLETED":
        raise ValidationError("Reports can only be generated for completed analyses")

    # report_format is client-supplied and is interpolated into the output
    # filename, so it must be an allowlisted token and never a path fragment.
    report_format = str(report_format or "pdf").strip().lower()
    if report_format not in SUPPORTED_REPORT_FORMATS:
        raise ValidationError(
            f"Unsupported report format. Allowed: {sorted(SUPPORTED_REPORT_FORMATS)}"
        )

    evidence = get_evidence(user, run.evidence_id)
    case = evidence.case

    from flask import current_app

    report_root = Path(current_app.config["REPORT_DIR"])
    case_dir = report_root / "cases" / str(case.id)
    case_dir.mkdir(parents=True, exist_ok=True)

    rpt_num = _allocate_report_number()
    fname = f"{rpt_num}-{uuid.uuid4().hex[:8]}.{report_format.lower()}"
    output_path = case_dir / fname
    upload_root = Path(current_app.config["ROOT_DIR"])

    audit_q = AuditLog.query.filter(
        (AuditLog.case_id == case.id) | (AuditLog.user_id == user.id)
    ).order_by(AuditLog.timestamp.asc())
    audit_events = audit_q.all()

    face_rows = (
        FaceVerification.query.filter(
            (FaceVerification.investigation_id == str(run.investigation_id or ""))
            | (FaceVerification.evidence_id == evidence.id)
        )
        .order_by(FaceVerification.id.desc())
        .all()
    )

    try:
        _build_pdf(
            output_path,
            report_number=rpt_num,
            case=case,
            evidence=evidence,
            run=run,
            audit_events=audit_events,
            generator=user,
            notes=investigator_notes,
            face_verifications=face_rows,
        )
    except Exception as exc:
        logger.exception("PDF generation failed for analysis=%s", analysis_id)
        raise ValidationError(f"Failed to generate report: {type(exc).__name__}") from exc

    if not output_path.is_file():
        raise ValidationError("Report generation produced no output file")

    size = output_path.stat().st_size
    digest = hash_file(output_path, algorithm="sha256")
    resolved = output_path.resolve()
    root_resolved = upload_root.resolve()
    try:
        rel = resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        rel = resolved.relative_to(report_root.resolve()).as_posix()

    report = InvestigationReport(
        report_number=rpt_num,
        case_id=case.id,
        evidence_id=evidence.id,
        analysis_id=run.id,
        investigation_id=str(run.investigation_id or ""),
        generated_by_user_id=user.id,
        report_format=report_format.lower(),
        storage_path=rel,
        file_size_bytes=int(size),
        sha256_hash=digest,
        investigator_notes=investigator_notes,
    )
    db.session.add(report)
    db.session.flush()
    record_audit(
        AuditEventType.REPORT_GENERATED,
        user_id=user.id,
        case_id=case.id,
        evidence_id=evidence.id,
        analysis_id=run.id,
        details={
            "report_number": rpt_num,
            "sha256": digest,
            "size": int(size),
            "format": report_format.lower(),
        },
    )
    db.session.commit()
    logger.info("Report %s generated for analysis=%s", rpt_num, analysis_id)
    return report


def report_to_dict(report: InvestigationReport) -> dict[str, Any]:
    return {
        "report_id": report.id,
        "report_number": report.report_number,
        "case_id": report.case_id,
        "evidence_id": report.evidence_id,
        "analysis_id": report.analysis_id,
        "investigation_id": report.investigation_id,
        "generated_by": report.generated_by_user_id,
        "format": report.report_format,
        "storage_path": report.storage_path,
        "size_bytes": report.file_size_bytes,
        "sha256": report.sha256_hash,
        "title": report.title,
        "investigator_notes": report.investigator_notes,
        "generated_at": report.generated_at.isoformat() if report.generated_at else None,
    }
