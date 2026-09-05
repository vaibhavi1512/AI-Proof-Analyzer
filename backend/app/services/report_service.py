"""Investigation PDF report generator (reportlab)."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
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
from backend.app.models.enums import AuditEventType
from backend.app.services.analysis_service import get_analysis
from backend.app.services.evidence_service import get_evidence

logger = logging.getLogger("maya.backend.reports")

SUPPORTED_REPORT_FORMATS = {"pdf"}

_DISCLAIMER = (
    "This MAYA Media Authenticity Analyzer report is generated for investigative "
    "and academic training purposes only. AI-driven authenticity assessments are "
    "probabilistic and should not be treated as definitive forensic conclusions. "
    "Predictions and confidence scores reflect the model's learned distribution "
    "over the training corpus and may fail on novel manipulation methods, "
    "out-of-distribution media, or low-quality inputs. Explainability visualizations "
    "indicate the regions that most influenced the classifier, not a human-legible "
    "proof of manipulation. Independent forensic verification by qualified personnel "
    "is always recommended before any operational or legal decision."
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


def _extract_advanced_xai(run: AnalysisRun) -> dict[str, Any]:
    if not run.raw_result_json:
        return {}
    try:
        return json.loads(run.raw_result_json).get("advanced_xai_results") or {}
    except Exception:
        return {}


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
        return Image(str(p), width=w * scale, height=h * scale)
    except Exception:
        return None


def _kv_table(rows: list[tuple[str, str]]) -> Table:
    t = Table(rows, hAlign="LEFT", colWidths=[5.2 * cm, 11.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2FF")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1F2A44")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def _section_title(text: str, styles) -> Paragraph:
    return Paragraph(
        f"<b>{text}</b>",
        ParagraphStyle(
            "sec",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#183B8A"),
            spaceBefore=10,
            spaceAfter=6,
            borderPadding=(0, 0, 2, 0),
        ),
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
    story.append(_kv_table([(k, Paragraph(v, styles["Normal"])) for k, v in fv_rows]))

    from flask import current_app

    root = Path(current_app.config["ROOT_DIR"])
    if latest.artifact_dir:
        ref_img = _safe_image(str(root / latest.artifact_dir / "reference_face.png"), max_w=7 * cm, max_h=7 * cm)
        evd_img = _safe_image(str(root / latest.artifact_dir / "evidence_face.png"), max_w=7 * cm, max_h=7 * cm)
        if ref_img or evd_img:
            story.append(Spacer(1, 3 * mm))
            if ref_img:
                story.append(Paragraph("<i>Reference face crop</i>", styles["Normal"]))
                story.append(ref_img)
            if evd_img:
                story.append(Paragraph("<i>Evidence face crop</i>", styles["Normal"]))
                story.append(evd_img)


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


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
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        title=f"MAYA Investigation Report — {run.investigation_id}",
        author="MAYA Media Authenticity Analyzer",
        subject=f"Case {case.case_number}",
    )
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].fontSize = 9

    story: list[Any] = []

    # Header
    header_tbl = Table(
        [[
            Paragraph(
                "<b><font size='16' color='#183B8A'>MAYA</font></b><br/>"
                "<font size='9'>Media Authenticity Analyzer</font>",
                ParagraphStyle("brand", alignment=TA_LEFT, leading=14),
            ),
            Paragraph(
                "<b>INVESTIGATION REPORT</b><br/>"
                f"<font size='8'>Generated {_utcnow().isoformat(timespec='seconds')} UTC</font>",
                ParagraphStyle("hd", alignment=TA_CENTER, textColor=colors.HexColor("#1F2A44"), leading=12),
            ),
        ]],
        colWidths=[8 * cm, 9 * cm],
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
            ]
        )
    )
    story.append(header_tbl)
    story.append(Spacer(1, 0.5 * cm))

    # Core identifiers
    ids = [
        ("Investigation ID", str(run.investigation_id or "")),
        ("Report Number", report_number),
        ("Case Number", str(case.case_number)),
        ("Case Title", str(case.title)),
        ("Evidence ID", f"EVD-{evidence.id} ({evidence.original_filename})"),
        ("Analyst / Investigator", generator.full_name or generator.username),
        ("Generated At (UTC)", _utcnow().isoformat(timespec="seconds")),
    ]
    story.append(_section_title("Investigation Summary", styles))
    story.append(_kv_table([(k, v) for k, v in ids]))

    # Prediction
    story.append(_section_title("Authenticity Assessment", styles))
    confidence = f"{float(run.confidence or 0.0):.2f}%"
    pred = str(run.prediction or "N/A")
    pred_color = colors.HexColor("#106F3A") if pred == "REAL" else colors.HexColor("#8A1430")
    pred_rows = [
        ("Prediction", f"<b><font color='#{pred_color.hexval()[2:]}'>{pred}</font></b>"),
        ("Confidence", confidence),
        ("Model", str(run.model_name or "")),
        ("Model Version", str(run.model_version or "")),
        ("Dataset Version", str(run.dataset_version or "")),
        ("Analysis Status", str(run.status or "")),
    ]
    if run.trust_score is not None:
        pred_rows.append(("XAI Trust Score", f"{float(run.trust_score):.2f}"))
    if run.quality_score is not None:
        pred_rows.append(("XAI Quality Score", f"{float(run.quality_score):.2f}"))
    if run.started_at:
        pred_rows.append(("Started At", run.started_at.isoformat(timespec="seconds")))
    if run.completed_at:
        pred_rows.append(("Completed At", run.completed_at.isoformat(timespec="seconds")))
    story.append(_kv_table([(k, Paragraph(v, styles["Normal"])) for k, v in pred_rows]))

    _append_face_verification_section(story, styles, face_verifications or [])

    # Evidence metadata
    story.append(_section_title("Evidence Integrity & Metadata", styles))
    ev_rows = [
        ("Original Filename", str(evidence.original_filename)),
        ("Stored Identifier", str(evidence.stored_filename)),
        ("SHA-256 Digest", str(evidence.sha256_hash)),
        ("MIME Type", str(evidence.mime_type)),
        ("Size (bytes)", f"{evidence.file_size_bytes:,}"),
        ("Uploaded By (user id)", f"{evidence.uploaded_by_user_id}"),
        ("Uploaded At", evidence.uploaded_at.isoformat(timespec="seconds") if evidence.uploaded_at else ""),
        ("Evidence Status", str(evidence.status)),
    ]
    story.append(_kv_table([(k, v) for k, v in ev_rows]))

    # XAI summary
    story.append(_section_title("Explainability (XAI) Summary", styles))
    adv = _extract_advanced_xai(run)
    xai_rows: list[tuple[str, str]] = []
    if run.explainer_name:
        xai_rows.append(("Primary Explainer", str(run.explainer_name)))
    methods_run = adv.get("methods_run") or []
    if methods_run:
        xai_rows.append(("Methods Executed", ", ".join(str(m) for m in methods_run)))
    trust = adv.get("trust")
    if trust:
        grade = trust.get("grade") if isinstance(trust, dict) else None
        xai_rows.append(("Trust Grade", str(grade) if grade else ""))
    if not xai_rows:
        xai_rows.append(("Status", "No additional explainability stages were requested for this analysis run."))
    story.append(_kv_table([(k, v) for k, v in xai_rows]))

    # XAI visualizations
    primary_overlay = _safe_image(run.overlay_path, max_w=14 * cm, max_h=9 * cm)
    primary_heatmap = _safe_image(run.heatmap_path, max_w=9 * cm, max_h=8 * cm)
    fig_rows = []
    if primary_overlay or primary_heatmap:
        story.append(_section_title("XAI Visualizations — Primary", styles))
        if primary_overlay:
            story.append(Paragraph("<i>Explainer overlay on evidence</i>", styles["Normal"]))
            story.append(primary_overlay)
            story.append(Spacer(1, 4 * mm))
        if primary_heatmap and primary_heatmap is not primary_overlay:
            story.append(Paragraph("<i>Raw attention heatmap</i>", styles["Normal"]))
            story.append(primary_heatmap)
            story.append(Spacer(1, 4 * mm))

    # Advanced XAI refs (page break if needed)
    if adv:
        story.append(PageBreak())
        story.append(_section_title("Advanced XAI — Artifact References", styles))
        ref_rows: list[tuple[str, str]] = []
        artifact_paths = adv.get("artifact_paths") or {}
        for k, v in artifact_paths.items():
            ref_rows.append((str(k).replace("_", " ").title(), str(v)))
        shap = adv.get("shap")
        if isinstance(shap, dict) and shap.get("enabled"):
            ref_rows.append(("SHAP", f"heatmap={shap.get('heatmap')} overlay={shap.get('overlay')}"))
        faith = adv.get("faithfulness")
        if isinstance(faith, dict) and faith.get("enabled"):
            ref_rows.append(("Faithfulness", f"comparison={faith.get('comparison')}"))
        cf = adv.get("counterfactual")
        if isinstance(cf, dict) and cf.get("enabled"):
            ref_rows.append(("Counterfactual", f"comparison={cf.get('comparison')}"))
        fusion = adv.get("fusion")
        if isinstance(fusion, dict) and fusion.get("enabled"):
            ref_rows.append(("Explanation Fusion", f"fused={fusion.get('fused_explanation')}"))
        trust_dict = adv.get("trust")
        if isinstance(trust_dict, dict) and trust_dict.get("enabled"):
            comps = trust_dict.get("components")
            comp_str = json.dumps(comps, indent=2) if isinstance(comps, (dict, list)) else str(comps)
            ref_rows.append(("Trust Components", comp_str[:600]))
        if ref_rows:
            story.append(_kv_table([(k, Paragraph(v, styles["Normal"])) for k, v in ref_rows]))

    # Audit timeline
    story.append(PageBreak())
    story.append(_section_title("Investigation Audit Timeline", styles))
    timeline_rows = [("Timestamp (UTC)", "Event", "Case/Evd/Ana", "Details")]
    for ev in audit_events[:80]:
        timeline_rows.append((
            ev.timestamp.isoformat(timespec="seconds") if ev.timestamp else "",
            str(ev.event_type),
            f"C={ev.case_id or '-'} E={ev.evidence_id or '-'} A={ev.analysis_id or '-'}",
            str(ev.details_json)[:160],
        ))
    t = Table(timeline_rows, hAlign="LEFT", repeatRows=1,
              colWidths=[4.2 * cm, 3.5 * cm, 3.5 * cm, 5.8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#C9D2E5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t)

    # Investigator notes
    if notes:
        story.append(Spacer(1, 0.4 * cm))
        story.append(_section_title("Investigator Notes", styles))
        story.append(Paragraph(str(notes), styles["Normal"]))

    # Disclaimer
    story.append(Spacer(1, 0.5 * cm))
    story.append(_section_title("Limitations & Disclaimer", styles))
    story.append(
        Paragraph(
            _DISCLAIMER,
            ParagraphStyle(
                "disc",
                parent=styles["Normal"],
                textColor=colors.HexColor("#4A4A4A"),
                fontSize=8.2,
                leading=10.5,
            ),
        )
    )

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)


def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    w, h = A4
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(
        1.8 * cm,
        0.9 * cm,
        "MAYA Media Authenticity Analyzer — Confidential Investigative Work Product",
    )
    canvas.drawRightString(w - 1.8 * cm, 0.9 * cm, f"Page {doc.page}")
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
