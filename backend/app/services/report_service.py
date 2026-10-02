"""Investigation PDF report generator (reportlab).

Presentation improvements (report V2) adjust title, authenticity interpretation,
integrity summary layout, XAI captions/disclaimers, audit-table wrapping, and
legal/readiness wording. AI inference, XAI, face verification, and API behaviour
are unchanged — only the generated PDF content and layout are modified.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
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
from backend.app.exceptions import AuthorizationError, ReportFileMissingError, ValidationError
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

_VIDEO_AGGREGATION_NOTE = (
    "The current video result is a video-level aggregation of frame-level evidence. "
    "It is not a dedicated temporal deepfake classifier."
)

_VIDEO_XAI_NOTE = (
    "Grad-CAM provides a visual explanation of regions influencing the model "
    "prediction. It is not independent proof of manipulation."
)

_VIDEO_TEMPORAL_LIMIT = (
    "Temporal face tracking is supporting forensic analysis of sampled frames. "
    "It does not perform continuous identity verification and is not proof of "
    "manipulation or that a detected face is fake."
)

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
_FRAME_DIR_RE = re.compile(r"^frame_(\d+)$")
_MAX_FRAME_TABLE_ROWS = 80
_GALLERY_COLS = 3
_THUMB_MAX_W = 5.1 * cm
_THUMB_MAX_H = 3.4 * cm
_THUMB_MAX_PX = 360

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
    styles.add(
        ParagraphStyle(
            "GalleryMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            alignment=TA_CENTER,
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


def _report_is_video(evidence: Evidence, raw: dict[str, Any]) -> bool:
    if str(evidence.media_type or "").strip().lower() == "video":
        return True
    return str(raw.get("media_type") or "").strip().lower() == "video"


def _fmt_optional(value: Any, *, suffix: str = "") -> str:
    if value is None or value == "":
        return UNAVAILABLE
    return f"{value}{suffix}"


def _fmt_seconds(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    return f"{number:.2f} s"


def _fmt_frame_confidence(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if 0.0 <= number <= 1.0:
        return f"{number * 100.0:.1f}%"
    return f"{number:.1f}%"


def _video_payload_probs(raw: dict[str, Any]) -> dict[str, Any]:
    """Video stores probabilities at the top level, not under investigation."""

    return {
        "real_probability": raw.get("real_probability"),
        "fake_probability": raw.get("fake_probability"),
        "confidence": raw.get("confidence"),
        "confidence_level": raw.get("confidence_level"),
        "threshold": (raw.get("aggregation") or {}).get("threshold", raw.get("threshold")),
        "threshold_decision": raw.get("threshold_decision"),
        "prediction": raw.get("prediction"),
        "model_name": raw.get("model_name"),
        "model_version": raw.get("model_version"),
    }


def _resolve_existing_file(*candidates: Path | str | None) -> Path | None:
    for item in candidates:
        if not item:
            continue
        path = Path(item)
        try:
            if path.is_file() and path.suffix.lower() not in _VIDEO_SUFFIXES:
                return path
        except OSError:
            continue
    return None


def _frame_jpeg_path(item: dict[str, Any], upload_root: Path) -> Path | None:
    stored = item.get("frame_path")
    if not stored:
        return None
    stored_path = Path(str(stored))
    return _resolve_existing_file(stored_path, upload_root / stored_path)


def _write_pdf_thumbnail(source: Path, dest: Path) -> Path | None:
    """Write a downscaled JPEG for PDF embedding. Never overwrites the source frame."""

    try:
        from PIL import Image as PILImage

        dest.parent.mkdir(parents=True, exist_ok=True)
        with PILImage.open(source) as src:
            image = src.convert("RGB")
            image.thumbnail((_THUMB_MAX_PX, _THUMB_MAX_PX), PILImage.Resampling.LANCZOS)
            if image.width <= 0 or image.height <= 0:
                return None
            image.save(dest, format="JPEG", quality=72, optimize=True)
        return dest if dest.is_file() else None
    except Exception:
        logger.warning("Failed to build PDF thumbnail for %s", source, exc_info=True)
        return None


def _thumbnail_image(path: Path, *, max_w: float, max_h: float) -> Image | None:
    """Embed a JPEG using pixel aspect ratio so JPEG DPI does not inflate layout."""

    try:
        from PIL import Image as PILImage

        with PILImage.open(path) as src:
            px_w, px_h = src.size
        if px_w <= 0 or px_h <= 0:
            return None
        ratio = px_w / px_h
        if max_w / max_h > ratio:
            draw_h = max_h
            draw_w = max_h * ratio
        else:
            draw_w = max_w
            draw_h = max_w / ratio
        flowable = Image(str(path), width=draw_w, height=draw_h)
        flowable.hAlign = "CENTER"
        return flowable
    except Exception:
        logger.warning("Failed to embed PDF thumbnail for %s", path, exc_info=True)
        return _safe_image(str(path), max_w=max_w, max_h=max_h)


def _gallery_cell(
    item: dict[str, Any],
    styles,
    *,
    xai: bool,
    upload_root: Path,
    thumb_dir: Path | None,
) -> list[Any]:
    number = item.get("frame_number", "—")
    pred = str(item.get("prediction") or "—")
    lines = [
        Paragraph(escape(f"Frame {number}"), styles["GalleryMeta"]),
        Paragraph(escape(_fmt_seconds(item.get("timestamp_seconds"))), styles["GalleryMeta"]),
        Paragraph(
            escape(f"{pred}  {_fmt_frame_confidence(item.get('confidence'))}"),
            styles["GalleryMeta"],
        ),
    ]
    if xai:
        lines.append(Paragraph("XAI", styles["GalleryMeta"]))
    jpeg = _frame_jpeg_path(item, upload_root)
    thumb = None
    if jpeg is not None:
        if thumb_dir is not None:
            dest = thumb_dir / f"frame_{number}.jpg"
            written = _write_pdf_thumbnail(jpeg, dest)
            if written is not None:
                thumb = _thumbnail_image(written, max_w=_THUMB_MAX_W, max_h=_THUMB_MAX_H)
        else:
            thumb = _thumbnail_image(jpeg, max_w=_THUMB_MAX_W, max_h=_THUMB_MAX_H)
    if thumb is None:
        lines.insert(0, Paragraph("Frame image unavailable", styles["GalleryMeta"]))
    else:
        lines.insert(0, thumb)
    return lines


def _append_video_frame_gallery(
    story: list[Any],
    styles,
    frames: list[dict[str, Any]],
    *,
    xai_numbers: set[int],
    thumb_dir: Path | None = None,
) -> None:
    story.append(_section_title("Video Frame Gallery", styles))
    story.append(
        Paragraph(
            "Every sampled/analyzed frame from the stored analysis is shown as a "
            "thumbnail. These stills are extracted JPEGs, not the MP4 container. "
            "Thumbnails are downscaled for the PDF only; source frames are unchanged.",
            styles["Caption"],
        )
    )
    if not frames:
        story.append(
            Paragraph("No sampled frames were stored for this analysis.", styles["BodyJustified"])
        )
        return
    from flask import current_app

    upload_root = Path(current_app.config["UPLOAD_DIR"])
    col_w = (_CONTENT_WIDTH - 8) / _GALLERY_COLS
    inner_w = col_w - 8
    cells = []
    for item in frames:
        try:
            number = int(item.get("frame_number"))
        except (TypeError, ValueError):
            number = None
        lines = _gallery_cell(
            item,
            styles,
            xai=number in xai_numbers if number is not None else False,
            upload_root=upload_root,
            thumb_dir=thumb_dir,
        )
        inner = Table([[line] for line in lines], colWidths=[inner_w])
        inner.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        cells.append(inner)
    rows: list[list[Any]] = []
    for index in range(0, len(cells), _GALLERY_COLS):
        row = cells[index : index + _GALLERY_COLS]
        while len(row) < _GALLERY_COLS:
            row.append(Paragraph("", styles["GalleryMeta"]))
        rows.append(row)
    table = Table(rows, colWidths=[col_w] * _GALLERY_COLS, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E5")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E4E9F2")),
            ]
        )
    )
    story.append(table)


def _video_xai_dirs(run: AnalysisRun) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    seen: set[int] = set()
    if run.artifact_dir:
        gradcam = Path(run.artifact_dir) / "xai" / "gradcam"
        try:
            if gradcam.is_dir():
                for child in sorted(gradcam.iterdir()):
                    match = _FRAME_DIR_RE.fullmatch(child.name)
                    if not match or not child.is_dir():
                        continue
                    number = int(match.group(1))
                    if number in seen:
                        continue
                    seen.add(number)
                    found.append((number, child))
        except OSError:
            pass
    if not found and run.heatmap_path:
        parent = Path(run.heatmap_path).parent
        match = re.search(r"frame_(\d+)", parent.name)
        if match:
            found.append((int(match.group(1)), parent))
    return found


def _frame_lookup(frames: list[Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in frames:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("frame_number"))
        except (TypeError, ValueError):
            continue
        out[number] = item
    return out


def _append_report_banner(
    story: list[Any],
    styles,
    *,
    report_number: str,
    case: Case,
) -> None:
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


def _append_integrity_block(
    story: list[Any],
    styles,
    evidence: Evidence,
    *,
    extra_rows: list[tuple[str, str]] | None = None,
) -> str:
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
    if extra_rows:
        ev_rows.extend(extra_rows)
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
    return integrity_status


def _append_audit_and_close(
    story: list[Any],
    styles,
    *,
    evidence: Evidence,
    run: AnalysisRun,
    audit_events: list[AuditLog],
    generator_notes: str | None,
    integrity_status: str,
    pred: str,
    confidence_val: float | None,
    face_verifications: list[FaceVerification],
    extra_readiness: list[tuple[str, str]] | None = None,
    final_paragraphs: list[str] | None = None,
    xai_recorded: bool | None = None,
) -> None:
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
            "Yes"
            if (
                bool(run.overlay_path or run.heatmap_path)
                if xai_recorded is None
                else xai_recorded
            )
            else "No",
        ),
        (
            "Face reference verification",
            "Yes" if face_verifications else "Not performed for this investigation",
        ),
    ]
    if extra_readiness:
        readiness_rows.extend(extra_readiness)
    story.append(_kv_table(readiness_rows, styles))
    story.append(Paragraph(escape(_READINESS_CLOSING), styles["BodyJustified"]))

    story.append(_section_title("Investigation Audit Timeline", styles))
    story.append(
        Paragraph(
            "The following timeline lists stored audit events for this case only. "
            "Timestamps and event types are taken from the AuditLog without alteration.",
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
    for paragraph in final_paragraphs or [_assessment_narrative(pred, confidence_val)]:
        story.append(Paragraph(escape(paragraph), styles["BodyJustified"]))

    if generator_notes:
        story.append(_section_title("Investigator Notes", styles))
        story.append(Paragraph(escape(str(generator_notes)), styles["BodyJustified"]))

    story.append(_section_title("Limitations & Disclaimer", styles))
    story.append(Paragraph(escape(_DISCLAIMER), styles["Disclaimer"]))


def _build_frame_table(frames: list[dict[str, Any]], styles) -> LongTable:
    header = [
        Paragraph("Frame", styles["CellHeader"]),
        Paragraph("Time", styles["CellHeader"]),
        Paragraph("Prediction", styles["CellHeader"]),
        Paragraph("Confidence", styles["CellHeader"]),
        Paragraph("REAL p", styles["CellHeader"]),
        Paragraph("FAKE p", styles["CellHeader"]),
    ]
    data: list[list[Any]] = [header]
    shown = frames[:_MAX_FRAME_TABLE_ROWS]
    for item in shown:
        real = _fmt_prob(item.get("real_probability")) or "—"
        fake = _fmt_prob(item.get("fake_probability")) or "—"
        data.append(
            [
                Paragraph(escape(str(item.get("frame_number", "—"))), styles["Cell"]),
                Paragraph(escape(_fmt_seconds(item.get("timestamp_seconds"))), styles["Cell"]),
                Paragraph(escape(str(item.get("prediction") or "—")), styles["Cell"]),
                Paragraph(escape(_fmt_frame_confidence(item.get("confidence"))), styles["Cell"]),
                Paragraph(escape(real), styles["Cell"]),
                Paragraph(escape(fake), styles["Cell"]),
            ]
        )
    col_widths = [2.0 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 3.9 * cm, _CONTENT_WIDTH - 13.5 * cm]
    table = LongTable(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183B8A")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E5")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#183B8A")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _append_video_xai_section(
    story: list[Any],
    styles,
    *,
    run: AnalysisRun,
    frames: list[dict[str, Any]],
) -> int:
    story.append(PageBreak())
    story.append(_section_title("XAI / Grad-CAM", styles))
    story.append(Paragraph(escape(_XAI_CAPTION), styles["BodyJustified"]))
    story.append(Paragraph(escape(_VIDEO_XAI_NOTE), styles["BodyJustified"]))
    dirs = _video_xai_dirs(run)
    lookup = _frame_lookup(frames)
    from flask import current_app

    upload_root = Path(current_app.config["UPLOAD_DIR"])
    included = 0
    if not dirs:
        story.append(
            Paragraph(
                "No Grad-CAM visualization artifacts were available for this analysis run. "
                "Missing optional explainability does not invalidate the stored verdict.",
                styles["BodyJustified"],
            )
        )
        return 0
    for number, folder in dirs:
        meta = lookup.get(number) or {}
        story.append(_section_title(f"Selected XAI frame {number}", styles))
        rows = [
            ("Frame number", str(number)),
            ("Timestamp", _fmt_seconds(meta.get("timestamp_seconds"))),
            ("Frame prediction", str(meta.get("prediction") or UNAVAILABLE)),
            ("Frame confidence", _fmt_frame_confidence(meta.get("confidence")) if meta else UNAVAILABLE),
        ]
        real = _fmt_prob(meta.get("real_probability"))
        fake = _fmt_prob(meta.get("fake_probability"))
        if real:
            rows.append(("REAL probability", real))
        if fake:
            rows.append(("FAKE probability", fake))
        story.append(_kv_table(rows, styles))

        jpeg = None
        stored = meta.get("frame_path")
        if stored:
            stored_path = Path(str(stored))
            jpeg = _resolve_existing_file(stored_path, upload_root / stored_path)
        original = _resolve_existing_file(folder / "original.png", jpeg)
        heatmap = _resolve_existing_file(folder / "heatmap.png")
        overlay = _resolve_existing_file(folder / "overlay.png")
        if original:
            img = _safe_image(str(original), max_w=12 * cm, max_h=8 * cm)
            if img:
                story.append(
                    _captioned_figure(
                        img,
                        f"Figure: Extracted JPEG / original still for frame {number} (not the MP4 container).",
                        styles,
                    )
                )
        if heatmap:
            img = _safe_image(str(heatmap), max_w=12 * cm, max_h=8 * cm)
            if img:
                story.append(
                    _captioned_figure(
                        img,
                        f"Figure: Grad-CAM heatmap for frame {number}.",
                        styles,
                    )
                )
        if overlay:
            img = _safe_image(str(overlay), max_w=14 * cm, max_h=9 * cm)
            if img:
                story.append(
                    _captioned_figure(
                        img,
                        f"Figure: Grad-CAM overlay for frame {number}.",
                        styles,
                    )
                )
        if original or heatmap or overlay:
            included += 1
        else:
            story.append(
                Paragraph(
                    f"Grad-CAM files for frame {number} were referenced but not readable on disk.",
                    styles["BodyJustified"],
                )
            )
    return included


def _video_summary(
    *,
    evidence: Evidence,
    run: AnalysisRun,
    raw: dict[str, Any],
    aggregation: dict[str, Any],
    temporal: dict[str, Any] | None,
    xai_count: int,
    integrity_status: str,
    pred: str,
    confidence_val: float | None,
) -> list[str]:
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    analyzed = aggregation.get("total_frames_analyzed")
    real_n = aggregation.get("real_frame_count")
    fake_n = aggregation.get("fake_frame_count")
    fake_pct = aggregation.get("fake_frame_percentage")
    parts = [
        (
            f"System analysis indicates that EVD-{evidence.id} ({evidence.original_filename}) "
            f"is stored video evidence. Integrity status at report time was {integrity_status} "
            f"based on the SHA-256 hash recorded at ingest."
        ),
        (
            f"Based on the frame-level model outputs, {analyzed if analyzed is not None else UNAVAILABLE} "
            f"sampled frames were analyzed "
            f"({real_n if real_n is not None else '—'} REAL, "
            f"{fake_n if fake_n is not None else '—'} FAKE"
            + (f", {float(fake_pct):.2f}% FAKE" if isinstance(fake_pct, (int, float)) else "")
            + ")."
        ),
        (
            f"The video-level result is {pred}"
            + (f" at {float(confidence_val):.2f}% model confidence" if confidence_val is not None else "")
            + f" using {aggregation.get('method') or 'mean_frame_probability'}. "
            + _VIDEO_AGGREGATION_NOTE
        ),
    ]
    if isinstance(temporal, dict) and temporal.get("error"):
        parts.append(
            "Temporal face analysis was not available for this run "
            f"({temporal.get('error')}). Further human/forensic review may be required."
        )
    elif isinstance(temporal, dict):
        gaps = temporal.get("tracking_gaps") or []
        gap_n = len(gaps) if isinstance(gaps, list) else 0
        track_n = temporal.get("track_count")
        parts.append(
            "Temporal analysis detected "
            f"{temporal.get('frames_with_face', '—')} sampled frame(s) with a face "
            f"({temporal.get('detector') or 'unspecified detector'}), "
            f"{track_n if track_n is not None else '—'} track(s), "
            f"and {gap_n} tracking gap record(s). "
            + _VIDEO_TEMPORAL_LIMIT
        )
        if gap_n:
            parts.append(
                "Tracking is fragmented: gaps were recorded on the sampled timeline. "
                "That is a tracking observation, not a manipulation finding."
            )
    else:
        parts.append("Temporal face analysis was not recorded for this analysis.")
    if xai_count:
        parts.append(
            f"Grad-CAM artifacts were included for {xai_count} selected frame(s). "
            + _VIDEO_XAI_NOTE
        )
    else:
        parts.append("No Grad-CAM artifacts were available to include in this report.")
    if metadata:
        parts.append(
            "Container metadata used in this report was taken from the stored Phase 2 "
            "extraction record; it was not re-measured at report time."
        )
    parts.append(
        "Further human/forensic review may be required. This report does not determine "
        "legal admissibility and does not prove authenticity or manipulation."
    )
    parts.append(_assessment_narrative(pred, confidence_val))
    return parts


_LSTM_XAI_NAME = re.compile(
    r"^(?:temporal_contribution|gradcam_contact_sheet|gradcam_frame_\d+)\.png$"
)
_LSTM_FRAME_NAME = re.compile(r"^gradcam_frame_(\d+)\.png$")


def _video_confidence_percent(video: dict[str, Any]) -> float | None:
    """Probability of the predicted class, as a percentage. No second model."""

    prediction = video.get("prediction")
    try:
        p_fake = float(video.get("p_fake"))
    except (TypeError, ValueError):
        return None
    if prediction == "FAKE":
        score = p_fake
    elif prediction == "REAL":
        score = 1.0 - p_fake
    else:
        return None
    if score != score:  # NaN
        return None
    return score * 100.0


def _fmt_count(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNAVAILABLE
    return str(int(value))


def _lstm_video_xai_dir(run: AnalysisRun) -> Path | None:
    """Return this analysis's video XAI directory, or None if it is not contained."""

    if not run.artifact_dir:
        return None
    from flask import current_app

    from backend.app.utils.paths import is_within

    try:
        artifacts_root = (Path(current_app.config["ROOT_DIR"]) / "artifacts").resolve()
        artifact_dir = Path(run.artifact_dir).resolve()
    except (OSError, RuntimeError):
        return None
    if not is_within(artifacts_root, artifact_dir):
        return None
    folder = (artifact_dir / "xai" / "video").resolve()
    if not is_within(artifact_dir, folder):
        return None
    return folder


def _lstm_video_xai_file(run: AnalysisRun, filename: str) -> Path | None:
    """Resolve one allow-listed video XAI file that belongs to this analysis."""

    if not _LSTM_XAI_NAME.fullmatch(str(filename)):
        return None
    folder = _lstm_video_xai_dir(run)
    if folder is None:
        return None
    from backend.app.utils.paths import is_within

    try:
        target = (folder / str(filename)).resolve()
    except OSError:
        return None
    if not is_within(folder, target) or not target.is_file():
        return None
    return target


def _lstm_artifact_names(run: AnalysisRun, raw: dict[str, Any]) -> list[str]:
    recorded = raw.get("xai_artifact_names")
    if isinstance(recorded, list):
        return [str(name) for name in recorded if _LSTM_XAI_NAME.fullmatch(str(name))]
    folder = _lstm_video_xai_dir(run)
    if folder is None or not folder.is_dir():
        return []
    found: list[str] = []
    try:
        for child in folder.iterdir():
            if child.is_file() and _LSTM_XAI_NAME.fullmatch(child.name):
                found.append(child.name)
    except OSError:
        return []
    return found


def _select_gradcam_names(names: list[str], limit: int = 4) -> list[str]:
    frames = []
    for name in names:
        match = _LSTM_FRAME_NAME.fullmatch(name)
        if match:
            frames.append((int(match.group(1)), name))
    frames.sort()
    if len(frames) <= limit:
        return [name for _, name in frames]
    if limit < 1:
        return []
    picked: list[str] = []
    last = len(frames) - 1
    for slot in range(limit):
        index = round(slot * last / (limit - 1)) if limit > 1 else 0
        name = frames[index][1]
        if name not in picked:
            picked.append(name)
    return picked


def _temporal_contribution_drawing(frames: list[Any]):
    """Bar chart of stored relative contribution. Values are not printed."""

    values: list[float] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        importance = frame.get("temporal_importance")
        if isinstance(importance, (int, float)) and not isinstance(importance, bool):
            values.append(float(importance))
    if not values:
        return None
    from reportlab.graphics.shapes import Drawing, Line, Rect, String

    width, height = 460, 150
    drawing = Drawing(width, height)
    peak = max(max(values), 1e-8)
    left, bottom, plot_h, plot_w = 16, 22, 100, width - 28
    slot = plot_w / len(values)
    drawing.add(Line(left, bottom, left + plot_w, bottom, strokeColor=colors.HexColor("#98A2B3")))
    for index, value in enumerate(values):
        bar_h = max(plot_h * (value / peak), 0.0)
        drawing.add(
            Rect(
                left + index * slot + 1,
                bottom,
                max(slot - 2, 1),
                bar_h,
                fillColor=colors.HexColor("#183B8A"),
                strokeColor=None,
            )
        )
    drawing.add(
        String(
            left,
            6,
            "Sampled frame order. Bar height is relative contribution, not probability.",
            fontSize=7,
            fillColor=colors.HexColor("#475467"),
        )
    )
    return drawing


def _append_lstm_xai(
    story: list[Any],
    styles,
    *,
    run: AnalysisRun,
    raw: dict[str, Any],
    video: dict[str, Any],
) -> bool:
    """Embed stored temporal and Grad-CAM artifacts. Returns whether any were included."""

    xai = video.get("xai") if isinstance(video.get("xai"), dict) else {}
    available = video.get("xai_available") is True
    story.append(_section_title("Explainable AI", styles))
    if not available:
        story.append(
            Paragraph(
                "Explainable AI was not available for this analysis. "
                "No temporal contribution or Grad-CAM artifact is included.",
                styles["BodyJustified"],
            )
        )
        return False

    names = _lstm_artifact_names(run, raw)
    included = False
    temporal_file = (
        _lstm_video_xai_file(run, "temporal_contribution.png")
        if "temporal_contribution.png" in names
        else None
    )
    story.append(Paragraph("Relative temporal contribution", styles["Heading2"]))
    story.append(
        Paragraph(
            escape(
                str(xai.get("temporal_wording") or "frames with higher relative contribution")
                + ". This chart shows relative contribution to the model prediction. "
                "It is not a probability and it is not a confidence score."
            ),
            styles["BodyJustified"],
        )
    )
    temporal_image = _safe_image(str(temporal_file) if temporal_file else None, max_w=_CONTENT_WIDTH, max_h=8 * cm)
    if temporal_image is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(temporal_image)
        included = True
    else:
        drawing = _temporal_contribution_drawing(xai.get("frames") or [])
        if drawing is not None:
            story.append(Spacer(1, 2 * mm))
            story.append(drawing)
            included = True
        else:
            story.append(
                Paragraph(
                    "Temporal explanation was not available for this analysis.",
                    styles["BodyJustified"],
                )
            )

    story.append(Paragraph("Regions contributing to the model prediction", styles["Heading2"]))
    story.append(
        Paragraph(
            escape(
                str(xai.get("spatial_wording") or "regions contributing to the model prediction")
                + ". These regions explain model behavior. They are not confirmed manipulated pixels "
                "and they are not proof of tampering."
            ),
            styles["BodyJustified"],
        )
    )
    contact = (
        _lstm_video_xai_file(run, "gradcam_contact_sheet.png")
        if "gradcam_contact_sheet.png" in names
        else None
    )
    contact_image = _safe_image(str(contact) if contact else None, max_w=_CONTENT_WIDTH, max_h=9 * cm)
    frame_names = _select_gradcam_names(names)
    frame_paths = [path for name in frame_names if (path := _lstm_video_xai_file(run, name))]
    if contact_image is None and not frame_paths:
        story.append(
            Paragraph(
                "Visual explanation was unavailable for this analysis. "
                "No Grad-CAM artifact was stored for this run.",
                styles["BodyJustified"],
            )
        )
        return included
    if contact_image is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(contact_image)
        story.append(Paragraph("Grad-CAM contact sheet for this analysis.", styles["Caption"]))
        included = True
    shown = 0
    for path in frame_paths:
        image = _safe_image(str(path), max_w=_CONTENT_WIDTH, max_h=7.5 * cm)
        if image is None:
            continue
        story.append(Spacer(1, 2 * mm))
        story.append(image)
        shown += 1
        included = True
    if shown:
        total = sum(1 for name in names if _LSTM_FRAME_NAME.fullmatch(name))
        story.append(
            Paragraph(
                f"Grad-CAM frames shown: {shown} of {total} stored for this analysis.",
                styles["Caption"],
            )
        )
    return included


def _append_lstm_video_report(
    story: list[Any],
    styles,
    *,
    report_number: str,
    case: Case,
    evidence: Evidence,
    run: AnalysisRun,
    generator: User,
    raw: dict[str, Any],
    audit_events: list[AuditLog],
    notes: str | None,
    face_verifications: list[FaceVerification],
) -> None:
    """Report the stored 16-frame LSTM result for this analysis only."""

    video = raw.get("video_analysis") if isinstance(raw.get("video_analysis"), dict) else {}
    _append_report_banner(story, styles, report_number=report_number, case=case)
    story.append(_section_title("Case and Evidence", styles))
    story.append(
        _kv_table(
            [
                ("Case ID", str(case.case_number)),
                ("Case Title", str(case.title)),
                ("Evidence ID", f"EVD-{evidence.id}"),
                ("Original filename", str(evidence.original_filename or "")),
                ("Media type", "VIDEO"),
                ("File size", f"{evidence.file_size_bytes:,} bytes"),
                ("Investigation ID", str(run.investigation_id or "")),
                ("Report Number", report_number),
                ("Analysis ID", str(run.id)),
                ("Analyst / Investigator", generator.full_name or generator.username),
                (
                    "Uploaded At",
                    evidence.uploaded_at.isoformat(timespec="seconds") if evidence.uploaded_at else UNAVAILABLE,
                ),
                ("Generated At (UTC)", _utcnow().isoformat(timespec="seconds")),
            ],
            styles,
        )
    )
    integrity_status = _append_integrity_block(
        story,
        styles,
        evidence,
        extra_rows=[("Evidence type", "VIDEO")],
    )
    pred = str(video.get("prediction") or run.prediction or "N/A")
    if pred not in {"REAL", "FAKE"}:
        pred = str(run.prediction or "N/A")
    confidence_val = _video_confidence_percent(video)
    pred_color = colors.HexColor("#106F3A") if pred == "REAL" else colors.HexColor("#8A1430")
    story.append(_section_title("Analysis Summary", styles))
    story.append(
        _kv_table(
            [
                (
                    "Prediction",
                    f"<b><font color='#{pred_color.hexval()[2:]}'>{escape(pred)}</font></b>",
                ),
                ("Confidence Score", _fmt_pct(confidence_val)),
                ("Model", str(video.get("model_name") or run.model_name or UNAVAILABLE)),
                ("Model version", str(video.get("model_version") or run.model_version or UNAVAILABLE)),
                ("Frames analyzed", _fmt_count(video.get("frames_analyzed"))),
                ("Face-crop frames", _fmt_count(video.get("face_crop_frames"))),
                ("Full-frame fallback frames", _fmt_count(video.get("fallback_frames"))),
                (
                    "XAI available",
                    "Yes" if video.get("xai_available") is True else "No",
                ),
            ],
            styles,
        )
    )
    story.append(
        Paragraph(
            "Confidence Score is the model-assigned probability of the predicted class. "
            "It is not proof and it is not a legal conclusion.",
            styles["Caption"],
        )
    )
    try:
        fallback_n = int(video.get("fallback_frames"))
    except (TypeError, ValueError):
        fallback_n = None
    if fallback_n is not None and fallback_n > 0:
        story.append(
            Paragraph(
                "Some sampled frames did not contain a detectable face and were analyzed "
                "using the full-frame fallback. A missing face is not treated as FAKE.",
                styles["BodyJustified"],
            )
        )
    xai_included = _append_lstm_xai(story, styles, run=run, raw=raw, video=video)
    _append_audit_and_close(
        story,
        styles,
        evidence=evidence,
        run=run,
        audit_events=audit_events,
        generator_notes=notes,
        integrity_status=integrity_status,
        pred=pred,
        confidence_val=confidence_val,
        face_verifications=face_verifications,
        xai_recorded=bool(video.get("xai_available") is True and xai_included),
        final_paragraphs=[
            _assessment_narrative(pred, confidence_val),
            "This report describes the stored 16-frame video analysis. "
            "It does not determine legal admissibility.",
        ],
    )


def _append_video_report(
    story: list[Any],
    styles,
    *,
    report_number: str,
    case: Case,
    evidence: Evidence,
    run: AnalysisRun,
    audit_events: list[AuditLog],
    generator: User,
    notes: str | None,
    face_verifications: list[FaceVerification],
    raw: dict[str, Any],
    thumb_dir: Path | None = None,
) -> None:
    if isinstance(raw.get("video_analysis"), dict):
        _append_lstm_video_report(
            story,
            styles,
            report_number=report_number,
            case=case,
            evidence=evidence,
            run=run,
            generator=generator,
            raw=raw,
            audit_events=audit_events,
            notes=notes,
            face_verifications=face_verifications,
        )
        return
    _append_report_banner(story, styles, report_number=report_number, case=case)
    probs = _video_payload_probs(raw)
    aggregation = raw.get("aggregation") if isinstance(raw.get("aggregation"), dict) else {}
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    frames = raw.get("frames") if isinstance(raw.get("frames"), list) else []
    suspicious = raw.get("suspicious_frames") if isinstance(raw.get("suspicious_frames"), list) else []
    temporal = raw.get("temporal_analysis") if isinstance(raw.get("temporal_analysis"), dict) else None

    story.append(_section_title("Investigation Identification", styles))
    story.append(
        _kv_table(
            [
                ("Investigation ID", str(run.investigation_id or "")),
                ("Report Number", report_number),
                ("Case Number", str(case.case_number)),
                ("Case Title", str(case.title)),
                ("Case Status", str(case.status or "")),
                ("Evidence ID", f"EVD-{evidence.id}"),
                ("Analysis ID", str(run.id)),
                ("Analyst / Investigator", generator.full_name or generator.username),
                ("Generated At (UTC)", _utcnow().isoformat(timespec="seconds")),
            ],
            styles,
        )
    )

    extra_integrity = [
        ("Evidence type", "VIDEO"),
        ("Storage path", str(evidence.storage_path or UNAVAILABLE)),
    ]
    integrity_status = _append_integrity_block(
        story, styles, evidence, extra_rows=extra_integrity
    )

    story.append(PageBreak())
    story.append(_section_title("Video Metadata", styles))
    sampled = metadata.get("extracted_frame_count")
    if sampled is None:
        sampled = aggregation.get("total_frames") or len(frames)
    story.append(
        _kv_table(
            [
                ("Duration", _fmt_seconds(metadata.get("duration_seconds"))),
                ("Resolution", (
                    f"{metadata.get('width')} × {metadata.get('height')}"
                    if metadata.get("width") is not None and metadata.get("height") is not None
                    else UNAVAILABLE
                )),
                ("FPS", _fmt_optional(metadata.get("fps"))),
                ("Container frame count", _fmt_optional(metadata.get("frame_count"))),
                ("Sampling rate", _fmt_optional(metadata.get("sampling_rate_fps"), suffix=" FPS")),
                ("Sampling interval", _fmt_optional(metadata.get("sample_interval_frames"), suffix=" frames")),
                ("Sampled / analyzed frames", str(sampled)),
            ],
            styles,
        )
    )
    story.append(
        Paragraph(
            "These values are copied from the stored Phase 2 extraction record.",
            styles["Caption"],
        )
    )

    story.append(_section_title("Video Authenticity Result", styles))
    confidence_val = float(run.confidence) if run.confidence is not None else None
    if confidence_val is None and probs.get("confidence") is not None:
        try:
            confidence_val = float(probs["confidence"])
        except (TypeError, ValueError):
            confidence_val = None
    pred = str(run.prediction or probs.get("prediction") or "N/A")
    pred_color = colors.HexColor("#106F3A") if pred == "REAL" else colors.HexColor("#8A1430")
    stored_level = probs.get("confidence_level") if isinstance(probs.get("confidence_level"), str) else None
    pred_rows: list[tuple[str, str]] = [
        (
            "Prediction",
            f"<b><font color='#{pred_color.hexval()[2:]}'>{escape(pred)}</font></b>",
        ),
        ("Model Confidence", _fmt_pct(confidence_val)),
        ("Confidence Interpretation", _confidence_interpretation(confidence_val, stored_level)),
        ("Model", str(run.model_name or probs.get("model_name") or "")),
        ("Model Version", str(run.model_version or probs.get("model_version") or "")),
        ("Analysis Status", str(run.status or "")),
        ("Aggregation method", str(aggregation.get("method") or "mean_frame_probability")),
    ]
    real_disp = _fmt_prob(probs.get("real_probability"))
    fake_disp = _fmt_prob(probs.get("fake_probability"))
    if real_disp is not None:
        pred_rows.append(("REAL Class Probability", real_disp))
    if fake_disp is not None:
        pred_rows.append(("FAKE Class Probability", fake_disp))
    threshold = aggregation.get("threshold", probs.get("threshold"))
    if threshold is not None:
        try:
            pred_rows.append(("Decision Threshold", f"{float(threshold):.4f}"))
        except (TypeError, ValueError):
            pass
    if isinstance(probs.get("threshold_decision"), str) and probs.get("threshold_decision"):
        pred_rows.append(("Threshold decision", str(probs["threshold_decision"])))
    if run.started_at:
        pred_rows.append(("Started At", run.started_at.isoformat(timespec="seconds")))
    if run.completed_at:
        pred_rows.append(("Completed At", run.completed_at.isoformat(timespec="seconds")))
    story.append(_kv_table(pred_rows, styles))
    story.append(Paragraph(escape(_VIDEO_AGGREGATION_NOTE), styles["BodyJustified"]))
    story.append(
        Paragraph(
            "Note: Model confidence is the probability assigned to the predicted class. "
            "It must not be read as a percentage of authenticity or as a legal certainty.",
            styles["Caption"],
        )
    )

    story.append(PageBreak())
    story.append(_section_title("Frame Analysis", styles))
    story.append(
        _kv_table(
            [
                ("Total frames analyzed", _fmt_optional(aggregation.get("total_frames_analyzed"))),
                ("REAL frame count", _fmt_optional(aggregation.get("real_frame_count"))),
                ("FAKE frame count", _fmt_optional(aggregation.get("fake_frame_count"))),
                (
                    "FAKE frame percentage",
                    (
                        f"{float(aggregation['fake_frame_percentage']):.2f}%"
                        if isinstance(aggregation.get("fake_frame_percentage"), (int, float))
                        else UNAVAILABLE
                    ),
                ),
                ("Suspicious / FAKE frames", str(len(suspicious))),
            ],
            styles,
        )
    )
    if suspicious:
        labels = []
        for item in suspicious:
            if isinstance(item, dict):
                labels.append(str(item.get("frame_number", "?")))
        if labels:
            story.append(
                Paragraph(
                    "Suspicious/FAKE frame numbers: " + escape(", ".join(labels)),
                    styles["BodyJustified"],
                )
            )
    valid_frames = [item for item in frames if isinstance(item, dict)]
    if valid_frames:
        story.append(_build_frame_table(valid_frames, styles))
        if len(valid_frames) > _MAX_FRAME_TABLE_ROWS:
            story.append(
                Paragraph(
                    f"Table shows the first {_MAX_FRAME_TABLE_ROWS} of {len(valid_frames)} stored frames.",
                    styles["Caption"],
                )
            )
    else:
        story.append(
            Paragraph("No frame-level rows were stored for this analysis.", styles["BodyJustified"])
        )

    xai_numbers = {number for number, _folder in _video_xai_dirs(run)}
    _append_video_frame_gallery(
        story,
        styles,
        valid_frames,
        xai_numbers=xai_numbers,
        thumb_dir=thumb_dir,
    )

    story.append(_section_title("Temporal Face Analysis", styles))
    story.append(Paragraph(escape(_VIDEO_TEMPORAL_LIMIT), styles["BodyJustified"]))
    if not temporal:
        story.append(
            Paragraph(
                "Temporal face analysis was not recorded. Missing optional temporal "
                "output does not prevent report generation.",
                styles["BodyJustified"],
            )
        )
    elif temporal.get("error"):
        story.append(
            _kv_table(
                [
                    ("Status", "Not available"),
                    ("Recorded error", str(temporal.get("error"))),
                    ("Method", str(temporal.get("method") or UNAVAILABLE)),
                ],
                styles,
            )
        )
        story.append(
            Paragraph(
                "Temporal analysis failed or was skipped during the original run. "
                "This report does not re-run detection.",
                styles["Caption"],
            )
        )
    else:
        primary = temporal.get("primary_track") if isinstance(temporal.get("primary_track"), dict) else {}
        gaps = temporal.get("tracking_gaps") if isinstance(temporal.get("tracking_gaps"), list) else []
        temporal_rows = [
            ("Detector", str(temporal.get("detector") or UNAVAILABLE)),
            ("Method", str(temporal.get("method") or UNAVAILABLE)),
            ("Face detection rate", _fmt_optional(temporal.get("face_detection_rate"))),
            ("Frames with face", _fmt_optional(temporal.get("frames_with_face"))),
            ("Track count", _fmt_optional(temporal.get("track_count"))),
            ("Average centroid movement", _fmt_optional(temporal.get("average_centroid_movement"))),
            ("Average bounding-box area change", _fmt_optional(temporal.get("average_bbox_area_change"))),
            ("Tracking gaps", str(len(gaps))),
        ]
        if primary:
            temporal_rows.extend(
                [
                    ("Primary track ID", _fmt_optional(primary.get("track_id"))),
                    ("Primary coverage", _fmt_optional(primary.get("coverage"))),
                    ("Primary frames tracked", _fmt_optional(primary.get("frames_tracked"))),
                    ("Primary start", _fmt_seconds(primary.get("start_timestamp"))),
                    ("Primary end", _fmt_seconds(primary.get("end_timestamp"))),
                ]
            )
        story.append(_kv_table(temporal_rows, styles))
        if gaps:
            story.append(
                Paragraph(
                    "Tracking is fragmented. Gap records below are copied from the stored "
                    "analysis and are not a finding of manipulation.",
                    styles["Caption"],
                )
            )
            gap_lines = []
            for gap in gaps[:12]:
                if not isinstance(gap, dict):
                    continue
                gap_lines.append(
                    f"{_fmt_seconds(gap.get('start_timestamp'))}–{_fmt_seconds(gap.get('end_timestamp'))} "
                    f"({gap.get('missing_frame_count', '?')} missing sampled frames)"
                )
            if gap_lines:
                story.append(Paragraph(escape("; ".join(gap_lines)), styles["BodyJustified"]))
        sampling_note = temporal.get("sampling_note")
        if sampling_note:
            story.append(Paragraph(escape(str(sampling_note)), styles["Caption"]))

    xai_count = _append_video_xai_section(story, styles, run=run, frames=valid_frames)
    _append_face_verification_section(story, styles, face_verifications)

    extra_readiness = [
        ("Evidence type", "VIDEO"),
        ("Aggregation method", str(aggregation.get("method") or "mean_frame_probability")),
        ("Temporal analysis", "Failed" if temporal and temporal.get("error") else ("Recorded" if temporal else "Not recorded")),
        ("XAI frames included", str(xai_count)),
    ]
    _append_audit_and_close(
        story,
        styles,
        evidence=evidence,
        run=run,
        audit_events=audit_events,
        generator_notes=notes,
        integrity_status=integrity_status,
        pred=pred,
        confidence_val=confidence_val,
        face_verifications=face_verifications,
        extra_readiness=extra_readiness,
        final_paragraphs=_video_summary(
            evidence=evidence,
            run=run,
            raw=raw,
            aggregation=aggregation,
            temporal=temporal,
            xai_count=xai_count,
            integrity_status=integrity_status,
            pred=pred,
            confidence_val=confidence_val,
        ),
    )


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
    raw = _extract_raw_payload(run)
    if _report_is_video(evidence, raw):
        thumb_dir = output_path.parent / f".{output_path.stem}_thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        try:
            _append_video_report(
                story,
                styles,
                report_number=report_number,
                case=case,
                evidence=evidence,
                run=run,
                audit_events=audit_events,
                generator=generator,
                notes=notes,
                face_verifications=face_verifications or [],
                raw=raw,
                thumb_dir=thumb_dir,
            )
            doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
        finally:
            shutil.rmtree(thumb_dir, ignore_errors=True)
        return

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
            "The following timeline lists stored audit events for this case only. "
            "Timestamps and event types are taken from the AuditLog without alteration.",
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


def latest_report_for_analysis(analysis_id: int) -> InvestigationReport | None:
    """Newest stored report for an analysis, if one has already been generated."""

    return (
        InvestigationReport.query.filter_by(analysis_id=analysis_id)
        .order_by(InvestigationReport.generated_at.desc(), InvestigationReport.id.desc())
        .first()
    )


def resolve_stored_report_file(report: InvestigationReport) -> Path:
    """Locate the PDF recorded for this report. Client paths are never accepted.

    The same containment rules as the download endpoint apply: the file must
    resolve inside REPORT_DIR. A path that points at some other readable file
    is rejected instead of being opened.
    """

    from flask import current_app

    from backend.app.utils.paths import resolve_within

    report_root = Path(current_app.config["REPORT_DIR"]).resolve()
    root = Path(current_app.config["ROOT_DIR"]).resolve()
    stored = str(report.storage_path or "")
    candidates = [
        resolve_within(report_root, stored),
        resolve_within(root, stored),
    ]
    full_path = next(
        (
            path
            for path in candidates
            if path is not None and path.is_file() and resolve_within(report_root, path)
        ),
        None,
    )
    if full_path is None:
        escaped = (root / stored).resolve()
        inside = (report_root / stored).resolve()
        if escaped.is_file() or inside.is_file():
            raise AuthorizationError("Report path failed safety check")
        raise ReportFileMissingError("The report file is not available.")
    return full_path


def ensure_analysis_report(user: User, analysis_id: int) -> InvestigationReport:
    """Return the current report, generating it once when none exists yet."""

    run = get_analysis(user, analysis_id)
    existing = latest_report_for_analysis(run.id)
    if existing is not None:
        return existing
    return generate_investigation_report(user, analysis_id)


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

    # Only this analysis's case. Matching the investigator's user id as well
    # pulled login and other-case events into every report.
    audit_events = (
        AuditLog.query.filter(AuditLog.case_id == run.case_id)
        .order_by(AuditLog.timestamp.asc())
        .all()
    )

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
    # One new report row produces at most one automatic email. Opening the
    # analysis page, downloading the PDF, or listing reports does not call this.
    from backend.app.services.email_service import deliver_generated_report

    deliver_generated_report(user, report)
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
        "email_delivery": _email_delivery_view(report),
    }


def _email_delivery_view(report: InvestigationReport) -> dict[str, Any]:
    """Safe status for the UI. Never includes SMTP settings or the full address."""

    from backend.app.services.email_service import mask_email

    status = report.email_status
    if status == "sent":
        message = "Report generated and emailed to your registered email address."
    elif status in {"unavailable", "failed"}:
        message = "Report generated successfully, but email delivery is unavailable."
    else:
        message = None
    masked = None
    generator = report.generator
    if status == "sent" and generator is not None and generator.email:
        masked = mask_email(generator.email)
    return {
        "status": status,
        "message": message,
        "masked_recipient": masked,
    }
