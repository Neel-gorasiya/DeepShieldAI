import io
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from utils.security import escape_reportlab_text


def generate_pdf_report(record: dict) -> bytes:
    """
    Generates a secure, cleanly styled PDF verification report in memory.
    All user-supplied and dynamic values are XML-escaped to prevent injection and crashes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=2,
    )

    header_sub_style = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
    )

    section_heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    body_bold_style = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
    )

    mono_style = ParagraphStyle(
        "Mono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
    )

    story = []

    # Title & Branding
    story.append(Paragraph("DeepShield AI", header_title_style))
    story.append(Paragraph("AI-Powered Deepfake Detection &amp; Forensic Media Verification", header_sub_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    # Parse metadata
    metadata = {}
    if record.get("metadata_json"):
        try:
            metadata = json.loads(record["metadata_json"])
        except Exception:
            metadata = {}

    sha256_raw = record.get("sha256") or metadata.get("sha256") or "N/A"
    processing_time_raw = record.get("processing_time") or metadata.get("processing_time") or "N/A"
    model_name_raw = record.get("model_name") or metadata.get("model_name") or "Multi-Modal Neural Detector"

    file_size_str = "N/A"
    if record.get("file_size"):
        sz = record["file_size"]
        file_size_str = f"{round(sz / 1024, 1)} KB ({sz:,} bytes)" if sz >= 1024 else f"{sz} bytes"

    # Escape dynamic values for ReportLab XML flowables
    safe_filename = escape_reportlab_text(record.get("filename", "N/A"))
    safe_media_type = escape_reportlab_text(str(record.get("media_type", "N/A")).capitalize())
    safe_sha256 = escape_reportlab_text(sha256_raw)
    safe_model_name = escape_reportlab_text(model_name_raw)
    safe_proc_time = escape_reportlab_text(processing_time_raw)
    safe_created_at = escape_reportlab_text(str(record.get("created_at", "N/A")))
    safe_verdict = escape_reportlab_text(str(record.get("prediction", "Unknown Verdict")).upper())

    # Verdict Highlight Box
    category = record.get("verdict_category", "alert")
    verdict_bg = colors.HexColor("#f1f5f9")
    verdict_text_color = colors.HexColor("#1e293b")
    if category == "ai":
        verdict_bg = colors.HexColor("#fee2e2")
        verdict_text_color = colors.HexColor("#991b1b")
    elif category == "human":
        verdict_bg = colors.HexColor("#dcfce7")
        verdict_text_color = colors.HexColor("#166534")
    elif category == "alert":
        verdict_bg = colors.HexColor("#fef3c7")
        verdict_text_color = colors.HexColor("#92400e")

    verdict_para = Paragraph(
        f"<b>ANALYSIS VERDICT:</b> {safe_verdict}",
        ParagraphStyle("VerdictText", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=11, leading=15, textColor=verdict_text_color)
    )

    verdict_table = Table([[verdict_para]], colWidths=[522])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), verdict_bg),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 10))

    # General Information Table
    story.append(Paragraph("1. Audit &amp; Artifact Information", section_heading_style))
    audit_data = [
        [Paragraph("Verification Audit ID", body_bold_style), Paragraph(f"#{record.get('id', 0):04d}", body_style)],
        [Paragraph("Analysis Date / Time", body_bold_style), Paragraph(safe_created_at, body_style)],
        [Paragraph("Artifact Filename", body_bold_style), Paragraph(safe_filename, body_style)],
        [Paragraph("Media Modality", body_bold_style), Paragraph(safe_media_type, body_style)],
        [Paragraph("File Payload Size", body_bold_style), Paragraph(file_size_str, body_style)],
        [Paragraph("SHA-256 Checksum", body_bold_style), Paragraph(safe_sha256, mono_style)],
    ]
    t1 = Table(audit_data, colWidths=[150, 372])
    t1.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 8))

    # Model & Probability Metrics
    story.append(Paragraph("2. Model Architecture &amp; Probability Metrics", section_heading_style))
    ai_score = f"{record.get('ai_score', 0.0)}%"
    hu_score = f"{record.get('human_score', 0.0)}%"

    metrics_data = [
        [Paragraph("Detection Model", body_bold_style), Paragraph(safe_model_name, body_style)],
        [Paragraph("Inference Duration", body_bold_style), Paragraph(safe_proc_time, body_style)],
        [Paragraph("AI / Synthetic Probability", body_bold_style), Paragraph(ai_score, body_bold_style)],
        [Paragraph("Human Authenticity Probability", body_bold_style), Paragraph(hu_score, body_bold_style)],
    ]
    t2 = Table(metrics_data, colWidths=[150, 372])
    t2.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 8))

    # Forensic Explanation & Findings
    story.append(Paragraph("3. Forensic Findings &amp; Diagnostics", section_heading_style))
    raw_reason = record.get("reason") or "No diagnostic telemetry provided."
    safe_reason = escape_reportlab_text(raw_reason)
    story.append(Paragraph(safe_reason, body_style))
    story.append(Spacer(1, 6))

    # Additional media metadata if present
    details = metadata.get("details", {})
    if details and isinstance(details, dict):
        detail_rows = []
        for k, v in details.items():
            formatted_key = escape_reportlab_text(str(k).replace("_", " ").title())
            formatted_val = escape_reportlab_text(str(v))
            detail_rows.append([Paragraph(formatted_key, body_bold_style), Paragraph(formatted_val, body_style)])

        if detail_rows:
            t3 = Table(detail_rows, colWidths=[150, 372])
            t3.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f8fafc")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t3)
            story.append(Spacer(1, 8))

    # Disclaimer & Integrity
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#e2e8f0"), spaceAfter=8))
    disclaimer_text = (
        "<b>Notice &amp; Disclaimer:</b> DeepShield AI generates probabilistic authenticity assessments using multi-modal "
        "neural feature matching and signal processing. Forensic indicators provide statistical evidence rather than absolute proof. "
        "Results should be interpreted within a comprehensive verification methodology."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
