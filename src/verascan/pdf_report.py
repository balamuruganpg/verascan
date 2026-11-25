"""Paginated PDF reports, independent of the legacy HTML template."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)

if TYPE_CHECKING:
    from verascan.report import ContaminationReport


def _glyph_map() -> dict[int, int]:
    if "Verascan" not in pdfmetrics.getRegisteredFontNames():
        fonts = Path(__file__).parent / "fonts"
        pdfmetrics.registerFont(TTFont("Verascan", str(fonts / "DejaVuSans.ttf")))
    return pdfmetrics.getFont("Verascan").face.charToGlyph


def _show_text(text: str) -> str:
    """Keep characters the PDF font can draw, and name the ones it cannot.

    ReportLab emits a non-BMP code point as an odd-length hex string, which
    readers drop. A cmap hit is not enough for those characters.
    """
    glyphs = _glyph_map()
    parts: list[str] = []
    for char in text:
        code = ord(char)
        drawable = char == "\n" or (
            code <= 0xFFFF and code in glyphs and glyphs[code] != 0
        )
        parts.append(char if drawable else f"[U+{code:04X} omitted]")
    return "".join(parts)


def _safe(text: str) -> str:
    # Name controls the font cannot draw. Stripping them first turned a NUL
    # into the surrounding letters and left no omission note.
    visible = _show_text(text)
    return html.escape(visible).replace("\n", "<br/>")


def _diff(train: str, eval_text: str) -> str:
    from verascan.report import _word_diff_opcodes

    # ``str.split`` drops tab and vertical tab before a word is drawn.
    # Naming those controls first keeps the same ``[U+XXXX omitted]`` mark
    # ``_show_text`` uses for an emoji the font cannot draw.
    train_words, eval_words, opcodes = _word_diff_opcodes(_show_text(train), _show_text(eval_text))
    parts: list[str] = []
    for tag, i, j, k, end in opcodes:
        if tag == "equal":
            parts.append(_safe(" ".join(train_words[i:j])))
        if tag in {"delete", "replace"}:
            parts.append(
                f'<font color="#9f1239" backColor="#ffe4e6"><strike>{_safe(" ".join(train_words[i:j]))}</strike></font>'
            )
        if tag in {"insert", "replace"}:
            parts.append(
                f'<font color="#166534" backColor="#dcfce7">{_safe(" ".join(eval_words[k:end]))}</font>'
            )
    return " ".join(parts) or "(empty text)"


def write_pdf(report: ContaminationReport, path: str) -> None:
    """Write a summary cover, match table and full word-level differences."""
    from verascan.report import format_match_score
    # Bundled DejaVu fonts cover Latin, Greek and Cyrillic without system fonts.
    fonts = Path(__file__).parent / "fonts"
    for name, filename in (
        ("Verascan", "DejaVuSans.ttf"),
        ("Verascan-Bold", "DejaVuSans-Bold.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "Verascan",
        normal="Verascan",
        bold="Verascan-Bold",
        italic="Verascan",
        boldItalic="Verascan-Bold",
    )
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "Verascan"
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            fontName="Verascan-Bold",
            fontSize=25,
            leading=32,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            "Rate",
            fontName="Verascan-Bold",
            fontSize=42,
            leading=52,
            textColor=colors.HexColor("#72e5bd"),
        )
    )
    styles.add(
        ParagraphStyle(
            "Diff", fontName="Verascan", fontSize=9, leading=15, spaceAfter=16, splitLongWords=True
        )
    )
    styles.add(ParagraphStyle("Cell", fontName="Verascan", fontSize=8, leading=11))
    styles.add(
        ParagraphStyle(
            "CoverLabel",
            fontName="Verascan-Bold",
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#72e5bd"),
        )
    )
    styles.add(
        ParagraphStyle(
            "CoverNote",
            fontName="Verascan",
            fontSize=9,
            leading=15,
            textColor=colors.HexColor("#bccbdb"),
        )
    )
    styles.add(
        ParagraphStyle(
            "TileValue",
            fontName="Verascan-Bold",
            fontSize=22,
            leading=32,
            textColor=colors.HexColor("#15263a"),
        )
    )
    styles.add(
        ParagraphStyle(
            "TileLabel",
            fontName="Verascan",
            fontSize=8,
            leading=13,
            textColor=colors.HexColor("#53677e"),
        )
    )
    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=44,
        leftMargin=44,
        topMargin=54,
        bottomMargin=48,
        title="Verascan contamination report",
        author="Verascan",
    )
    data = report.to_dict()
    flagged = len(report.contaminated_indices())
    category = report.contamination_status
    status = {"Clean": "CLEAN", "Low": "LOW OVERLAP", "High": "HIGH OVERLAP"}[category]
    rate_style = styles["Rate"]
    rate_label = report.formatted_contamination_rate
    rate_column = doc.width * 0.44
    left_padding, right_padding = 22, 18
    usable_width = rate_column - left_padding - right_padding
    text_width = pdfmetrics.stringWidth(rate_label, rate_style.fontName, rate_style.fontSize)
    if text_width >= usable_width:
        # Leave room for rounding in ReportLab's paragraph layout.
        scale = (usable_width - 1) / text_width
        rate_style.fontSize *= scale
        rate_style.leading *= scale
    if flagged:
        rate_style.textColor = colors.HexColor("#ffb4c4" if category == "High" else "#f8c978")
    cover = LongTable(
        [
            [
                [
                    Paragraph("AUDIT REPORT", styles["CoverLabel"]),
                    Spacer(1, 15),
                    Paragraph("Contamination<br/>assessment", styles["CoverTitle"]),
                    Spacer(1, 18),
                    Paragraph(
                        "A review of overlap between training data and evaluation examples.",
                        styles["CoverNote"],
                    ),
                ],
                [
                    Paragraph(status, styles["CoverLabel"]),
                    Spacer(1, 8),
                    Paragraph(_safe(rate_label), rate_style),
                    Spacer(1, 8),
                    Paragraph(
                        f"{flagged:,} / {report.eval_size:,} evaluation examples flagged"
                        if report.eval_size else "No evaluation examples were available.",
                        styles["CoverNote"],
                    ),
                ],
            ]
        ],
        colWidths=[doc.width - rate_column, rate_column],
        cornerRadii=[12, 12, 12, 12],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#101e2c")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), left_padding),
                ("RIGHTPADDING", (0, 0), (-1, -1), right_padding),
                ("TOPPADDING", (0, 0), (-1, -1), 30),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 30),
            ]
        )
    )
    metrics = LongTable(
        [
            [
                [Paragraph(value, styles["TileValue"]), Paragraph(label, styles["TileLabel"])]
                for label, value in (
                    ("Training examples", f"{report.train_size:,}"),
                    ("Evaluation examples", f"{report.eval_size:,}"),
                    ("Matching pairs", f"{len(report.matches):,}"),
                )
            ]
        ],
        colWidths=[doc.width / 3] * 3,
    )
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#edf3f7")),
                ("TOPPADDING", (0, 0), (-1, -1), 17),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 17),
                ("LEFTPADDING", (0, 0), (-1, -1), 18),
            ]
        )
    )
    story: list[Any] = [
        Paragraph("VERASCAN / DATA INTEGRITY", styles["Heading3"]),
        Spacer(1, 16),
        cover,
        Spacer(1, 18),
        metrics,
        Spacer(1, 24),
        Paragraph("Audit configuration", styles["Heading2"]),
    ]
    summary = [
        ("Methods", ", ".join(report.methods_used) or "None"),
        ("Benchmarks", ", ".join(data.get("benchmarks", [])) or "Custom evaluation set"),
        *report.matching_criteria,
        ("Generated (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
    ]
    table = LongTable(
        [
            [Paragraph(_safe(k), styles["Normal"]), Paragraph(_safe(v), styles["Normal"])]
            for k, v in summary
        ],
        colWidths=[175, doc.width - 175],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f1f5f9"), colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 18),
            Paragraph(
                "The rate counts unique evaluation examples, not matching pairs. "
                "Similarity identifies potential leakage and should be reviewed in context.",
                styles["Normal"],
            ),
            PageBreak(),
            Paragraph("Match index", styles["Heading1"]),
        ]
    )
    if not report.matches:
        story.append(Paragraph("No matches met the detection criteria.", styles["Normal"]))
    else:
        rows = [
            [
                Paragraph(label, styles["Cell"])
                for label in ("Pair", "Eval row", "Train row", "Method", "Benchmark", "Score")
            ]
        ]
        for i, match in enumerate(data["matches"], 1):
            rows.append(
                [
                    Paragraph(_safe(str(value)), styles["Cell"])
                    for value in (
                        i,
                        match["eval_index"],
                        match["train_index"],
                        match["method"],
                        match.get("benchmark", "-"),
                        format_match_score(match["score"]),
                    )
                ]
            )
        table = LongTable(rows, repeatRows=1, colWidths=[36, 57, 63, 80, doc.width - 290, 54])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend(
            [
                table,
                PageBreak(),
                Paragraph("Word-level differences", styles["Heading1"]),
                Paragraph(
                    "Red strikethrough: training-only words. Green: evaluation-only words. "
                    "Unmarked words appear in both texts.",
                    styles["Normal"],
                ),
                Spacer(1, 16),
            ]
        )
        for i, match in enumerate(data["matches"], 1):
            label = (
                f"Pair {i} | Eval {match['eval_index']} / Train {match['train_index']} | "
                f"{match['method']} | {match.get('benchmark', 'custom')} | "
                f"{format_match_score(match['score'])}"
            )
            story.append(Paragraph(_safe(label), styles["Heading3"]))
            story.append(Paragraph(_diff(match["train_text"], match["eval_text"]), styles["Diff"]))

    def page_decoration(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(44, 38, A4[0] - 44, 38)
        canvas.setFont("Verascan", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(44, 25, f"Verascan {data['version']} | Contamination audit")
        canvas.drawRightString(A4[0] - 44, 25, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
