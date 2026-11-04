"""PDF reports preserve summary, match data and long word differences."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader

from verascan import AuditReport, ContaminationReport, MatchRecord


def test_pdf_audit_summary_table_and_highlighted_diff(tmp_path: Path) -> None:
    report = AuditReport(
        train_size=20,
        eval_size=4,
        methods_used=["semantic"],
        benchmarks=["gsm8k"],
        eval_records=[{"benchmark": "gsm8k"}] * 4,
        matches=[MatchRecord(0, 3, "The quick green fox", "The quick brown fox", 0.95, "semantic")],
    )
    path = tmp_path / "audit.pdf"
    report.to_pdf(str(path))
    assert path.stat().st_size > 1000
    reader = PdfReader(path)
    assert len(reader.pages) >= 2
    text = "\n".join(page.extract_text() for page in reader.pages)
    for expected in ("25.0%", "20", "4", "gsm8k", "semantic", "0.9500", "brown", "green"):
        assert expected in text
    assert reader.metadata.title == "Verascan contamination report"
    cover = reader.pages[0].extract_text()
    for expected in (
        "AUDIT REPORT",
        "25.0%",
        "Training examples",
        "Evaluation examples",
        "gsm8k",
        "semantic",
    ):
        assert expected in cover


def test_pdf_empty_report(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    ContaminationReport().to_pdf(str(path))
    text = " ".join(p.extract_text() for p in PdfReader(path).pages)
    assert "0.0%" in text and "No matches" in text


def test_pdf_paginates_long_text_and_match_table(tmp_path: Path) -> None:
    report = ContaminationReport(
        train_size=60,
        eval_size=60,
        methods_used=["exact"],
        matches=[
            MatchRecord(
                i, i, f"eval-{i} " + "word " * 350, f"train-{i} " + "word " * 350, 1.0, "exact"
            )
            for i in range(60)
        ],
    )
    path = tmp_path / "long.pdf"
    report.to_pdf(str(path))
    pages = PdfReader(path).pages
    assert len(pages) > 10
    assert "eval-59" in " ".join(p.extract_text() for p in pages)


def test_pdf_escapes_markup_and_preserves_unicode(tmp_path: Path) -> None:
    text = '<script>alert("café")</script> & naïve <br/> αβ'
    report = ContaminationReport(
        train_size=1,
        eval_size=1,
        matches=[MatchRecord(0, 0, text, text, 1.0, "exact")],
    )
    path = tmp_path / "safe.pdf"
    report.to_pdf(str(path))
    content = " ".join(p.extract_text() for p in PdfReader(path).pages)
    assert '<script>alert("café")</script>' in content
    assert "αβ" in content and "naïve" in content


@pytest.mark.parametrize("command", ["check", "audit"])
def test_cpu_cli_pdf_output(tmp_path: Path, command: str) -> None:
    from typer.testing import CliRunner

    from verascan.cli import app

    train = tmp_path / "train.jsonl"
    train.write_text('{"text":"hello world"}\n', encoding="utf-8")
    path = tmp_path / "report.pdf"
    args = [command, "--train", str(train), "--output", str(path), "--no-progress"]
    args += ["--eval", str(train)] if command == "check" else ["--synthetic"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert path.read_bytes().startswith(b"%PDF-")
