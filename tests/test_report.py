"""Tests for ContaminationReport and reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verascan.report import ContaminationReport, MatchRecord, _word_diff_html

# ---------- MatchRecord ------------------------------------------------- #


def test_match_record_creation() -> None:
    m = MatchRecord(
        eval_index=0,
        train_index=1,
        eval_text="hello",
        train_text="hello",
        score=1.0,
        method="exact",
    )
    assert m.score == 1.0
    assert m.method == "exact"


# ---------- word diff ---------------------------------------------------- #


def test_word_diff_identical() -> None:
    html = _word_diff_html("hello world", "hello world")
    assert "<del>" not in html
    assert "<ins>" not in html


def test_word_diff_replacement() -> None:
    html = _word_diff_html("the cat sat", "the dog sat")
    assert "<del>cat</del>" in html
    assert "<ins>dog</ins>" in html


def test_word_diff_insertion() -> None:
    html = _word_diff_html("a b", "a x b")
    assert "<ins>" in html


# ---------- ContaminationReport ------------------------------------------ #


@pytest.fixture()
def sample_report() -> ContaminationReport:
    return ContaminationReport(
        train_size=100,
        eval_size=10,
        methods_used=["exact", "fuzzy"],
        threshold=0.85,
        matches=[
            MatchRecord(0, 5, "eval text A", "train text A", 1.0, "exact"),
            MatchRecord(1, 10, "eval text B", "train text B similar", 0.9, "fuzzy"),
            MatchRecord(2, 20, "eval text C", "train text C alike", 0.87, "fuzzy"),
        ],
    )


def test_contamination_rate(sample_report: ContaminationReport) -> None:
    assert sample_report.contamination_rate == pytest.approx(3 / 10)


def test_contamination_rate_empty() -> None:
    r = ContaminationReport(eval_size=0)
    assert r.contamination_rate == 0.0


def test_flagged_min_score(sample_report: ContaminationReport) -> None:
    high = sample_report.flagged(min_score=0.95)
    assert len(high) == 1
    assert high[0].score == 1.0


def test_counts(sample_report: ContaminationReport) -> None:
    assert sample_report.exact_count == 1
    assert sample_report.fuzzy_count == 2
    assert sample_report.semantic_count == 0


def test_summary_prints(
    sample_report: ContaminationReport, capsys: pytest.CaptureFixture[str]
) -> None:
    text = sample_report.summary()
    assert "30.0%" in text
    captured = capsys.readouterr()
    assert "Verascan" in captured.out


def test_to_dict(sample_report: ContaminationReport) -> None:
    d = sample_report.to_dict()
    assert d["train_size"] == 100
    assert d["total_matches"] == 3
    assert len(d["matches"]) == 3


def test_to_json(sample_report: ContaminationReport, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    sample_report.to_json(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["eval_size"] == 10
    assert len(data["matches"]) == 3


def test_to_html(sample_report: ContaminationReport, tmp_path: Path) -> None:
    out = tmp_path / "report.html"
    sample_report.to_html(str(out))
    html = out.read_text(encoding="utf-8")
    assert "<html" in html
    assert "Verascan" in html
    assert "exact" in html
    assert "fuzzy" in html


def test_empty_report_html(tmp_path: Path) -> None:
    r = ContaminationReport(train_size=50, eval_size=20, methods_used=["exact"])
    out = tmp_path / "empty.html"
    r.to_html(str(out))
    html = out.read_text(encoding="utf-8")
    assert "No contamination detected" in html
    assert "0 of 20 evaluation examples" in html


def test_html_escaping_special_characters(tmp_path: Path) -> None:
    """Verify that code/tags like <script> or <div> are safely escaped in HTML."""
    r = ContaminationReport(
        train_size=5,
        eval_size=1,
        methods_used=["exact"],
        threshold=0.85,
        matches=[
            MatchRecord(
                eval_index=0,
                train_index=0,
                eval_text="<script>alert('xss')</script> & <div>test</div>",
                train_text="<script>alert('train')</script> & <div>test</div>",
                score=1.0,
                method="exact",
            )
        ],
    )
    out = tmp_path / "escaped.html"
    r.to_html(str(out))
    html_content = out.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in html_content
    assert "&lt;div&gt;" in html_content
    assert "<script>alert" not in html_content


def test_html_interactive_elements(sample_report: ContaminationReport, tmp_path: Path) -> None:
    """Verify interactive UI elements and buttons exist in generated HTML."""
    out = tmp_path / "interactive.html"
    sample_report.to_html(str(out))
    html_content = out.read_text(encoding="utf-8")
    assert "searchInput" in html_content
    assert "filterMethod" in html_content
    assert "toggleAllDetails" in html_content
    assert "progress-bar" in html_content
