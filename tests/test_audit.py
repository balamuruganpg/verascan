"""Tests for the verascan.audit() benchmark contamination feature."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import verascan
from verascan.audit import AuditReport
from verascan.benchmarks import SYNTHETIC_BENCHMARKS, normalize_benchmarks


def test_normalize_benchmarks_valid():
    assert normalize_benchmarks(["mmlu", "GSM8K"]) == ["mmlu", "gsm8k"]
    assert normalize_benchmarks("humaneval, mmlu") == ["humaneval", "mmlu"]
    assert normalize_benchmarks(["mmlu", "mmlu"]) == ["mmlu"]


def test_audit_unknown_benchmark():
    with pytest.raises(ValueError, match="Unknown benchmark 'unknown_benchmark'"):
        verascan.audit(train=["Hello world"], benchmarks=["unknown_benchmark"])

    with pytest.raises(ValueError, match="At least one benchmark must be specified"):
        verascan.audit(train=["Hello world"], benchmarks=[])


def test_audit_missing_datasets_extra():
    with patch("verascan.benchmarks.check_datasets_available") as mock_check:
        mock_check.side_effect = ImportError(
            "Hugging Face 'datasets' is required for benchmark presets. "
            "Install with: pip install verascan[hf]"
        )
        with pytest.raises(ImportError, match="pip install verascan\\[hf\\]"):
            verascan.audit(train=["Hello world"], benchmarks=["gsm8k"], synthetic=False)


def test_audit_synthetic_stand_in_direct():
    train = ["This is completely unrelated training text about astronomy and stars."]
    report = verascan.audit(
        train=train,
        benchmarks=["gsm8k", "humaneval"],
        synthetic=True,
        methods=["exact"],
    )

    assert isinstance(report, AuditReport)
    assert report.train_size == 1
    assert report.eval_size == len(SYNTHETIC_BENCHMARKS["gsm8k"]) + len(
        SYNTHETIC_BENCHMARKS["humaneval"]
    )
    assert report.total_matches == 0
    assert report.contamination_rate == 0.0
    assert report.benchmark_counts == {"gsm8k": 0, "humaneval": 0}


def test_audit_hf_download_unavailable_with_allow_synthetic(caplog):
    with (
        patch("verascan.benchmarks.check_datasets_available"),
        patch(
            "verascan.benchmarks._load_hf_gsm8k",
            side_effect=ConnectionError("Simulated network outage"),
        ),
    ):
        # When allow_synthetic=False, raises RuntimeError
        with pytest.raises(RuntimeError, match="Failed to load benchmark 'gsm8k'"):
            verascan.audit(
                train=["Some train text"],
                benchmarks=["gsm8k"],
                synthetic=False,
                allow_synthetic=False,
            )

        # When allow_synthetic=True, falls back to synthetic stand-in with warning
        with caplog.at_level(logging.WARNING):
            report = verascan.audit(
                train=["Some train text"],
                benchmarks=["gsm8k"],
                synthetic=False,
                allow_synthetic=True,
            )
            assert report.eval_size == len(SYNTHETIC_BENCHMARKS["gsm8k"])
            assert "Using synthetic stand-in" in caplog.text


def test_audit_missing_datasets_with_allow_synthetic(caplog):
    with (
        patch(
            "verascan.benchmarks.check_datasets_available",
            side_effect=ImportError("Hugging Face 'datasets' is required"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        report = verascan.audit(
            train=["Some train text"],
            benchmarks=["gsm8k"],
            synthetic=False,
            allow_synthetic=True,
        )
        assert report.eval_size == len(SYNTHETIC_BENCHMARKS["gsm8k"])
        assert "Using synthetic stand-in" in caplog.text


def test_audit_finds_exact_planted_leak_gsm8k():
    planted_question = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    train = [
        "Unrelated text on biology.",
        planted_question,
        "Another clean text about history.",
    ]

    report = verascan.audit(
        train=train,
        benchmarks=["gsm8k"],
        synthetic=True,
        methods=["exact"],
    )

    assert report.exact_count == 1
    assert report.total_matches == 1
    assert report.contamination_rate > 0.0
    assert report.benchmark_counts["gsm8k"] == 1

    match = report.matches[0]
    assert match.method == "exact"
    assert match.train_index == 1
    assert match.eval_text == planted_question


def test_audit_finds_planted_leaks_humaneval_and_mmlu():
    planted_code = SYNTHETIC_BENCHMARKS["humaneval"][0]["prompt"]
    planted_mmlu = SYNTHETIC_BENCHMARKS["mmlu"][0]["question"]

    train = [
        planted_code,
        "Clean training example 1.",
        planted_mmlu,
        "Clean training example 2.",
    ]

    report = verascan.audit(
        train=train,
        benchmarks=["humaneval", "mmlu"],
        synthetic=True,
        methods=["exact"],
    )

    assert report.total_matches == 2
    assert report.benchmark_counts["humaneval"] == 1
    assert report.benchmark_counts["mmlu"] == 1

    by_bm = report.by_benchmark()
    assert len(by_bm["humaneval"]) == 1
    assert len(by_bm["mmlu"]) == 1
    assert by_bm["humaneval"][0].train_text == planted_code
    assert by_bm["mmlu"][0].train_text == planted_mmlu


def test_audit_fuzzy_and_ngram_planted_leak():
    base = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    near_dup = base.replace("breakfast every morning", "breakfast in the morning")
    train = [near_dup]

    report = verascan.audit(
        train=train,
        benchmarks=["gsm8k"],
        synthetic=True,
        methods=["fuzzy"],
        threshold=0.80,
    )

    assert report.fuzzy_count >= 1
    assert report.benchmark_counts["gsm8k"] >= 1


def test_audit_custom_train_column():
    planted_question = SYNTHETIC_BENCHMARKS["gsm8k"][1]["question"]
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "instruction": ["Clean prompt", planted_question],
        }
    )

    report = verascan.audit(
        train=df,
        benchmarks=["gsm8k"],
        column="instruction",
        synthetic=True,
        methods=["exact"],
    )

    assert report.total_matches == 1
    assert report.matches[0].train_index == 1


def test_audit_report_methods_and_exports(tmp_path: Path):
    planted = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    train = [planted]

    report = verascan.audit(
        train=train,
        benchmarks=["gsm8k", "mmlu"],
        synthetic=True,
        methods=["exact"],
    )

    # 1. Summary
    summary_text = report.summary()
    assert "Verascan Benchmark Audit Report" in summary_text
    assert "gsm8k" in summary_text
    assert "mmlu" in summary_text
    assert "1 contaminated" in summary_text

    # 2. to_dict
    data = report.to_dict()
    assert data["benchmarks"] == ["gsm8k", "mmlu"]
    assert data["benchmark_counts"] == {"gsm8k": 1, "mmlu": 0}
    assert data["matches"][0]["benchmark"] == "gsm8k"

    # 3. to_json
    json_path = tmp_path / "audit_report.json"
    report.to_json(str(json_path))
    assert json_path.exists()

    # 4. to_html
    html_path = tmp_path / "audit_report.html"
    report.to_html(str(html_path))
    assert html_path.exists()
    html_content = html_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content


def test_audit_ngram_planted_leak():
    # Long shared phrase (13-gram overlap) from GSM8K problem
    base = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    # Append some prefix and suffix to test ngram detection
    train = [f"In a math textbook problem: {base} Solve carefully."]

    report = verascan.audit(
        train=train,
        benchmarks=["gsm8k"],
        synthetic=True,
        methods=["ngram"],
        ngram_n=8,
    )

    assert report.ngram_count >= 1
    assert report.benchmark_counts["gsm8k"] >= 1


def test_audit_semantic_planted_leak():
    from verascan.engines.semantic import is_available

    if not is_available():
        pytest.skip("Semantic dependencies not installed")

    # Paraphrase of MMLU question 1
    paraphrase = (
        "In eukaryotic organisms, what is the primary role and function "
        "carried out by the cellular ribosome?"
    )
    train = [paraphrase]

    report = verascan.audit(
        train=train,
        benchmarks=["mmlu"],
        synthetic=True,
        methods=["semantic"],
        threshold=0.75,
    )

    assert report.semantic_count >= 1
    assert report.benchmark_counts["mmlu"] >= 1


def test_audit_train_file_path(tmp_path: Path):
    import json

    planted = SYNTHETIC_BENCHMARKS["humaneval"][0]["prompt"]
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"text": planted}) + "\n", encoding="utf-8")

    report = verascan.audit(
        train=str(train_file),
        benchmarks=["humaneval"],
        synthetic=True,
        methods=["exact"],
    )

    assert report.total_matches == 1
    assert report.benchmark_counts["humaneval"] == 1


def test_audit_cleaned_eval():
    planted = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    train = [planted]

    report = verascan.audit(
        train=train,
        benchmarks=["gsm8k"],
        synthetic=True,
        methods=["exact"],
    )

    cleaned = report.cleaned_eval()
    assert isinstance(cleaned, pd.DataFrame)
    # Original 5 synthetic examples minus 1 contaminated example
    assert len(cleaned) == 4
    # The planted question should no longer be in the cleaned eval
    assert planted not in cleaned["text"].tolist()
