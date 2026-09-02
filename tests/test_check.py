"""Tests for the main verascan.check() function."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import verascan
from verascan.report import ContaminationReport


def test_check_with_lists() -> None:
    train = ["The cat sat on the mat.", "A totally different sentence."]
    eval_ = ["The cat sat on the mat.", "Something else entirely."]

    report = verascan.check(train, eval_, show_progress=False)
    assert isinstance(report, ContaminationReport)
    assert report.train_size == 2
    assert report.eval_size == 2
    assert report.exact_count == 1
    assert report.contamination_rate == 0.5


def test_check_with_dataframe() -> None:
    train_df = pd.DataFrame({"text": ["hello world", "foo bar"]})
    eval_df = pd.DataFrame({"text": ["hello world", "baz qux"]})

    report = verascan.check(train_df, eval_df, show_progress=False)
    assert report.exact_count == 1
    assert report.contamination_rate == 0.5


def test_check_with_jsonl(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    eval_file = tmp_path / "eval.jsonl"

    train_file.write_text(
        json.dumps({"text": "example A"}) + "\n" + json.dumps({"text": "example B"}) + "\n",
        encoding="utf-8",
    )
    eval_file.write_text(
        json.dumps({"text": "example A"}) + "\n" + json.dumps({"text": "example C"}) + "\n",
        encoding="utf-8",
    )

    report = verascan.check(str(train_file), str(eval_file), show_progress=False)
    assert report.exact_count == 1
    assert report.contamination_rate == 0.5


def test_check_fuzzy_only() -> None:
    train = ["The quick brown fox jumps over the lazy dog."]
    eval_ = ["The quick brown fox jumps over a lazy dog."]

    report = verascan.check(
        train,
        eval_,
        methods=["fuzzy"],
        threshold=0.6,
        show_progress=False,
    )
    assert report.exact_count == 0
    assert report.fuzzy_count == 1
    assert report.contamination_rate == 1.0


def test_check_exact_and_fuzzy_no_double_count() -> None:
    """An exact match should NOT also be reported as a fuzzy match."""
    train = ["Identical sentence here."]
    eval_ = ["Identical sentence here."]

    report = verascan.check(
        train,
        eval_,
        methods=["exact", "fuzzy"],
        threshold=0.5,
        show_progress=False,
    )
    assert report.exact_count == 1
    assert report.fuzzy_count == 0  # excluded because already matched
    assert report.contamination_rate == 1.0


def test_check_invalid_method() -> None:
    with pytest.raises(ValueError, match="Unknown method 'invalid'"):
        verascan.check(["a"], ["b"], methods=["invalid"])  # type: ignore[list-item]


def test_check_custom_column() -> None:
    train_df = pd.DataFrame({"content": ["alpha", "beta"]})
    eval_df = pd.DataFrame({"content": ["alpha", "gamma"]})

    report = verascan.check(train_df, eval_df, column="content", show_progress=False)
    assert report.exact_count == 1


def test_full_pipeline_html_json(tmp_path: Path) -> None:
    train = ["Sentence one.", "Sentence two.", "Sentence three."]
    eval_ = ["Sentence one.", "Sentence two modified slightly."]

    report = verascan.check(
        train,
        eval_,
        methods=["exact", "fuzzy"],
        threshold=0.6,
        show_progress=False,
    )
    html_path = tmp_path / "report.html"
    json_path = tmp_path / "report.json"

    report.to_html(str(html_path))
    report.to_json(str(json_path))

    assert html_path.exists()
    assert json_path.exists()
    assert "Verascan" in html_path.read_text(encoding="utf-8")


def test_check_ngram_only() -> None:
    """methods=["ngram"] finds shared 13-grams end-to-end via verascan.check."""
    shared = " ".join(f"tok{i}" for i in range(13))
    train = [shared + " train-tail"]
    eval_ = [shared + " eval-tail", "totally different short text"]

    report = verascan.check(
        train,
        eval_,
        methods=["ngram"],
        show_progress=False,
    )
    assert report.exact_count == 0
    assert report.fuzzy_count == 0
    assert report.ngram_count == 1
    assert report.contamination_rate == 0.5
    assert report.matches[0].method == "ngram"
