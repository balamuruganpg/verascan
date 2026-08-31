"""Tests for cleaned evaluation-set export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import verascan
from verascan.report import ContaminationReport, MatchRecord


def test_cleaned_eval_drops_contaminated_list_rows() -> None:
    train = ["hello world", "other train"]
    eval_ = ["hello world", "clean example", "hello world", "another clean"]
    report = verascan.check(train, eval_, methods=["exact"], show_progress=False)
    cleaned = report.cleaned_eval()
    assert cleaned == ["clean example", "another clean"]
    assert isinstance(cleaned, list)
    assert report.contaminated_eval() == ["hello world", "hello world"]


def test_cleaned_eval_preserves_dataframe_columns() -> None:
    train = pd.DataFrame({"text": ["leak me"]})
    eval_df = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "text": ["leak me", "keep me", "also keep"],
            "label": ["pos", "neg", "neu"],
        }
    )
    report = verascan.check(train, eval_df, methods=["exact"], show_progress=False)
    cleaned = report.cleaned_eval()
    assert isinstance(cleaned, pd.DataFrame)
    assert list(cleaned.columns) == ["id", "text", "label"]
    assert list(cleaned["text"]) == ["keep me", "also keep"]
    assert list(cleaned["id"]) == [20, 30]
    assert list(cleaned["label"]) == ["neg", "neu"]


def test_cleaned_eval_empty_contamination_equals_original() -> None:
    train = ["alpha"]
    eval_ = ["beta", "gamma"]
    report = verascan.check(train, eval_, methods=["exact"], show_progress=False)
    assert report.matches == []
    assert report.cleaned_eval() == eval_


def test_cleaned_eval_all_contaminated_is_empty() -> None:
    train = ["x", "y"]
    eval_ = ["x", "y"]
    report = verascan.check(train, eval_, methods=["exact"], show_progress=False)
    assert report.cleaned_eval() == []
    assert report.contaminated_eval() == eval_


def test_to_cleaned_jsonl_roundtrip(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    train_path = tmp_path / "train.jsonl"
    eval_rows = [
        {"text": "duplicate line", "id": 1},
        {"text": "unique eval", "id": 2},
        {"text": "also unique", "id": 3},
    ]
    train_path.write_text(json.dumps({"text": "duplicate line"}) + "\n", encoding="utf-8")
    eval_path.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n", encoding="utf-8")

    report = verascan.check(str(train_path), str(eval_path), methods=["exact"], show_progress=False)
    out = tmp_path / "cleaned.jsonl"
    report.to_cleaned(str(out))

    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert loaded == [{"text": "unique eval", "id": 2}, {"text": "also unique", "id": 3}]


def test_to_cleaned_csv_roundtrip(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.csv"
    train_path = tmp_path / "train.csv"
    pd.DataFrame({"text": ["same", "keep"], "tag": ["a", "b"]}).to_csv(eval_path, index=False)
    pd.DataFrame({"text": ["same"]}).to_csv(train_path, index=False)

    report = verascan.check(str(train_path), str(eval_path), methods=["exact"], show_progress=False)
    out = tmp_path / "cleaned.csv"
    report.to_cleaned(str(out))

    cleaned = pd.read_csv(out)
    assert list(cleaned.columns) == ["text", "tag"]
    assert list(cleaned["text"]) == ["keep"]
    assert list(cleaned["tag"]) == ["b"]


def test_to_cleaned_list_uses_column_name(tmp_path: Path) -> None:
    report = verascan.check(
        ["dup"],
        ["dup", "ok"],
        methods=["exact"],
        column="prompt",
        show_progress=False,
    )
    out = tmp_path / "cleaned.jsonl"
    report.to_cleaned(str(out))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert rows == [{"prompt": "ok"}]


def test_to_contaminated_jsonl(tmp_path: Path) -> None:
    report = verascan.check(["dup"], ["dup", "ok"], methods=["exact"], show_progress=False)
    out = tmp_path / "contaminated.jsonl"
    report.to_contaminated(str(out))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert rows == [{"text": "dup"}]


def test_to_cleaned_rejects_bad_extension(tmp_path: Path) -> None:
    report = verascan.check(["a"], ["b"], methods=["exact"], show_progress=False)
    with pytest.raises(ValueError, match="Unsupported export extension"):
        report.to_cleaned(str(tmp_path / "cleaned.parquet"))


def test_cleaned_eval_requires_retained_eval() -> None:
    report = ContaminationReport(
        train_size=1,
        eval_size=2,
        matches=[MatchRecord(0, 0, "x", "x", 1.0, "exact")],
        methods_used=["exact"],
    )
    with pytest.raises(ValueError, match="not retained"):
        report.cleaned_eval()


def test_to_json_does_not_embed_eval_payload() -> None:
    report = verascan.check(["a"], ["a", "b"], methods=["exact"], show_progress=False)
    payload = report.to_dict()
    assert "eval_texts" not in payload
    assert "eval_records" not in payload


def test_custom_column_dataframe_preserved() -> None:
    train = pd.DataFrame({"prompt": ["q1"]})
    eval_df = pd.DataFrame({"prompt": ["q1", "q2"], "meta": [1, 2]})
    report = verascan.check(train, eval_df, column="prompt", methods=["exact"], show_progress=False)
    cleaned = report.cleaned_eval()
    assert list(cleaned["prompt"]) == ["q2"]
    assert list(cleaned["meta"]) == [2]


def test_empty_dataframe_keeps_columns() -> None:
    train = pd.DataFrame({"text": ["x"]})
    eval_df = pd.DataFrame({"text": pd.Series(dtype=str), "label": pd.Series(dtype=str)})
    report = verascan.check(train, eval_df, methods=["exact"], show_progress=False)
    cleaned = report.cleaned_eval()
    assert isinstance(cleaned, pd.DataFrame)
    assert list(cleaned.columns) == ["text", "label"]
    assert len(cleaned) == 0
