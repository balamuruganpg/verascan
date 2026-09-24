"""Tests for verascan.loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from verascan.loaders import load_eval_payload, load_texts

# ---------- list[str] --------------------------------------------------- #


def test_load_list_of_strings() -> None:
    data = ["hello", "world"]
    assert load_texts(data) == ["hello", "world"]


def test_load_list_rejects_non_strings_or_non_dicts() -> None:
    with pytest.raises(TypeError, match="must be strings or all elements must be dictionaries"):
        load_texts(["hello", 123])  # type: ignore[list-item]


def test_load_list_of_dicts_default_column() -> None:
    data = [{"text": "alpha", "score": 1}, {"text": "beta", "score": 2}]
    assert load_texts(data) == ["alpha", "beta"]


def test_load_list_of_dicts_custom_column() -> None:
    data = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
    ]
    assert load_texts(data, column="question") == ["What is 2+2?", "Capital of France?"]


def test_load_list_of_dicts_missing_column() -> None:
    data = [{"question": "What is 2+2?"}, {"answer": "Paris"}]
    with pytest.raises(KeyError, match=r"Key 'question' not found in dict at index 1"):
        load_texts(data, column="question")


# ---------- pandas DataFrame -------------------------------------------- #


def test_load_dataframe_default_column() -> None:
    df = pd.DataFrame({"text": ["alpha", "beta"], "label": [0, 1]})
    assert load_texts(df) == ["alpha", "beta"]


def test_load_dataframe_custom_column() -> None:
    df = pd.DataFrame({"content": ["gamma", "delta"]})
    assert load_texts(df, column="content") == ["gamma", "delta"]


def test_load_dataframe_missing_column() -> None:
    df = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(KeyError, match="not found"):
        load_texts(df, column="text")


# ---------- CSV --------------------------------------------------------- #


def test_load_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("text,id\nfoo,1\nbar,2\n", encoding="utf-8")
    assert load_texts(str(csv_file)) == ["foo", "bar"]


def test_load_csv_missing_column(tmp_path: Path) -> None:
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x,y\n1,2\n", encoding="utf-8")
    with pytest.raises(KeyError, match="not found in CSV"):
        load_texts(str(csv_file), column="text")


# ---------- JSONL ------------------------------------------------------- #


def test_load_jsonl(tmp_path: Path) -> None:
    jl_file = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"text": "first line"}),
        json.dumps({"text": "second line"}),
    ]
    jl_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert load_texts(str(jl_file)) == ["first line", "second line"]


def test_load_jsonl_missing_key(tmp_path: Path) -> None:
    jl_file = tmp_path / "data.jsonl"
    jl_file.write_text(json.dumps({"wrong": "value"}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match="Key 'text' not found"):
        load_texts(str(jl_file))


def test_load_jsonl_bad_json(tmp_path: Path) -> None:
    jl_file = tmp_path / "data.jsonl"
    jl_file.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_texts(str(jl_file))


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    jl_file = tmp_path / "data.jsonl"
    content = f"\n{json.dumps({'text': 'ok'})}\n\n"
    jl_file.write_text(content, encoding="utf-8")
    assert load_texts(str(jl_file)) == ["ok"]


# ---------- error handling ---------------------------------------------- #


def test_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Not a data file"):
        load_texts("non_existent_file_12345.jsonl")


def test_loader_dataset_name_string_error() -> None:
    with pytest.raises(
        FileNotFoundError,
        match=r"Not a data file: 'imdb'\. Pass a file path, list\[str\], DataFrame, or Hugging Face Dataset object\.",
    ):
        load_texts("imdb")


def test_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "data.parquet"
    bad_file.write_text("fake parquet", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_texts(str(bad_file))


def test_unsupported_type() -> None:
    with pytest.raises(TypeError, match="Unsupported data source type"):
        load_texts(12345)  # type: ignore[arg-type]


# ---------- load_eval_payload ------------------------------------------- #


def test_load_eval_payload_list_has_no_records() -> None:
    loaded = load_eval_payload(["hello", "world"])
    assert loaded.texts == ["hello", "world"]
    assert loaded.records is None
    assert loaded.columns is None


def test_load_eval_payload_list_of_dicts() -> None:
    data = [
        {"question": "What is 2+2?", "answer": "4", "id": 101},
        {"question": "What is 3+3?", "answer": "6", "id": 102},
    ]
    loaded = load_eval_payload(data, column="question")
    assert loaded.texts == ["What is 2+2?", "What is 3+3?"]
    assert loaded.records == data
    assert loaded.columns == ["question", "answer", "id"]


def test_load_eval_payload_dataframe_keeps_columns() -> None:
    df = pd.DataFrame({"text": ["alpha"], "label": ["x"]})
    loaded = load_eval_payload(df)
    assert loaded.texts == ["alpha"]
    assert loaded.columns == ["text", "label"]
    assert loaded.records is not None
    assert loaded.records[0]["text"] == "alpha"
    assert loaded.records[0]["label"] == "x"


def test_load_eval_payload_jsonl_keeps_extra_keys(tmp_path: Path) -> None:
    jl_file = tmp_path / "data.jsonl"
    jl_file.write_text(
        json.dumps({"text": "first", "id": 1})
        + "\n"
        + json.dumps({"text": "second", "id": 2})
        + "\n",
        encoding="utf-8",
    )
    loaded = load_eval_payload(str(jl_file))
    assert loaded.texts == ["first", "second"]
    assert loaded.columns == ["text", "id"]
    assert loaded.records == [{"text": "first", "id": 1}, {"text": "second", "id": 2}]
