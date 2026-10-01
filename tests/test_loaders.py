"""Tests for verascan.loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


class _MockDataset:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data
        self.column_names = list(data[0].keys()) if data else []

    def __getitem__(self, key: str) -> list[Any]:
        return [row[key] for row in self._data]

    def __iter__(self) -> Any:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._data)


def test_load_dataset_by_name_train_default_split(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any, Any]] = []

    def mock_load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((name, args, kwargs))
        return _MockDataset([{"text": "Sample 1", "label": 0}, {"text": "Sample 2", "label": 1}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    texts = load_texts("imdb", column="text")
    assert texts == ["Sample 1", "Sample 2"]
    assert len(calls) == 1
    assert calls[0][0] == "imdb"
    assert calls[0][2].get("split") == "train[:2000]"


def test_load_dataset_by_name_eval_default_split(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any, Any]] = []

    def mock_load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((name, args, kwargs))
        return _MockDataset([{"text": "Eval sample"}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    payload = load_eval_payload("imdb", column="text", default_split="test")
    assert payload.texts == ["Eval sample"]
    assert calls[0][2].get("split") == "test"


def test_load_dataset_by_name_custom_split_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any, Any]] = []

    def mock_load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append((name, args, kwargs))
        return _MockDataset([{"question": "What is 1+1?", "answer": "2"}])

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    payload = load_eval_payload("gsm8k", column="question", split="train[:500]", config="main")
    assert payload.texts == ["What is 1+1?"]
    assert calls[0][0] == "gsm8k"
    assert calls[0][1] == ("main",)
    assert calls[0][2].get("split") == "train[:500]"


def test_load_dataset_missing_datasets_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r'pip install "verascan\[hf\]"'):
        load_texts("imdb")


def test_load_dataset_neither_file_nor_loadable(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    def mock_load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
        raise ValueError(f"Dataset '{name}' does not exist on Hugging Face hub")

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    with pytest.raises(
        ValueError, match="Could not load 'not_a_real_dataset_123' as a Hugging Face dataset"
    ):
        load_texts("not_a_real_dataset_123")


def test_check_with_dataset_names(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    import verascan

    def mock_load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
        split = kwargs.get("split")
        if split == "train[:500]":
            return _MockDataset(
                [{"question": "How many clips?"}, {"question": "Unique train question."}]
            )
        if split == "test[:200]":
            return _MockDataset(
                [{"question": "How many clips?"}, {"question": "Unique eval question."}]
            )
        return _MockDataset([{"question": "Default"}])

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    report = verascan.check(
        train="gsm8k",
        eval="gsm8k",
        column="question",
        train_split="train[:500]",
        eval_split="test[:200]",
        config="main",
        methods=["exact"],
    )
    assert report.total_matches == 1
    assert report.matches[0].eval_text == "How many clips?"


def test_audit_with_dataset_name(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    import verascan
    from verascan.benchmarks import SYNTHETIC_BENCHMARKS

    leak_text = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]

    def mock_load_dataset(name: str, *args: Any, **kwargs: Any) -> Any:
        return _MockDataset(
            [{"text": leak_text}, {"text": "A completely unrelated training sentence."}]
        )

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    report = verascan.audit(
        train="imdb",
        benchmarks=["mmlu", "gsm8k"],
        column="text",
        split="train[:2000]",
        synthetic=True,
        methods=["exact"],
    )
    assert report.benchmark_counts["gsm8k"] >= 1


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
