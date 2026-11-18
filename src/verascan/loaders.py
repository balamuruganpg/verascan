"""Data loading utilities — normalise any supported input into a list of strings.

Supported sources:
- ``list[str]`` and ``list[dict]``
- :class:`pandas.DataFrame`
- CSV file path
- JSON or JSONL file path
- HuggingFace ``datasets.Dataset`` (when ``datasets`` is installed)
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, NamedTuple, TextIO, cast

import pandas as pd

# Type alias for anything the public API accepts as a dataset.
DataInput = str | pd.DataFrame | list[str] | list[dict[str, Any]] | Any

_HF_DATASET_TYPE: type | None = None


class LoadedEval(NamedTuple):
    """Texts plus original tabular rows (when the source was not a bare list)."""

    texts: list[str]
    records: list[dict[str, Any]] | None = None
    columns: list[str] | None = None


def _get_hf_dataset_type() -> type | None:
    """Lazily resolve ``datasets.Dataset`` to avoid hard dependency."""
    global _HF_DATASET_TYPE  # noqa: PLW0603
    if _HF_DATASET_TYPE is None:
        try:
            from datasets import Dataset  # type: ignore[import-untyped]

            _HF_DATASET_TYPE = Dataset
        except ImportError:
            pass
    return _HF_DATASET_TYPE


def _load_csv(path: str, column: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in CSV file '{path}'. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def _text_value(value: Any, column: str, location: str) -> str:
    if not isinstance(value, str) and pd.api.types.is_scalar(value) and pd.isna(value):
        raise ValueError(f"Missing text for '{column}' {location}.")
    return str(value)


def _records_payload(rows: Iterable[tuple[Any, str]], column: str) -> LoadedEval:
    texts: list[str] = []
    records: list[dict[str, Any]] = []
    columns: dict[str, None] = {}
    for row, location in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid record {location}: expected a JSON object.")
        if column not in row:
            raise KeyError(
                f"Key '{column}' not found {location}. Available keys: {list(row.keys())}"
            )
        texts.append(_text_value(row[column], column, location))
        records.append(row)
        for key in row:
            columns.setdefault(key, None)
    return LoadedEval(texts, records, list(columns) or [column])


def _jsonl_rows(fh: TextIO, path: str) -> Iterator[tuple[Any, str]]:
    for line_no, line in enumerate(fh, start=1):
        if not line.strip():
            continue
        location = f"on line {line_no} of '{path}'"
        try:
            yield json.loads(line), location
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON {location}: {exc}") from exc


def _load_json_payload(path: str, column: str, *, lines: bool) -> LoadedEval:
    with open(path, encoding="utf-8") as fh:
        if lines:
            return _records_payload(_jsonl_rows(fh, path), column)
        try:
            document = json.load(fh)
        except json.JSONDecodeError as exc:
            # Older releases treated .json files as JSON Lines.
            if exc.msg == "Extra data" or not exc.doc.strip():
                fh.seek(0)
                return _records_payload(_jsonl_rows(fh, path), column)
            raise ValueError(f"Invalid JSON in '{path}': {exc}") from exc
        rows = document if isinstance(document, list) else [document]
        return _records_payload(
            ((row, f"at row {index} of '{path}'") for index, row in enumerate(rows, 1)),
            column,
        )


def _load_dataframe(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    return [
        _text_value(value, column, f"at row {index} of the DataFrame")
        for index, value in enumerate(df[column], 1)
    ]


def _load_hf_dataset(ds: Any, column: str) -> list[str]:
    if column not in ds.column_names:
        raise KeyError(
            f"Column '{column}' not found in HuggingFace Dataset. "
            f"Available columns: {ds.column_names}"
        )
    return [
        _text_value(value, column, f"at row {index} of the Hugging Face dataset")
        for index, value in enumerate(ds[column], 1)
    ]


def _is_file_path_syntax(source: str) -> bool:
    """Check if *source* syntax indicates it was intended as a local file path."""
    p = Path(source)
    if source.startswith(("./", ".\\", "../", "..\\", "/", "\\")) or p.is_absolute():
        return True
    return p.suffix.lower() in {".csv", ".jsonl", ".json", ".tsv", ".parquet", ".txt"}


def _process_hf_dataset(ds: Any, column: str) -> LoadedEval:
    """Extract texts and tabular records from a Hugging Face Dataset or DatasetDict."""
    if hasattr(ds, "keys") and not hasattr(ds, "column_names"):
        if "train" in ds:
            ds = ds["train"]
        elif "test" in ds:
            ds = ds["test"]
        else:
            first_key = next(iter(ds.keys()))
            ds = ds[first_key]

    texts = _load_hf_dataset(ds, column)
    if hasattr(ds, "to_pandas"):
        frame = ds.to_pandas()
        return LoadedEval(
            texts=texts,
            records=cast(list[dict[str, Any]], frame.to_dict(orient="records")),
            columns=list(frame.columns),
        )
    return LoadedEval(
        texts=texts,
        records=[{column: text} for text in texts],
        columns=list(getattr(ds, "column_names", [column])),
    )


def load_eval_payload(
    source: DataInput,
    *,
    column: str = "text",
    split: str | None = None,
    config: str | None = None,
    default_split: str = "train[:2000]",
) -> LoadedEval:
    """Load *source* as texts plus original records (when tabular).

    Returns a :class:`LoadedEval`. ``records`` / ``columns`` are ``None`` when
    *source* is a ``list[str]`` so callers can round-trip list vs table types.
    CSV cells are read as literal strings. Other tabular inputs reject missing
    text values (including null and NaN) with ``ValueError``. Non-missing
    values are converted to strings, and literal empty strings are valid text.
    """
    # --- list[str] / list[dict] -------------------------------------------
    if isinstance(source, list):
        if not source:
            return LoadedEval(texts=[], records=None, columns=None)
        if all(isinstance(item, str) for item in source):
            return LoadedEval(texts=cast(list[str], source), records=None, columns=None)
        if all(isinstance(item, dict) for item in source):
            return _records_payload(
                ((item, f"in dict at index {idx}") for idx, item in enumerate(source)), column
            )
        raise TypeError(
            "When passing a list, all elements must be strings or all elements must be dictionaries."
        )

    # --- pandas DataFrame -------------------------------------------------
    if isinstance(source, pd.DataFrame):
        texts = _load_dataframe(source, column)
        return LoadedEval(
            texts=texts,
            records=cast(list[dict[str, Any]], source.to_dict(orient="records")),
            columns=list(source.columns),
        )

    # --- HuggingFace Dataset (optional) -----------------------------------
    hf_type = _get_hf_dataset_type()
    if hf_type is not None and isinstance(source, hf_type):
        return _process_hf_dataset(source, column)

    # --- file path or HuggingFace dataset name ----------------------------
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        path_obj = Path(path)

        # Existing local file
        if path_obj.exists() and path_obj.is_file():
            ext = path_obj.suffix.lower()
            if ext == ".csv":
                df = _load_csv(path, column)
                return LoadedEval(
                    texts=df[column].tolist(),
                    records=cast(list[dict[str, Any]], df.to_dict(orient="records")),
                    columns=list(df.columns),
                )
            if ext in {".jsonl", ".json"}:
                return _load_json_payload(path, column, lines=ext == ".jsonl")

            raise ValueError(
                f"Unsupported file extension '{ext}' for '{path}'. Supported: .csv, .jsonl, .json"
            )

        # Non-existent file path syntax
        if _is_file_path_syntax(path):
            raise FileNotFoundError(
                f"Not a data file: '{path}'. "
                f"Pass a file path, list[str], DataFrame, or Hugging Face Dataset object."
            )

        # Non-file string: attempt Hugging Face dataset load
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                f"Cannot load dataset '{path}': datasets library is not installed. "
                'Install with pip install "verascan[hf]".'
            ) from exc

        target_split = split if split is not None else default_split
        try:
            if config:
                ds = load_dataset(path, config, split=target_split)
            else:
                ds = load_dataset(path, split=target_split)
        except Exception as exc:
            raise ValueError(
                f"Could not load '{path}' as a Hugging Face dataset (and it is not an existing local file): {exc}"
            ) from exc

        return _process_hf_dataset(ds, column)

    raise TypeError(
        f"Unsupported data source type: {type(source).__name__}. "
        f"Expected list[str], pandas DataFrame, file path (CSV/JSONL), "
        f"or HuggingFace Dataset."
    )


def load_texts(
    source: DataInput,
    *,
    column: str = "text",
    split: str | None = None,
    config: str | None = None,
    default_split: str = "train[:2000]",
) -> list[str]:
    """Load *source* and return a flat list of text strings.

    CSV cells retain their original text, including empty cells. Missing text
    values in other tabular sources raise ``ValueError``; literal strings such
    as ``"None"``, ``"nan"`` and ``""`` remain valid.

    Parameters
    ----------
    source:
        One of:

        - a ``list[str]``
        - a :class:`pandas.DataFrame` containing a text column
        - a file path ending in ``.csv``, ``.jsonl`` or ``.json``
        - a HuggingFace ``datasets.Dataset`` or dataset name string
    column:
        Name of the text column to extract (used for DataFrames / Datasets / CSV / JSONL).
    split:
        Split to load when *source* is a Hugging Face dataset name.
    config:
        Optional dataset configuration when *source* is a Hugging Face dataset name.
    default_split:
        Default split to use if *split* is not provided (default: ``"train[:2000]"``).

    Returns
    -------
    list[str]
    """
    return load_eval_payload(
        source,
        column=column,
        split=split,
        config=config,
        default_split=default_split,
    ).texts


def write_dataset(
    data: list[str] | pd.DataFrame,
    path: str,
    *,
    column: str,
    texts: list[str] | None = None,
) -> None:
    """Export tabular data, using the loaded texts when column types were inferred."""
    dest = Path(path)
    ext = dest.suffix.lower()
    if ext not in {".csv", ".jsonl", ".json"}:
        raise ValueError(
            f"Unsupported export extension '{ext}' for '{path}'. Supported: .csv, .jsonl, .json"
        )
    if isinstance(data, list):
        frame = pd.DataFrame({column: data})
    elif texts is not None:
        frame = data.assign(**{column: texts})
    else:
        frame = data
    if ext == ".csv":
        # Quoting all fields preserves whitespace and embedded CR/LF, including in one-column CSVs.
        frame.to_csv(dest, index=False, quoting=csv.QUOTE_ALL)
    else:
        frame.to_json(dest, orient="records", lines=ext == ".jsonl", force_ascii=False)
