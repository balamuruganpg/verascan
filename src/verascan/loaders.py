"""Data loading utilities — normalise any supported input into a list of strings.

Supported sources:
- ``list[str]``
- :class:`pandas.DataFrame`
- CSV file path
- JSONL file path
- HuggingFace ``datasets.Dataset`` (when ``datasets`` is installed)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

# Type alias for anything the public API accepts as a dataset.
DataInput = str | pd.DataFrame | list[str] | Any

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
    df = pd.read_csv(path)
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in CSV file '{path}'. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def _load_jsonl_payload(path: str, column: str) -> LoadedEval:
    texts: list[str] = []
    records: list[dict[str, Any]] = []
    columns: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of '{path}': {exc}") from exc
            if column not in obj:
                raise KeyError(
                    f"Key '{column}' not found on line {line_no} of '{path}'. "
                    f"Available keys: {list(obj.keys())}"
                )
            records.append(obj)
            texts.append(str(obj[column]))
            for key in obj:
                if key not in columns:
                    columns.append(key)
    if not columns:
        columns = [column]
    return LoadedEval(texts=texts, records=records, columns=columns)


def _load_dataframe(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    return df[column].astype(str).tolist()


def _load_hf_dataset(ds: Any, column: str) -> list[str]:
    if column not in ds.column_names:
        raise KeyError(
            f"Column '{column}' not found in HuggingFace Dataset. "
            f"Available columns: {ds.column_names}"
        )
    return [str(x) for x in ds[column]]


def load_eval_payload(source: DataInput, *, column: str = "text") -> LoadedEval:
    """Load *source* as texts plus original records (when tabular).

    Returns a :class:`LoadedEval`. ``records`` / ``columns`` are ``None`` when
    *source* is a ``list[str]`` so callers can round-trip list vs table types.
    """
    # --- list[str] --------------------------------------------------------
    if isinstance(source, list):
        if not all(isinstance(item, str) for item in source):
            raise TypeError("When passing a list, every element must be a string.")
        return LoadedEval(texts=source, records=None, columns=None)

    # --- pandas DataFrame -------------------------------------------------
    if isinstance(source, pd.DataFrame):
        texts = _load_dataframe(source, column)
        return LoadedEval(
            texts=texts,
            records=source.to_dict(orient="records"),
            columns=list(source.columns),
        )

    # --- HuggingFace Dataset (optional) -----------------------------------
    hf_type = _get_hf_dataset_type()
    if hf_type is not None and isinstance(source, hf_type):
        texts = _load_hf_dataset(source, column)
        if hasattr(source, "to_pandas"):
            frame = source.to_pandas()
            return LoadedEval(
                texts=texts,
                records=frame.to_dict(orient="records"),
                columns=list(frame.columns),
            )
        return LoadedEval(
            texts=texts,
            records=[{column: text} for text in texts],
            columns=list(getattr(source, "column_names", [column])),
        )

    # --- file path --------------------------------------------------------
    if isinstance(source, (str, os.PathLike)):
        path = str(source)
        if not Path(path).exists():
            raise FileNotFoundError(f"Data file not found: '{path}'")

        ext = Path(path).suffix.lower()
        if ext == ".csv":
            df = _load_csv(path, column)
            return LoadedEval(
                texts=df[column].astype(str).tolist(),
                records=df.to_dict(orient="records"),
                columns=list(df.columns),
            )
        if ext in {".jsonl", ".json"}:
            return _load_jsonl_payload(path, column)

        raise ValueError(
            f"Unsupported file extension '{ext}' for '{path}'. Supported: .csv, .jsonl, .json"
        )

    raise TypeError(
        f"Unsupported data source type: {type(source).__name__}. "
        f"Expected list[str], pandas DataFrame, file path (CSV/JSONL), "
        f"or HuggingFace Dataset."
    )


def load_texts(source: DataInput, *, column: str = "text") -> list[str]:
    """Load *source* and return a flat list of text strings.

    Parameters
    ----------
    source:
        One of:

        - a ``list[str]``
        - a :class:`pandas.DataFrame` containing a text column
        - a file path ending in ``.csv`` or ``.jsonl``
        - a HuggingFace ``datasets.Dataset``
    column:
        Name of the text column to extract (used for DataFrames / Datasets / CSV / JSONL).

    Returns
    -------
    list[str]

    Raises
    ------
    KeyError
        If the requested *column* does not exist in the source.
    FileNotFoundError
        If *source* is a path that does not exist.
    ValueError
        If *source* is an unsupported type or file format.
    """
    return load_eval_payload(source, column=column).texts
