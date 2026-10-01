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
from typing import Any, NamedTuple, cast

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
    """
    # --- list[str] / list[dict] -------------------------------------------
    if isinstance(source, list):
        if not source:
            return LoadedEval(texts=[], records=None, columns=None)
        if all(isinstance(item, str) for item in source):
            return LoadedEval(texts=cast(list[str], source), records=None, columns=None)
        if all(isinstance(item, dict) for item in source):
            texts: list[str] = []
            columns: list[str] = []
            for idx, item in enumerate(source):
                if column not in item:
                    raise KeyError(
                        f"Key '{column}' not found in dict at index {idx}. "
                        f"Available keys: {list(item.keys())}"
                    )
                texts.append(str(item[column]))
                for key in item:
                    if key not in columns:
                        columns.append(key)
            if not columns:
                columns = [column]
            return LoadedEval(
                texts=texts,
                records=cast(list[dict[str, Any]], source),
                columns=columns,
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
                    texts=df[column].astype(str).tolist(),
                    records=cast(list[dict[str, Any]], df.to_dict(orient="records")),
                    columns=list(df.columns),
                )
            if ext in {".jsonl", ".json"}:
                return _load_jsonl_payload(path, column)

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

    Parameters
    ----------
    source:
        One of:

        - a ``list[str]``
        - a :class:`pandas.DataFrame` containing a text column
        - a file path ending in ``.csv`` or ``.jsonl``
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
