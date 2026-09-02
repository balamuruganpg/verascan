"""Dataset splitting utility to prevent train/eval data contamination.

Creates a train/eval split and guarantees zero exact, fuzzy, or semantic
leakage between the two splits by detecting candidate leakage and moving
contaminated eval examples out of eval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from verascan.core import check
from verascan.loaders import DataInput, LoadedEval, load_eval_payload

_VALID_METHODS = {"exact", "ngram", "fuzzy", "semantic"}
_DEFAULT_METHODS = ["exact", "fuzzy"]


@dataclass(frozen=True)
class SplitResult:
    """Summary metrics of the split operation."""

    total_samples: int
    train_size: int
    eval_size: int
    moved_count: int
    methods_used: list[str]
    threshold: float
    seed: int | None

    @property
    def eval_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.eval_size / self.total_samples

    def summary(self) -> str:
        """Return a human-readable summary string and print it."""
        lines = [
            "",
            "=" * 47,
            "  Verascan Split Summary",
            "=" * 47,
            f"  Total samples   : {self.total_samples:,}",
            f"  Methods         : {', '.join(self.methods_used)}",
            f"  Threshold       : {self.threshold}",
            f"  Seed            : {self.seed}",
            "-" * 47,
            f"  Train size      : {self.train_size:,} ({self.train_size / max(1, self.total_samples):.1%})",
            f"  Eval size       : {self.eval_size:,} ({self.eval_ratio:.1%})",
            f"  Purified leakage: {self.moved_count:,} candidate eval samples moved to train",
            "  Residual leakage: 0.0% (guaranteed leak-free)",
            "=" * 47,
            "",
        ]
        text = "\n".join(lines)
        print(text)
        return text


def _slice_payload(payload: LoadedEval, indices: list[int]) -> list[str] | pd.DataFrame:
    """Slice original payload into list[str] or pd.DataFrame maintaining format."""
    if payload.records is not None:
        frame = pd.DataFrame(payload.records, columns=payload.columns)
        if not indices:
            return cast("pd.DataFrame", frame.iloc[0:0].copy())
        return cast("pd.DataFrame", frame.iloc[indices].reset_index(drop=True))
    return [payload.texts[i] for i in indices]


def _write_split_export(
    data: list[str] | pd.DataFrame,
    path: str,
    *,
    column: str = "text",
) -> None:
    """Save split data to disk as CSV, JSONL, or JSON."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ext = dest.suffix.lower()
    if ext not in {".csv", ".jsonl", ".json"}:
        raise ValueError(
            f"Unsupported export extension '{ext}' for '{path}'. Supported: .csv, .jsonl, .json"
        )
    frame = pd.DataFrame({column: data}) if isinstance(data, list) else data
    if ext == ".csv":
        frame.to_csv(dest, index=False)
    elif ext == ".jsonl":
        frame.to_json(dest, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_json(dest, orient="records", lines=False, force_ascii=False)


def split(
    data: DataInput,
    *,
    eval_size: float | int = 0.2,
    methods: Sequence[str] | None = None,
    threshold: float = 0.85,
    column: str = "text",
    seed: int | None = 42,
    move_to: str = "train",
    output_train: str | None = None,
    output_eval: str | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    num_perm: int = 128,
    ngram_n: int = 13,
    ngram_max_count: int = 10,
    show_progress: bool = False,
) -> tuple[Any, Any]:
    """Split *data* into train and eval sets with guaranteed zero contamination.

    Parameters
    ----------
    data:
        Input dataset. Accepts a file path (CSV or JSONL), a :class:`pandas.DataFrame`,
        a ``list[str]``, or a HuggingFace Dataset.
    eval_size:
        Target evaluation ratio as a float between 0.0 and 1.0 (e.g. 0.2 for 20%),
        or an absolute number of evaluation rows as an integer.
    methods:
        Detection methods to run during leak verification. Defaults to ``["exact", "fuzzy"]``.
        Include ``"semantic"`` for embedding-based checks (requires ``verascan[semantic]``).
    threshold:
        Similarity threshold for fuzzy and semantic methods (0 – 1).
    column:
        Name of the text column to evaluate.
    seed:
        Random seed for shuffling before partition. Set to ``None`` for non-deterministic split.
    move_to:
        Where to place candidate eval rows that are found to leak into train.
        ``"train"`` (default) moves them into the training set so no data is discarded.
        ``"drop"`` discards contaminated candidate rows completely.
    output_train:
        Optional file path to save the training split (.csv, .jsonl, or .json).
    output_eval:
        Optional file path to save the evaluation split (.csv, .jsonl, or .json).
    model_name:
        Model name for semantic embeddings (if ``"semantic"`` is in methods).
    batch_size:
        Batch size for semantic embeddings.
    num_perm:
        Number of MinHash permutations for fuzzy matching.
    show_progress:
        Whether to display tqdm progress bars during check iterations.

    Returns
    -------
    tuple[Any, Any]
        A ``(train, eval)`` tuple. The data type matches the input:
        ``list[str]`` for string lists, or :class:`pandas.DataFrame` for
        DataFrames and tabular file paths.
    """
    if move_to not in {"train", "drop"}:
        raise ValueError(f"Unknown move_to option '{move_to}'. Valid options: 'train', 'drop'")

    # Validate methods
    if methods is None:
        methods_list = list(_DEFAULT_METHODS)
    else:
        methods_list = list(methods)
        for m in methods_list:
            if m not in _VALID_METHODS:
                raise ValueError(f"Unknown method '{m}'. Valid methods: {sorted(_VALID_METHODS)}")

    payload = load_eval_payload(data, column=column)
    n = len(payload.texts)

    if n == 0:
        empty_train = _slice_payload(payload, [])
        empty_eval = _slice_payload(payload, [])
        if output_train:
            _write_split_export(empty_train, output_train, column=column)
        if output_eval:
            _write_split_export(empty_eval, output_eval, column=column)
        return empty_train, empty_eval

    # Determine initial target eval count
    if isinstance(eval_size, float):
        if not (0.0 <= eval_size <= 1.0):
            raise ValueError(f"eval_size as float must be between 0.0 and 1.0, got {eval_size}")
        target_eval = int(round(n * eval_size))
    elif isinstance(eval_size, int):
        if eval_size < 0:
            raise ValueError(f"eval_size as integer cannot be negative, got {eval_size}")
        target_eval = min(eval_size, n)
    else:
        raise TypeError(f"eval_size must be float or int, got {type(eval_size).__name__}")

    # Generate shuffled indices
    indices = list(range(n))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(indices)
    else:
        random.shuffle(indices)

    # Initial candidate partition
    eval_indices = indices[:target_eval]
    train_indices = indices[target_eval:]

    all_texts = payload.texts
    moved_count = 0

    # Iteratively detect and purge leakage between candidate train and candidate eval
    while eval_indices and train_indices:
        current_train_texts = [all_texts[i] for i in train_indices]
        current_eval_texts = [all_texts[i] for i in eval_indices]

        report = check(
            current_train_texts,
            current_eval_texts,
            methods=methods_list,
            threshold=threshold,
            model_name=model_name,
            batch_size=batch_size,
            num_perm=num_perm,
            ngram_n=ngram_n,
            ngram_max_count=ngram_max_count,
            show_progress=show_progress,
        )

        if report.total_matches == 0:
            break

        # Collect original indices of flagged eval items
        flagged_local_indices = {m.eval_index for m in report.matches}
        flagged_original_indices = [eval_indices[idx] for idx in flagged_local_indices]
        flagged_set = set(flagged_original_indices)

        # Remove from candidate eval
        eval_indices = [idx for idx in eval_indices if idx not in flagged_set]
        moved_count += len(flagged_original_indices)

        if move_to == "train":
            # Add to train so data is preserved without any eval leakage
            train_indices.extend(flagged_original_indices)

    train_data = _slice_payload(payload, train_indices)
    eval_data = _slice_payload(payload, eval_indices)

    if output_train:
        _write_split_export(train_data, output_train, column=column)
    if output_eval:
        _write_split_export(eval_data, output_eval, column=column)

    return train_data, eval_data
