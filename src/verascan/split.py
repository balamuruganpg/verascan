"""Dataset splitting utility to prevent train/eval data contamination.

Creates a train/eval split by detecting overlap with the selected methods and
moving or dropping contaminated evaluation examples.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from verascan.core import _validate_methods, check
from verascan.loaders import DataInput, LoadedEval, load_eval_payload, write_dataset
from verascan.report import (
    calculate_contamination_rate,
    format_contamination_rate,
    matching_criteria,
)


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
    residual_leakage: str | None = None

    @property
    def eval_ratio(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.eval_size / self.total_samples

    def summary(self) -> str:
        """Return a summary with sizes relative to the input dataset."""
        dropped = self.total_samples - self.train_size - self.eval_size
        methods = ", ".join(self.methods_used)
        residual = (
            self.residual_leakage
            if self.residual_leakage is not None
            else f"0.0% with {methods} on retained rows"
        )
        lines = [
            "",
            "=" * 47,
            "  Verascan Split Summary",
            "=" * 47,
            f"  Total samples   : {self.total_samples:,}",
            f"  Methods         : {methods}",
            *[
                f"  {label:<16}: {value}"
                for label, value in matching_criteria(self.methods_used, self.threshold)
            ],
            f"  Seed            : {self.seed}",
            "-" * 47,
            f"  Train split     : {self.train_size:,} samples "
            f"({self.train_size / max(1, self.total_samples):.1%} of input)",
            f"  Eval split      : {self.eval_size:,} samples ({self.eval_ratio:.1%} of input)",
            f"  Dropped rows    : {dropped:,}",
            f"  Purified leakage: {self.moved_count:,} candidate eval samples moved to train",
            f"  Residual leakage: {residual}",
            "=" * 47,
            "",
        ]
        text = "\n".join(lines)
        return text


def _slice_payload(payload: LoadedEval, indices: list[int]) -> list[str] | pd.DataFrame:
    """Slice original payload into list[str] or pd.DataFrame maintaining format."""
    if payload.records is not None:
        frame = pd.DataFrame(payload.records, columns=payload.columns)
        if not indices:
            return cast("pd.DataFrame", frame.iloc[0:0].copy())
        return cast("pd.DataFrame", frame.iloc[indices].reset_index(drop=True))
    return [payload.texts[i] for i in indices]


def split(
    data: DataInput,
    *,
    eval_size: float | int = 0.2,
    methods: Sequence[str] | None = None,
    threshold: float = 0.85,
    column: str = "text",
    split: str | None = None,
    config: str | None = None,
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
    """Split *data* into train and eval sets, removing overlap detected by the selected methods.

    Parameters
    ----------
    data:
        Input dataset. Accepts a file path (CSV or JSONL), a :class:`pandas.DataFrame`,
        a ``list[str]``, or a HuggingFace Dataset / dataset name.
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
    split:
        Split to load when *data* is a Hugging Face dataset name (default: ``"train[:2000]"``).
    config:
        Optional dataset configuration when *data* is a Hugging Face dataset name.
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
    methods_list = _validate_methods(methods)
    payload = load_eval_payload(data, column=column, split=split, config=config)
    train, evaluation, _ = _split_loaded(
        payload,
        eval_size=eval_size,
        methods=methods_list,
        threshold=threshold,
        column=column,
        seed=seed,
        move_to=move_to,
        output_train=output_train,
        output_eval=output_eval,
        model_name=model_name,
        batch_size=batch_size,
        num_perm=num_perm,
        ngram_n=ngram_n,
        ngram_max_count=ngram_max_count,
        show_progress=show_progress,
    )
    return train, evaluation


def _split_loaded(
    payload: LoadedEval,
    *,
    eval_size: float | int,
    methods: list[str],
    threshold: float,
    column: str,
    seed: int | None,
    move_to: str,
    output_train: str | None,
    output_eval: str | None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    num_perm: int = 128,
    ngram_n: int = 13,
    ngram_max_count: int = 10,
    show_progress: bool = False,
) -> tuple[list[str] | pd.DataFrame, list[str] | pd.DataFrame, SplitResult]:
    if (
        output_train
        and output_eval
        and Path(output_train).resolve() == Path(output_eval).resolve()
    ):
        raise ValueError("--output-train and --output-eval must use different paths.")
    if move_to not in {"train", "drop"}:
        raise ValueError(f"Unknown move_to option '{move_to}'. Valid options: 'train', 'drop'")

    # Determine initial target eval count
    n = len(payload.texts)
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
            methods=methods,
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
        if move_to == "train":
            # Add to train so data is preserved without any eval leakage
            train_indices.extend(flagged_original_indices)
            moved_count += len(flagged_original_indices)

    residual_leakage = _ngram_blind_residual(
        all_texts,
        train_indices,
        eval_indices,
        methods,
        ngram_n,
        ngram_max_count,
    )
    train_data = _slice_payload(payload, train_indices)
    eval_data = _slice_payload(payload, eval_indices)

    for output, data, indices in (
        (output_train, train_data, train_indices),
        (output_eval, eval_data, eval_indices),
    ):
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            write_dataset(data, output, column=column, texts=[all_texts[i] for i in indices])

    summary = SplitResult(
        n,
        len(train_indices),
        len(eval_indices),
        moved_count,
        methods,
        threshold,
        seed,
        residual_leakage=residual_leakage,
    )
    return train_data, eval_data, summary


def _capped_eval_indexes(
    train_texts: list[str],
    eval_texts: list[str],
    ngram_n: int,
    ngram_max_count: int,
) -> set[int]:
    """Eval rows that share a training gram dropped by the frequency cap."""
    from collections import Counter

    from verascan.engines.ngram import _word_ngrams

    doc_freq: Counter[str] = Counter()
    for text in train_texts:
        for gram in set(_word_ngrams(text, ngram_n)):
            doc_freq[gram] += 1
    capped = {gram for gram, count in doc_freq.items() if count >= ngram_max_count}
    if not capped:
        return set()
    shared: set[int] = set()
    for index, text in enumerate(eval_texts):
        if set(_word_ngrams(text, ngram_n)) & capped:
            shared.add(index)
    return shared


def _ngram_blind_residual(
    texts: list[str],
    train_indices: list[int],
    eval_indices: list[int],
    methods: list[str],
    ngram_n: int,
    ngram_max_count: int,
) -> str | None:
    """Describe overlap an n-gram split cannot see.

    N-gram overlap needs ``ngram_n`` words, and a gram that appears in at
    least ``ngram_max_count`` training rows is dropped before scoring. Either
    blind spot can leave copies in both splits. Rows that share one capped
    gram are not exact copies, so an exact rematch alone reports nothing.
    The scan has already moved every pair the selected methods can see.
    """
    if not train_indices or not eval_indices:
        return None
    from verascan.engines.exact import find_exact_matches
    from verascan.engines.ngram import _word_ngrams

    train_texts = [texts[index] for index in train_indices]
    eval_texts = [texts[index] for index in eval_indices]
    if set(methods) == {"ngram"}:
        exact = find_exact_matches(train_texts, eval_texts, show_progress=False)
        if exact:
            flagged = len({match.eval_index for match in exact})
            shown = format_contamination_rate(
                calculate_contamination_rate(flagged, len(eval_texts))
            )
            short = False
            long = False
            for match in exact:
                if not _word_ngrams(match.train_text, ngram_n) or not _word_ngrams(
                    match.eval_text, ngram_n
                ):
                    short = True
                else:
                    long = True
            reasons: list[str] = []
            if short:
                reasons.append(f"ngram cannot compare rows shorter than {ngram_n} words")
            if long:
                reasons.append(
                    f"ngram dropped grams appearing in at least {ngram_max_count} training rows"
                )
            if reasons:
                detail = reasons[0] if len(reasons) == 1 else ", and ".join(reasons)
                return f"{shown} verbatim overlap remains; {detail}"

    capped = _capped_eval_indexes(train_texts, eval_texts, ngram_n, ngram_max_count)
    if not capped:
        return None
    shown = format_contamination_rate(calculate_contamination_rate(len(capped), len(eval_texts)))
    if "ngram" in methods:
        return (
            f"{shown} n-gram overlap remains; "
            f"ngram dropped grams appearing in at least {ngram_max_count} training rows"
        )
    return (
        f"{shown} n-gram overlap remains; "
        f"retained rows share an n-gram appearing in at least {ngram_max_count} training rows"
    )
