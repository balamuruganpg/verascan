"""Core orchestration — the public ``verascan.check()`` function."""

from __future__ import annotations

from collections.abc import Sequence

from verascan.engines.exact import find_exact_matches
from verascan.engines.fuzzy import find_fuzzy_matches
from verascan.loaders import DataInput, load_texts
from verascan.report import ContaminationReport, MatchRecord

_VALID_METHODS = {"exact", "fuzzy", "semantic"}
_DEFAULT_METHODS = ["exact", "fuzzy"]


def check(
    train: DataInput,
    eval: DataInput,
    *,
    methods: Sequence[str] | None = None,
    threshold: float = 0.85,
    column: str = "text",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    num_perm: int = 128,
    show_progress: bool = True,
) -> ContaminationReport:
    """Scan *eval* for contamination against *train*.

    Parameters
    ----------
    train, eval:
        Training / evaluation data. Accepts a file path (CSV / JSONL),
        a :class:`pandas.DataFrame`, a ``list[str]``, or a HuggingFace Dataset.
    methods:
        Detection methods to run. Defaults to ``["exact", "fuzzy"]``.
        Add ``"semantic"`` for embedding-based matching (requires ``verascan[semantic]``).
    threshold:
        Similarity threshold for fuzzy and semantic methods (0 – 1).
    column:
        Column name to extract text from (when inputs are DataFrames/CSVs/JSONL).
    model_name:
        HuggingFace model name for semantic embeddings (default: ``"all-MiniLM-L6-v2"``).
    batch_size:
        Batch size for semantic embedding computation.
    num_perm:
        Number of MinHash permutations for fuzzy matching.
    show_progress:
        Whether to display tqdm progress bars.

    Returns
    -------
    ContaminationReport
    """
    # Validate methods
    if methods is None:
        methods_list = list(_DEFAULT_METHODS)
    else:
        methods_list = list(methods)
        for method_name in methods_list:
            if method_name not in _VALID_METHODS:
                raise ValueError(
                    f"Unknown method '{method_name}'. Valid methods: {sorted(_VALID_METHODS)}"
                )

    # 1. Load texts
    train_texts = load_texts(train, column=column)
    eval_texts = load_texts(eval, column=column)

    all_matches: list[MatchRecord] = []

    # --- exact ----------------------------------------------------------- #
    if "exact" in methods_list:
        exact_matches = find_exact_matches(train_texts, eval_texts, show_progress=show_progress)
        all_matches.extend(exact_matches)

    # --- fuzzy ----------------------------------------------------------- #
    if "fuzzy" in methods_list:
        # Exclude already-matched eval indices to avoid double-counting.
        matched_eval = {match.eval_index for match in all_matches}
        fuzzy_eval_texts = [t for i, t in enumerate(eval_texts) if i not in matched_eval]
        idx_map = [i for i in range(len(eval_texts)) if i not in matched_eval]

        fuzzy_matches = find_fuzzy_matches(
            train_texts,
            fuzzy_eval_texts,
            threshold=threshold,
            num_perm=num_perm,
            show_progress=show_progress,
        )
        # Remap eval_index back to original positions.
        for match in fuzzy_matches:
            all_matches.append(
                MatchRecord(
                    eval_index=idx_map[match.eval_index],
                    train_index=match.train_index,
                    eval_text=match.eval_text,
                    train_text=match.train_text,
                    score=match.score,
                    method=match.method,
                )
            )

    # --- semantic -------------------------------------------------------- #
    if "semantic" in methods_list:
        from verascan.engines.semantic import find_semantic_matches

        # Exclude already-matched eval indices.
        matched_eval_sem = {match.eval_index for match in all_matches}
        sem_eval_texts = [t for i, t in enumerate(eval_texts) if i not in matched_eval_sem]
        idx_map_sem = [i for i in range(len(eval_texts)) if i not in matched_eval_sem]
        sem_matches = find_semantic_matches(
            train_texts,
            sem_eval_texts,
            threshold=threshold,
            model_name=model_name,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        for match in sem_matches:
            all_matches.append(
                MatchRecord(
                    eval_index=idx_map_sem[match.eval_index],
                    train_index=match.train_index,
                    eval_text=match.eval_text,
                    train_text=match.train_text,
                    score=match.score,
                    method=match.method,
                )
            )

    # Sort matches by eval_index, then by score descending.
    all_matches.sort(key=lambda item: (item.eval_index, -item.score))

    return ContaminationReport(
        train_size=len(train_texts),
        eval_size=len(eval_texts),
        matches=all_matches,
        methods_used=methods_list,
        threshold=threshold,
    )
