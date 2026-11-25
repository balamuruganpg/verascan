"""Audit training datasets against popular public benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from verascan.benchmarks import (
    load_benchmark,
    normalize_benchmarks,
)
from verascan.core import _validate_methods, check
from verascan.loaders import DataInput, _is_file_path_syntax
from verascan.report import ContaminationReport, MatchRecord


@dataclass
class AuditReport(ContaminationReport):
    """Contamination report specialized for public benchmark audits."""

    benchmarks: list[str] = field(default_factory=list)

    def _benchmark_labels(self) -> dict[str, str]:
        stand_ins = {
            row["benchmark"]
            for row in self.eval_records or []
            if row.get("synthetic") is True
        }
        return {
            name: f"{name} (synthetic)" if name in stand_ins else name
            for name in self.benchmarks
        }

    def _label(self, name: str) -> str:
        return self._benchmark_labels().get(name, name)

    @property
    def benchmark_counts(self) -> dict[str, int]:
        """Count of contaminated evaluation examples per benchmark label.

        Keys match :meth:`by_benchmark`, :meth:`summary`, and :meth:`to_dict`
        (synthetic stand-ins use ``"gsm8k (synthetic)"``).
        """
        labels = self._benchmark_labels()
        counts = {labels[name]: 0 for name in self.benchmarks}
        if self.eval_records is None:
            return counts
        for idx in self.contaminated_indices():
            if idx < len(self.eval_records):
                label = self._label(str(self.eval_records[idx].get("benchmark", "unknown")))
                counts[label] = counts.get(label, 0) + 1
        return counts

    def by_benchmark(self) -> dict[str, list[MatchRecord]]:
        """Return matches grouped by the same benchmark label as the JSON report."""
        labels = self._benchmark_labels()
        grouped: dict[str, list[MatchRecord]] = {labels[name]: [] for name in self.benchmarks}
        if self.eval_records is None:
            return grouped
        for match in self.matches:
            if match.eval_index < len(self.eval_records):
                label = self._label(
                    str(self.eval_records[match.eval_index].get("benchmark", "unknown"))
                )
                grouped.setdefault(label, []).append(match)
        return grouped

    def summary(self) -> str:
        """Return a human-readable summary string."""
        unique_eval = self.contaminated_indices()
        labels = self._benchmark_labels()
        lines = [
            "",
            "=" * 47,
            "  Verascan Benchmark Audit Report",
            "=" * 47,
            f"  Train size      : {self.train_size:,}",
            f"  Benchmarks      : {', '.join(labels.values())}",
            f"  Eval size       : {self.eval_size:,}",
            f"  Methods         : {', '.join(self.methods_used)}",
            *[f"  {label:<16}: {value}" for label, value in self.matching_criteria],
            "-" * 47,
            f"  Total matches   : {len(self.matches)}",
            f"  Contaminated    : {len(unique_eval)} / {self.eval_size} eval samples "
            f"({self.formatted_contamination_rate})",
        ]
        if self.exact_count:
            lines.append(f"    Exact matches : {self.exact_count}")
        if self.ngram_count:
            lines.append(f"    N-gram matches: {self.ngram_count}")
        if self.fuzzy_count:
            lines.append(f"    Fuzzy matches : {self.fuzzy_count}")
        if self.semantic_count:
            lines.append(f"    Semantic hits : {self.semantic_count}")

        if self.benchmarks:
            lines.append("-" * 47)
            lines.append("  Benchmark Contamination:")
            bm_counts = self.benchmark_counts
            for bm in self.benchmarks:
                label = labels[bm]
                c = bm_counts.get(label, 0)
                lines.append(f"    - {label:<12}: {c} contaminated")

        lines.append("=" * 47)
        lines.append("")
        text = "\n".join(lines)
        return text

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict, including benchmark breakdowns."""
        data = super().to_dict()
        labels = self._benchmark_labels()
        data["synthetic"] = any(name != label for name, label in labels.items())
        data["benchmarks"] = [labels[name] for name in self.benchmarks]
        data["benchmark_counts"] = {
            labels[name]: self.benchmark_counts.get(labels[name], 0) for name in self.benchmarks
        }
        if self.eval_records is not None:
            for match_dict in data.get("matches", []):
                idx = match_dict.get("eval_index")
                if idx is not None and idx < len(self.eval_records):
                    name = self.eval_records[idx].get("benchmark", "unknown")
                    match_dict["benchmark"] = labels.get(name, name)
        return data


def audit(
    train: DataInput,
    benchmarks: str | Sequence[str] = ("mmlu", "gsm8k", "humaneval"),
    *,
    methods: Sequence[str] | None = None,
    threshold: float = 0.85,
    column: str = "text",
    split: str | None = None,
    train_split: str | None = None,
    config: str | None = None,
    train_config: str | None = None,
    benchmark_split: str = "test",
    synthetic: bool = False,
    allow_synthetic: bool = False,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    num_perm: int = 128,
    ngram_n: int = 13,
    ngram_max_count: int = 10,
    show_progress: bool = True,
) -> AuditReport:
    """Audit *train* dataset against public benchmark presets.

    Parameters
    ----------
    train:
        Training data to scan. Accepts a file path (CSV or JSONL),
        a :class:`pandas.DataFrame`, a ``list[str]``, or a HuggingFace Dataset / dataset name.
    benchmarks:
        One or more benchmark preset names to check against.
        Supported presets: ``"mmlu"``, ``"gsm8k"``, ``"humaneval"``.
    methods:
        Detection methods to run (default: ``["exact", "fuzzy"]``).
    threshold:
        Similarity threshold for fuzzy and semantic methods (0 – 1).
    column:
        Column name in *train* to extract text from.
    split:
        Alias for the training split if it starts with "train", or for the
        benchmark split otherwise. A training split requires a dataset name.
    train_split:
        Specific split to load when *train* is a Hugging Face dataset name.
    config:
        Dataset configuration name when *train* is a Hugging Face dataset name.
    train_config:
        Specific configuration name for *train*.
    benchmark_split:
        Split for benchmark presets (default: ``"test"``). A non-default value
        takes precedence over a benchmark split supplied through *split*.
    synthetic:
        If True, use synthetic stand-ins for offline / fast testing.
    allow_synthetic:
        If True, fall back to synthetic stand-ins if Hugging Face is unreachable.
    model_name:
        Model name for semantic embeddings if ``"semantic"`` is in *methods*.
    batch_size:
        Batch size for semantic embeddings.
    num_perm:
        Number of MinHash permutations for fuzzy matching.
    ngram_n:
        Word n-gram size for the ``ngram`` method.
    ngram_max_count:
        Document frequency cutoff for the ``ngram`` method.
    show_progress:
        Whether to display tqdm progress bars.

    Returns
    -------
    AuditReport
        A specialized :class:`ContaminationReport` containing benchmark audit results.
    """
    methods = _validate_methods(methods)
    normalized_bms = normalize_benchmarks(benchmarks)

    eff_train_split = train_split
    eff_bm_split = benchmark_split
    if split is not None:
        if split.startswith("train"):
            if eff_train_split is None:
                eff_train_split = split
        elif benchmark_split == "test":
            eff_bm_split = split
    if eff_train_split is not None and (
        not isinstance(train, str) or _is_file_path_syntax(train) or Path(train).is_file()
    ):
        raise ValueError("An audit train split requires a Hugging Face dataset name.")

    eff_train_config = train_config if train_config is not None else config

    all_records: list[dict[str, Any]] = []
    for bm_name in normalized_bms:
        records = load_benchmark(
            bm_name,
            split=eff_bm_split,
            synthetic=synthetic,
            allow_synthetic=allow_synthetic,
        )
        all_records.extend(records)

    eval_df = pd.DataFrame(all_records) if all_records else pd.DataFrame(columns=["text"])
    # *column* selects training text only. Benchmark answer/subject fields stay intact.
    base_report = check(
        train=train,
        eval=eval_df,
        methods=methods,
        threshold=threshold,
        column=column,
        eval_column="text",
        train_split=eff_train_split,
        config=eff_train_config,
        model_name=model_name,
        batch_size=batch_size,
        num_perm=num_perm,
        ngram_n=ngram_n,
        ngram_max_count=ngram_max_count,
        show_progress=show_progress,
    )

    return AuditReport(
        train_size=base_report.train_size,
        eval_size=base_report.eval_size,
        matches=base_report.matches,
        methods_used=base_report.methods_used,
        threshold=base_report.threshold,
        eval_texts=base_report.eval_texts,
        eval_records=base_report.eval_records,
        eval_columns=base_report.eval_columns,
        eval_column="text",
        benchmarks=normalized_bms,
    )
