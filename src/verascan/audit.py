"""Audit training datasets against popular public benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from verascan.benchmarks import (
    load_benchmark,
    normalize_benchmarks,
)
from verascan.core import check
from verascan.loaders import DataInput
from verascan.report import ContaminationReport, MatchRecord


@dataclass
class AuditReport(ContaminationReport):
    """Contamination report specialized for public benchmark audits."""

    benchmarks: list[str] = field(default_factory=list)

    @property
    def benchmark_counts(self) -> dict[str, int]:
        """Count of contaminated evaluation examples per benchmark."""
        counts = {b: 0 for b in self.benchmarks}
        if self.eval_records is None:
            return counts
        flagged_indices = self.contaminated_indices()
        for idx in flagged_indices:
            if idx < len(self.eval_records):
                bm = self.eval_records[idx].get("benchmark", "unknown")
                counts[bm] = counts.get(bm, 0) + 1
        return counts

    def by_benchmark(self) -> dict[str, list[MatchRecord]]:
        """Return matches grouped by benchmark name."""
        grouped: dict[str, list[MatchRecord]] = {b: [] for b in self.benchmarks}
        if self.eval_records is None:
            return grouped
        for match in self.matches:
            if match.eval_index < len(self.eval_records):
                bm = self.eval_records[match.eval_index].get("benchmark", "unknown")
                grouped.setdefault(bm, []).append(match)
        return grouped

    def summary(self) -> str:
        """Return a human-readable summary string and print it."""
        unique_eval = self.contaminated_indices()
        lines = [
            "",
            "=" * 47,
            "  Verascan Benchmark Audit Report",
            "=" * 47,
            f"  Train size      : {self.train_size:,}",
            f"  Benchmarks      : {', '.join(self.benchmarks)}",
            f"  Eval size       : {self.eval_size:,}",
            f"  Methods         : {', '.join(self.methods_used)}",
            f"  Threshold       : {self.threshold}",
            "-" * 47,
            f"  Total matches   : {len(self.matches)}",
            f"  Contaminated    : {len(unique_eval)} / {self.eval_size} eval samples "
            f"({self.contamination_rate:.1%})",
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
                c = bm_counts.get(bm, 0)
                lines.append(f"    - {bm:<12}: {c} contaminated")

        lines.append("=" * 47)
        lines.append("")
        text = "\n".join(lines)
        print(text)
        return text

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict, including benchmark breakdowns."""
        data = super().to_dict()
        data["benchmarks"] = self.benchmarks
        data["benchmark_counts"] = self.benchmark_counts
        if self.eval_records is not None:
            for match_dict in data.get("matches", []):
                idx = match_dict.get("eval_index")
                if idx is not None and idx < len(self.eval_records):
                    match_dict["benchmark"] = self.eval_records[idx].get("benchmark", "unknown")
        return data


def audit(
    train: DataInput,
    benchmarks: str | Sequence[str] = ("mmlu", "gsm8k", "humaneval"),
    *,
    methods: Sequence[str] | None = None,
    threshold: float = 0.85,
    column: str = "text",
    split: str = "test",
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
        a :class:`pandas.DataFrame`, a ``list[str]``, or a HuggingFace Dataset.
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
        Split to evaluate for benchmark datasets (default: ``"test"``).
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
    normalized_bms = normalize_benchmarks(benchmarks)

    all_records: list[dict[str, Any]] = []
    for bm_name in normalized_bms:
        records = load_benchmark(
            bm_name,
            split=split,
            synthetic=synthetic,
            allow_synthetic=allow_synthetic,
        )
        all_records.extend(records)

    eval_df = pd.DataFrame(all_records)
    if column != "text" and "text" in eval_df.columns:
        eval_df[column] = eval_df["text"]

    base_report = check(
        train=train,
        eval=eval_df,
        methods=methods,
        threshold=threshold,
        column=column,
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
        eval_column=column,
        benchmarks=normalized_bms,
    )
