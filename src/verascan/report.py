"""ContaminationReport and supporting data structures."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
from jinja2 import Template
from markupsafe import Markup

from verascan._version import __version__
from verascan.loaders import write_dataset

_HIGH_CONTAMINATION_RATE = 0.05

# ---------------------------------------------------------------------------
# MatchRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRecord:
    """A single flagged pair of texts."""

    eval_index: int
    train_index: int
    eval_text: str
    train_text: str
    score: float
    method: str  # "exact" | "ngram" | "fuzzy" | "semantic"
    diff: str | None = None  # word-level diff HTML (populated by report)


def _steps_to_opcodes(
    steps: list[tuple[int, int, int]],
) -> list[tuple[str, int, int, int, int]]:
    """Merge one-token edits into the opcode spans the reports render."""
    tags = ("equal", "replace", "delete", "insert")
    opcodes: list[tuple[str, int, int, int, int]] = []
    for how, i, j in steps:
        tag = tags[how]
        if how == 0:
            span = (tag, i, i + 1, j, j + 1)
        elif how == 1:
            span = (tag, i, i + 1, j, j + 1)
        elif how == 2:
            span = (tag, i, i + 1, j, j)
        else:
            span = (tag, i, i, j, j + 1)
        if opcodes and opcodes[-1][0] == tag:
            previous = opcodes[-1]
            opcodes[-1] = (tag, previous[1], span[2], previous[3], span[4])
        else:
            opcodes.append(span)
    return opcodes


def _align_words_linear(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Diagonal alignment for sequences too long for the edit table.

    A mismatch whose next tokens agree is a substitution. That keeps a
    repeated word matched in place instead of striking a shifted copy.
    """
    steps: list[tuple[int, int, int]] = []
    i = 0
    j = 0
    n = len(a)
    m = len(b)
    while i < n and j < m:
        if a[i] == b[j]:
            steps.append((0, i, j))
            i += 1
            j += 1
        elif i + 1 < n and j + 1 < m and a[i + 1] == b[j + 1]:
            steps.append((1, i, j))
            i += 1
            j += 1
        elif j + 1 < m and a[i] == b[j + 1] and not (i + 1 < n and a[i + 1] == b[j]):
            steps.append((3, i, j))
            j += 1
        elif i + 1 < n and a[i + 1] == b[j]:
            steps.append((2, i, j))
            i += 1
        else:
            steps.append((1, i, j))
            i += 1
            j += 1
    while i < n:
        steps.append((2, i, j))
        i += 1
    while j < m:
        steps.append((3, i, j))
        j += 1
    return _steps_to_opcodes(steps)


def _align_words(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Levenshtein word alignment. Ties stay on the diagonal.

    One substitution costs the same as one insertion or one deletion. When
    those costs tie, the diagonal edit wins, so a repeated token is replaced
    where it changed. A shifted longest-match can instead strike copies that
    appear on both sides and draw the real change at the other end.
    """
    if a == b or Counter(a) == Counter(b):
        # Same words, including a moved copy. Striking one of them deletes a
        # token both sides still contain. Only a real count change is marked.
        return [("equal", 0, len(a), 0, len(b))] if a else []
    n = len(a)
    m = len(b)
    if n * m > 1_000_000:
        return _align_words_linear(a, b)
    width = m + 1
    dist = [0] * ((n + 1) * width)
    parent = bytearray((n + 1) * width)

    def at(i: int, j: int) -> int:
        return i * width + j

    for i in range(1, n + 1):
        dist[at(i, 0)] = i
        parent[at(i, 0)] = 2
    for j in range(1, m + 1):
        dist[at(0, j)] = j
        parent[at(0, j)] = 3
    for i in range(1, n + 1):
        token = a[i - 1]
        for j in range(1, m + 1):
            diagonal = dist[at(i - 1, j - 1)] + (0 if token == b[j - 1] else 1)
            best = diagonal
            how = 0 if token == b[j - 1] else 1
            delete = dist[at(i - 1, j)] + 1
            if delete < best:
                best = delete
                how = 2
            insert = dist[at(i, j - 1)] + 1
            if insert < best:
                best = insert
                how = 3
            dist[at(i, j)] = best
            parent[at(i, j)] = how
    steps: list[tuple[int, int, int]] = []
    i = n
    j = m
    while i > 0 or j > 0:
        how = parent[at(i, j)]
        if how in (0, 1):
            steps.append((how, i - 1, j - 1))
            i -= 1
            j -= 1
        elif how == 2:
            steps.append((2, i - 1, j))
            i -= 1
        else:
            steps.append((3, i, j - 1))
            j -= 1
    steps.reverse()
    return _steps_to_opcodes(steps)


def _word_diff_opcodes(
    train: str, eval_text: str
) -> tuple[list[str], list[str], list[tuple[str, int, int, int, int]]]:
    """Word alignment shared by the HTML report and the PDF report.

    See :func:`_align_words`. A repeated word that still appears on both
    sides stays unmarked; only the tokens that actually changed are struck.
    """
    train_words = train.split()
    eval_words = eval_text.split()
    return train_words, eval_words, _align_words(train_words, eval_words)


# A normalized vector dotted with itself can miss 1 by a fraction of a millionth.
# 0.999991 is far enough below 1 that rounding must not call it perfect.
_AGREEMENT_EPSILON = 1e-6


def stored_similarity(score: float, threshold: float) -> float:
    """Four-decimal similarity that stays faithful to the admitting comparison.

    Fuzzy Jaccard estimates and semantic cosines use this one rule:

    - round to four decimal places;
    - a value of at least 1, or within a millionth of 1, stays 1;
    - any lower value is never stored as 1 (rounding 0.99996 or 0.999991
      must not invent a perfect score);
    - if rounding would fall under *threshold*, store the next four-decimal
      value that still meets *threshold*;
    - if that adjustment would itself be a manufactured 1, store *score*
      unchanged so ``flagged(min_score=threshold)`` still contains the pair.
    """
    clamped = max(-1.0, min(1.0, float(score)))
    stored = round(clamped, 4)
    if clamped < 1.0 - _AGREEMENT_EPSILON and stored >= 1.0:
        stored = 0.9999
    if stored < threshold:
        bumped = math.floor(float(threshold) * 10000 + 1) / 10000
        if bumped > 1.0:
            bumped = 1.0
        if (bumped >= 1.0 and clamped < 1.0) or bumped < threshold:
            return clamped
        stored = bumped
    return stored


def format_match_score(score: float) -> str:
    """Format a stored match score so HTML and PDF show the same digits.

    Four decimals are the usual label. A score kept below 1 so it would not
    be stored as a perfect match must not round back to ``1.0000``.
    """
    rendered = f"{score:.4f}"
    if not score < 1.0 or rendered != "1.0000":
        return rendered
    for decimals in range(5, 16):
        finer = f"{score:.{decimals}f}"
        if finer[0] != "1":
            return finer.rstrip("0").rstrip(".")
    return f"{score:.16g}"


def _word_diff_html(a: str, b: str) -> str:
    """Return an HTML fragment highlighting word-level differences.

    Escapes all input strings safely to prevent raw HTML/script injection
    when datasets contain code, XML, or special characters.
    """
    a_words, b_words, opcodes = _word_diff_opcodes(a, b)
    a_words = [html.escape(word) for word in a_words]
    b_words = [html.escape(word) for word in b_words]
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            parts.append(" ".join(a_words[i1:i2]))
        elif tag == "replace":
            parts.append(
                f"<del>{' '.join(a_words[i1:i2])}</del> <ins>{' '.join(b_words[j1:j2])}</ins>"
            )
        elif tag == "delete":
            parts.append(f"<del>{' '.join(a_words[i1:i2])}</del>")
        elif tag == "insert":
            parts.append(f"<ins>{' '.join(b_words[j1:j2])}</ins>")
    return " ".join(parts)


def format_contamination_rate(rate: float) -> str:
    """Format contamination rate as a percentage string, never showing '0.0%' if rate > 0."""
    if rate <= 0.0:
        return "0.0%"
    pct = rate * 100
    standard = f"{rate:.1%}"
    if rate < _HIGH_CONTAMINATION_RATE and standard == "5.0%":
        precise = f"{pct:.6f}".rstrip("0").rstrip(".")
        return f"{precise}%" if precise != "5" else "<5.0%"
    # 0.9995 is not the whole evaluation set. One decimal would print 100.0%.
    if rate < 1.0 and standard == "100.0%":
        for decimals in range(2, 16):
            precise = f"{pct:.{decimals}f}".rstrip("0").rstrip(".")
            if precise != "100":
                return f"{precise}%"
        return f"{pct:.16g}%"
    if standard != "0.0%":
        return standard
    for decimals in (3, 4, 5, 6):
        val = f"{pct:.{decimals}f}".rstrip("0").rstrip(".")
        if val and float(val) > 0:
            return f"{val}%"
    return f"{pct:.6g}%"


def calculate_contamination_rate(flagged: int, eval_size: int) -> float:
    """Return the fraction of evaluation examples flagged by any method."""
    return flagged / eval_size if eval_size else 0.0


def contamination_status(total_matches: int, rate: float) -> str:
    """Classify a report using the published Clean, Low and High scale."""
    if not total_matches:
        return "Clean"
    return "Low" if rate < _HIGH_CONTAMINATION_RATE else "High"


def threshold_methods(methods: Sequence[str]) -> list[str]:
    return [method for method in methods if method in {"fuzzy", "semantic"}]


def matching_criteria(methods: Sequence[str], threshold: float) -> list[tuple[str, str]]:
    """Describe the criterion for engines with similarity or overlap scores."""
    criteria = []
    scored_methods = threshold_methods(methods)
    if scored_methods:
        criteria.append(("Threshold", f"{threshold} ({', '.join(scored_methods)})"))
    if "ngram" in methods:
        criteria.append(("N-gram", "any retained overlap"))
    return criteria


# ---------------------------------------------------------------------------
# ContaminationReport
# ---------------------------------------------------------------------------


@dataclass
class ContaminationReport:
    """Holds all contamination results and provides export helpers."""

    train_size: int = 0
    eval_size: int = 0
    matches: list[MatchRecord] = field(default_factory=list)
    methods_used: list[str] = field(default_factory=list)
    threshold: float = 0.85
    eval_texts: list[str] = field(default_factory=list, repr=False)
    eval_records: list[dict[str, Any]] | None = field(default=None, repr=False)
    eval_columns: list[str] | None = field(default=None, repr=False)
    eval_column: str = "text"

    # ---- query helpers -------------------------------------------------- #

    def flagged(self, *, min_score: float = 0.0) -> list[MatchRecord]:
        """Return matches at or above *min_score*."""
        return [m for m in self.matches if m.score >= min_score]

    @property
    def total_matches(self) -> int:
        """Total number of flagged matches."""
        return len(self.matches)

    @property
    def contamination_rate(self) -> float:
        """Fraction of eval examples that appear in at least one match."""
        return calculate_contamination_rate(len(self.contaminated_indices()), self.eval_size)

    @property
    def formatted_contamination_rate(self) -> str:
        """Contamination rate formatted as a percentage string, never showing '0.0%' if matches exist."""
        return format_contamination_rate(self.contamination_rate)

    @property
    def contamination_status(self) -> str:
        return contamination_status(self.total_matches, self.contamination_rate)

    @property
    def matching_criteria(self) -> list[tuple[str, str]]:
        return matching_criteria(self.methods_used, self.threshold)

    @property
    def exact_count(self) -> int:
        return sum(1 for m in self.matches if m.method == "exact")

    @property
    def fuzzy_count(self) -> int:
        return sum(1 for m in self.matches if m.method == "fuzzy")

    @property
    def ngram_count(self) -> int:
        return sum(1 for m in self.matches if m.method == "ngram")

    @property
    def semantic_count(self) -> int:
        return sum(1 for m in self.matches if m.method == "semantic")

    # ---- display -------------------------------------------------------- #

    def summary(self) -> str:
        """Return a human-readable summary string."""
        unique_eval = {m.eval_index for m in self.matches}
        lines = [
            "",
            "=" * 47,
            "  Verascan Contamination Report",
            "=" * 47,
            f"  Train size      : {self.train_size:,}",
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
        lines.append("=" * 47)
        lines.append("")
        text = "\n".join(lines)
        return text

    # ---- serialisation -------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict."""
        return {
            "version": __version__,
            "train_size": self.train_size,
            "eval_size": self.eval_size,
            "threshold": self.threshold,
            "methods_used": self.methods_used,
            "threshold_methods": threshold_methods(self.methods_used),
            "contamination_rate": self.contamination_rate,
            "total_matches": len(self.matches),
            "exact_matches": self.exact_count,
            "ngram_matches": self.ngram_count,
            "fuzzy_matches": self.fuzzy_count,
            "semantic_matches": self.semantic_count,
            "matches": [
                {
                    "eval_index": m.eval_index,
                    "train_index": m.train_index,
                    "eval_text": m.eval_text,
                    "train_text": m.train_text,
                    "score": m.score,
                    "method": m.method,
                }
                for m in self.matches
            ],
        }

    def to_json(self, path: str) -> None:
        """Write the report to a JSON file."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    def to_pdf(self, path: str) -> None:
        """Write a paginated PDF with summary, match table and highlighted differences."""
        from verascan.pdf_report import write_pdf

        write_pdf(self, path)

    def to_html(self, path: str) -> None:
        """Render a self-contained, modern HTML report."""
        report = self.to_dict()
        enriched = []
        for m, match_dict in zip(self.matches, report["matches"]):
            benchmark = match_dict.get("benchmark")
            if (
                benchmark is None
                and self.eval_records is not None
                and m.eval_index < len(self.eval_records)
            ):
                benchmark = self.eval_records[m.eval_index].get("benchmark")
            enriched.append(
                {
                    **asdict(m),
                    "benchmark": benchmark,
                    "score_label": format_match_score(m.score),
                    "diff_html": Markup(_word_diff_html(m.train_text, m.eval_text)),
                }
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        html_out = _HTML_TEMPLATE.render(
            report=report,
            matches=enriched,
            status=self.contamination_status,
            criteria=self.matching_criteria,
            flagged_examples=len(self.contaminated_indices()),
            contamination_pct=format_contamination_rate(report["contamination_rate"]),
            contamination_pct_raw=report["contamination_rate"] * 100,
            timestamp=timestamp,
            version=__version__,
        )
        Path(path).write_text(html_out, encoding="utf-8")

    # ---- cleaned eval export -------------------------------------------- #

    def contaminated_indices(self) -> set[int]:
        """Return eval indices that appear in at least one match."""
        return {m.eval_index for m in self.matches}

    def _eval_row_count(self) -> int:
        if self.eval_records is not None:
            return len(self.eval_records)
        if self.eval_texts:
            return len(self.eval_texts)
        return self.eval_size

    def _require_eval_source(self) -> None:
        if self.eval_records is None and not self.eval_texts and self.eval_size > 0:
            raise ValueError(
                "Cannot export eval examples because the original evaluation "
                "set was not retained. Run verascan.check() so the report "
                "keeps eval data, or set `eval_texts` / `eval_records`."
            )

    def _slice_eval(self, indices: list[int]) -> list[str] | pd.DataFrame:
        self._require_eval_source()
        if self.eval_records is not None:
            frame = pd.DataFrame(self.eval_records, columns=self.eval_columns)
            if not indices:
                return cast("pd.DataFrame", frame.iloc[0:0].copy())
            return cast("pd.DataFrame", frame.iloc[indices].reset_index(drop=True))
        return [self.eval_texts[i] for i in indices]

    def cleaned_eval(self) -> list[str] | pd.DataFrame:
        """Return evaluation examples that were **not** flagged as contaminated.

        The return type matches the original eval input to :func:`verascan.check`:
        a ``list[str]`` for list inputs, or a :class:`pandas.DataFrame` (original
        columns preserved) for DataFrame / CSV / JSONL / HuggingFace Dataset
        inputs.

        When no contamination is found the cleaned set equals the original eval.
        """
        return self._slice_eval(self._eval_indices(contaminated=False))

    def contaminated_eval(self) -> list[str] | pd.DataFrame:
        """Return evaluation examples that were flagged as contaminated.

        Same return-type rules as :meth:`cleaned_eval`. Rows follow original
        eval order; each index appears once even if it matched multiple times.
        """
        return self._slice_eval(self._eval_indices(contaminated=True))

    def _eval_indices(self, *, contaminated: bool) -> list[int]:
        flagged = self.contaminated_indices()
        return [i for i in range(self._eval_row_count()) if (i in flagged) == contaminated]

    def to_cleaned(self, path: str) -> None:
        """Write the cleaned evaluation set to *path*.

        Format is inferred from the suffix:

        * ``.csv`` — CSV with header
        * ``.jsonl`` — JSON Lines
        * ``.json`` — JSON array of records

        List inputs are written using the original text column name (default
        ``"text"``). Tabular inputs keep every original column.
        """
        self._write_eval_export(self._eval_indices(contaminated=False), path)

    def to_contaminated(self, path: str) -> None:
        """Write flagged evaluation examples to *path*. See :meth:`to_cleaned`."""
        self._write_eval_export(self._eval_indices(contaminated=True), path)

    def _write_eval_export(self, indices: list[int], path: str) -> None:
        data = self._slice_eval(indices)
        texts = [self.eval_texts[i] for i in indices] if self.eval_texts else None
        write_dataset(data, path, column=self.eval_column, texts=texts)


# ---------------------------------------------------------------------------
# Jinja2 HTML template (Modern, responsive, self-contained, interactive)
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = Path(__file__).with_name("report.html")
_HTML_TEMPLATE = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"), autoescape=True)
