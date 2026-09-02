"""ContaminationReport and supporting data structures."""

from __future__ import annotations

import difflib
import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import pandas as pd
from jinja2 import Template

from verascan._version import __version__

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


def _word_diff_html(a: str, b: str) -> str:
    """Return an HTML fragment highlighting word-level differences.

    Escapes all input strings safely to prevent raw HTML/script injection
    when datasets contain code, XML, or special characters.
    """
    a_words = [html.escape(w) for w in a.split()]
    b_words = [html.escape(w) for w in b.split()]
    sm = difflib.SequenceMatcher(None, a_words, b_words)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
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
        if self.eval_size == 0:
            return 0.0
        unique_eval = {m.eval_index for m in self.matches}
        return len(unique_eval) / self.eval_size

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
        """Return a human-readable summary string and print it."""
        unique_eval = {m.eval_index for m in self.matches}
        lines = [
            "",
            "=" * 47,
            "  Verascan Contamination Report",
            "=" * 47,
            f"  Train size      : {self.train_size:,}",
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
        lines.append("=" * 47)
        lines.append("")
        text = "\n".join(lines)
        print(text)
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
            "contamination_rate": round(self.contamination_rate, 6),
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
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def to_html(self, path: str) -> None:
        """Render a self-contained, modern HTML report."""
        enriched = []
        for m in self.matches:
            diff_html = _word_diff_html(m.train_text, m.eval_text)
            enriched.append(
                {
                    **asdict(m),
                    "diff_html": diff_html,
                    "eval_text_escaped": html.escape(m.eval_text),
                    "train_text_escaped": html.escape(m.train_text),
                }
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        html_out = _HTML_TEMPLATE.render(
            report=self.to_dict(),
            matches=enriched,
            contamination_pct=f"{self.contamination_rate:.1%}",
            contamination_pct_raw=round(self.contamination_rate * 100, 2),
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
        flagged = self.contaminated_indices()
        keep = [i for i in range(self._eval_row_count()) if i not in flagged]
        return self._slice_eval(keep)

    def contaminated_eval(self) -> list[str] | pd.DataFrame:
        """Return evaluation examples that were flagged as contaminated.

        Same return-type rules as :meth:`cleaned_eval`. Rows follow original
        eval order; each index appears once even if it matched multiple times.
        """
        flagged = self.contaminated_indices()
        keep = [i for i in range(self._eval_row_count()) if i in flagged]
        return self._slice_eval(keep)

    def to_cleaned(self, path: str) -> None:
        """Write the cleaned evaluation set to *path*.

        Format is inferred from the suffix:

        * ``.csv`` — CSV with header
        * ``.jsonl`` — JSON Lines
        * ``.json`` — JSON array of records

        List inputs are written using the original text column name (default
        ``"text"``). Tabular inputs keep every original column.
        """
        self._write_eval_export(self.cleaned_eval(), path)

    def to_contaminated(self, path: str) -> None:
        """Write flagged evaluation examples to *path*. See :meth:`to_cleaned`."""
        self._write_eval_export(self.contaminated_eval(), path)

    def _write_eval_export(self, data: list[str] | pd.DataFrame, path: str) -> None:
        dest = Path(path)
        ext = dest.suffix.lower()
        if ext not in {".csv", ".jsonl", ".json"}:
            raise ValueError(
                f"Unsupported export extension '{ext}' for '{path}'. Supported: .csv, .jsonl, .json"
            )
        frame = pd.DataFrame({self.eval_column: data}) if isinstance(data, list) else data
        if ext == ".csv":
            frame.to_csv(dest, index=False)
        elif ext == ".jsonl":
            frame.to_json(dest, orient="records", lines=True, force_ascii=False)
        else:
            frame.to_json(dest, orient="records", lines=False, force_ascii=False)


# ---------------------------------------------------------------------------
# Jinja2 HTML template (Modern, responsive, self-contained, interactive)
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = Path(__file__).with_name("report.html")
_HTML_TEMPLATE = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
