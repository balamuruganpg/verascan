"""ContaminationReport and supporting data structures."""

from __future__ import annotations

import difflib
import html
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    method: str  # "exact" | "fuzzy" | "semantic"
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

    # ---- query helpers -------------------------------------------------- #

    def flagged(self, *, min_score: float = 0.0) -> list[MatchRecord]:
        """Return matches at or above *min_score*."""
        return [m for m in self.matches if m.score >= min_score]

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


# ---------------------------------------------------------------------------
# Jinja2 HTML template (Modern, responsive, self-contained, interactive)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Verascan Report — Data Contamination Analysis</title>
<style>
  :root {
    --bg-page: #090d16;
    --bg-surface: #111827;
    --bg-surface-elevated: #1f2937;
    --bg-surface-hover: #283548;
    --border-subtle: #1f293d;
    --border-card: #2d3748;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --primary: #38bdf8;
    --primary-dim: rgba(56, 189, 248, 0.12);
    --exact: #f43f5e;
    --exact-dim: rgba(244, 63, 94, 0.14);
    --fuzzy: #f59e0b;
    --fuzzy-dim: rgba(245, 158, 11, 0.14);
    --semantic: #6366f1;
    --semantic-dim: rgba(99, 102, 241, 0.14);
    --success: #10b981;
    --success-dim: rgba(16, 185, 129, 0.14);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: var(--font-sans);
    background-color: var(--bg-page);
    color: var(--text-primary);
    line-height: 1.5;
    padding: 2rem 1.25rem 4rem;
    min-height: 100vh;
  }

  .container {
    max-width: 1240px;
    margin: 0 auto;
  }

  /* Header */
  header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 1rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .brand-logo {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, #0284c7, #6366f1);
    border-radius: var(--radius-md);
    font-size: 1.35rem;
  }
  .brand-title h1 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
  }
  .brand-title p {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  .meta-badge-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border: 1px solid transparent;
  }
  .badge-default {
    background: var(--bg-surface-elevated);
    color: var(--text-secondary);
    border-color: var(--border-card);
  }

  /* Status Banner */
  .status-banner {
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-card);
  }
  .status-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 1rem;
  }
  .status-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .status-indicator {
    width: 14px;
    height: 14px;
    border-radius: 50%;
  }
  .status-clean { background: var(--success); box-shadow: 0 0 12px var(--success); }
  .status-warn { background: var(--fuzzy); box-shadow: 0 0 12px var(--fuzzy); }
  .status-danger { background: var(--exact); box-shadow: 0 0 12px var(--exact); }

  .status-text h2 {
    font-size: 1.15rem;
    font-weight: 600;
  }
  .status-text p {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }
  .progress-container {
    background: var(--bg-surface-elevated);
    border-radius: 9999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.5rem;
    display: flex;
  }
  .progress-bar {
    height: 100%;
    transition: width 0.4s ease;
  }
  .progress-bar-clean { background: var(--success); }
  .progress-bar-warn { background: var(--fuzzy); }
  .progress-bar-danger { background: var(--exact); }

  /* Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
  }
  .metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    box-shadow: var(--shadow-card);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .metric-card:hover {
    border-color: var(--primary);
  }
  .metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0.35rem 0;
    letter-spacing: -0.02em;
    color: var(--text-primary);
  }
  .metric-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
  }

  /* Method Badges */
  .method-exact {
    background: var(--exact-dim);
    color: var(--exact);
    border-color: rgba(244, 63, 94, 0.3);
  }
  .method-fuzzy {
    background: var(--fuzzy-dim);
    color: var(--fuzzy);
    border-color: rgba(245, 158, 11, 0.3);
  }
  .method-semantic {
    background: var(--semantic-dim);
    color: var(--semantic);
    border-color: rgba(99, 102, 241, 0.3);
  }

  /* Controls & Filters Bar */
  .controls-bar {
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }
  .filter-tabs {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .filter-btn {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-card);
    color: var(--text-secondary);
    padding: 0.45rem 0.9rem;
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .filter-btn:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }
  .filter-btn.active {
    background: var(--primary);
    color: #04101e;
    border-color: var(--primary);
  }
  .count-chip {
    background: rgba(0, 0, 0, 0.25);
    padding: 0.1rem 0.4rem;
    border-radius: 9999px;
    font-size: 0.72rem;
  }
  .filter-btn.active .count-chip {
    background: rgba(0, 0, 0, 0.2);
    color: #04101e;
  }

  .search-box {
    display: flex;
    align-items: center;
    position: relative;
    min-width: 220px;
    flex-grow: 1;
    max-width: 360px;
  }
  .search-input {
    width: 100%;
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-sm);
    padding: 0.5rem 0.85rem;
    font-size: 0.85rem;
    color: var(--text-primary);
    outline: none;
    transition: border-color 0.15s;
  }
  .search-input:focus {
    border-color: var(--primary);
  }
  .search-input::placeholder {
    color: var(--text-muted);
  }

  .view-options {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .action-btn {
    background: var(--bg-surface-elevated);
    border: 1px solid var(--border-card);
    color: var(--text-secondary);
    padding: 0.45rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .action-btn:hover {
    background: var(--bg-surface-hover);
    color: var(--text-primary);
  }

  /* Matches Container */
  .matches-list {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  .match-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    overflow: hidden;
    box-shadow: var(--shadow-card);
    transition: border-color 0.15s ease;
  }
  .match-card:hover {
    border-color: var(--border-subtle);
  }
  .match-card-header {
    background: var(--bg-surface-elevated);
    padding: 0.75rem 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    border-bottom: 1px solid var(--border-card);
  }
  .match-card-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .match-idx {
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
  .score-badge {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.25rem 0.6rem;
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
  }

  .match-card-body {
    padding: 1.25rem;
    display: grid;
    gap: 1rem;
  }
  .text-block {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }
  .text-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .text-label-tag {
    font-family: var(--font-mono);
    color: var(--text-muted);
    font-weight: normal;
  }
  .text-content {
    background: #0d121f;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #e2e8f0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: var(--font-sans);
  }
  .diff-content {
    background: #0b111e;
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    font-size: 0.9rem;
    line-height: 1.65;
    word-break: break-word;
  }

  /* Diff Styling */
  del {
    background: rgba(244, 63, 94, 0.22);
    color: #fda4af;
    text-decoration: line-through;
    padding: 0.15rem 0.35rem;
    border-radius: 3px;
    font-weight: 500;
  }
  ins {
    background: rgba(16, 185, 129, 0.22);
    color: #6ee7b7;
    text-decoration: none;
    padding: 0.15rem 0.35rem;
    border-radius: 3px;
    font-weight: 600;
  }

  /* Empty State */
  .empty-state {
    text-align: center;
    padding: 4rem 1.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-lg);
    color: var(--text-secondary);
  }
  .empty-icon {
    font-size: 2.8rem;
    margin-bottom: 1rem;
  }
  .empty-title {
    font-size: 1.25rem;
    color: var(--text-primary);
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  /* Footer */
  footer {
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
    font-size: 0.8rem;
    color: var(--text-muted);
  }
  footer a {
    color: var(--primary);
    text-decoration: none;
  }
  footer a:hover {
    text-decoration: underline;
  }

  @media (max-width: 768px) {
    body { padding: 1rem 0.75rem 3rem; }
    .status-header { flex-direction: column; align-items: flex-start; }
    .controls-bar { flex-direction: column; align-items: stretch; }
    .search-box { max-width: 100%; }
  }
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <header>
    <div class="brand">
      <div class="brand-logo">🔍</div>
      <div class="brand-title">
        <h1>Verascan Report</h1>
        <p>Data Contamination &amp; Leakage Audit</p>
      </div>
    </div>
    <div class="meta-badge-group">
      <span class="badge badge-default">Threshold: {{ report.threshold }}</span>
      <span class="badge badge-default">Methods: {{ report.methods_used | join(', ') }}</span>
      <span class="badge badge-default">{{ timestamp }}</span>
    </div>
  </header>

  <!-- Status Banner -->
  <section class="status-banner">
    <div class="status-header">
      <div class="status-title">
        {% if report.contamination_rate == 0 %}
          <div class="status-indicator status-clean"></div>
          <div class="status-text">
            <h2>Clean — No Contamination Detected</h2>
            <p>0 of {{ report.eval_size }} evaluation examples leaked into the training set.</p>
          </div>
        {% elif report.contamination_rate < 0.03 %}
          <div class="status-indicator status-warn"></div>
          <div class="status-text">
            <h2>Low Risk Contamination ({{ contamination_pct }})</h2>
            <p>{{ report.total_matches }} matching pairs detected across datasets.</p>
          </div>
        {% else %}
          <div class="status-indicator status-danger"></div>
          <div class="status-text">
            <h2>Significant Contamination Detected ({{ contamination_pct }})</h2>
            <p>{{ report.total_matches }} matching pairs detected — evaluation results may be inflated.</p>
          </div>
        {% endif %}
      </div>
      <div class="status-score">
        <span class="metric-value" style="font-size:1.6rem; color: {% if report.contamination_rate == 0 %}var(--success){% elif report.contamination_rate < 0.03 %}var(--fuzzy){% else %}var(--exact){% endif %};">
          {{ contamination_pct }}
        </span>
      </div>
    </div>
    <div class="progress-container">
      <div class="progress-bar {% if report.contamination_rate == 0 %}progress-bar-clean{% elif report.contamination_rate < 0.03 %}progress-bar-warn{% else %}progress-bar-danger{% endif %}"
           style="width: {{ [contamination_pct_raw, 100]|min }}%;"></div>
    </div>
  </section>

  <!-- Metrics Grid -->
  <section class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Evaluation Set Size</div>
      <div class="metric-value">{{ report.eval_size | e }}</div>
      <div class="metric-sub">Total eval samples audited</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Training Set Size</div>
      <div class="metric-value">{{ report.train_size | e }}</div>
      <div class="metric-sub">Reference training corpus</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Total Matches Flagged</div>
      <div class="metric-value">{{ report.total_matches | e }}</div>
      <div class="metric-sub">Pairs meeting threshold</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Detection Breakdown</div>
      <div style="display:flex; gap:0.4rem; margin-top:0.6rem; flex-wrap:wrap;">
        <span class="badge method-exact">Exact: {{ report.exact_matches }}</span>
        <span class="badge method-fuzzy">Fuzzy: {{ report.fuzzy_matches }}</span>
        {% if report.semantic_matches > 0 or 'semantic' in report.methods_used %}
        <span class="badge method-semantic">Semantic: {{ report.semantic_matches }}</span>
        {% endif %}
      </div>
      <div class="metric-sub" style="margin-top:0.5rem;">Across {{ report.methods_used | length }} engine(s)</div>
    </div>
  </section>

  {% if matches %}
  <!-- Controls Bar -->
  <section class="controls-bar">
    <div class="filter-tabs">
      <button class="filter-btn active" onclick="filterMethod('all')" id="btn-all">
        All <span class="count-chip">{{ matches | length }}</span>
      </button>
      {% if report.exact_matches > 0 %}
      <button class="filter-btn" onclick="filterMethod('exact')" id="btn-exact">
        Exact <span class="count-chip">{{ report.exact_matches }}</span>
      </button>
      {% endif %}
      {% if report.fuzzy_matches > 0 %}
      <button class="filter-btn" onclick="filterMethod('fuzzy')" id="btn-fuzzy">
        Fuzzy <span class="count-chip">{{ report.fuzzy_matches }}</span>
      </button>
      {% endif %}
      {% if report.semantic_matches > 0 %}
      <button class="filter-btn" onclick="filterMethod('semantic')" id="btn-semantic">
        Semantic <span class="count-chip">{{ report.semantic_matches }}</span>
      </button>
      {% endif %}
    </div>

    <div class="search-box">
      <input type="text" class="search-input" id="searchInput" placeholder="Search text or index..." oninput="handleSearch()">
    </div>

    <div class="view-options">
      <button class="action-btn" onclick="toggleAllDetails()" id="toggleDetailsBtn">Collapse All</button>
    </div>
  </section>

  <!-- Flagged Pairs List -->
  <section class="matches-list" id="matchesList">
    {% for m in matches %}
    <div class="match-card" data-method="{{ m.method }}" data-score="{{ m.score }}" data-eval-idx="{{ m.eval_index }}" data-train-idx="{{ m.train_index }}" data-text="{{ m.eval_text_escaped }} {{ m.train_text_escaped }}">
      <div class="match-card-header">
        <div class="match-card-left">
          <span class="match-idx">#{{ loop.index }}</span>
          <span class="badge method-{{ m.method }}">{{ m.method }}</span>
          <span class="score-badge">Score: {{ "%.3f" | format(m.score) }}</span>
        </div>
        <div style="font-size:0.8rem; color:var(--text-secondary);">
          Eval [<code>#{{ m.eval_index }}</code>] &harr; Train [<code>#{{ m.train_index }}</code>]
        </div>
      </div>
      <div class="match-card-body">
        <div class="text-block">
          <div class="text-label">
            <span>Word-Level Diff Comparison</span>
            <span class="text-label-tag">Train &rarr; Eval Diff</span>
          </div>
          <div class="diff-content">{{ m.diff_html }}</div>
        </div>

        <div class="text-block raw-text-block">
          <div class="text-label">
            <span>Evaluation Text</span>
            <span class="text-label-tag">Index #{{ m.eval_index }}</span>
          </div>
          <div class="text-content">{{ m.eval_text_escaped }}</div>
        </div>

        <div class="text-block raw-text-block">
          <div class="text-label">
            <span>Training Text</span>
            <span class="text-label-tag">Index #{{ m.train_index }}</span>
          </div>
          <div class="text-content">{{ m.train_text_escaped }}</div>
        </div>
      </div>
    </div>
    {% endfor %}
  </section>
  {% else %}
  <div class="empty-state">
    <div class="empty-icon">✅</div>
    <div class="empty-title">No contamination detected</div>
    <p>None of the evaluation samples exceeded the similarity threshold against the training set.</p>
  </div>
  {% endif %}

  <!-- Footer -->
  <footer>
    <div>Generated by <strong>Verascan v{{ version }}</strong></div>
    <div><a href="https://github.com/balamuruganpg/verascan" target="_blank" rel="noopener">Documentation &amp; Source</a></div>
  </footer>

</div>

<script>
  let currentMethod = 'all';
  let searchQuery = '';
  let detailsExpanded = true;

  function filterMethod(method) {
    currentMethod = method;
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.getElementById('btn-' + method);
    if (activeBtn) activeBtn.classList.add('active');
    applyFilters();
  }

  function handleSearch() {
    searchQuery = document.getElementById('searchInput').value.toLowerCase().trim();
    applyFilters();
  }

  function applyFilters() {
    const cards = document.querySelectorAll('.match-card');
    cards.forEach(card => {
      const method = card.getAttribute('data-method');
      const text = card.getAttribute('data-text').toLowerCase();
      const evalIdx = card.getAttribute('data-eval-idx');
      const trainIdx = card.getAttribute('data-train-idx');

      const matchesMethod = (currentMethod === 'all' || method === currentMethod);
      const matchesSearch = (!searchQuery || text.includes(searchQuery) || evalIdx === searchQuery || trainIdx === searchQuery);

      if (matchesMethod && matchesSearch) {
        card.style.display = 'block';
      } else {
        card.style.display = 'none';
      }
    });
  }

  function toggleAllDetails() {
    detailsExpanded = !detailsExpanded;
    const blocks = document.querySelectorAll('.raw-text-block');
    blocks.forEach(b => {
      b.style.display = detailsExpanded ? 'flex' : 'none';
    });
    document.getElementById('toggleDetailsBtn').innerText = detailsExpanded ? 'Collapse All' : 'Expand All';
  }
</script>
</body>
</html>
"""
)
