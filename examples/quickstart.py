"""Verascan Quickstart Example.

Demonstrates exact, fuzzy, and semantic data contamination detection
between training and evaluation sets, with terminal summary and report exports.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import verascan
from verascan.engines import semantic_available

# ── 1. Sample Datasets ──────────────────────────────────

train_corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming industries worldwide.",
    "Python is a versatile and expressive programming language.",
    "Data quality is essential for training reliable foundation models.",
    "Open-source software drives global technological innovation.",
    "Neural networks learn hierarchical representations from raw data.",
    "Transfer learning drastically reduces the need for labeled data.",
    "Attention mechanisms revolutionized natural language processing.",
]

eval_corpus = [
    "The quick brown fox jumps over the lazy dog.",  # Exact duplicate
    "Machine learning is transforming industries.",  # Near-duplicate / minor truncation
    "Rust is a fast and memory-safe systems language.",  # Clean / no contamination
    "Data quality is essential for reliable ML models.",  # Near-duplicate / phrasing variation
]

# ── 2. Run Contamination Scan ─────────────────────────────

methods = ["exact", "fuzzy"]
if semantic_available():
    methods.append("semantic")

print(f"Running Verascan audit using methods: {', '.join(methods)} ...\n")

report = verascan.check(
    train=train_corpus,
    eval=eval_corpus,
    methods=methods,
    threshold=0.5,
    show_progress=True,
)

# ── 3. Inspect Summary & Flagged Pairs ──────────────────────────

report.summary()

print(f"Contamination Rate: {report.contamination_rate:.1%}")
print(f"Exact Matches     : {report.exact_count}")
print(f"Fuzzy Matches     : {report.fuzzy_count}")
if "semantic" in methods:
    print(f"Semantic Matches  : {report.semantic_count}")
print()

print("Flagged Contamination Pairs (score >= 0.50):")
for match in report.flagged(min_score=0.5):
    print(
        f"  [{match.method:8s}] score={match.score:.3f} | Eval #{match.eval_index} <-> Train #{match.train_index}"
    )
    print(f"    Eval : {match.eval_text}")
    print(f"    Train: {match.train_text}")
    print()

# ── 4. Export Reports ───────────────────────────────

report.to_html("quickstart_report.html")
report.to_json("quickstart_report.json")
report.to_cleaned("quickstart_eval_cleaned.jsonl")
print("Saved interactive HTML report to: quickstart_report.html")
print("Saved machine-readable JSON to : quickstart_report.json")
print("Saved decontaminated eval set to: quickstart_eval_cleaned.jsonl")
