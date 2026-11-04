"""JSON loading and optional local Streamlit dashboard launcher."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


def load_report(path: str) -> dict[str, Any]:
    """Read and validate JSON reports, including the 0.5.0 export schema."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"Invalid Verascan JSON report: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("A Verascan report must be a JSON object.")
    for key in ("train_size", "eval_size"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid report field '{key}': expected a non-negative integer.")
    matches = data.get("matches")
    if not isinstance(matches, list):
        raise ValueError("Invalid report: 'matches' must be a list.")
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("Invalid report: each match must be an object.")
        for field in ("eval_text", "train_text", "method"):
            if not isinstance(match.get(field), str):
                raise ValueError(f"Invalid report match field '{field}': expected text.")
        for field, size in (("eval_index", "eval_size"), ("train_index", "train_size")):
            index = match.get(field)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < data[size]:
                raise ValueError(f"Invalid report match '{field}': row index is out of bounds.")
        score = match.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("Invalid report match score: expected a number between -1 and 1.")
        if "benchmark" in match and not isinstance(match["benchmark"], str):
            raise ValueError("Invalid report match benchmark: expected text.")
    for field in ("methods_used", "benchmarks"):
        value = data.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"Invalid report field '{field}': expected a list of names.")
    return cast(dict[str, Any], data)


def demo_report() -> dict[str, Any]:
    """Return five labeled planted pairs without reading or writing user data."""
    from verascan import AuditReport, MatchRecord

    pairs = [
        ("What is the capital of France?", "What is the capital of France?", "exact", "mmlu", 1.0),
        (
            "A shop sells 12 apples at $2 each. What is the total?",
            "A shop sells 12 apples at $2 each. What is the total?",
            "exact",
            "gsm8k",
            1.0,
        ),
        (
            "Explain why plants need sunlight to grow.",
            "Explain why green plants need sunlight for growth.",
            "semantic",
            "mmlu",
            0.9632,
        ),
        (
            "def add(a, b):\n    return a + b",
            "def add(x, y):\n    return x + y",
            "fuzzy",
            "humaneval",
            0.9184,
        ),
        (
            "A train travels 60 miles in two hours. Find its average speed.",
            "A train covers 60 miles in 2 hours. Calculate its average speed.",
            "semantic",
            "gsm8k",
            0.8841,
        ),
    ]
    report = AuditReport(
        train_size=1000,
        eval_size=100,
        methods_used=["exact", "fuzzy", "semantic"],
        benchmarks=["mmlu", "gsm8k", "humaneval"],
        threshold=0.85,
        matches=[
            MatchRecord(i, 10 + i, evaluation, training, score, method)
            for i, (training, evaluation, method, _, score) in enumerate(pairs)
        ],
        eval_records=[{"benchmark": pair[3]} for pair in pairs],
    )
    return report.to_dict()


def launch_dashboard(path: str, *, demo: bool = False) -> int:
    """Run the packaged dashboard using this interpreter, bound to localhost."""
    if not demo:
        load_report(path)
    if importlib.util.find_spec("streamlit") is None:
        raise ImportError(
            'The dashboard requires Streamlit. Install pip install "verascan[dashboard]".'
        )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).with_name("dashboard_app.py")),
            "--server.address=127.0.0.1",
            "--browser.gatherUsageStats=false",
            "--theme.base=dark",
            "--theme.primaryColor=#ffffff",
            "--theme.backgroundColor=#000000",
            "--theme.secondaryBackgroundColor=#141414",
            "--theme.textColor=#ffffff",
            "--",
            str(Path(path).resolve()),
        ]
        + (["--demo"] if demo else []),
        check=False,
    ).returncode
