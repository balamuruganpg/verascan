"""Monochrome dashboard tokens and safe, theme-aware table presentation."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def theme_palette(light: bool) -> dict[str, str]:
    """Keep all non-status colors on the grayscale axis."""
    if light:
        return {
            "bg": "#ffffff",
            "surface": "#fafafa",
            "card": "#f5f5f5",
            "field": "#ffffff",
            "raised": "#ebebeb",
            "fg": "#000000",
            "muted": "#666666",
            "subtle": "#737373",
            "border": "#e5e5e5",
            "border-hover": "#a3a3a3",
            "clean": "#15803d",
            "low": "#a16207",
            "high": "#b91c1c",
        }
    return {
        "bg": "#000000",
        "surface": "#0a0a0a",
        "card": "#101010",
        "field": "#141414",
        "raised": "#232323",
        "fg": "#ffffff",
        "muted": "#a3a3a3",
        "subtle": "#8c8c8c",
        "border": "#262626",
        "border-hover": "#737373",
        "clean": "#4ade80",
        "low": "#fbbf24",
        "high": "#f87171",
    }


def dashboard_css(light: bool) -> str:
    """Apply a per-session theme without altering server or other users' settings."""
    tokens = "; ".join(f"--vs-{name}: {value}" for name, value in theme_palette(light).items())
    base = Path(__file__).with_name("dashboard.css").read_text(encoding="utf-8")
    return f":root {{color-scheme: {'light' if light else 'dark'}; {tokens};}}\n{base}"


def match_table_html(rows: list[dict[str, Any]]) -> str:
    """Render a bounded page of pairs; escape every dataset string."""
    body: list[str] = []
    for row in rows:
        score = float(row["score"])
        bar = round(max(0, min(1, score)) * 100)
        evaluation, training = int(row["eval_index"]), int(row["train_index"])
        cells = {
            key: html.escape(str(row[key]))
            for key in ("method", "benchmark", "train_text", "eval_text")
        }
        body.append(
            f'<tr data-eval="{evaluation}" data-train="{training}">'
            f'<td><span class="vs-score">{score:.4f}</span><span class="vs-score-track" aria-hidden="true"><i style="width:{bar}%"></i></span></td>'
            f'<td><span class="vs-method">{cells["method"]}</span></td>'
            f'<td class="vs-benchmark">{cells["benchmark"]}</td>'
            f'<td><div class="vs-table-text">{cells["train_text"]}</div></td>'
            f'<td><div class="vs-table-text">{cells["eval_text"]}</div></td>'
            f'<td class="vs-ref">T{training}<br/>E{evaluation}</td></tr>'
        )
    headers = "".join(
        f'<th scope="col">{name}</th>'
        for name in ("Score", "Method", "Benchmark", "Training text", "Evaluation text", "Rows")
    )
    return (
        '<div class="vs-table-wrap"><table class="vs-match-table">'
        '<caption class="vs-sr-only">Matching pairs on this page</caption>'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )
