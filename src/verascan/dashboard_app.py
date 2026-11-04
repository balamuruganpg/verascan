"""Local report review workspace, launched by ``verascan dashboard``."""

from __future__ import annotations

import html
import math
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from verascan._version import __version__
from verascan.dashboard import demo_report, load_report
from verascan.dashboard_style import dashboard_css, match_table_html, theme_palette
from verascan.report import _word_diff_html

st.set_page_config(
    page_title="Verascan | Audit overview", page_icon=":material/verified_user:", layout="wide"
)


@st.cache_data(ttl=60, max_entries=4)
def read_report(path: str, modified_ns: int) -> dict[str, Any]:
    """Invalidate cached JSON when its file changes."""
    return load_report(path)


demo = "--demo" in sys.argv[1:]
try:
    if len(sys.argv) < 2:
        raise ValueError("Open a report with: verascan dashboard report.json")
    report_path = Path(sys.argv[1])
    data = (
        demo_report()
        if demo
        else read_report(str(report_path.resolve()), report_path.stat().st_mtime_ns)
    )
except (OSError, ValueError) as exc:
    st.error(f"Cannot open report: {exc}")
    st.stop()

columns = ["eval_index", "train_index", "method", "benchmark", "score", "eval_text", "train_text"]
frame = pd.DataFrame(data["matches"]).reindex(columns=columns)
frame["benchmark"] = frame["benchmark"].fillna("custom")
flagged = frame["eval_index"].nunique()
rate = flagged / data["eval_size"] if data["eval_size"] else 0.0
percent = f"{rate * 100:.6f}".rstrip("0").rstrip(".")
rate_label = (percent if "." in percent else percent + ".0") + "%"
if flagged and float(percent) == 0:
    rate_label = f"{rate * 100:.6g}%"
status, tone = (
    ("Clean", "clean") if not flagged else (("Low", "low") if rate < 0.05 else ("High", "high"))
)
if not data["eval_size"]:
    status, tone = "No evaluation data", "neutral"
methods = list(dict.fromkeys(data.get("methods_used", []) + frame["method"].tolist()))
benchmarks = list(dict.fromkeys(data.get("benchmarks", []) + frame["benchmark"].tolist()))
source = "Demo / planted matches" if demo else report_path.name

with st.container(key="masthead"):
    brand, document, appearance = st.columns([4, 2.7, 1.5], vertical_alignment="center")
    with brand:
        st.html(
            """<div class="vs-brand"><span class="vs-logo" aria-hidden="true"><span class="vs-logo-mark">V</span></span><span>verascan<span class="vs-brand-sub">EVALUATION INTEGRITY</span></span></div>"""
        )
    with document:
        st.html(
            f'<div class="vs-document"><span class="vs-doc-icon" aria-hidden="true">▤</span>{html.escape(source)}</div>'
        )
    with appearance:
        light = st.toggle("Light mode", key="light_theme")
palette = theme_palette(light)
st.html(f"<style>{dashboard_css(light)}</style>")
st.html(
    """<div class="vs-page-heading"><span class="vs-eyebrow">WORKSPACE <span class="vs-slash">/</span> AUDIT REPORT</span><h1>Audit overview<span class="vs-heading-dot">.</span></h1><p>A clear picture of your data. Every overlap, in context.</p></div>"""
)
if demo:
    st.html(
        '<div class="vs-demo"><strong>DEMO REPORT</strong><span>Five planted matches for exploration. Your real audit is unchanged.</span></div>'
    )

headline = (
    "No overlap detected.<br/>A clear evaluation set."
    if not flagged
    else "Overlap detected.<br/>Review the evidence."
)
description = f"{flagged:,} of {data['eval_size']:,} evaluation examples flagged across {len(frame):,} matching pairs."
if not data["eval_size"]:
    headline = "There is no evaluation data to assess."
    description = "Run an audit with evaluation examples to measure contamination."
st.html(f"""
<section class="vs-hero vs-{tone}" aria-label="Audit status">
<div class="vs-hero-copy"><span class="vs-status"><span class="vs-live-dot"></span>{status}</span>
<h2>{headline}</h2><p>{description}</p>
<div class="vs-tags"><span>{html.escape(", ".join(methods) or "No methods recorded")}</span>
<span>{html.escape(", ".join(benchmarks) or "Custom evaluation set")}</span></div></div>
<div class="vs-rate{" vs-rate-precise" if len(rate_label) > 7 else ""}"><div class="vs-eyebrow">CONTAMINATION RATE</div><strong>{rate_label}</strong>
<span>{flagged:,} / {data["eval_size"]:,} evaluation examples</span>
<div class="vs-rate-track" aria-hidden="true"><i style="width:{min(100, rate * 100):.6f}%"></i></div></div></section>
""")

with st.container(horizontal=True, key="metrics"):
    st.metric("Training examples", f"{data['train_size']:,}", border=True)
    st.metric("Evaluation examples", f"{data['eval_size']:,}", border=True)
    st.metric("Matching pairs", f"{len(frame):,}", border=True)
    st.metric("Contamination rate", rate_label, border=True)

if frame.empty:
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        if data["eval_size"]:
            st.html("""<section class="vs-empty"><div class="vs-check">&#10003;</div>
<span class="vs-eyebrow">AUDIT COMPLETE</span><h2>No contamination detected</h2>
<p>No evaluation examples met the matching criteria in this report.
Your results are ready to review and share.</p>
<div class="vs-empty-note">A clean result applies to the examples, methods and threshold used in this audit.</div></section>""")
        else:
            st.html(
                '<div class="vs-notice" role="status"><strong>No evaluation data</strong><p>No evaluation examples were audited. An empty report cannot establish a clean result.</p></div>'
            )
    with right:
        st.html(f"""<section class="vs-scope"><span class="vs-eyebrow">AUDIT SCOPE</span>
<h3>What was checked</h3><dl><dt>Detection methods</dt><dd>{html.escape(", ".join(methods) or "None recorded")}</dd>
<dt>Benchmarks</dt><dd>{html.escape(", ".join(benchmarks) or "Custom evaluation set")}</dd>
<dt>Similarity threshold</dt><dd>{html.escape(str(data.get("threshold", "Not recorded")))}</dd>
<dt>Report</dt><dd>{html.escape(source)}</dd></dl></section>""")
else:
    st.html(
        '<div class="vs-section-title"><h2>Where overlap appears</h2><p>Matching pairs across the complete report.</p></div>'
    )
    for container, field, title in zip(
        st.columns(2, gap="large"),
        ("method", "benchmark"),
        ("Matches by method", "Matches by benchmark"),
    ):
        with container, st.container(key=f"chart_{field}"):
            st.subheader(title)
            counts = frame[field].value_counts().rename_axis("Category").reset_index(name="Pairs")
            chart = (
                alt.Chart(counts)
                .mark_bar(color=palette["fg"], cornerRadiusEnd=3, size=14)
                .encode(
                    x=alt.X(
                        "Pairs:Q",
                        title=None,
                        axis=alt.Axis(
                            tickMinStep=1,
                            gridColor=palette["border"],
                            labelColor=palette["muted"],
                            domain=False,
                            tickColor=palette["border"],
                        ),
                    ),
                    y=alt.Y(
                        "Category:N",
                        title=None,
                        sort="-x",
                        axis=alt.Axis(
                            labelColor=palette["fg"],
                            labelFontSize=12,
                            labelPadding=12,
                            domain=False,
                            ticks=False,
                        ),
                    ),
                    tooltip=["Category:N", "Pairs:Q"],
                )
                .properties(height=150)
                .configure(background=palette["card"], font="Segoe UI")
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, alt=title, width="stretch", theme=None)

    st.html(
        '<div class="vs-section-title"><h2>Inspect the evidence</h2><p>Filter matching pairs, then compare their original text.</p></div>'
    )
    with st.container(key="filters"):
        a, b, c = st.columns([1, 1, 1], gap="large")
        with a:
            selected_methods = st.multiselect("Method", methods, default=methods, key="methods")
        with b:
            selected_benchmarks = st.multiselect(
                "Benchmark", benchmarks, default=benchmarks, key="benchmarks"
            )
        with c:
            min_score = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.01, key="score")
        query = st.text_input(
            "Search text",
            placeholder="Search training and evaluation text…",
            key="query",
            icon=":material/search:",
        )
    filtered = frame.loc[
        frame["method"].isin(selected_methods)
        & frame["benchmark"].isin(selected_benchmarks)
        & (frame["score"] >= min_score)
    ]
    if query:
        filtered = filtered.loc[
            filtered["eval_text"].str.contains(query, case=False, regex=False, na=False)
            | filtered["train_text"].str.contains(query, case=False, regex=False, na=False)
        ]
    st.caption(
        f"{len(filtered):,} of {len(frame):,} matching pairs · Scores describe similarity, not certainty of leakage."
    )
    if filtered.empty:
        st.html(
            '<div class="vs-notice" role="status"><strong>No pairs match these filters.</strong><p>Lower the minimum score or clear your search to explore more evidence.</p></div>'
        )
    else:
        pages = math.ceil(len(filtered) / 25)
        if st.session_state.get("page", 1) > pages:
            st.session_state["page"] = 1
        page = 1
        if pages > 1:
            page = int(
                st.number_input(
                    "Page", min_value=1, max_value=pages, value=1, step=1, key="page", width=140
                )
            )
        visible = filtered.iloc[(page - 1) * 25 : page * 25]
        st.html(match_table_html(visible.to_dict(orient="records")))
        selected = st.selectbox(
            "Compare a pair",
            visible.index.tolist(),
            key="pair",
            format_func=lambda idx: (
                f"Eval {frame.loc[idx, 'eval_index']} / Train {frame.loc[idx, 'train_index']} · {frame.loc[idx, 'benchmark']} · {frame.loc[idx, 'method']} · {frame.loc[idx, 'score']:.4f}"
            ),
        )
        row = frame.loc[selected]
        for container, field, label in zip(
            st.columns(2, gap="large"),
            ("train_text", "eval_text"),
            ("Training text", "Evaluation text"),
        ):
            with container:
                st.html(
                    f'<section class="vs-text-card"><span class="vs-eyebrow">{label}</span><div class="vs-pair-text">{html.escape(str(row[field]))}</div></section>'
                )
        with st.expander("Word-level differences", expanded=True):
            st.caption("Strikethrough: removed from training. Underline: added in evaluation.")
            st.html(
                f'<div class="vs-diff">{_word_diff_html(row["train_text"], row["eval_text"])}</div>'
            )

st.html(f"""<footer class="vs-footer"><span><strong>verascan</strong> / v{__version__}</span>
<span>Clean = 0% · Low &lt; 5% · High ≥ 5% · Display rounded to 6 decimals</span>
<a href="https://github.com/balamuruganpg/verascan" target="_blank" rel="noopener noreferrer">View source ↗</a></footer>""")
