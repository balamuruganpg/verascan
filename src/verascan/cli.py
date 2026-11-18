"""Command-line interface for Verascan."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer

from verascan._version import __version__
from verascan.report import ContaminationReport

_USER_ERRORS = (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError)

app = typer.Typer(
    name="verascan",
    help="Detect data contamination between training and evaluation sets.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
)


def _fail(exc: Exception) -> NoReturn:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2) from exc


def _write_report(
    report: ContaminationReport, path: str, *, default: str, label_format: bool = True
) -> None:
    extension = Path(path).suffix.lower()
    if extension == ".pdf":
        kind, writer = "PDF", report.to_pdf
    elif extension == ".html" or (extension != ".json" and default == "html"):
        kind, writer = "HTML", report.to_html
    else:
        kind, writer = "JSON", report.to_json
    writer(path)
    label = f"{kind} report" if label_format else "Report"
    typer.echo(f"{label} written to: {path}")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"verascan {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Verascan — detect data contamination between training and evaluation sets."""


@app.command()
def check(
    train: str = typer.Option(..., "--train", "-t", help="Path to training data (CSV or JSONL)."),
    eval: str = typer.Option(..., "--eval", "-e", help="Path to evaluation data (CSV or JSONL)."),
    methods: str = typer.Option(
        "exact,fuzzy",
        "--methods",
        "-m",
        help="Comma-separated detection methods: exact, ngram, fuzzy, semantic.",
    ),
    threshold: float = typer.Option(0.85, "--threshold", help="Similarity threshold (0-1)."),
    column: str = typer.Option("text", "--column", "-c", help="Name of the text column."),
    fail_above: float = typer.Option(
        1.0,
        "--fail-above",
        help="Exit with code 1 if contamination rate exceeds this (useful for CI).",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write report to this path (.pdf, .json, or secondary .html).",
    ),
    output_cleaned: str | None = typer.Option(
        None,
        "--output-cleaned",
        help="Write decontaminated eval set to this path (.csv, .jsonl, or .json).",
    ),
    train_split: str | None = typer.Option(
        None,
        "--train-split",
        help="Hugging Face split for training data (default: train[:2000]).",
    ),
    eval_split: str | None = typer.Option(
        None,
        "--eval-split",
        help="Hugging Face split for evaluation data (default: test).",
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        help="Hugging Face dataset configuration name.",
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bars."),
) -> None:
    """Run a contamination check and print a summary."""
    from verascan.core import check as run_check

    method_list = [m.strip() for m in methods.split(",") if m.strip()]

    try:
        report = run_check(
            train=train,
            eval=eval,
            methods=method_list,
            threshold=threshold,
            column=column,
            train_split=train_split,
            eval_split=eval_split,
            config=config,
            show_progress=not no_progress,
        )
        typer.echo(report.summary())
        if output:
            _write_report(report, output, default="json")
        if output_cleaned:
            report.to_cleaned(output_cleaned)
            typer.echo(f"Cleaned eval written to: {output_cleaned}")
    except _USER_ERRORS as exc:
        _fail(exc)

    # CI gate.
    if report.contamination_rate > fail_above:
        typer.secho(
            f"\n✘ FAIL: Contamination rate {report.contamination_rate:.2%} "
            f"exceeds threshold {fail_above:.2%}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def gpu(
    train: str = typer.Option(
        ..., "--train", "-t", help="Training file or Hugging Face dataset name."
    ),
    benchmarks: str = typer.Option("mmlu,gsm8k,humaneval", "--benchmarks", "-b"),
    column: str = typer.Option("text", "--column", "-c"),
    device: str = typer.Option("cuda", "--device", help="NVIDIA CUDA device: cuda or cuda:N."),
    batch_size: int = typer.Option(
        256, "--batch-size", min=1, help="Embedding/index/search batch size."
    ),
    threshold: float = typer.Option(0.85, "--threshold", min=0.0, max=1.0),
    model_name: str = typer.Option("all-MiniLM-L6-v2", "--model-name"),
    max_train: int | None = typer.Option(None, "--max-train", min=1, help="Limit training rows."),
    max_eval: int | None = typer.Option(
        None, "--max-eval", min=1, help="Limit rows per benchmark."
    ),
    train_split: str | None = typer.Option(
        None, "--train-split", help="HF training split (default: train)."
    ),
    benchmark_split: str = typer.Option("test", "--benchmark-split"),
    config: str | None = typer.Option(None, "--config", help="HF training dataset configuration."),
    output: str = typer.Option(
        "report.pdf", "--output", "-o", help="Report path: .pdf, .json, or .html."
    ),
    output_json: str | None = typer.Option(
        None, "--output-json", help="JSON sidecar for the dashboard."
    ),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use offline benchmark stand-ins."),
    no_progress: bool = typer.Option(False, "--no-progress"),
) -> None:
    """Audit public benchmarks using NVIDIA CUDA embeddings and GPU-preferred FAISS."""
    from verascan.gpu import cuda_info, gpu_audit

    try:
        extension = Path(output).suffix.lower()
        if extension not in {".pdf", ".json", ".html"}:
            raise ValueError("GPU report output must end in .pdf, .json, or .html.")
        if output_json and Path(output).resolve() == Path(output_json).resolve():
            raise ValueError("--output and --output-json must use different paths.")
        info = cuda_info(device)
        typer.echo(
            f"CUDA device: {info['name']} ({info['device']}); runtime {info['cuda_version']}"
        )
        typer.echo(
            f"Memory: {info['free_memory'] / 1024**3:.2f} GiB free / "
            f"{info['total_memory'] / 1024**3:.2f} GiB total; batch size {batch_size}"
        )
        report = gpu_audit(
            train,
            benchmarks,
            column=column,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            model_name=model_name,
            max_train=max_train,
            max_eval=max_eval,
            train_split=train_split,
            config=config,
            benchmark_split=benchmark_split,
            synthetic=synthetic,
            show_progress=not no_progress,
        )
        typer.echo(report.summary())
        _write_report(report, output, default="json", label_format=False)
        if output_json:
            report.to_json(output_json)
            typer.echo(f"JSON report written to: {output_json}")
    except _USER_ERRORS as exc:
        _fail(exc)


@app.command()
def dashboard(
    report: str = typer.Argument(..., help="Path to a Verascan JSON report."),
    demo: bool = typer.Option(
        False, "--demo", help="Preview five planted pairs without changing the report."
    ),
) -> None:
    """Open a local interactive Streamlit report dashboard."""
    from verascan.dashboard import launch_dashboard

    try:
        code = launch_dashboard(report, demo=demo)
    except _USER_ERRORS as exc:
        _fail(exc)
    raise typer.Exit(code=code)


def _parse_eval_size(value: str) -> float | int:
    try:
        return int(value)
    except ValueError:
        try:
            ratio = float(value)
        except ValueError as exc:
            raise ValueError("--eval-size requires a ratio or an integer row count.") from exc
        return int(ratio) if ratio > 1 and ratio.is_integer() else ratio


@app.command()
def split(
    input: str = typer.Option(..., "--input", "-i", help="Path to input dataset (CSV or JSONL)."),
    eval_size: str = typer.Option(
        "0.2", "--eval-size", help="Evaluation ratio (e.g. 1.0) or integer row count (e.g. 1)."
    ),
    output_train: str | None = typer.Option(
        None, "--output-train", help="Path to write train split (.csv, .jsonl, or .json)."
    ),
    output_eval: str | None = typer.Option(
        None, "--output-eval", help="Path to write eval split (.csv, .jsonl, or .json)."
    ),
    methods: str = typer.Option(
        "exact,fuzzy",
        "--methods",
        "-m",
        help="Comma-separated detection methods: exact, ngram, fuzzy, semantic.",
    ),
    threshold: float = typer.Option(0.85, "--threshold", help="Similarity threshold (0-1)."),
    column: str = typer.Option("text", "--column", "-c", help="Name of the text column."),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducible shuffling."),
    move_to: str = typer.Option(
        "train", "--move-to", help="Action for contaminated candidates: 'train' or 'drop'."
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bars."),
) -> None:
    """Create a train/eval split and remove overlap found by the selected methods."""
    from verascan.core import _validate_methods
    from verascan.loaders import load_eval_payload
    from verascan.split import _split_loaded

    try:
        method_list = _validate_methods([m.strip() for m in methods.split(",") if m.strip()])
        parsed_eval_size = _parse_eval_size(eval_size)
        payload = load_eval_payload(input, column=column)
        _, _, summary = _split_loaded(
            payload,
            eval_size=parsed_eval_size,
            methods=method_list,
            threshold=threshold,
            column=column,
            seed=seed,
            move_to=move_to,
            output_train=output_train,
            output_eval=output_eval,
            show_progress=not no_progress,
        )
    except _USER_ERRORS as exc:
        _fail(exc)

    typer.echo(summary.summary())
    if output_train:
        typer.echo(f"  Train set written to: {output_train}")
    if output_eval:
        typer.echo(f"  Eval set written to : {output_eval}")


@app.command()
def audit(
    train: str = typer.Option(..., "--train", "-t", help="Path to training data (CSV or JSONL)."),
    benchmarks: str = typer.Option(
        "mmlu,gsm8k,humaneval",
        "--benchmarks",
        "-b",
        help="Comma-separated benchmark presets: mmlu, gsm8k, humaneval.",
    ),
    methods: str = typer.Option(
        "exact,fuzzy",
        "--methods",
        "-m",
        help="Comma-separated detection methods: exact, ngram, fuzzy, semantic.",
    ),
    threshold: float = typer.Option(0.85, "--threshold", help="Similarity threshold (0-1)."),
    column: str = typer.Option(
        "text", "--column", "-c", help="Name of the text column in train data."
    ),
    fail_above: float = typer.Option(
        1.0,
        "--fail-above",
        help="Exit with code 1 if contamination rate exceeds this (useful for CI).",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write report to this path (.pdf, .json, or secondary .html).",
    ),
    output_json: str | None = typer.Option(
        None,
        "--output-json",
        help="Write machine-readable JSON report to this path.",
    ),
    synthetic: bool = typer.Option(
        False,
        "--synthetic",
        help="Use small synthetic stand-ins for testing (no Hugging Face download).",
    ),
    split: str | None = typer.Option(
        None,
        "--split",
        help="Hugging Face split or slice. Names starting with 'train' apply to training data; other names apply to benchmarks.",
    ),
    train_split: str | None = typer.Option(
        None,
        "--train-split",
        help="Explicit training dataset split (default: train[:2000]).",
    ),
    benchmark_split: str = typer.Option(
        "test", "--benchmark-split", help="Explicit split for benchmark datasets."
    ),
    config: str | None = typer.Option(
        None,
        "--config",
        help="Hugging Face dataset configuration name for train data.",
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bars."),
) -> None:
    """Audit a training dataset against popular public evaluation benchmarks."""
    from verascan.audit import audit as run_audit

    benchmark_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    method_list = [m.strip() for m in methods.split(",") if m.strip()]

    try:
        report = run_audit(
            train=train,
            benchmarks=benchmark_list,
            methods=method_list,
            threshold=threshold,
            column=column,
            split=split,
            train_split=train_split,
            benchmark_split=benchmark_split,
            config=config,
            synthetic=synthetic,
            show_progress=not no_progress,
        )
        typer.echo(report.summary())
        if output:
            _write_report(
                report,
                output,
                default="html",
                label_format=Path(output).suffix.lower() in {".pdf", ".html", ".json"},
            )
        if output_json:
            report.to_json(output_json)
            typer.echo(f"JSON report written to: {output_json}")
    except _USER_ERRORS as exc:
        _fail(exc)

    # CI gate.
    if report.contamination_rate > fail_above:
        typer.secho(
            f"\n✘ FAIL: Contamination rate {report.contamination_rate:.2%} "
            f"exceeds threshold {fail_above:.2%}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
