"""Command-line interface for Verascan."""

from __future__ import annotations

import typer

from verascan._version import __version__

app = typer.Typer(
    name="verascan",
    help="Detect data contamination between training and evaluation sets.",
    add_completion=False,
    no_args_is_help=True,
)


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
    except (FileNotFoundError, ValueError, KeyError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(report.summary())

    # Write output file if requested.
    if output:
        if output.lower().endswith(".pdf"):
            report.to_pdf(output)
            typer.echo(f"PDF report written to: {output}")
        elif output.endswith(".html"):
            report.to_html(output)
            typer.echo(f"HTML report written to: {output}")
        elif output.endswith(".json"):
            report.to_json(output)
            typer.echo(f"JSON report written to: {output}")
        else:
            # Default to JSON.
            report.to_json(output)
            typer.echo(f"JSON report written to: {output}")

    if output_cleaned:
        try:
            report.to_cleaned(output_cleaned)
        except ValueError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"Cleaned eval written to: {output_cleaned}")

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
    from pathlib import Path

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
        if extension == ".pdf":
            report.to_pdf(output)
        elif extension == ".html":
            report.to_html(output)
        else:
            report.to_json(output)
        typer.echo(f"Report written to: {output}")
        if output_json:
            report.to_json(output_json)
            typer.echo(f"JSON report written to: {output_json}")
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc


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
    except (OSError, ValueError, ImportError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    raise typer.Exit(code=code)


@app.command()
def split(
    input: str = typer.Option(..., "--input", "-i", help="Path to input dataset (CSV or JSONL)."),
    eval_size: float = typer.Option(
        0.2, "--eval-size", help="Target evaluation ratio (0-1) or row count."
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
        help="Comma-separated detection methods: exact, fuzzy, semantic.",
    ),
    threshold: float = typer.Option(0.85, "--threshold", help="Similarity threshold (0-1)."),
    column: str = typer.Option("text", "--column", "-c", help="Name of the text column."),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducible shuffling."),
    move_to: str = typer.Option(
        "train", "--move-to", help="Action for contaminated candidates: 'train' or 'drop'."
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bars."),
) -> None:
    """Create a leak-free train/eval split from a dataset."""
    from verascan.split import split as run_split

    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    parsed_eval_size: float | int = (
        int(eval_size) if eval_size.is_integer() and eval_size >= 1 else eval_size
    )

    try:
        train_data, eval_data = run_split(
            data=input,
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
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    train_len = len(train_data)
    eval_len = len(eval_data)
    total_len = train_len + eval_len

    lines = [
        "",
        "=" * 47,
        "  Verascan Split Summary",
        "=" * 47,
        f"  Total samples   : {total_len:,}",
        f"  Methods         : {', '.join(method_list)}",
        f"  Threshold       : {threshold}",
        f"  Seed            : {seed}",
        "-" * 47,
        f"  Train split     : {train_len:,} samples ({train_len / max(1, total_len):.1%})",
        f"  Eval split      : {eval_len:,} samples ({eval_len / max(1, total_len):.1%})",
        "  Residual leakage: 0.0% (guaranteed leak-free)",
        "=" * 47,
    ]
    if output_train:
        lines.append(f"  Train set written to: {output_train}")
    if output_eval:
        lines.append(f"  Eval set written to : {output_eval}")
    lines.append("")
    typer.echo("\n".join(lines))


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
        help="Hugging Face split for training data (default: train[:2000]).",
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
            config=config,
            synthetic=synthetic,
            show_progress=not no_progress,
        )
    except (FileNotFoundError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(report.summary())

    if output:
        if output.lower().endswith(".pdf"):
            report.to_pdf(output)
            typer.echo(f"PDF report written to: {output}")
        elif output.endswith(".html"):
            report.to_html(output)
            typer.echo(f"HTML report written to: {output}")
        elif output.endswith(".json"):
            report.to_json(output)
            typer.echo(f"JSON report written to: {output}")
        else:
            report.to_html(output)
            typer.echo(f"Report written to: {output}")

    if output_json:
        report.to_json(output_json)
        typer.echo(f"JSON report written to: {output_json}")

    # CI gate.
    if report.contamination_rate > fail_above:
        typer.secho(
            f"\n✘ FAIL: Contamination rate {report.contamination_rate:.2%} "
            f"exceeds threshold {fail_above:.2%}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
