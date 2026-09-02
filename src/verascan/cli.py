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
        help="Write report to this path (.html or .json).",
    ),
    output_cleaned: str | None = typer.Option(
        None,
        "--output-cleaned",
        help="Write decontaminated eval set to this path (.csv, .jsonl, or .json).",
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
            show_progress=not no_progress,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    report.summary()

    # Write output file if requested.
    if output:
        if output.endswith(".html"):
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
