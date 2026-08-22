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
        help="Comma-separated detection methods: exact, fuzzy, semantic.",
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

    # CI gate.
    if report.contamination_rate > fail_above:
        typer.secho(
            f"\n✘ FAIL: Contamination rate {report.contamination_rate:.2%} "
            f"exceeds threshold {fail_above:.2%}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
