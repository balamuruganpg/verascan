"""Tests for the Typer-based CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from verascan._version import __version__
from verascan.cli import app

runner = CliRunner()


def _make_data_files(tmp_path: Path) -> tuple[str, str]:
    train = tmp_path / "train.jsonl"
    eval_ = tmp_path / "eval.jsonl"
    train.write_text(
        '{"text":"hello world"}\n{"text":"foo bar"}\n',
        encoding="utf-8",
    )
    eval_.write_text(
        '{"text":"hello world"}\n{"text":"something different"}\n',
        encoding="utf-8",
    )
    return str(train), str(eval_)


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"verascan {__version__}" in result.stdout


def test_check_basic(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--no-progress",
        ],
    )
    assert result.exit_code == 0
    assert "Verascan" in result.stdout
    assert "50.0%" in result.stdout


def test_check_with_output_json(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    out_json = str(tmp_path / "report.json")
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--output",
            out_json,
            "--no-progress",
        ],
    )
    assert result.exit_code == 0
    assert Path(out_json).exists()
    data = json.loads(Path(out_json).read_text(encoding="utf-8"))
    assert data["total_matches"] == 1


def test_check_with_output_html(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    out_html = str(tmp_path / "report.html")
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--output",
            out_html,
            "--no-progress",
        ],
    )
    assert result.exit_code == 0
    assert Path(out_html).exists()
    assert "<html" in Path(out_html).read_text(encoding="utf-8")


def test_check_fail_above(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--fail-above",
            "0.0",
            "--no-progress",
        ],
    )
    # contamination_rate is 50% which exceeds 0%, so should exit 1.
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_check_file_not_found() -> None:
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            "/nope/train.jsonl",
            "--eval",
            "/nope/eval.jsonl",
            "--no-progress",
        ],
    )
    assert result.exit_code == 2


def test_check_methods_flag(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--methods",
            "exact",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0


def test_check_output_cleaned_jsonl(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    out_cleaned = str(tmp_path / "cleaned.jsonl")
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--output-cleaned",
            out_cleaned,
            "--no-progress",
        ],
    )
    assert result.exit_code == 0
    assert Path(out_cleaned).exists()
    rows = [
        json.loads(line)
        for line in Path(out_cleaned).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows == [{"text": "something different"}]
    assert "Cleaned eval written to" in result.stdout


def test_check_output_cleaned_bad_extension(tmp_path: Path) -> None:
    train, eval_ = _make_data_files(tmp_path)
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            train,
            "--eval",
            eval_,
            "--output-cleaned",
            str(tmp_path / "cleaned.parquet"),
            "--no-progress",
        ],
    )
    assert result.exit_code == 2


def test_cli_audit_help() -> None:
    result = runner.invoke(app, ["audit", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "--train" in result.stdout
    assert "--benchmarks" in result.stdout
    assert "--synthetic" in result.stdout
    assert "--fail-above" in result.stdout


def test_cli_audit_synthetic_planted_leak(tmp_path: Path) -> None:
    from verascan.benchmarks import SYNTHETIC_BENCHMARKS

    planted = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"text": planted}) + "\n", encoding="utf-8")

    out_html = tmp_path / "audit.html"
    out_json = tmp_path / "audit.json"

    result = runner.invoke(
        app,
        [
            "audit",
            "--train",
            str(train_file),
            "--benchmarks",
            "gsm8k",
            "--synthetic",
            "--methods",
            "exact",
            "--output",
            str(out_html),
            "--output-json",
            str(out_json),
            "--no-progress",
        ],
    )

    assert result.exit_code == 0
    assert "Verascan Benchmark Audit Report" in result.stdout
    assert "gsm8k" in result.stdout
    assert "1 contaminated" in result.stdout
    assert out_html.exists()
    assert out_json.exists()

    # Verify benchmark filter and badge in generated HTML
    html_text = out_html.read_text(encoding="utf-8")
    assert "badge-benchmark" in html_text
    assert "filterBenchmark" in html_text
    assert "Audited Benchmarks" in html_text


def test_cli_audit_unknown_benchmark(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"text":"some data"}\n', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "audit",
            "--train",
            str(train_file),
            "--benchmarks",
            "nonexistent_benchmark",
            "--synthetic",
            "--no-progress",
        ],
    )

    assert result.exit_code == 2
    assert "Error:" in result.stderr
    assert "Unknown benchmark 'nonexistent_benchmark'" in result.stderr


def test_cli_audit_fail_above(tmp_path: Path) -> None:
    from verascan.benchmarks import SYNTHETIC_BENCHMARKS

    planted = SYNTHETIC_BENCHMARKS["gsm8k"][0]["question"]
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"text": planted}) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "audit",
            "--train",
            str(train_file),
            "--benchmarks",
            "gsm8k",
            "--synthetic",
            "--methods",
            "exact",
            "--fail-above",
            "0.0",
            "--no-progress",
        ],
    )

    assert result.exit_code == 1
    assert "FAIL" in result.stderr
