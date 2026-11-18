"""Regression coverage for the VeraScan 0.6.0.post1 review."""

from __future__ import annotations

import json
import re
import sys
from html import unescape
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import verascan
from verascan.audit import AuditReport
from verascan.cli import app
from verascan.loaders import load_texts
from verascan.report import ContaminationReport, MatchRecord

runner = CliRunner()


def _jsonl(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    return path


@pytest.fixture
def rows(tmp_path: Path) -> Path:
    return _jsonl(tmp_path / "rows.jsonl", ["shared"])


def test_csv_na_tokens_stay_distinct(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text('text\nNA\nnull\nN/A\n\n""\n"  "\n00123\n1e2\n10\n', encoding="utf-8")
    texts = ["NA", "null", "N/A", "", "  ", "00123", "1e2", "10"]
    assert load_texts(path) == texts

    for eval_source in (path, _jsonl(tmp_path / "rows.jsonl", texts)):
        report = verascan.check(["NA", "123", "100.0"], eval_source, methods=["exact"])
        assert report.contaminated_indices() == {0}

    exported = tmp_path / "export.csv"
    verascan.check([], texts + ["a\rb", "one\r\ntwo"], methods=["exact"]).to_cleaned(str(exported))
    assert load_texts(exported) == texts + ["a\rb", "one\r\ntwo"]


@pytest.mark.parametrize("source_type", ["jsonl", "dataframe", "records", "hf"])
def test_missing_text_cannot_match_literal_none_or_nan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_type: str
) -> None:
    records = [{"text": "None"}, {"text": "nan"}, {"text": None}, {"text": float("nan")}]
    if source_type == "jsonl":
        source = tmp_path / "missing.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in records[:-1]), encoding="utf-8")
    elif source_type == "dataframe":
        source = pd.DataFrame(records)
    elif source_type == "hf":

        class Dataset:
            column_names = ["text"]

            def __getitem__(self, column: str) -> list:
                return [row[column] for row in records]

        monkeypatch.setattr("verascan.loaders._HF_DATASET_TYPE", Dataset)
        source = Dataset()
    else:
        source = records

    with pytest.raises(ValueError, match="Missing text"):
        verascan.check(["None", "nan"], source, methods=["exact"])
    assert load_texts(records[:2]) == ["None", "nan"]


@pytest.mark.parametrize("body", ['[\n  {"text": "alpha"}\n]', '{\n  "text": "alpha"\n}'])
def test_json_document_loads_objects_and_arrays(tmp_path: Path, body: str) -> None:
    path = tmp_path / "records.json"
    path.write_text(body, encoding="utf-8")
    assert load_texts(path) == ["alpha"]


@pytest.mark.parametrize("extension", ["csv", "json", "jsonl"])
def test_dataset_exports_reload_through_load_texts(tmp_path: Path, extension: str) -> None:
    texts = ["NA", "null", "", 'A "quote", plus\na newline', "00123"]
    eval_data = pd.DataFrame({"prompt": texts, "id": range(len(texts))})
    report = verascan.check(["NA", ""], eval_data, column="prompt", methods=["exact"])
    for write, expected in [
        (report.to_cleaned, [texts[i] for i in (1, 3, 4)]),
        (report.to_contaminated, [texts[i] for i in (0, 2)]),
    ]:
        path = tmp_path / f"report.{extension}"
        write(str(path))
        assert load_texts(path, column="prompt") == expected

    train_path = tmp_path / f"train.{extension}"
    eval_path = tmp_path / f"eval.{extension}"
    train, evaluation = verascan.split(
        eval_data,
        column="prompt",
        eval_size=0.6,
        methods=["exact"],
        output_train=str(train_path),
        output_eval=str(eval_path),
    )
    assert load_texts(train_path, column="prompt") == train["prompt"].tolist()
    assert load_texts(eval_path, column="prompt") == evaluation["prompt"].tolist()

    verascan.split(
        eval_data, column="prompt", eval_size=1.0, methods=["exact"], output_train=str(train_path)
    )
    assert load_texts(train_path, column="prompt") == []


@pytest.mark.parametrize("extension", ["csv", "json", "jsonl"])
def test_dataset_exports_preserve_text_before_dataframe_inference(
    tmp_path: Path, extension: str
) -> None:
    data = [{"text": 9_007_199_254_740_993}, {"text": 1.5}]
    path = tmp_path / f"numeric.{extension}"
    texts = load_texts(data)
    verascan.check([], data, methods=["exact"]).to_cleaned(str(path))
    assert load_texts(path) == texts
    verascan.split(data, eval_size=1.0, methods=["exact"], seed=0, output_eval=str(path))
    assert load_texts(path) == texts


@pytest.mark.parametrize("body", ['"ordinary text"\n', '[{"text": "value"}]\n', "null\n"])
def test_invalid_jsonl_row_is_cli_error(tmp_path: Path, rows: Path, body: str) -> None:
    invalid = tmp_path / "bad.jsonl"
    invalid.write_text(body, encoding="utf-8")
    result = runner.invoke(
        app, ["check", "--train", str(rows), "--eval", str(invalid), "--methods", "exact"]
    )
    assert result.exit_code == 2
    assert result.stderr.startswith("Error: ") and "expected a JSON object" in result.stderr
    assert "Traceback" not in result.output


def test_empty_methods_rejected_by_check_and_split() -> None:
    with pytest.raises(ValueError, match="At least one detection method"):
        verascan.check(["shared"], ["shared"], methods=[])
    with pytest.raises(ValueError, match="At least one detection method"):
        verascan.split([], methods=[])


@pytest.mark.parametrize(
    "command, options",
    [
        ("check", ["--train", "{rows}", "--eval", "{rows}", "--fail-above", "0"]),
        ("audit", ["--train", "{rows}", "--synthetic"]),
        ("split", ["--input", "{rows}", "--eval-size", "1.0"]),
    ],
)
@pytest.mark.parametrize("methods", ["", ",", "   "])
def test_empty_methods_escapes_no_cli_gate(
    rows: Path, command: str, options: list[str], methods: str
) -> None:
    args = [option.format(rows=rows) for option in options]
    result = runner.invoke(app, [command, *args, "--methods", methods])
    assert result.exit_code == 2
    assert result.stderr.startswith("Error: At least one detection method")


@pytest.mark.parametrize(
    "eval_size, unique_matches, html_status, pdf_status, dashboard_status",
    [
        (3_000_000, 1, "Low Risk", "LOW OVERLAP", "Low"),
        (100, 4, "Low Risk", "LOW OVERLAP", "Low"),
        (1001, 50, "Low Risk", "LOW OVERLAP", "Low"),
        (100, 5, "High Risk", "HIGH OVERLAP", "High"),
        (100, 0, "Clean", "CLEAN", "Clean"),
        (0, 0, "Clean", "CLEAN", "Clean"),
    ],
)
def test_contamination_rate_and_status_agree_across_surfaces(
    tmp_path: Path,
    eval_size: int,
    unique_matches: int,
    html_status: str,
    pdf_status: str,
    dashboard_status: str,
) -> None:
    from pypdf import PdfReader

    matches = [MatchRecord(i, 0, "shared", "shared", 1.0, "exact") for i in range(unique_matches)]
    matches += matches[:1]  # A second pair does not flag another evaluation row.
    report = ContaminationReport(train_size=1, eval_size=eval_size, matches=matches)
    paths = {ext: tmp_path / f"report.{ext}" for ext in ("json", "html", "pdf")}
    report.to_json(str(paths["json"]))
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["contamination_rate"] == (unique_matches / eval_size if eval_size else 0)
    report.to_html(str(paths["html"]))
    rendered = paths["html"].read_text(encoding="utf-8")
    assert f"<h2>{html_status}" in rendered
    width = re.search(r"style='width: ([\d.eE+-]+)%;'", rendered)
    assert width and float(width[1]) == pytest.approx(data["contamination_rate"] * 100)
    report.to_pdf(str(paths["pdf"]))
    cover = PdfReader(paths["pdf"]).pages[0].extract_text()
    assert pdf_status in cover
    assert report.formatted_contamination_rate in cover.splitlines()
    if unique_matches and dashboard_status == "Low":
        assert report.formatted_contamination_rate not in {"0.0%", "5.0%"}

    from verascan.dashboard import report_metrics

    assert report_metrics(data) == (unique_matches, data["contamination_rate"], dashboard_status)


def test_audit_split_routes_benchmark_slices_and_explicit_train_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import import_module

    module = import_module("verascan.audit")
    benchmark_splits = []
    train_splits = []

    def load_benchmark(name: str, *, split: str, **kwargs: object) -> list[dict]:
        benchmark_splits.append(split)
        return [{"text": "result", "benchmark": name}]

    def load_train(source: str, *, split: str | None, **kwargs: object) -> list[str]:
        train_splits.append(split)
        return ["train"]

    monkeypatch.setattr(module, "load_benchmark", load_benchmark)
    monkeypatch.setattr("verascan.core.load_texts", load_train)
    for split, explicit_train in [
        ("test[:100]", None),
        ("latest", None),
        ("pretrain", None),
        ("train[:10]", None),
        ("test[:3]", "custom_train"),
    ]:
        verascan.audit(
            "org/dataset",
            benchmarks=["gsm8k"],
            methods=["exact"],
            split=split,
            train_split=explicit_train,
        )
    assert benchmark_splits == ["test[:100]", "latest", "pretrain", "test", "test[:3]"]
    assert train_splits == [None, None, None, "train[:10]", "custom_train"]


def test_audit_rejects_ignored_local_train_split(rows: Path) -> None:
    with pytest.raises(ValueError, match="train split requires a Hugging Face dataset name"):
        verascan.audit(rows, benchmarks=["gsm8k"], synthetic=True, split="train[:10]")


def test_synthetic_stand_ins_are_labeled_per_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    from verascan import benchmarks

    def unavailable(split: str) -> list[dict]:
        raise OSError("offline")

    monkeypatch.setattr(benchmarks, "check_datasets_available", lambda: None)
    monkeypatch.setattr(benchmarks, "_load_hf_gsm8k", unavailable)
    monkeypatch.setattr(
        benchmarks, "_load_hf_mmlu", lambda split: [{"benchmark": "mmlu", "text": "real row"}]
    )
    report = verascan.audit(
        ["real row", benchmarks.SYNTHETIC_BENCHMARKS["gsm8k"][0]["text"]],
        benchmarks=["gsm8k", "mmlu"],
        allow_synthetic=True,
        methods=["exact"],
    )
    data = report.to_dict()
    assert data["synthetic"] is True
    assert data["benchmarks"] == ["gsm8k (synthetic)", "mmlu"]
    assert data["benchmark_counts"] == {"gsm8k (synthetic)": 1, "mmlu": 1}
    assert [match["benchmark"] for match in data["matches"]] == data["benchmarks"]
    assert "gsm8k (synthetic), mmlu" in report.summary()


@pytest.mark.parametrize(
    "command, extension, prefix",
    [
        ("check", "HtML", "<!doctype html>"),
        ("audit", "JsON", "{"),
    ],
)
def test_report_output_suffix_is_case_insensitive(
    tmp_path: Path, rows: Path, command: str, extension: str, prefix: str
) -> None:
    output = tmp_path / f"report.{extension}"
    args = ["--eval", str(rows)] if command == "check" else ["--benchmarks", "gsm8k", "--synthetic"]
    result = runner.invoke(
        app, [command, "--train", str(rows), "--methods", "exact", "--output", str(output), *args]
    )
    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8").lstrip().lower().startswith(prefix)


def test_cli_eval_size_one_point_zero_is_full_ratio(tmp_path: Path) -> None:
    rows = _jsonl(tmp_path / "rows.jsonl", [f"row {i}" for i in range(10)])
    output = tmp_path / "eval.jsonl"
    for argument, expected_count in [("1.0", 10), ("1", 1), ("0.5", 5), ("10", 10)]:
        result = runner.invoke(
            app,
            [
                "split",
                "--input",
                str(rows),
                "--eval-size",
                argument,
                "--methods",
                "exact",
                "--output-eval",
                str(output),
            ],
        )
        assert result.exit_code == 0, result.output
        assert len(load_texts(output)) == expected_count


def test_drop_summary_uses_input_size_and_reports_dropped_rows(tmp_path: Path) -> None:
    rows = _jsonl(tmp_path / "rows.jsonl", ["LEAK"] * 6 + [f"unique {i}" for i in range(4)])
    result = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(rows),
            "--eval-size",
            "0.5",
            "--methods",
            "exact",
            "--move-to",
            "drop",
            "--seed",
            "0",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Total samples   : 10" in result.output
    assert "Dropped rows    : 3" in result.output
    assert "5 samples (50.0% of input)" in result.output
    assert "2 samples (20.0% of input)" in result.output
    assert "0.0% with exact on retained rows" in result.output
    assert "guaranteed leak-free" not in result.output


def test_fuzzy_threshold_one_matches_identical_signatures(rows: Path) -> None:
    from verascan.engines.fuzzy import find_fuzzy_matches

    hits = find_fuzzy_matches(
        ["shared", "different", "SHARED "],
        ["  Shared", "unrelated"],
        threshold=1.0,
        show_progress=False,
    )
    assert [(m.eval_index, m.train_index, m.score) for m in hits] == [(0, 0, 1.0), (0, 2, 1.0)]
    result = runner.invoke(
        app, ["check", "--train", str(rows), "--eval", str(rows), "--threshold", "1"]
    )
    assert result.exit_code == 0, result.output


def test_ngram_report_labels_retained_overlap_and_scopes_threshold(tmp_path: Path) -> None:
    shared = " ".join(f"word{i}" for i in range(13))
    evaluation = shared + " " + " ".join(f"extra{i}" for i in range(13))
    report = verascan.check([shared], [evaluation], methods=["ngram"], threshold=0.99)
    assert report.matches[0].score < 0.99
    assert "any retained overlap" in report.summary()
    assert "0.99" not in report.summary()
    html_path = tmp_path / "ngram.html"
    report.to_html(str(html_path))
    assert "any retained overlap" in html_path.read_text(encoding="utf-8")


def test_html_escapes_benchmark_text_and_attributes(tmp_path: Path) -> None:
    name = """R&D <Round "1"> ('pilot')"""
    report = AuditReport(
        train_size=1,
        eval_size=1,
        benchmarks=[name],
        matches=[MatchRecord(0, 0, "shared", "shared", 1.0, "exact")],
        eval_records=[{"benchmark": name}],
    )
    output = tmp_path / "safe.html"
    report.to_html(str(output))
    rendered = output.read_text(encoding="utf-8")
    assert name not in rendered
    attributes = re.findall(r"data-benchmark='([^']*)'", rendered)
    assert [unescape(value) for value in attributes].count(name) == 2
    assert "onclick='filterBenchmark(this)'" in rendered


@pytest.mark.parametrize("command", ["check", "split"])
def test_missing_semantic_extra_has_clean_cli_error(
    rows: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "faiss", None)
    if command == "check":
        args = ["--train", str(rows), "--eval", str(rows)]
    else:
        _jsonl(rows, ["shared", "different"])
        args = ["--input", str(rows), "--eval-size", "0.5"]
    result = runner.invoke(app, [command, *args, "--methods", "semantic", "--no-progress"])
    assert result.exit_code == 2
    assert result.stderr.startswith("Error: Semantic matching is unavailable")
    assert "Traceback" not in result.output


def test_report_output_path_error_is_a_cli_user_error(tmp_path: Path, rows: Path) -> None:
    output = tmp_path / "directory.json"
    output.mkdir()
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            str(rows),
            "--eval",
            str(rows),
            "--methods",
            "exact",
            "--output",
            str(output),
            "--no-progress",
        ],
    )
    assert result.exit_code == 2
    assert result.stderr.startswith("Error: ")
    assert "Traceback" not in result.output


def test_invalid_cli_option_uses_error_contract() -> None:
    result = runner.invoke(app, ["check", "--unknown-option"])
    assert result.exit_code == 2
    assert "Error: No such option" in result.stderr
    assert "Traceback" not in result.output
