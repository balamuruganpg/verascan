"""Dashboard launch and actual Streamlit widget behavior."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from verascan import ContaminationReport, MatchRecord
from verascan.cli import app


def _match_table(view: Any) -> str:
    """Inspect the semantic table actually presented to the user."""
    return next(
        element.proto.body
        for element in view.get("html")
        if 'class="vs-match-table"' in element.proto.body
    )


def _visible_pairs(view: Any) -> list[tuple[str, str]]:
    return re.findall(r'<tr data-eval="(\d+)" data-train="(\d+)">', _match_table(view))


def test_dashboard_command_exists() -> None:
    result = CliRunner().invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0 and "REPORT" in result.output.upper()


def test_dashboard_launch_preserves_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit", minversion="1.66")
    path = tmp_path / "report with spaces.json"
    ContaminationReport().to_json(str(path))
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(app, ["dashboard", str(path)])
    assert result.exit_code == 0, result.output
    assert len(commands) == 1
    command = commands[0]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert "--server.address=127.0.0.1" in command
    assert command[-2:] == ["--", str(path.resolve())]
    assert Path(command[4]).is_file()


@pytest.mark.parametrize("payload", ["not-json", "[]", '{"matches": "wrong"}'])
def test_dashboard_invalid_report_is_clear(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    result = CliRunner().invoke(app, ["dashboard", str(path)])
    assert result.exit_code == 2
    assert "report" in result.output.lower()
    assert "traceback" not in result.output.lower()


def _dashboard_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: ContaminationReport
) -> Any:
    pytest.importorskip("streamlit", minversion="1.66")
    from streamlit.testing.v1 import AppTest

    path = tmp_path / "report.json"
    report.to_json(str(path))
    spec = importlib.util.find_spec("verascan.dashboard_app")
    assert spec is not None and spec.origin is not None
    monkeypatch.setattr(sys, "argv", [spec.origin, str(path)])
    return AppTest.from_file(spec.origin, default_timeout=20).run()


def test_dashboard_filters_and_literal_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = ContaminationReport(
        train_size=2,
        eval_size=3,
        methods_used=["exact", "semantic"],
        matches=[
            MatchRecord(0, 0, "[needle] <script>hi</script>", "train", 1.0, "exact"),
            MatchRecord(1, 1, "other", "training", 0.87, "semantic"),
        ],
    )
    view = _dashboard_app(tmp_path, monkeypatch, report)
    assert not view.exception
    assert view.metric[3].value == "66.666667%"
    assert len(view.get("vega_lite_chart")) == 2
    assert len(_visible_pairs(view)) == 2
    view.text_input(key="query").set_value("[needle]").run()
    assert not view.exception and _visible_pairs(view) == [("0", "0")]
    assert "&lt;script&gt;" in _match_table(view) and "<script>" not in _match_table(view)
    view.text_input(key="query").set_value("").run()
    view.multiselect(key="methods").set_value(["semantic"]).run()
    assert _visible_pairs(view) == [("1", "1")]
    assert "0.8700" in _match_table(view)
    view.slider(key="score").set_value(0.95).run()
    assert not view.exception
    assert any(
        'class="vs-notice"' in element.proto.body and "No pairs match" in element.proto.body
        for element in view.get("html")
    )


def test_dashboard_empty_legacy_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    view = _dashboard_app(tmp_path, monkeypatch, ContaminationReport())
    assert not view.exception
    assert view.metric[3].value == "0.0%"
    assert any(
        'class="vs-notice"' in element.proto.body and "No evaluation examples" in element.proto.body
        for element in view.get("html")
    )
    assert not view.get("vega_lite_chart")


def test_dashboard_clean_report_has_no_charts_or_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = _dashboard_app(tmp_path, monkeypatch, ContaminationReport(train_size=100, eval_size=20))
    assert not view.exception
    assert not view.get("vega_lite_chart")
    assert not view.multiselect and not view.slider
    html = " ".join(element.proto.body for element in view.get("html"))
    assert "No contamination detected" in html and "Clean" in html


def test_demo_launch_does_not_modify_real_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit", minversion="1.66")
    path = tmp_path / "report.json"
    ContaminationReport(train_size=64, eval_size=16).to_json(str(path))
    before = path.read_bytes()
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = CliRunner().invoke(app, ["dashboard", str(path), "--demo"])
    assert result.exit_code == 0, result.output
    assert commands[0][-3:] == ["--", str(path.resolve()), "--demo"]
    assert "--theme.base=dark" in commands[0]
    assert path.read_bytes() == before


def test_demo_dashboard_displays_planted_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit", minversion="1.66")
    from streamlit.testing.v1 import AppTest

    spec = importlib.util.find_spec("verascan.dashboard_app")
    assert spec is not None and spec.origin is not None
    monkeypatch.setattr(sys, "argv", [spec.origin, "unused.json", "--demo"])
    view = AppTest.from_file(spec.origin, default_timeout=20).run()
    assert not view.exception
    assert len(_visible_pairs(view)) == 5
    assert all(method in _match_table(view) for method in ("exact", "fuzzy", "semantic"))
    assert len(view.get("vega_lite_chart")) == 2
    assert view.selectbox(key="pair").value is not None


def test_dashboard_clamps_page_when_filter_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = ContaminationReport(
        train_size=30,
        eval_size=30,
        methods_used=["exact"],
        matches=[MatchRecord(i, i, f"row{i} ", f"row{i}", 1.0, "exact") for i in range(30)],
    )
    view = _dashboard_app(tmp_path, monkeypatch, report)
    view.number_input(key="page").set_value(2).run()
    view.text_input(key="query").set_value("row0 ").run()
    assert not view.exception and len(_visible_pairs(view)) == 1


@pytest.mark.parametrize(
    "eval_size, status, percentage",
    [(1000, "Low", "0.1%"), (20, "High", "5.0%"), (10**10, "Low", "1e-08%")],
)
def test_status_preserves_nonzero_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_size: int, status: str, percentage: str
) -> None:
    report = ContaminationReport(
        train_size=1,
        eval_size=eval_size,
        methods_used=["exact"],
        matches=[MatchRecord(0, 0, "shared", "shared", 1.0, "exact")],
    )
    view = _dashboard_app(tmp_path, monkeypatch, report)
    assert not view.exception and view.metric[3].value == percentage
    rendered = " ".join(element.proto.body for element in view.get("html"))
    assert f"</span>{status}</span>" in rendered


def test_dashboard_paginates_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = ContaminationReport(
        train_size=30,
        eval_size=30,
        methods_used=["exact"],
        matches=[MatchRecord(i, i, f"row{i}", f"row{i}", 1.0, "exact") for i in range(30)],
    )
    view = _dashboard_app(tmp_path, monkeypatch, report)
    assert not view.exception and len(_visible_pairs(view)) == 25
    view.number_input(key="page").set_value(2).run()
    assert not view.exception and len(_visible_pairs(view)) == 5
    assert _visible_pairs(view)[-1] == ("29", "29")


def test_dashboard_benchmark_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from verascan import AuditReport

    report = AuditReport(
        train_size=1,
        eval_size=2,
        methods_used=["semantic"],
        benchmarks=["gsm8k", "mmlu"],
        eval_records=[{"benchmark": "gsm8k"}, {"benchmark": "mmlu"}],
        matches=[
            MatchRecord(0, 0, "math", "a", 0.9, "semantic"),
            MatchRecord(1, 0, "knowledge", "a", 0.95, "semantic"),
        ],
    )
    view = _dashboard_app(tmp_path, monkeypatch, report)
    view.multiselect(key="benchmarks").set_value(["mmlu"]).run()
    assert not view.exception and _visible_pairs(view) == [("1", "0")]
    assert "mmlu" in _match_table(view)


def test_header_theme_toggle_preserves_report_and_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = ContaminationReport(
        train_size=2,
        eval_size=2,
        methods_used=["exact"],
        matches=[MatchRecord(i, i, f"example {i}", f"example {i}", 1.0, "exact") for i in range(2)],
    )
    view = _dashboard_app(tmp_path, monkeypatch, report)
    view.text_input(key="query").set_value("example 1").run()
    view.toggle(key="light_theme").set_value(True).run()
    assert not view.exception
    assert _visible_pairs(view) == [("1", "1")]
    assert view.text_input(key="query").value == "example 1"
    styles = " ".join(element.proto.body for element in view.get("html"))
    assert "--vs-bg: #ffffff" in styles
    assert view.metric[3].value == "100.0%"
    view.toggle(key="light_theme").set_value(False).run()
    assert not view.exception and _visible_pairs(view) == [("1", "1")]
    styles = " ".join(element.proto.body for element in view.get("html"))
    assert "--vs-bg: #000000" in styles
