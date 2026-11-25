"""Regression coverage for the VeraScan 0.6.1 review.

Each test pins a public result that 0.6.1 got wrong. Semantic cases inject a
fake encoder so they run without the optional extras.
"""

from __future__ import annotations

import json
import math
import re
import sys
import zlib
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from typer.testing import CliRunner

import verascan
from verascan.benchmarks import SYNTHETIC_BENCHMARKS
from verascan.cli import app
from verascan.dashboard import load_report
from verascan.engines.fuzzy import find_fuzzy_matches
from verascan.engines.ngram import find_ngram_matches
from verascan.engines.semantic import find_semantic_matches
from verascan.loaders import load_texts, write_dataset
from verascan.pdf_report import _diff
from verascan.report import ContaminationReport, MatchRecord, _word_diff_html

runner = CliRunner()
_PHRASE = ("unique", "contamination", "phrase", "alpha", "beta", "gamma")


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text or "")


def _jsonl(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    return path


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _hash_embed(text: str) -> np.ndarray:
    """Bag-of-tokens stand-in that truncates at the MiniLM content window."""
    tokens = text.split()[:254]
    vector = np.zeros(4096, dtype=np.float64)
    for token in tokens:
        vector[1 + (zlib.adler32(token.encode()) % 4095)] += 1.0
    if all(token in tokens for token in _PHRASE):
        vector[0] = 20.0
    if vector.sum() == 0.0:
        vector[1] = 1.0
    vector /= np.linalg.norm(vector)
    return vector.astype(np.float32)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        del pair
        return 2


class _IndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        scores = queries @ self.vectors.T
        k = min(int(k), scores.shape[1])
        order = np.argsort(-scores, axis=1)[:, :k]
        top = np.take_along_axis(scores, order, axis=1)
        return top.astype(np.float32), order.astype(np.int64)


@pytest.fixture
def semantic_stub(monkeypatch: pytest.MonkeyPatch):
    """Install sentence-transformers and faiss doubles and restore the probe."""
    import verascan.engines.semantic as semantic

    state: dict[str, object] = {
        "encode": lambda texts: np.vstack([_hash_embed(text) for text in texts]),
        "tokenizer": None,
        "max_seq_length": None,
    }
    transformers = ModuleType("sentence_transformers")
    faiss_mod = ModuleType("faiss")
    faiss_mod.IndexFlatIP = _IndexFlatIP  # type: ignore[attr-defined]

    class SentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.max_seq_length = state["max_seq_length"]
            self.tokenizer = state["tokenizer"]

        def encode(self, texts: list[str], **_kwargs: object) -> np.ndarray:
            encoder = state["encode"]
            assert callable(encoder)
            return encoder(list(texts))

    transformers.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", transformers)
    monkeypatch.setitem(sys.modules, "faiss", faiss_mod)
    monkeypatch.setattr(semantic, "_PROBED", False)
    monkeypatch.setattr(semantic, "_SEMANTIC_ERROR", None)
    return state


def test_csv_nul_round_trip_matches_jsonl(tmp_path: Path) -> None:
    raw = "hel\x00lo"
    source = tmp_path / "in.csv"
    source.write_bytes(b'text\n"hel\x00lo"\n')
    assert load_texts(source) == [raw]

    exported = tmp_path / "out.csv"
    write_dataset([raw], str(exported), column="text")
    assert load_texts(exported) == [raw]
    assert load_texts(exported) == load_texts(_jsonl(tmp_path / "rows.jsonl", [raw]))
    assert verascan.check(["hel"], [raw], methods=["exact"]).total_matches == 0
    assert verascan.check([raw], load_texts(exported), methods=["exact"]).total_matches == 1


def test_csv_and_jsonl_accept_the_same_bom(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    jsonl_path = tmp_path / "rows.jsonl"
    json_path = tmp_path / "rows.json"
    csv_path.write_bytes(b"\xef\xbb\xbftext\nhello\n")
    jsonl_path.write_bytes(b'\xef\xbb\xbf{"text": "hello"}\n')
    json_path.write_bytes(b'\xef\xbb\xbf{"text": "one"}\n{"text": "two"}\n')
    assert load_texts(csv_path) == ["hello"]
    assert load_texts(jsonl_path) == ["hello"]
    assert load_texts(json_path) == ["one", "two"]

    latin = tmp_path / "latin.csv"
    latin.write_bytes("text\ncafé\n".encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        load_texts(latin)
    semicolon = tmp_path / "semi.csv"
    semicolon.write_text("text;note\nhello;world\n", encoding="utf-8")
    with pytest.raises(KeyError, match="text;note"):
        load_texts(semicolon)
    tsv = tmp_path / "rows.tsv"
    tsv.write_text("text\thello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_texts(tsv)


def test_audit_preserves_answer_and_subject_fields() -> None:
    question = SYNTHETIC_BENCHMARKS["gsm8k"][0]["text"]
    train = pd.DataFrame({"answer": [question, "18", "unrelated"]})
    report = verascan.audit(
        train,
        benchmarks=["gsm8k"],
        synthetic=True,
        methods=["exact"],
        column="answer",
    )
    assert report.contamination_rate == pytest.approx(0.2)
    assert [(match.train_index, match.eval_index) for match in report.matches] == [(0, 0)]
    flagged = report.contaminated_eval()
    assert isinstance(flagged, pd.DataFrame)
    assert flagged.iloc[0]["answer"] == "18"

    mmlu = SYNTHETIC_BENCHMARKS["mmlu"][0]["text"]
    subject_train = pd.DataFrame({"subject": [mmlu, "biology", "unrelated"]})
    subject_report = verascan.audit(
        subject_train,
        benchmarks=["mmlu"],
        synthetic=True,
        methods=["exact"],
        column="subject",
    )
    subject_rows = subject_report.contaminated_eval()
    assert isinstance(subject_rows, pd.DataFrame)
    assert subject_rows.iloc[0]["subject"] == "biology"
    assert all(match.train_index != 1 for match in subject_report.matches)


def test_html_and_pdf_word_diffs_mark_the_same_word(tmp_path: Path) -> None:
    words = ["other"] * 245
    for index in range(1, 245, 3):
        if words.count("the") == 80:
            break
        words[index] = "the"
    words[10] = "the"
    assert words.count("the") == 80
    changed = words.copy()
    changed[10] = "CHANGED"
    train = " ".join(words)
    evaluation = " ".join(changed)

    html = _word_diff_html(train, evaluation)
    deleted = re.findall(r"<del>(.*?)</del>", html)
    inserted = re.findall(r"<ins>(.*?)</ins>", html)
    struck = re.findall(r"<strike>(.*?)</strike>", _diff(train, evaluation))
    assert deleted == ["the"]
    assert inserted == ["CHANGED"]
    assert struck == ["the"]

    report = ContaminationReport(
        train_size=1,
        eval_size=1,
        matches=[MatchRecord(0, 0, evaluation, train, 1.0, "exact")],
        methods_used=["exact"],
        eval_texts=[evaluation],
    )
    pdf_path = tmp_path / "diff.pdf"
    report.to_pdf(str(pdf_path))
    assert "CHANGED" in _pdf_text(pdf_path)


def test_semantic_threshold_is_between_zero_and_one(tmp_path: Path) -> None:
    report = verascan.check(["same"], ["same"], methods=["exact"], threshold=1.5)
    assert report.total_matches == 1
    for threshold in (1.5, 85, -0.2, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="Semantic threshold must be between 0 and 1"):
            find_semantic_matches(["same"], ["same"], threshold=threshold, show_progress=False)
    with pytest.raises(ValueError, match="Fuzzy threshold must be between 0 and 1"):
        find_fuzzy_matches(["same"], ["other"], threshold=1.5, show_progress=False)

    rows = _jsonl(tmp_path / "rows.jsonl", ["The capital of France is Paris."])
    for bad in ("1.5", "85", "-0.2", "nan"):
        result = runner.invoke(
            app,
            [
                "check",
                "--train",
                str(rows),
                "--eval",
                str(rows),
                "--methods",
                "semantic",
                "--threshold",
                bad,
                "--no-progress",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "Semantic threshold must be between 0 and 1" in _plain(result.stderr)
        assert "Traceback" not in result.output


def test_semantic_passages_score_text_past_the_window(semantic_stub: dict[str, object]) -> None:
    semantic_stub["tokenizer"] = _Tokenizer()
    semantic_stub["max_seq_length"] = 256
    prefix = " ".join(f"word{i}" for i in range(400))
    ending_a = prefix + " ALPHA ending about physics and gravity"
    ending_b = prefix + " BETA ending about cooking and recipes"
    phrase = "unique contamination phrase alpha beta gamma"
    shared_tail = prefix + " " + phrase
    other_tail = " ".join(f"other{i}" for i in range(400)) + " " + phrase

    diverged = find_semantic_matches([ending_a], [ending_b], show_progress=False)
    assert len(diverged) == 1
    assert 0.85 <= diverged[0].score < 1.0

    late = find_semantic_matches([shared_tail], [other_tail], show_progress=False)
    assert len(late) == 1
    assert 0.85 <= late[0].score < 1.0

    identical = find_semantic_matches([ending_a], [ending_a], show_progress=False)
    assert identical[0].score == 1.0
    short = find_semantic_matches(
        ["The cat sat on the mat."], ["The cat sat on the mat."], show_progress=False
    )
    assert short[0].score == 1.0
    for text in ("", " ", "\n"):
        self_match = find_semantic_matches([text], [text], show_progress=False)
        assert self_match[0].score == 1.0
    unrelated = find_semantic_matches(
        [""], ["I need to buy groceries today."], threshold=0.0, show_progress=False
    )
    assert unrelated[0].score < 1.0
    assert (
        find_semantic_matches(
            [""], ["I need to buy groceries today."], show_progress=False
        )
        == []
    )


def test_cpu_semantic_keeps_every_neighbor_and_a_loadable_score(
    semantic_stub: dict[str, object], tmp_path: Path
) -> None:
    row = "identical training row"
    semantic_stub["encode"] = lambda texts: np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))
    matches = find_semantic_matches([row] * 15, [row], threshold=0.5, show_progress=False)
    assert {match.train_index for match in matches} == set(range(15))
    assert len(matches) == 15
    combined = verascan.check([row] * 15, [row], methods=["exact", "semantic"], show_progress=False)
    assert combined.exact_count == 15
    assert combined.semantic_count == 0

    semantic_stub["encode"] = lambda texts: np.tile(np.array([[2.0, 0.0]], dtype=np.float32), (len(texts), 1))
    oversized = find_semantic_matches(["a"], ["a"], threshold=0.5, show_progress=False)
    assert oversized[0].score == 1.0
    report = verascan.check(["a"], ["a"], methods=["semantic"], threshold=0.5, show_progress=False)
    score_path = tmp_path / "scores.json"
    report.to_json(str(score_path))
    assert load_report(str(score_path))["matches"][0]["score"] == 1.0

    raw = np.float32(0.8500009775161743)
    threshold = float(raw)

    def encode_borderline(texts: list[str]) -> np.ndarray:
        rows = [
            [float(raw), 0.0] if text == "query" else [1.0, 0.0]
            for text in texts
        ]
        return np.asarray(rows, dtype=np.float32)

    semantic_stub["encode"] = encode_borderline
    borderline = verascan.check(
        ["doc"],
        ["query"],
        methods=["semantic"],
        threshold=threshold,
        show_progress=False,
    )
    assert borderline.matches
    assert borderline.flagged(min_score=borderline.threshold)
    assert threshold <= borderline.matches[0].score <= 1.0
    kept = tmp_path / "kept.json"
    borderline.to_json(str(kept))
    assert load_report(str(kept))["matches"][0]["score"] == borderline.matches[0].score


def test_json_report_rejects_nonfinite_threshold(tmp_path: Path) -> None:
    report = ContaminationReport(
        train_size=1,
        eval_size=1,
        matches=[],
        methods_used=["exact"],
        threshold=float("nan"),
    )
    destination = tmp_path / "nan.json"
    with pytest.raises(ValueError):
        report.to_json(str(destination))

    payload = {
        "train_size": 1,
        "eval_size": 1,
        "matches": [],
        "methods_used": ["semantic"],
        "threshold": float("nan"),
    }
    nan_path = tmp_path / "loaded.json"
    nan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid report threshold"):
        load_report(str(nan_path))
    payload["threshold"] = True
    nan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid report threshold"):
        load_report(str(nan_path))
    payload["threshold"] = 1.5
    nan_path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_report(str(nan_path))["threshold"] == 1.5


def test_ngram_overlap_stays_visible_in_html_and_pdf(tmp_path: Path) -> None:
    shared = " ".join(f"w{i}" for i in range(13))
    long_eval = shared + " " + " ".join(f"e{i}" for i in range(20_000))
    stored = find_ngram_matches([shared], [long_eval], show_progress=False)
    assert stored[0].score == 0.0001

    visible_eval = shared + " " + " ".join(f"e{i}" for i in range(2_500))
    visible = find_ngram_matches([shared], [visible_eval], show_progress=False)
    assert visible[0].score == 0.0004
    report = ContaminationReport(
        train_size=1,
        eval_size=1,
        matches=[MatchRecord(0, 0, "eval row", "train row", visible[0].score, "ngram")],
        methods_used=["ngram"],
        eval_texts=["eval row"],
    )
    html_path = tmp_path / "ngram.html"
    pdf_path = tmp_path / "ngram.pdf"
    report.to_html(str(html_path))
    report.to_pdf(str(pdf_path))
    html = html_path.read_text(encoding="utf-8")
    assert "Score: 0.0004" in html
    assert "Score: 0.000<" not in html
    assert "0.0004" in _pdf_text(pdf_path)


def test_html_banner_counts_examples_and_pairs(tmp_path: Path) -> None:
    sentence = "the same sentence for the banner"
    report = verascan.check(
        [sentence, sentence, sentence],
        [sentence, "other one", "other two", "other three"],
        methods=["exact"],
    )
    assert report.contamination_rate == pytest.approx(0.25)
    assert report.total_matches == 3
    html_path = tmp_path / "banner.html"
    report.to_html(str(html_path))
    html = html_path.read_text(encoding="utf-8")
    assert "1 of 4 evaluation examples flagged (25.0%)" in html
    assert "3 matching pairs detected" in html
    assert "evaluation results may be inflated" in html


def test_fail_above_rejects_nan_and_prints_the_summary_percent(tmp_path: Path) -> None:
    rows = _jsonl(tmp_path / "rows.jsonl", ["shared"])
    for command, extra in (
        ("check", ["--eval", str(rows)]),
        ("audit", ["--benchmarks", "gsm8k", "--synthetic"]),
    ):
        for bad in ("nan", "inf"):
            result = runner.invoke(
                app,
                [
                    command,
                    "--train",
                    str(rows),
                    "--methods",
                    "exact",
                    "--no-progress",
                    "--fail-above",
                    bad,
                    *extra,
                ],
            )
            assert result.exit_code == 2, result.output
            assert "--fail-above must be a finite number" in _plain(result.stderr)
            assert "FAIL:" not in _plain(result.stderr)

    above = runner.invoke(
        app,
        [
            "check",
            "--train",
            str(rows),
            "--eval",
            str(rows),
            "--methods",
            "exact",
            "--no-progress",
            "--fail-above",
            "2",
        ],
    )
    assert above.exit_code == 0, above.output

    train = _jsonl(tmp_path / "train.jsonl", ["shared"])
    evaluation = _jsonl(tmp_path / "eval.jsonl", ["shared"] + [f"row {i}" for i in range(29_999)])
    result = runner.invoke(
        app,
        [
            "check",
            "--train",
            str(train),
            "--eval",
            str(evaluation),
            "--methods",
            "exact",
            "--no-progress",
            "--fail-above",
            "0",
        ],
    )
    assert result.exit_code == 1, result.output
    stderr = _plain(result.stderr)
    assert "0.003%" in result.stdout
    assert "0.003%" in stderr
    assert "0.00% exceeds threshold 0.00%" not in stderr
    assert "exceeds threshold 0.0%" in stderr


def test_colliding_and_unknown_output_paths_fail_before_write(tmp_path: Path) -> None:
    rows = _jsonl(tmp_path / "rows.jsonl", ["shared"])
    sentinel = tmp_path / "same.json"
    sentinel.write_text("sentinel", encoding="utf-8")
    other = tmp_path / "nested" / ".." / "same.json"
    check = runner.invoke(
        app,
        [
            "check",
            "--train",
            str(rows),
            "--eval",
            str(rows),
            "--methods",
            "exact",
            "--no-progress",
            "--output",
            str(sentinel),
            "--output-cleaned",
            str(other),
        ],
    )
    assert check.exit_code == 2, check.output
    assert "different paths" in _plain(check.stderr)
    assert sentinel.read_text(encoding="utf-8") == "sentinel"

    audit_path = tmp_path / "clobber.html"
    audit_path.write_text("sentinel", encoding="utf-8")
    audit = runner.invoke(
        app,
        [
            "audit",
            "--train",
            str(rows),
            "--benchmarks",
            "gsm8k",
            "--synthetic",
            "--methods",
            "exact",
            "--no-progress",
            "--output",
            str(audit_path),
            "--output-json",
            str(tmp_path / "sub" / ".." / "clobber.html"),
        ],
    )
    assert audit.exit_code == 2, audit.output
    assert audit_path.read_text(encoding="utf-8") == "sentinel"

    split_path = tmp_path / "split.jsonl"
    split_path.write_text("sentinel", encoding="utf-8")
    split = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(rows),
            "--eval-size",
            "0.5",
            "--methods",
            "exact",
            "--output-train",
            str(split_path),
            "--output-eval",
            str(tmp_path / "down" / ".." / "split.jsonl"),
        ],
    )
    assert split.exit_code == 2, split.output
    assert split_path.read_text(encoding="utf-8") == "sentinel"

    for command, extra in (
        ("check", ["--eval", str(rows)]),
        ("audit", ["--benchmarks", "gsm8k", "--synthetic"]),
    ):
        unknown = tmp_path / f"{command}-report.txt"
        result = runner.invoke(
            app,
            [
                command,
                "--train",
                str(rows),
                "--methods",
                "exact",
                "--no-progress",
                "--output",
                str(unknown),
                *extra,
            ],
        )
        assert result.exit_code == 2, result.output
        message = _plain(result.stderr)
        assert ".pdf" in message and ".json" in message and ".html" in message
        assert not unknown.exists()


def test_benchmark_counts_use_the_json_label() -> None:
    question = SYNTHETIC_BENCHMARKS["gsm8k"][0]["text"]
    report = verascan.audit([question], benchmarks=["gsm8k"], synthetic=True, methods=["exact"])
    assert report.benchmark_counts == {"gsm8k (synthetic)": 1}
    assert report.benchmark_counts == report.to_dict()["benchmark_counts"]
    assert set(report.by_benchmark()) == set(report.benchmark_counts)
    assert report.to_dict()["matches"][0]["benchmark"] == "gsm8k (synthetic)"


def test_fuzzy_keeps_pairs_whose_minhash_estimate_clears_the_threshold() -> None:
    train = "atgwkfhhuomwvsnbmwsnyvwbfociwfoqprtyabpkj obz zngruc xeamvnk"
    evaluation = "atgwkfhhuomwvanbmwsnyvwbfociwfoqprtyabpkj obz zngruc xeamvnk"
    matches = find_fuzzy_matches([train], [evaluation], threshold=0.8, show_progress=False)
    assert len(matches) == 1
    assert matches[0].score >= 0.8

    _train_split, eval_split = verascan.split(
        [train, evaluation],
        eval_size=1,
        methods=["fuzzy"],
        threshold=0.8,
        seed=0,
        show_progress=False,
    )
    assert eval_split == []

    everything = find_fuzzy_matches(
        ["alpha bravo", "completely unrelated zebra"],
        ["alpha bravo", "totally different yacht"],
        threshold=0.0,
        show_progress=False,
    )
    assert len(everything) == 4
    assert any(match.score == 0.0 for match in everything)


def test_ngram_only_split_does_not_claim_zero_residual_for_short_copies(tmp_path: Path) -> None:
    rows = ["same words here"] * 4 + ["other words there"]
    source = _jsonl(tmp_path / "rows.jsonl", rows)
    result = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(source),
            "--eval-size",
            "0.5",
            "--methods",
            "ngram",
            "--seed",
            "1",
            "--no-progress",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = result.stdout
    assert "Residual leakage: 0.0%" not in summary
    assert (
        "Residual leakage: 100.0% verbatim overlap remains; "
        "ngram cannot compare rows shorter than 13 words"
    ) in summary


def test_pdf_records_emoji_the_font_cannot_draw(tmp_path: Path) -> None:
    text = "hello 😀 café Δέλτα"
    report = verascan.check([text], [text], methods=["exact"])
    pdf_path = tmp_path / "emoji.pdf"
    report.to_pdf(str(pdf_path))
    extracted = _pdf_text(pdf_path)
    assert "1F600" in extracted
    assert "omitted" in extracted
    assert "café" in extracted
    assert "Δέλτα" in extracted

    nul = verascan.check(["hel\x00lo"], ["hel\x00lo"], methods=["exact"])
    nul_path = tmp_path / "nul.pdf"
    nul.to_pdf(str(nul_path))
    nul_text = _pdf_text(nul_path)
    assert "U+0000" not in nul_text
    assert "hello" in nul_text
    assert math.isfinite(report.matches[0].score)
