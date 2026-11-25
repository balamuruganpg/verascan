"""Regression coverage for the VeraScan 0.6.2 verification.

Each test pins a result that 0.6.2 got wrong. Semantic cases inject a fake
encoder so they run without the optional extras.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pypdf import PdfReader
from typer.testing import CliRunner

import verascan
from verascan.cli import _distinct_fail_percents, app
from verascan.engines.semantic import find_semantic_matches
from verascan.pdf_report import _diff
from verascan.report import ContaminationReport, MatchRecord, _word_diff_html

runner = CliRunner()


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text or "")


def _jsonl(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    return path


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _orthogonal(texts: list[str]) -> np.ndarray:
    """Unit vectors that agree only when the strings are identical."""
    rows = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vector = np.empty(len(digest) * 8, dtype=np.float32)
        for index, byte in enumerate(digest):
            for bit in range(8):
                vector[index * 8 + bit] = 1.0 if byte & (1 << bit) else -1.0
        vector /= np.linalg.norm(vector)
        rows.append(vector)
    return np.vstack(rows)


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
def semantic_stub(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    import verascan.engines.semantic as semantic

    state: dict[str, object] = {"encode": _orthogonal}
    transformers = ModuleType("sentence_transformers")
    faiss_mod = ModuleType("faiss")
    faiss_mod.IndexFlatIP = _IndexFlatIP  # type: ignore[attr-defined]

    class SentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.max_seq_length = 256
            self.tokenizer = _Tokenizer()

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


def _contained_score(train: str, evaluation: str) -> float:
    matches = find_semantic_matches([train], [evaluation], show_progress=False)
    assert len(matches) == 1
    assert matches[0].score < 1.0
    return matches[0].score


def test_semantic_containment_scores_a_verbatim_ending(semantic_stub: dict[str, object]) -> None:
    del semantic_stub
    prefix = " ".join(f"word{index}" for index in range(400))
    phrase = "unique contamination phrase alpha beta gamma"
    words = " ".join(f"sharedsentence{index}" for index in range(40))
    long_phrase = prefix + " " + phrase
    long_words = prefix + " " + words

    assert _contained_score(long_phrase, phrase) >= 0.85
    assert _contained_score(long_words, words) >= 0.85
    assert _contained_score(long_words, " ".join(words.split()[:32])) >= 0.85
    assert _contained_score(long_words, " ".join(words.split()[:16])) >= 0.85

    middle = " ".join(f"word{index}" for index in range(200))
    middle += " " + phrase + " "
    middle += " ".join(f"word{index}" for index in range(200, 400))
    assert _contained_score(middle, phrase) >= 0.85

    # The whole evaluation row still fits in the model window.
    short_train = " ".join(f"word{index}" for index in range(30)) + " " + phrase
    assert _contained_score(short_train, phrase) >= 0.85

    assert find_semantic_matches([prefix], [phrase], show_progress=False) == []
    identical = find_semantic_matches([long_phrase], [long_phrase], show_progress=False)
    assert identical[0].score == 1.0
    assert find_semantic_matches([long_phrase], [""], show_progress=False) == []


def test_ngram_split_of_long_copies_does_not_claim_zero_residual(tmp_path: Path) -> None:
    sentence = " ".join(f"word{index}" for index in range(20))
    rows = [sentence] * 20
    source = _jsonl(tmp_path / "long-copies.jsonl", rows)
    result = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(source),
            "--eval-size",
            "0.25",
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
        "ngram dropped grams appearing in at least 10 training rows"
    ) in summary

    train, evaluation = verascan.split(
        rows, eval_size=0.25, methods=["ngram"], seed=1, show_progress=False
    )
    exact = verascan.check(train, evaluation, methods=["exact"], show_progress=False)
    ngram = verascan.check(train, evaluation, methods=["ngram"], show_progress=False)
    assert exact.contamination_rate == 1.0
    assert exact.total_matches == 75
    assert ngram.total_matches == 0

    eleven = _jsonl(tmp_path / "eleven.jsonl", [sentence] * 11)
    eleven_result = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(eleven),
            "--eval-size",
            "1",
            "--methods",
            "ngram",
            "--seed",
            "1",
            "--no-progress",
        ],
    )
    assert eleven_result.exit_code == 0, eleven_result.output
    assert "Residual leakage: 0.0%" not in eleven_result.stdout
    assert "verbatim overlap remains" in eleven_result.stdout

    cleared, held = verascan.split(
        rows,
        eval_size=0.25,
        methods=["ngram"],
        seed=1,
        ngram_max_count=100,
        show_progress=False,
    )
    assert held == []
    assert len(cleared) == 20


def test_fail_above_prints_two_distinct_percentages(tmp_path: Path) -> None:
    original_rate, original_gate = _distinct_fail_percents(1 / 30_000, 0.0)
    assert original_rate == "0.003%"
    assert original_gate == "0.0%"
    assert original_rate != original_gate

    cases = (
        (3, "0.333333333332", "33.3%"),
        (200, "0.004999999999", "0.5%"),
        (20_000, "4.9999999e-05", "0.005%"),
    )
    train = _jsonl(tmp_path / "train.jsonl", ["shared"])
    for size, gate, summary_rate in cases:
        evaluation = _jsonl(
            tmp_path / f"eval-{size}.jsonl",
            ["shared"] + [f"row {index}" for index in range(size - 1)],
        )
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
                gate,
            ],
        )
        assert result.exit_code == 1, result.output
        assert summary_rate in result.stdout
        stderr = _plain(result.stderr)
        found = re.search(r"Contamination rate (\S+) exceeds threshold (\S+)", stderr)
        assert found, stderr
        assert found.group(1) != found.group(2)


def test_fuzzy_stored_score_stays_above_the_admitting_threshold() -> None:
    train = "atgwkfhhuomwvsnbmwsnyvwbfociwfoqprtyabpkj"
    evaluation = "atgwkfhhuomwvsnbmwsnyvwbfowiwfoqprtyabpkj"
    threshold = 0.765625
    report = verascan.check(
        [train],
        [evaluation],
        methods=["fuzzy"],
        threshold=threshold,
        show_progress=False,
    )
    assert report.contamination_rate == 1.0
    assert report.matches[0].score >= report.threshold
    assert report.flagged(min_score=report.threshold)


def test_semantic_cosine_near_one_is_not_stored_as_perfect(
    semantic_stub: dict[str, object],
) -> None:
    def encode(texts: list[str]) -> np.ndarray:
        rows = [
            [0.99996, 0.0] if text == "query" else [1.0, 0.0]
            for text in texts
        ]
        return np.asarray(rows, dtype=np.float32)

    semantic_stub["encode"] = encode
    # This call has no tokenizer on the default stub path: the fixture always
    # installs one. Containment must not be what turns the dot product into 1.
    report = verascan.check(
        ["doc"],
        ["query"],
        methods=["semantic"],
        threshold=0.85,
        show_progress=False,
    )
    assert report.contamination_rate == 1.0
    assert report.matches[0].score == pytest.approx(0.9999)
    assert report.matches[0].score < 1.0


def _substitution_marks(length: int, indexes: list[int]) -> tuple[str, str]:
    train_words = ["the"] * length
    eval_words = ["the"] * length
    for index in indexes:
        eval_words[index] = "CHANGED"
    train = " ".join(train_words)
    evaluation = " ".join(eval_words)
    html = _word_diff_html(train, evaluation)
    pdf = _diff(train, evaluation)
    deleted = re.findall(r"<del>(.*?)</del>", html)
    inserted = re.findall(r"<ins>(.*?)</ins>", html)
    struck = re.findall(r"<strike>(.*?)</strike>", pdf)
    assert deleted == ["the"] * len(indexes)
    assert inserted == ["CHANGED"] * len(indexes)
    assert struck == ["the"] * len(indexes)
    assert html.split("<del>")[0].split().count("the") == indexes[0]
    assert pdf.split("<strike>")[0].split().count("the") == indexes[0]
    return html, pdf


def test_repeated_word_diff_does_not_strike_shared_copies(tmp_path: Path) -> None:
    for length, index in ((245, 10), (199, 10), (200, 10), (400, 10), (245, 80), (400, 200)):
        _substitution_marks(length, [index])
    _substitution_marks(400, [5, 250])

    train = " ".join(["the"] * 245)
    evaluation_words = ["the"] * 245
    evaluation_words[10] = "CHANGED"
    evaluation = " ".join(evaluation_words)
    report = ContaminationReport(
        train_size=1,
        eval_size=1,
        matches=[MatchRecord(0, 0, evaluation, train, 1.0, "exact")],
        methods_used=["exact"],
        eval_texts=[evaluation],
    )
    pdf_path = tmp_path / "repeated.pdf"
    report.to_pdf(str(pdf_path))
    assert "CHANGED" in _pdf_text(pdf_path)


def test_pdf_word_diff_notes_omitted_controls(tmp_path: Path) -> None:
    cases = (
        ("hel\x00lo", "U+0000"),
        ("hel\x00lo 😀", "U+0000"),
        ("tab\there", "U+0009"),
        ("vt\x0bhere", "U+000B"),
        ("bell\x07here", "U+0007"),
        ("del\x7fhere", "U+007F"),
    )
    for text, note in cases:
        rendered = _diff(text, text)
        assert note in rendered
        assert "omitted" in rendered
        report = verascan.check([text], [text], methods=["exact"])
        pdf_path = tmp_path / f"{note.replace('+', '')}.pdf"
        report.to_pdf(str(pdf_path))
        extracted = _pdf_text(pdf_path)
        assert note in extracted
        assert "omitted" in extracted
