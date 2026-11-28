"""Regression coverage for the VeraScan 0.6.4 near-variant hunt.

Each test pins one finding from that hunt. Semantic cases inject a fake
encoder so they run without the optional extras.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pypdf import PdfReader
from typer.testing import CliRunner

import verascan
from verascan.cli import app
from verascan.engines.semantic import find_semantic_matches
from verascan.pdf_report import _diff
from verascan.report import (
    _word_diff_html,
    format_contamination_rate,
    format_match_score,
    stored_similarity,
)

runner = CliRunner()


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text or "")


def _jsonl(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
    return path


def _pdf_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _orthogonal(texts: list[str]) -> np.ndarray:
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


def _pad(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_short_token_span_is_not_a_contained_passage(semantic_stub: dict[str, object]) -> None:
    del semantic_stub
    paragraph = (
        "The committee met on Tuesday and approved the budget after a long debate. "
        "Members asked for the final report before the next session."
    )
    long_pad = _pad(400)
    phrase = "unique contamination phrase alpha beta gamma"
    hosted = long_pad + " " + phrase

    for query in ("the", "The", "budget", "the budget", "on Tuesday", "approved the budget", "a"):
        assert find_semantic_matches([paragraph], [query], show_progress=False) == []

    probed = find_semantic_matches([long_pad], ["budget"], threshold=0, show_progress=False)
    assert probed
    assert probed[0].score < 0.85
    assert probed[0].score != pytest.approx(0.9999)

    contained = find_semantic_matches([hosted], [phrase], show_progress=False)
    assert len(contained) == 1
    assert contained[0].score == pytest.approx(0.9999)
    assert contained[0].score < 1.0


def test_one_edited_word_in_a_contained_passage_stays_above_the_default_threshold(
    semantic_stub: dict[str, object],
) -> None:
    del semantic_stub
    phrase = "unique contamination phrase alpha beta gamma"
    words = phrase.split()
    words[3] = "CHANGED"
    edited = " ".join(words)
    typo = phrase.replace("alpha", "alpah")
    shared = " ".join(f"sharedsentence{index}" for index in range(40))
    shared_words = shared.split()
    shared_words[20] = "sharedsentencX20"
    shared_typo = " ".join(shared_words)

    for pad_count in (20, 300):
        for place in ("start", "middle", "end"):
            pad = _pad(pad_count)
            if place == "start":
                host = phrase + " " + pad
            elif place == "end":
                host = pad + " " + phrase
            else:
                host = _pad(pad_count // 2) + " " + phrase + " " + _pad(pad_count // 2, "tail")
            verbatim = find_semantic_matches([host], [phrase], show_progress=False)
            assert verbatim[0].score == pytest.approx(0.9999)
            for query in (edited, typo):
                matches = find_semantic_matches([host], [query], show_progress=False)
                assert len(matches) == 1
                assert matches[0].score >= 0.85
                assert matches[0].score < 0.9999

    long_shared = _pad(300) + " " + shared
    assert find_semantic_matches([long_shared], [shared], show_progress=False)[0].score == pytest.approx(
        0.9999
    )
    edited_shared = find_semantic_matches([long_shared], [shared_typo], show_progress=False)
    assert len(edited_shared) == 1
    assert edited_shared[0].score >= 0.85
    assert edited_shared[0].score < 0.9999


def test_passage_split_across_two_documents_clears_the_default_threshold(
    semantic_stub: dict[str, object],
) -> None:
    del semantic_stub
    phrase = "unique contamination phrase alpha beta gamma"
    left, right = phrase.split()[:3], phrase.split()[3:]
    pad = _pad(80, "pad")
    documents = [pad + " " + " ".join(left), pad + " " + " ".join(right)]
    joined = find_semantic_matches(documents, [phrase], show_progress=False)
    assert len(joined) == 2
    assert {match.train_index for match in joined} == {0, 1}
    assert all(match.score >= 0.85 for match in joined)
    assert all(match.score < 0.9999 for match in joined)

    shared = " ".join(f"sharedsentence{index}" for index in range(12))
    half = shared.split()
    halves = [pad + " " + " ".join(half[:6]), pad + " " + " ".join(half[6:])]
    assert find_semantic_matches([halves[0]], [" ".join(half[:6])], show_progress=False)[
        0
    ].score == pytest.approx(0.9999)
    covered = find_semantic_matches(halves, [shared], show_progress=False)
    assert len(covered) == 2
    assert all(match.score >= 0.85 for match in covered)


def test_contained_passage_stays_below_one_when_the_threshold_exceeds_the_cap(
    semantic_stub: dict[str, object],
) -> None:
    del semantic_stub
    phrase = "unique contamination phrase alpha beta gamma"
    long = _pad(400) + " " + phrase
    for threshold in (0.85, 0.9999):
        matches = find_semantic_matches([long], [phrase], threshold=threshold, show_progress=False)
        assert len(matches) == 1
        assert matches[0].score == pytest.approx(0.9999)
        assert matches[0].score != 1.0
    for threshold in (0.99991, 0.99995, 1.0):
        assert (
            find_semantic_matches([long], [phrase], threshold=threshold, show_progress=False) == []
        )
    for threshold in (0.99991, 1.0):
        identical = find_semantic_matches([long], [long], threshold=threshold, show_progress=False)
        assert identical[0].score == 1.0


def test_capped_shared_gram_is_not_reported_as_zero_residual(tmp_path: Path) -> None:
    shared = " ".join(f"uniquegram{index}" for index in range(13))
    rows = [
        shared + " " + " ".join(f"pad{row}-{index}" for index in range(8)) for row in range(15)
    ]
    source = _jsonl(tmp_path / "capped.jsonl", rows)
    for methods in ("ngram", "exact,ngram", "ngram,fuzzy", "fuzzy"):
        result = runner.invoke(
            app,
            [
                "split",
                "--input",
                str(source),
                "--eval-size",
                "5",
                "--methods",
                methods,
                "--seed",
                "1",
                "--no-progress",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Residual leakage: 0.0%" not in result.stdout
        assert "at least 10 training rows" in result.stdout

    train, evaluation = verascan.split(
        rows, eval_size=5, methods=["ngram"], seed=1, show_progress=False
    )
    assert verascan.check(train, evaluation, methods=["ngram"], show_progress=False).total_matches == 0
    assert verascan.check(train, evaluation, methods=["exact"], show_progress=False).total_matches == 0

    cleared = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(source),
            "--eval-size",
            "0.4",
            "--methods",
            "ngram",
            "--seed",
            "1",
            "--no-progress",
        ],
    )
    assert cleared.exit_code == 0, cleared.output
    assert "Residual leakage: 0.0%" in cleared.stdout


def test_mixed_short_and_long_copies_name_both_ngram_limits(tmp_path: Path) -> None:
    sentence = " ".join(f"word{index}" for index in range(20))
    rows = [sentence] * 20 + ["too short row"] * 6
    source = _jsonl(tmp_path / "mixed.jsonl", rows)
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
    assert "ngram cannot compare rows shorter than 13 words" in summary
    assert "ngram dropped grams appearing in at least 10 training rows" in summary

    _train, evaluation = verascan.split(
        rows, eval_size=0.25, methods=["ngram"], seed=1, show_progress=False
    )
    assert any(len(row.split()) < 13 for row in evaluation)
    assert any(len(row.split()) >= 13 for row in evaluation)


def test_fail_above_separates_collapsed_percents_and_near_complete_rates(tmp_path: Path) -> None:
    assert format_contamination_rate(0.9995) == "99.95%"
    assert format_contamination_rate(0.9994) == "99.9%"
    assert format_contamination_rate(0.9999) == "99.99%"
    assert format_contamination_rate(1.0) == "100.0%"

    train = _jsonl(tmp_path / "train.jsonl", ["shared"])
    cases = (
        (10, math.nextafter(0.1, 0.0), "10.0%"),
        (3, math.nextafter(1 / 3, 0.0), "33.3%"),
        (100_000, math.nextafter(1 / 100_000, 0.0), "0.001%"),
        (10_000, 0.9998999999999999, "99.99%"),
    )
    for size, gate, summary_rate in cases:
        hits = 9999 if size == 10_000 else 1
        evaluation = _jsonl(
            tmp_path / f"eval-{size}.jsonl",
            ["shared"] * hits + [f"row {index}" for index in range(size - hits)],
        )
        gate_text = format(gate, ".17g")
        assert float(gate_text) == gate
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
                gate_text,
            ],
        )
        assert result.exit_code == 1, result.output
        assert summary_rate in result.stdout
        assert "(100.0%)" not in result.stdout
        found = re.search(
            r"Contamination rate (\S+) exceeds threshold (\S+)", _plain(result.stderr)
        )
        assert found, result.stderr
        assert found.group(1) != found.group(2)


def test_cosine_just_below_one_is_not_stored_as_perfect(
    semantic_stub: dict[str, object], tmp_path: Path
) -> None:
    encode_holder = {"score": 0.999991}

    def directed(texts: list[str]) -> np.ndarray:
        rows = [
            [encode_holder["score"], 0.0] if text == "query" else [1.0, 0.0] for text in texts
        ]
        return np.asarray(rows, dtype=np.float32)

    semantic_stub["encode"] = directed
    report = verascan.check(["doc"], ["query"], methods=["semantic"], threshold=0.85, show_progress=False)
    assert report.matches[0].score == pytest.approx(0.9999)
    assert report.matches[0].score < 1.0
    html_path = tmp_path / "near-one.html"
    report.to_html(str(html_path))
    html = html_path.read_text(encoding="utf-8")
    assert "Score: 1.0000" not in html
    assert "Score: 0.9999" in html

    encode_holder["score"] = 0.99996
    held = verascan.check(["doc"], ["query"], methods=["semantic"], threshold=0.85, show_progress=False)
    assert held.matches[0].score == pytest.approx(0.9999)
    assert stored_similarity(0.999991, 0.85) == pytest.approx(0.9999)
    assert stored_similarity(0.999991, 0.85) < 1.0


def test_raw_near_one_score_is_not_labeled_as_perfect(
    semantic_stub: dict[str, object], tmp_path: Path
) -> None:
    def encode(texts: list[str]) -> np.ndarray:
        rows = [[0.99996, 0.0] if text == "query" else [1.0, 0.0] for text in texts]
        return np.asarray(rows, dtype=np.float32)

    semantic_stub["encode"] = encode
    report = verascan.check(
        ["doc"], ["query"], methods=["semantic"], threshold=0.99996, show_progress=False
    )
    assert report.matches
    assert report.matches[0].score < 1.0
    assert report.flagged(min_score=report.threshold)
    label = format_match_score(report.matches[0].score)
    assert label != "1.0000"
    assert not label.startswith("1")
    html_path = tmp_path / "raw-label.html"
    report.to_html(str(html_path))
    html = html_path.read_text(encoding="utf-8")
    assert f"Score: {label}" in html
    assert "Score: 1.0000" not in html
    assert format_match_score(0.99995) != "1.0000"
    assert format_match_score(1.0) == "1.0000"


def test_moved_repeated_word_is_not_struck_when_counts_match() -> None:
    cases = (
        ("ALPHA the the", "the ALPHA the"),
        ("aa bb aa bb aa bb", "bb aa bb aa bb aa"),
        (" ".join(["ALPHA"] + ["the"] * 17), " ".join(["the"] * 8 + ["ALPHA"] + ["the"] * 9)),
        (" ".join(["aa", "bb"] * 750), " ".join(["bb", "aa"] * 750)),
    )
    for train, evaluation in cases:
        html = _word_diff_html(train, evaluation)
        pdf = _diff(train, evaluation)
        assert "<del>" not in html
        assert "<ins>" not in html
        assert "<strike>" not in pdf
        assert "#166534" not in pdf
        assert "#9f1239" not in pdf

    replaced = _word_diff_html("the the the", "the CHANGED the")
    assert "<del>the</del>" in replaced
    assert "<ins>CHANGED</ins>" in replaced


def test_tab_between_words_is_unchanged_in_html_and_pdf(tmp_path: Path) -> None:
    train = "alpha alpha beta alpha alpha"
    evaluation = "alpha alpha beta\t alpha alpha"
    html = _word_diff_html(train, evaluation)
    pdf = _diff(train, evaluation)
    assert "<del>" not in html
    assert "<ins>" not in html
    assert html == train
    assert "<strike>" not in pdf
    assert "#166534" not in pdf
    assert "beta" in pdf
    assert "U+0009" not in pdf

    named = _diff("tab\there", "tab\there")
    assert "U+0009" in named
    assert "omitted" in named
    assert "<strike>" not in named

    report = verascan.check([train], [evaluation], methods=["exact"])
    pdf_path = tmp_path / "tab.pdf"
    report.to_pdf(str(pdf_path))
    extracted = _pdf_text(pdf_path)
    assert "beta[U+0009" not in extracted
    assert "alpha alpha beta alpha alpha" in extracted


def test_default_scan_catches_zero_width_characters_and_cyrillic_homoglyphs() -> None:
    plain = "The committee published the quarterly contamination report on Tuesday."
    afternoon = plain.replace("Tuesday.", "Tuesday afternoon.")
    variants = (
        plain.replace(" ", " \u200b"),
        plain.replace(" ", "\u200b"),
        afternoon.replace(" ", "\u2060"),
        plain.replace("a", "\u0430"),
        plain.replace("published", "pub\u200blished"),
        plain.replace("published", "pub\u00adlished"),
        plain.replace(" ", "\u00a0"),
    )
    for variant in variants:
        source = afternoon if "\u2060" in variant else plain
        report = verascan.check([source], [variant], show_progress=False)
        assert report.contamination_rate == 1.0
        assert report.total_matches >= 1

    homoglyph = plain.replace("a", "\u0430")
    for methods in (["exact"], ["fuzzy"], ["exact", "fuzzy"]):
        caught = verascan.check([plain], [homoglyph], methods=methods, show_progress=False)
        assert caught.total_matches >= 1
