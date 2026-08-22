"""Tests for the fuzzy / near-duplicate detection engine."""

from __future__ import annotations

from verascan.engines.fuzzy import _shingle, find_fuzzy_matches


def test_shingle_basic() -> None:
    shingles = _shingle("hello", k=3)
    assert shingles == {"hel", "ell", "llo"}


def test_shingle_short_text() -> None:
    shingles = _shingle("hi", k=5)
    assert shingles == {"hi"}


def test_near_duplicate_found() -> None:
    train = [
        "The quick brown fox jumps over the lazy dog.",
        "Completely unrelated text about astronomy and stars.",
    ]
    # Minor edit: "a lazy dog" instead of "the lazy dog"
    eval_ = ["The quick brown fox jumps over a lazy dog."]

    matches = find_fuzzy_matches(train, eval_, threshold=0.6, show_progress=False)
    assert len(matches) == 1
    m = matches[0]
    assert m.eval_index == 0
    assert m.train_index == 0
    assert m.score >= 0.6
    assert m.method == "fuzzy"


def test_identical_texts_match() -> None:
    train = ["Identical text sentence here."]
    eval_ = ["Identical text sentence here."]
    matches = find_fuzzy_matches(train, eval_, threshold=0.95, show_progress=False)
    assert len(matches) == 1
    assert matches[0].score >= 0.95


def test_completely_different_no_match() -> None:
    train = ["Alpha beta gamma delta."]
    eval_ = ["One two three four."]
    matches = find_fuzzy_matches(train, eval_, threshold=0.8, show_progress=False)
    assert len(matches) == 0


def test_empty_inputs() -> None:
    assert find_fuzzy_matches([], [], show_progress=False) == []
    assert find_fuzzy_matches(["a"], [], show_progress=False) == []


def test_threshold_filtering() -> None:
    train = ["The weather today is sunny and warm with blue skies."]
    eval_ = ["The weather today is sunny and pleasant with blue skies."]

    # Low threshold should find it; very high threshold should not.
    matches_low = find_fuzzy_matches(train, eval_, threshold=0.5, show_progress=False)
    matches_high = find_fuzzy_matches(train, eval_, threshold=0.95, show_progress=False)
    assert len(matches_low) >= 1
    assert len(matches_high) == 0
