"""Tests for the exact matching engine."""

from __future__ import annotations

from verascan.engines.exact import find_exact_matches


def test_exact_matches_found(sample_texts: tuple[list[str], list[str]]) -> None:
    train, eval_ = sample_texts
    matches = find_exact_matches(train, eval_, show_progress=False)
    assert len(matches) == 1
    m = matches[0]
    assert m.eval_index == 0
    assert m.train_index == 0
    assert m.score == 1.0
    assert m.method == "exact"
    assert m.eval_text == "The quick brown fox jumps over the lazy dog."


def test_no_matches() -> None:
    train = ["apple", "banana"]
    eval_ = ["cherry", "date"]
    matches = find_exact_matches(train, eval_, show_progress=False)
    assert matches == []


def test_multiple_train_duplicates() -> None:
    """If the same text appears multiple times in train, all are reported."""
    train = ["hello world", "other text", "hello world"]
    eval_ = ["hello world"]
    matches = find_exact_matches(train, eval_, show_progress=False)
    assert len(matches) == 2
    assert {m.train_index for m in matches} == {0, 2}


def test_whitespace_normalisation() -> None:
    """Extra spaces, tabs, and newlines should still match."""
    train = ["  hello   world \n"]
    eval_ = ["hello world"]
    matches = find_exact_matches(train, eval_, show_progress=False)
    assert len(matches) == 1


def test_case_insensitive() -> None:
    train = ["Hello World"]
    eval_ = ["hello world"]
    matches = find_exact_matches(train, eval_, show_progress=False)
    assert len(matches) == 1


def test_empty_inputs() -> None:
    assert find_exact_matches([], [], show_progress=False) == []
    assert find_exact_matches(["a"], [], show_progress=False) == []
    assert find_exact_matches([], ["a"], show_progress=False) == []
