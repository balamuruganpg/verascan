"""Tests for the GPT-3-style word n-gram overlap engine."""

from __future__ import annotations

from verascan.engines.ngram import _clean_text, _word_ngrams, find_ngram_matches


def _thirteen_words(prefix: str = "word") -> str:
    """Build a 13-word sentence suitable for default n=13 grams."""
    return " ".join(f"{prefix}{i}" for i in range(13))


def test_clean_text_lowercase_and_punctuation() -> None:
    assert _clean_text("Hello, World!") == "hello world"
    assert _clean_text("  Foo... BAR???  ") == "foo bar"


def test_word_ngrams_basic() -> None:
    text = "one two three four five"
    assert _word_ngrams(text, n=3) == [
        "one two three",
        "two three four",
        "three four five",
    ]


def test_shared_13gram_hit() -> None:
    shared = _thirteen_words("shared")
    train = [shared + " train only tail"]
    eval_ = [shared + " eval only tail"]

    matches = find_ngram_matches(train, eval_, n=13, max_count=10, show_progress=False)
    assert len(matches) == 1
    m = matches[0]
    assert m.eval_index == 0
    assert m.train_index == 0
    assert m.method == "ngram"
    assert m.score > 0.0


def test_no_hit_clean_case() -> None:
    train = [_thirteen_words("alpha") + " end"]
    eval_ = [_thirteen_words("beta") + " end"]
    matches = find_ngram_matches(train, eval_, n=13, show_progress=False)
    assert matches == []


def test_short_texts_below_n_words() -> None:
    """Texts with fewer than n words cannot form an n-gram."""
    train = ["one two three four five"]
    eval_ = ["one two three four five"]
    matches = find_ngram_matches(train, eval_, n=13, show_progress=False)
    assert matches == []


def test_frequency_filter_drops_common_grams() -> None:
    """Grams appearing in >= max_count train docs are ignored."""
    shared = _thirteen_words("common")
    train = [shared + f" doc{i}" for i in range(3)]
    eval_ = [shared + " eval"]
    # max_count=3 → gram appears in 3 docs → dropped
    matches = find_ngram_matches(train, eval_, n=13, max_count=3, show_progress=False)
    assert matches == []
    # max_count=4 → retained → hit
    matches_kept = find_ngram_matches(train, eval_, n=13, max_count=4, show_progress=False)
    assert len(matches_kept) >= 1


def test_empty_inputs() -> None:
    assert find_ngram_matches([], [], show_progress=False) == []
    assert find_ngram_matches(["a"], [], show_progress=False) == []
    assert find_ngram_matches([], ["a"], show_progress=False) == []
