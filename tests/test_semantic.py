"""Tests for the semantic similarity detection engine.

Tests are grouped into two categories:
- Tests that ALWAYS run (availability check, error messages)
- Tests that require sentence-transformers + faiss (skipped otherwise)
"""

from __future__ import annotations

import pytest

from verascan.engines.semantic import is_available, require

# ---------- availability API --------------------------------------------- #


def test_is_available_returns_bool() -> None:
    """is_available() must return a bool regardless of environment."""
    result = is_available()
    assert isinstance(result, bool)


def test_require_when_available() -> None:
    """If deps are available, require() should not raise."""
    if not is_available():
        pytest.skip("semantic extras not usable in this environment")
    require()  # should not raise


def test_require_when_unavailable() -> None:
    """If deps are broken/missing, require() should raise ImportError."""
    if is_available():
        pytest.skip("semantic extras are available — can't test missing-dep path")
    with pytest.raises(ImportError, match="Semantic matching is unavailable"):
        require()


def test_error_message_is_helpful() -> None:
    """The error message should tell users how to fix it."""
    if is_available():
        pytest.skip("semantic extras are available")
    with pytest.raises(ImportError, match="pip install verascan"):
        require()


# ---------- functional tests (need real deps) ---------------------------- #


@pytest.mark.skipif(not is_available(), reason="semantic extras not usable")
class TestSemanticMatching:
    """Functional tests — only run when sentence-transformers + FAISS work."""

    def test_identical_text(self) -> None:
        from verascan.engines.semantic import find_semantic_matches

        train = ["The cat sat on the mat."]
        eval_ = ["The cat sat on the mat."]
        matches = find_semantic_matches(train, eval_, threshold=0.9, show_progress=False)
        assert len(matches) == 1
        assert matches[0].score >= 0.99
        assert matches[0].method == "semantic"
        assert matches[0].eval_index == 0
        assert matches[0].train_index == 0

    def test_semantically_similar(self) -> None:
        from verascan.engines.semantic import find_semantic_matches

        train = ["A dog is playing in the park."]
        eval_ = ["A puppy runs around in the garden."]
        matches = find_semantic_matches(train, eval_, threshold=0.3, show_progress=False)
        assert len(matches) >= 1
        assert matches[0].method == "semantic"
        assert matches[0].score > 0.3

    def test_unrelated_no_match(self) -> None:
        from verascan.engines.semantic import find_semantic_matches

        train = ["Quantum mechanics describes wave-particle duality."]
        eval_ = ["I need to buy groceries today."]
        matches = find_semantic_matches(train, eval_, threshold=0.9, show_progress=False)
        assert len(matches) == 0

    def test_empty_inputs(self) -> None:
        from verascan.engines.semantic import find_semantic_matches

        assert find_semantic_matches([], [], show_progress=False) == []
        assert find_semantic_matches(["a"], [], show_progress=False) == []
        assert find_semantic_matches([], ["a"], show_progress=False) == []

    def test_multiple_matches(self) -> None:
        from verascan.engines.semantic import find_semantic_matches

        train = [
            "Machine learning is a subset of artificial intelligence.",
            "The weather is sunny today.",
            "Deep learning uses neural networks.",
        ]
        eval_ = [
            "AI includes machine learning techniques.",
            "It is raining outside.",
        ]
        matches = find_semantic_matches(train, eval_, threshold=0.3, show_progress=False)
        # At threshold 0.3, should pick up at least the ML-related pairs.
        assert len(matches) >= 1
        methods = {m.method for m in matches}
        assert methods == {"semantic"}

    def test_threshold_filters_correctly(self) -> None:
        from verascan.engines.semantic import find_semantic_matches

        train = ["Python is a programming language."]
        eval_ = ["Java is a programming language."]
        low = find_semantic_matches(train, eval_, threshold=0.3, show_progress=False)
        high = find_semantic_matches(train, eval_, threshold=0.95, show_progress=False)
        assert len(low) >= len(high)


# ---------- integration with check() ------------------------------------ #


@pytest.mark.skipif(not is_available(), reason="semantic extras not usable")
def test_check_with_semantic_method() -> None:
    """Verify semantic integrates into the main check() pipeline."""
    import verascan

    train = ["The cat sat on the mat."]
    eval_ = ["The cat sat on the mat."]
    report = verascan.check(
        train,
        eval_,
        methods=["exact", "semantic"],
        threshold=0.9,
        show_progress=False,
    )
    assert report.exact_count == 1
    assert report.semantic_count == 0
    assert report.contamination_rate == 1.0


@pytest.mark.skipif(not is_available(), reason="semantic extras not usable")
def test_check_semantic_only() -> None:
    """Verify semantic-only mode works."""
    import verascan

    train = ["Dogs are loyal animals."]
    eval_ = ["Dogs are faithful pets."]
    report = verascan.check(
        train,
        eval_,
        methods=["semantic"],
        threshold=0.3,
        show_progress=False,
    )
    assert report.semantic_count >= 1
    assert report.exact_count == 0
    assert report.fuzzy_count == 0
