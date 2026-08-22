"""Shared test fixtures."""

import pytest


@pytest.fixture()
def sample_texts() -> tuple[list[str], list[str]]:
    train = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "Natural language processing enables computers to understand text.",
        "Deep learning uses multilayer neural networks.",
    ]
    eval_ = [
        "The quick brown fox jumps over the lazy dog.",  # exact
        "Machine learning is a branch of artificial intelligence.",  # near-dup
        "Quantum computing will change the future.",  # no match
    ]
    return train, eval_
