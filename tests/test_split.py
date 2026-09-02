"""Tests for the verascan.split prevention utility."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import verascan
from verascan.cli import app
from verascan.split import split

runner = CliRunner()


def test_split_list_of_strings():
    data = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming modern society and technology.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "The solar system consists of eight planets orbiting the Sun.",
        "Quantum mechanics describes nature at the smallest atomic scales.",
        "The Renaissance was a fervent period of European cultural revival.",
        "Machine learning algorithms build models based on sample data.",
        "The French Revolution began in 1789 with the storming of the Bastille.",
        "DNA contains the genetic instructions for all living organisms.",
        "Shakespeare wrote Romeo and Juliet in the late sixteenth century.",
    ]
    train, eval_set = split(data, eval_size=0.2, seed=42)

    assert isinstance(train, list)
    assert isinstance(eval_set, list)
    assert len(train) == 8
    assert len(eval_set) == 2
    assert len(train) + len(eval_set) == 10

    # Ensure zero overlap
    assert set(train).isdisjoint(set(eval_set))


def test_split_exact_leakage_purged():
    # Construct a dataset where an exact duplicate would naturally fall into both splits
    data = [
        "What is machine learning? ML is a subset of AI.",
        "Deep learning uses artificial neural networks.",
        "Natural language processing helps machines understand human language.",
        "Computer vision enables machines to see and process visual data.",
        "Reinforcement learning trains agents via reward feedback.",
        "What is machine learning? ML is a subset of AI.",  # Duplicate of index 0
    ]

    train, eval_set = split(data, eval_size=0.5, methods=["exact"], seed=42)

    # Verify using verascan.check that there is 0% contamination
    report = verascan.check(train=train, eval=eval_set, methods=["exact"])
    assert report.total_matches == 0
    assert report.contamination_rate == 0.0

    # The duplicated sentence must NOT appear in eval_set if it is in train
    for s in eval_set:
        assert s not in train


def test_split_fuzzy_leakage_purged():
    data = [
        "The quick brown fox jumps over the lazy dog and runs away quickly.",
        "Paris is the beautiful capital of France with the Eiffel Tower.",
        "Python is a versatile and readable general-purpose programming language.",
        "Quantum computing harnesses superposition and entanglement for computation.",
        "The quick brown fox jumps over the lazy dog and runs away fast.",  # Near duplicate
        "Supervised learning algorithms infer a function from labeled training data.",
    ]

    train, eval_set = split(data, eval_size=0.5, methods=["fuzzy"], threshold=0.75, seed=42)

    # Verify no fuzzy contamination remains
    report = verascan.check(train=train, eval=eval_set, methods=["fuzzy"], threshold=0.75)
    assert report.total_matches == 0
    assert report.contamination_rate == 0.0


def test_split_semantic_leakage_purged():
    from verascan.engines.semantic import is_available

    if not is_available():
        pytest.skip("semantic dependencies not installed")

    data = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above an inactive canine.",  # Semantic paraphrase
        "Photosynthesis generates oxygen from sunlight and water.",
        "Paris is the capital of France.",
        "Quantum computing relies on qubits and superposition.",
    ]

    train, eval_set = split(
        data,
        eval_size=0.4,
        methods=["semantic"],
        threshold=0.70,
        seed=42,
    )

    report = verascan.check(train=train, eval=eval_set, methods=["semantic"], threshold=0.70)
    assert report.total_matches == 0
    assert report.contamination_rate == 0.0


def test_split_dataframe_preserves_columns():
    df = pd.DataFrame(
        {
            "id": [101, 102, 103, 104, 105],
            "text": [
                "First sample record for NLP.",
                "Second sample record for NLP.",
                "Third sample record for NLP.",
                "Fourth sample record for NLP.",
                "Fifth sample record for NLP.",
            ],
            "label": ["A", "B", "A", "C", "B"],
            "difficulty": [1, 2, 3, 1, 2],
        }
    )

    train, eval_set = split(df, eval_size=0.4, column="text", seed=123)

    assert isinstance(train, pd.DataFrame)
    assert isinstance(eval_set, pd.DataFrame)
    assert list(train.columns) == ["id", "text", "label", "difficulty"]
    assert list(eval_set.columns) == ["id", "text", "label", "difficulty"]
    assert len(train) + len(eval_set) == len(df)


def test_split_seed_reproducibility():
    data = [f"Sample record text string {i}" for i in range(30)]

    train1, eval1 = split(data, eval_size=0.3, methods=["exact"], seed=99)
    train2, eval2 = split(data, eval_size=0.3, methods=["exact"], seed=99)
    train3, eval3 = split(data, eval_size=0.3, methods=["exact"], seed=100)

    assert train1 == train2
    assert eval1 == eval2
    assert eval1 != eval3


def test_split_integer_eval_size():
    data = [f"Item {i}" for i in range(15)]
    train, eval_set = split(data, eval_size=5, methods=["exact"], seed=42)

    assert len(eval_set) == 5
    assert len(train) == 10


def test_split_move_to_drop():
    data = [
        "Exact text copy A.",
        "Unique text B.",
        "Unique text C.",
        "Exact text copy A.",  # Duplicate of index 0
    ]

    # With move_to="drop", contaminated items are discarded rather than moved to train
    train, eval_set = split(data, eval_size=0.5, methods=["exact"], move_to="drop", seed=42)

    report = verascan.check(train=train, eval=eval_set, methods=["exact"])
    assert report.total_matches == 0
    # Duplicate was dropped
    assert len(train) + len(eval_set) <= len(data)


def test_split_output_files(tmp_path: Path):
    data = [
        {"id": 1, "text": "Record number one in data."},
        {"id": 2, "text": "Record number two in data."},
        {"id": 3, "text": "Record number three in data."},
        {"id": 4, "text": "Record number four in data."},
    ]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    train_out = tmp_path / "train.jsonl"
    eval_out = tmp_path / "eval.csv"

    train, eval_set = split(
        str(input_file),
        eval_size=0.5,
        output_train=str(train_out),
        output_eval=str(eval_out),
        seed=42,
    )

    assert train_out.exists()
    assert eval_out.exists()

    # Verify contents written can be read back
    loaded_train_df = pd.read_json(train_out, lines=True)
    assert len(loaded_train_df) == len(train)

    loaded_eval_df = pd.read_csv(eval_out)
    assert len(loaded_eval_df) == len(eval_set)


def test_split_cli_command(tmp_path: Path):
    data = [
        {"text": "Line 1 of training text."},
        {"text": "Line 2 of training text."},
        {"text": "Line 3 of training text."},
        {"text": "Line 4 of training text."},
        {"text": "Line 5 of training text."},
    ]
    input_file = tmp_path / "dataset.jsonl"
    with open(input_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

    train_file = tmp_path / "out_train.jsonl"
    eval_file = tmp_path / "out_eval.jsonl"

    result = runner.invoke(
        app,
        [
            "split",
            "--input",
            str(input_file),
            "--eval-size",
            "0.4",
            "--output-train",
            str(train_file),
            "--output-eval",
            str(eval_file),
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert "Verascan Split Summary" in result.output
    assert "Residual leakage: 0.0%" in result.output
    assert train_file.exists()
    assert eval_file.exists()


def test_split_edge_cases():
    # Empty input
    empty_train, empty_eval = split([], eval_size=0.2)
    assert empty_train == []
    assert empty_eval == []

    # eval_size = 0.0
    data = ["A", "B", "C"]
    train, eval_set = split(data, eval_size=0.0)
    assert len(train) == 3
    assert len(eval_set) == 0

    # Invalid eval_size float
    with pytest.raises(ValueError, match="eval_size as float must be between 0.0 and 1.0"):
        split(data, eval_size=1.5)

    with pytest.raises(ValueError, match="eval_size as float must be between 0.0 and 1.0"):
        split(data, eval_size=-0.2)

    # Invalid eval_size type
    with pytest.raises(TypeError, match="eval_size must be float or int"):
        split(data, eval_size="invalid")  # type: ignore[arg-type]

    # Invalid move_to
    with pytest.raises(ValueError, match="Unknown move_to option"):
        split(data, move_to="invalid_action")

    # Invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        split(data, methods=["nonexistent"])
