"""Benchmark preset loaders for public evaluation datasets."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

logger = logging.getLogger(__name__)

SUPPORTED_BENCHMARKS: set[str] = {"mmlu", "gsm8k", "humaneval"}


def check_datasets_available() -> None:
    """Verify that the Hugging Face ``datasets`` library is installed."""
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Hugging Face 'datasets' is required for benchmark presets. "
            "Install with: pip install verascan[hf]"
        ) from exc


def normalize_benchmarks(benchmarks: str | Sequence[str]) -> list[str]:
    """Validate and normalize benchmark preset names."""
    if isinstance(benchmarks, str):
        benchmarks_list = [b.strip() for b in benchmarks.split(",") if b.strip()]
    else:
        benchmarks_list = list(benchmarks)

    if not benchmarks_list:
        raise ValueError("At least one benchmark must be specified.")

    normalized: list[str] = []
    for b in benchmarks_list:
        clean_b = b.strip().lower()
        if clean_b not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark '{b}'. Supported benchmarks: {sorted(SUPPORTED_BENCHMARKS)}"
            )
        if clean_b not in normalized:
            normalized.append(clean_b)
    return normalized


# ---------------------------------------------------------------------------
# Synthetic benchmark stand-ins (for offline / CI testing)
# ---------------------------------------------------------------------------

SYNTHETIC_BENCHMARKS: dict[str, list[dict[str, Any]]] = {
    "gsm8k": [
        {
            "benchmark": "gsm8k",
            "text": (
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
                "and bakes muffins for her friends every day with four. She sells the remainder "
                "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does "
                "she make every day at the farmers' market?"
            ),
            "question": (
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
                "and bakes muffins for her friends every day with four. She sells the remainder "
                "at the farmers' market daily for $2 per fresh duck egg. How much in dollars does "
                "she make every day at the farmers' market?"
            ),
            "answer": "18",
        },
        {
            "benchmark": "gsm8k",
            "text": (
                "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                "How many bolts in total does it take?"
            ),
            "question": (
                "A robe takes 2 bolts of blue fiber and half that much white fiber. "
                "How many bolts in total does it take?"
            ),
            "answer": "3",
        },
        {
            "benchmark": "gsm8k",
            "text": (
                "Josh decides to try flipping a house. He buys a house for $80,000 and spends "
                "$50,000 in repairs. He then sells the house for $150,000. What is his profit in dollars?"
            ),
            "question": (
                "Josh decides to try flipping a house. He buys a house for $80,000 and spends "
                "$50,000 in repairs. He then sells the house for $150,000. What is his profit in dollars?"
            ),
            "answer": "20000",
        },
        {
            "benchmark": "gsm8k",
            "text": (
                "James writes a 3-page letter to 2 different friends twice a week. "
                "How many pages does he write a year?"
            ),
            "question": (
                "James writes a 3-page letter to 2 different friends twice a week. "
                "How many pages does he write a year?"
            ),
            "answer": "624",
        },
        {
            "benchmark": "gsm8k",
            "text": (
                "Every day, Wendi feeds each of her chickens 3 cups of mixed feed. "
                "If she has 20 chickens, how many cups of feed does she need for one week?"
            ),
            "question": (
                "Every day, Wendi feeds each of her chickens 3 cups of mixed feed. "
                "If she has 20 chickens, how many cups of feed does she need for one week?"
            ),
            "answer": "420",
        },
    ],
    "humaneval": [
        {
            "benchmark": "humaneval",
            "text": (
                "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
                '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
                "    given threshold.\n"
                "    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)\n"
                "    True\n"
                "    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)\n"
                "    False\n"
                '    """\n'
            ),
            "prompt": (
                "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n"
                '    """Check if in given list of numbers, are any two numbers closer to each other than\n'
                "    given threshold.\n"
                "    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)\n"
                "    True\n"
                "    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)\n"
                "    False\n"
                '    """\n'
            ),
            "task_id": "HumanEval/0",
        },
        {
            "benchmark": "humaneval",
            "text": (
                "def separate_paren_groups(paren_string: str) -> list[str]:\n"
                '    """Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n'
                "    separate those group into separate strings and return the list of those.\n"
                "    Separate groups are balanced, each group as a separate string.\n"
                "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
                "    ['()', '(())', '(()())']\n"
                '    """\n'
            ),
            "prompt": (
                "def separate_paren_groups(paren_string: str) -> list[str]:\n"
                '    """Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n'
                "    separate those group into separate strings and return the list of those.\n"
                "    Separate groups are balanced, each group as a separate string.\n"
                "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
                "    ['()', '(())', '(()())']\n"
                '    """\n'
            ),
            "task_id": "HumanEval/1",
        },
        {
            "benchmark": "humaneval",
            "text": (
                "def truncate_number(number: float) -> float:\n"
                '    """Given a positive floating point number, it can be decomposed into\n'
                "    and integer part (largest integer smaller than given number) and decimals\n"
                "    (leftover part always smaller than 1).\n\n"
                "    Return the decimal part of the number.\n"
                "    >>> truncate_number(3.5)\n"
                "    0.5\n"
                '    """\n'
            ),
            "prompt": (
                "def truncate_number(number: float) -> float:\n"
                '    """Given a positive floating point number, it can be decomposed into\n'
                "    and integer part (largest integer smaller than given number) and decimals\n"
                "    (leftover part always smaller than 1).\n\n"
                "    Return the decimal part of the number.\n"
                "    >>> truncate_number(3.5)\n"
                "    0.5\n"
                '    """\n'
            ),
            "task_id": "HumanEval/2",
        },
        {
            "benchmark": "humaneval",
            "text": (
                "def below_zero(operations: list[int]) -> bool:\n"
                '    """You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
                "    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n"
                "    at that point function should return True. Otherwise it should return False.\n"
                "    >>> below_zero([1, 2, 3])\n"
                "    False\n"
                "    >>> below_zero([1, 2, -4, 5])\n"
                "    True\n"
                '    """\n'
            ),
            "prompt": (
                "def below_zero(operations: list[int]) -> bool:\n"
                '    """You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
                "    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n"
                "    at that point function should return True. Otherwise it should return False.\n"
                "    >>> below_zero([1, 2, 3])\n"
                "    False\n"
                "    >>> below_zero([1, 2, -4, 5])\n"
                "    True\n"
                '    """\n'
            ),
            "task_id": "HumanEval/3",
        },
        {
            "benchmark": "humaneval",
            "text": (
                "def mean_absolute_deviation(numbers: list[float]) -> float:\n"
                '    """For a given list of input numbers, calculate Mean Absolute Deviation\n'
                "    around the mean of this dataset.\n"
                "    Mean Absolute Deviation is the average absolute difference between the\n"
                "    numbers and the mean:\n"
                "    MAD = average |x - x_mean|\n"
                "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
                "    1.0\n"
                '    """\n'
            ),
            "prompt": (
                "def mean_absolute_deviation(numbers: list[float]) -> float:\n"
                '    """For a given list of input numbers, calculate Mean Absolute Deviation\n'
                "    around the mean of this dataset.\n"
                "    Mean Absolute Deviation is the average absolute difference between the\n"
                "    numbers and the mean:\n"
                "    MAD = average |x - x_mean|\n"
                "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
                "    1.0\n"
                '    """\n'
            ),
            "task_id": "HumanEval/4",
        },
    ],
    "mmlu": [
        {
            "benchmark": "mmlu",
            "text": "Which of the following describes the primary function of the ribosome in a eukaryotic cell?",
            "question": "Which of the following describes the primary function of the ribosome in a eukaryotic cell?",
            "subject": "biology",
            "choices": [
                "DNA replication",
                "Protein synthesis",
                "Lipid metabolism",
                "Cellular respiration",
            ],
            "answer": 1,
        },
        {
            "benchmark": "mmlu",
            "text": "In classical mechanics, if the net external force acting on a system of particles is zero, which quantity is conserved?",
            "question": "In classical mechanics, if the net external force acting on a system of particles is zero, which quantity is conserved?",
            "subject": "physics",
            "choices": [
                "Total energy",
                "Total linear momentum",
                "Total angular momentum",
                "Kinetic energy",
            ],
            "answer": 1,
        },
        {
            "benchmark": "mmlu",
            "text": "What was the main purpose of the Marshall Plan following World War II?",
            "question": "What was the main purpose of the Marshall Plan following World War II?",
            "subject": "history",
            "choices": [
                "Rebuilding European economies and preventing communist spread",
                "Establishing military bases across the Pacific",
                "Creating the United Nations",
                "Developing atomic weapons technology",
            ],
            "answer": 0,
        },
        {
            "benchmark": "mmlu",
            "text": "Which algorithmic complexity best describes binary search on a sorted array of length n?",
            "question": "Which algorithmic complexity best describes binary search on a sorted array of length n?",
            "subject": "computer_science",
            "choices": [
                "O(1)",
                "O(log n)",
                "O(n)",
                "O(n log n)",
            ],
            "answer": 1,
        },
        {
            "benchmark": "mmlu",
            "text": "Under the United States Constitution, which branch has the power to declare war?",
            "question": "Under the United States Constitution, which branch has the power to declare war?",
            "subject": "jurisprudence",
            "choices": [
                "The Executive Branch",
                "The Legislative Branch (Congress)",
                "The Judicial Branch",
                "State Governors",
            ],
            "answer": 1,
        },
    ],
}


# ---------------------------------------------------------------------------
# Hugging Face Loaders
# ---------------------------------------------------------------------------


def _load_hf_gsm8k(split: str = "test") -> list[dict[str, Any]]:
    import datasets

    try:
        ds = datasets.load_dataset("gsm8k", "main", split=split)
    except Exception:
        ds = datasets.load_dataset("openai/gsm8k", "main", split=split)

    records: list[dict[str, Any]] = []
    for item in ds:
        q = str(item["question"])
        records.append(
            {
                "benchmark": "gsm8k",
                "text": q,
                "question": q,
                "answer": str(item.get("answer", "")),
            }
        )
    return records


def _load_hf_humaneval(split: str = "test") -> list[dict[str, Any]]:
    import datasets

    try:
        ds = datasets.load_dataset("openai_humaneval", split=split)
    except Exception:
        ds = datasets.load_dataset("openai/openai_humaneval", split=split)

    records: list[dict[str, Any]] = []
    for item in ds:
        p = str(item["prompt"])
        records.append(
            {
                "benchmark": "humaneval",
                "text": p,
                "prompt": p,
                "task_id": str(item.get("task_id", "")),
                "canonical_solution": str(item.get("canonical_solution", "")),
                "entry_point": str(item.get("entry_point", "")),
            }
        )
    return records


def _load_hf_mmlu(split: str = "test") -> list[dict[str, Any]]:
    import datasets

    ds = datasets.load_dataset("cais/mmlu", "all", split=split)
    records: list[dict[str, Any]] = []
    for item in ds:
        q = str(item["question"])
        records.append(
            {
                "benchmark": "mmlu",
                "text": q,
                "question": q,
                "subject": str(item.get("subject", "")),
                "choices": item.get("choices", []),
                "answer": item.get("answer", ""),
            }
        )
    return records


_HF_LOADERS: dict[str, Callable[[str], list[dict[str, Any]]]] = {
    "gsm8k": _load_hf_gsm8k,
    "humaneval": _load_hf_humaneval,
    "mmlu": _load_hf_mmlu,
}


def load_benchmark(
    name: str,
    *,
    split: str = "test",
    synthetic: bool = False,
    allow_synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Load a benchmark preset from Hugging Face or synthetic stand-in.

    Parameters
    ----------
    name:
        Benchmark preset name (``"mmlu"``, ``"gsm8k"``, or ``"humaneval"``).
    split:
        Dataset split to load (default: ``"test"``).
    synthetic:
        If True, return the synthetic stand-in directly without querying Hugging Face.
    allow_synthetic:
        If True and Hugging Face loading fails (e.g. offline/network error),
        fall back to the synthetic stand-in.

    Returns
    -------
    list[dict[str, Any]]
        List of records, each containing at least ``"benchmark"`` and ``"text"``.
    """
    clean_name = name.strip().lower()
    if clean_name not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{name}'. Supported benchmarks: {sorted(SUPPORTED_BENCHMARKS)}"
        )

    if synthetic:
        return [dict(row) for row in SYNTHETIC_BENCHMARKS[clean_name]]

    if clean_name == "gsm8k":
        loader = _load_hf_gsm8k
    elif clean_name == "humaneval":
        loader = _load_hf_humaneval
    elif clean_name == "mmlu":
        loader = _load_hf_mmlu
    else:
        raise ValueError(f"Unknown benchmark '{clean_name}'")

    check_datasets_available()
    try:
        return loader(split)
    except Exception as exc:
        if allow_synthetic:
            logger.warning(
                "Failed to download benchmark '%s' from Hugging Face: %s. Using synthetic stand-in.",
                clean_name,
                exc,
            )
            return [dict(row) for row in SYNTHETIC_BENCHMARKS[clean_name]]
        raise RuntimeError(
            f"Failed to load benchmark '{clean_name}' from Hugging Face: {exc}\n"
            "Ensure you have an active internet connection, or pass allow_synthetic=True."
        ) from exc
