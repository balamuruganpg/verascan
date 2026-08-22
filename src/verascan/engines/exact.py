"""Exact-match contamination detection via content hashing.

Uses SHA-256 hashes of normalised text for O(n+m) lookup.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from tqdm import tqdm

from verascan.report import MatchRecord


def _normalise(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for stable hashing."""
    return " ".join(text.lower().split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_exact_matches(
    train_texts: list[str],
    eval_texts: list[str],
    *,
    show_progress: bool = True,
) -> list[MatchRecord]:
    """Return exact duplicates between *train_texts* and *eval_texts*.

    Normalisation (lowercased, whitespace-collapsed) is applied before
    comparison so that trivially different formatting is ignored.

    Complexity: O(n + m) where n = len(train_texts), m = len(eval_texts).
    """
    # Build hash → list-of-train-indices mapping.
    train_index: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(
        tqdm(train_texts, desc="Hashing train", disable=not show_progress, leave=False)
    ):
        h = _hash(_normalise(text))
        train_index[h].append(idx)

    matches: list[MatchRecord] = []
    for eval_idx, eval_text in enumerate(
        tqdm(eval_texts, desc="Exact scan", disable=not show_progress, leave=False)
    ):
        h = _hash(_normalise(eval_text))
        if h in train_index:
            for train_idx in train_index[h]:
                matches.append(
                    MatchRecord(
                        eval_index=eval_idx,
                        train_index=train_idx,
                        eval_text=eval_text,
                        train_text=train_texts[train_idx],
                        score=1.0,
                        method="exact",
                    )
                )

    return matches
