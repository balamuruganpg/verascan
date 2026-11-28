"""Fuzzy / near-duplicate detection via MinHash signatures (datasketch).

Uses character-level n-gram shingling. Each pair's MinHash Jaccard estimate is
compared directly. An LSH index can drop a pair whose own estimate is already
above the threshold, so it is not used to decide membership.
"""

from __future__ import annotations

import numpy as np
from datasketch import MinHash
from tqdm import tqdm

from verascan.normalize import match_keys
from verascan.report import MatchRecord, stored_similarity

_SHINGLE_SIZE = 5  # character n-gram length


def _shingle(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    """Return the set of character k-grams in an already folded *text*."""
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _signature_pair(text: str, num_perm: int) -> tuple[np.ndarray, np.ndarray]:
    """MinHash both format-character readings. They coincide when there are none."""
    deleted, spaced = match_keys(text)
    primary = _make_minhash(_shingle(deleted), num_perm).hashvalues
    if deleted == spaced:
        return primary, primary
    return primary, _make_minhash(_shingle(spaced), num_perm).hashvalues


def _make_minhash(shingles: set[str], num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


def find_fuzzy_matches(
    train_texts: list[str],
    eval_texts: list[str],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    show_progress: bool = True,
) -> list[MatchRecord]:
    """Return pairs whose MinHash Jaccard estimate is at least *threshold*.

    Parameters
    ----------
    train_texts, eval_texts:
        Lists of raw text strings.
    threshold:
        Minimum estimated Jaccard similarity to report (0–1). ``0`` reports
        every pair, including estimates of zero, because zero still meets it.
    num_perm:
        Number of MinHash permutations (higher → more accurate but slower).
    show_progress:
        Show tqdm progress bars.
    """
    if isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("Fuzzy threshold must be between 0 and 1.")
    if num_perm < 2:
        raise ValueError("Fuzzy matching requires at least two MinHash permutations.")
    if not train_texts or not eval_texts:
        return []

    train_deleted: list[np.ndarray] = []
    train_spaced: list[np.ndarray] = []
    for text in tqdm(train_texts, desc="MinHash train", disable=not show_progress, leave=False):
        deleted, spaced = _signature_pair(text, num_perm)
        train_deleted.append(deleted)
        train_spaced.append(spaced)
    signatures = np.stack(train_deleted)
    spaced_signatures = np.stack(train_spaced)

    matches: list[MatchRecord] = []
    for eval_idx, eval_text in enumerate(
        tqdm(eval_texts, desc="Fuzzy scan", disable=not show_progress, leave=False)
    ):
        eval_deleted, eval_spaced = _signature_pair(eval_text, num_perm)
        deleted_scores = (signatures == eval_deleted).mean(axis=1)
        spaced_scores = (spaced_signatures == eval_spaced).mean(axis=1)
        scores = np.maximum(deleted_scores, spaced_scores)
        for train_idx in np.flatnonzero(scores >= threshold):
            # ``stored_similarity`` keeps the four-decimal score on the
            # admitting side of ``threshold``. Rounding 0.765625 down to
            # 0.7656 would hide a pair this scan just accepted.
            admitted = stored_similarity(float(scores[train_idx]), threshold)
            matches.append(
                MatchRecord(
                    eval_index=eval_idx,
                    train_index=int(train_idx),
                    eval_text=eval_text,
                    train_text=train_texts[int(train_idx)],
                    score=admitted,
                    method="fuzzy",
                )
            )

    return matches
