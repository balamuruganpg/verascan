"""Fuzzy / near-duplicate detection via MinHash + LSH (datasketch).

Uses character-level n-gram shingling with MinHash signatures and
Locality-Sensitive Hashing for fast approximate Jaccard similarity.
"""

from __future__ import annotations

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from verascan.report import MatchRecord

_SHINGLE_SIZE = 5  # character n-gram length


def _shingle(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    """Return the set of character k-grams in *text*."""
    text = " ".join(text.lower().split())  # normalise whitespace
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


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
    """Return near-duplicate pairs above *threshold* Jaccard similarity.

    Parameters
    ----------
    train_texts, eval_texts:
        Lists of raw text strings.
    threshold:
        Minimum estimated Jaccard similarity to report (0–1).
    num_perm:
        Number of MinHash permutations (higher → more accurate but slower).
    show_progress:
        Show tqdm progress bars.
    """
    # Build LSH index from training data.
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    train_minhashes: list[MinHash] = []

    for idx, text in enumerate(
        tqdm(train_texts, desc="MinHash train", disable=not show_progress, leave=False)
    ):
        mh = _make_minhash(_shingle(text), num_perm)
        train_minhashes.append(mh)
        lsh.insert(f"train_{idx}", mh)

    # Query each eval text.
    matches: list[MatchRecord] = []
    for eval_idx, eval_text in enumerate(
        tqdm(eval_texts, desc="Fuzzy scan", disable=not show_progress, leave=False)
    ):
        eval_mh = _make_minhash(_shingle(eval_text), num_perm)
        candidates = lsh.query(eval_mh)

        for key in candidates:
            train_idx = int(key.split("_", 1)[1])
            # Compute actual estimated Jaccard for the pair.
            score = eval_mh.jaccard(train_minhashes[train_idx])
            if score >= threshold:
                matches.append(
                    MatchRecord(
                        eval_index=eval_idx,
                        train_index=train_idx,
                        eval_text=eval_text,
                        train_text=train_texts[train_idx],
                        score=round(score, 4),
                        method="fuzzy",
                    )
                )

    return matches
