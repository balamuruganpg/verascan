"""GPT-3-style word n-gram overlap contamination detection.

Implements the classic Brown et al. (2020) open-data check: lowercase and
strip punctuation, build word *n*-grams (default 13), drop grams that are
too frequent in train, and flag eval examples that share any retained gram.

Inspired by the LLMSanitize ``gpt3`` open-data method
(https://github.com/ntunlp/LLMSanitize) — reimplemented to fit Verascan's
``MatchRecord`` / progress-bar conventions.
"""

from __future__ import annotations

import string
from collections import Counter, defaultdict

from tqdm import tqdm

from verascan.report import MatchRecord

_DEFAULT_N = 13
_DEFAULT_MAX_COUNT = 10


def _clean_text(text: str) -> str:
    """Lowercase and strip punctuation from each whitespace-split token."""
    return " ".join(word.strip(string.punctuation) for word in text.lower().split())


def _word_ngrams(text: str, n: int) -> list[str]:
    """Return ordered word *n*-grams from cleaned *text* (may be empty)."""
    words = _clean_text(text).split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def find_ngram_matches(
    train_texts: list[str],
    eval_texts: list[str],
    *,
    n: int = _DEFAULT_N,
    max_count: int = _DEFAULT_MAX_COUNT,
    min_collisions: int = 1,
    show_progress: bool = True,
) -> list[MatchRecord]:
    """Flag eval texts that share word *n*-grams with train.

    Parameters
    ----------
    train_texts, eval_texts:
        Lists of raw text strings.
    n:
        Word n-gram size (GPT-3 default: 13).
    max_count:
        Drop training n-grams that appear in at least this many *documents*
        (too frequent / uninformative). GPT-3 / LLMSanitize default: 10.
    min_collisions:
        Minimum number of shared retained n-grams to flag a pair (default: 1).
    show_progress:
        Show tqdm progress bars.

    Scoring
    -------
    For each (eval, train) pair that shares at least *min_collisions*
    retained n-grams, ``score`` is the fraction of the eval example's
    n-grams that also appear in that train document's retained set
    (overlap ratio in ``[0, 1]``).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if max_count < 1:
        raise ValueError(f"max_count must be >= 1, got {max_count}")
    if min_collisions < 1:
        raise ValueError(f"min_collisions must be >= 1, got {min_collisions}")

    # ngram → document frequency + list of train indices containing it
    doc_freq: Counter[str] = Counter()
    ngram_to_train: dict[str, list[int]] = defaultdict(list)
    train_ngram_sets: list[set[str]] = []

    for idx, text in enumerate(
        tqdm(train_texts, desc="N-gram train", disable=not show_progress, leave=False)
    ):
        grams = set(_word_ngrams(text, n))
        train_ngram_sets.append(grams)
        for gram in grams:
            doc_freq[gram] += 1
            ngram_to_train[gram].append(idx)

    # Drop too-frequent grams (keep only informative collisions).
    retained: set[str] = {g for g, c in doc_freq.items() if c < max_count}
    if not retained:
        return []

    matches: list[MatchRecord] = []
    for eval_idx, eval_text in enumerate(
        tqdm(eval_texts, desc="N-gram scan", disable=not show_progress, leave=False)
    ):
        eval_grams = _word_ngrams(eval_text, n)
        if not eval_grams:
            continue

        # Per-train collision counts (only retained grams).
        shared_counts: Counter[int] = Counter()
        for gram in eval_grams:
            if gram not in retained:
                continue
            for train_idx in ngram_to_train[gram]:
                shared_counts[train_idx] += 1

        for train_idx, collisions in shared_counts.items():
            if collisions < min_collisions:
                continue
            # Overlap vs this train doc's retained grams (unique eval grams
            # that appear in the train doc), normalised by eval gram count.
            eval_unique = set(eval_grams)
            train_retained = train_ngram_sets[train_idx] & retained
            overlap = len(eval_unique & train_retained)
            score = overlap / len(eval_unique) if eval_unique else 0.0
            matches.append(
                MatchRecord(
                    eval_index=eval_idx,
                    train_index=train_idx,
                    eval_text=eval_text,
                    train_text=train_texts[train_idx],
                    score=round(score, 4),
                    method="ngram",
                )
            )

    return matches
