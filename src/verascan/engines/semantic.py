"""Semantic similarity detection via sentence-transformers + FAISS.

This module is **optional** — install with ``pip install verascan[semantic]``.

When TensorFlow is also installed, sentence-transformers can fail to import
due to Keras version conflicts. We handle this gracefully: if the import
fails for *any* reason, we report clear instructions rather than a cryptic
traceback.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from tqdm import tqdm

import verascan._env  # noqa: F401
from verascan.report import MatchRecord, stored_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency availability (Lazy probing)
# ---------------------------------------------------------------------------

_PROBED: bool = False
_SEMANTIC_ERROR: str | None = None


def _probe_semantic_deps() -> str | None:
    """Try importing semantic deps and return an error string or ``None``."""
    global _PROBED, _SEMANTIC_ERROR  # noqa: PLW0603
    if _PROBED:
        return _SEMANTIC_ERROR

    errors: list[str] = []

    # -- sentence-transformers -------------------------------------------
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        errors.append("sentence-transformers (not installed)")
    except Exception as exc:
        # Covers Keras/TF conflicts, broken installs, etc.
        errors.append(f"sentence-transformers ({type(exc).__name__}: {exc})")

    # -- faiss -----------------------------------------------------------
    try:
        import faiss  # noqa: F401
    except ImportError:
        errors.append("faiss-cpu (not installed)")
    except Exception as exc:
        errors.append(f"faiss-cpu ({type(exc).__name__}: {exc})")

    _PROBED = True
    if errors:
        _SEMANTIC_ERROR = (
            "Semantic matching is unavailable.\n"
            "  Problems:\n" + "\n".join(f"    - {e}" for e in errors) + "\n\n"
            "  To fix, run:  pip install verascan[semantic]\n"
            "  If you have TensorFlow/Keras conflicts, also try:\n"
            "    pip install tf-keras\n"
            "    # or: pip uninstall tensorflow keras  (if you don't need TF)"
        )
    else:
        _SEMANTIC_ERROR = None

    return _SEMANTIC_ERROR


def is_available() -> bool:
    """Return ``True`` if semantic matching dependencies are usable."""
    return _probe_semantic_deps() is None


def require() -> None:
    """Raise :class:`ImportError` with actionable message if deps are broken."""
    err = _probe_semantic_deps()
    if err is not None:
        raise ImportError(err)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Passages overlap so a shared ending past the model window can still match.
_PASSAGE_WINDOW = 32
_PASSAGE_STRIDE = 16
# Every window must clear this before a score of 1.0 means the same document.
_SAME_WINDOW = 0.9995
_IDENTITY_CAP = 0.9999
_SIM_CHUNK = 4096
# A one-word or other short token span is not a contained passage.
_MIN_CONTAINED_TOKENS = 6
# One edited word inside a passage at least this long still clears 0.85.
_MIN_NEAR_WORDS = 6
_NEAR_CONTAINED_SCORE = 0.95
# A passage split across documents is covered by pieces at least this long.
_MIN_SPLIT_PIECE = 3
_MIN_SPLIT_EVAL_WORDS = 6
_SPLIT_COVER_SCORE = 0.90
_SPLIT_STEP_BUDGET = 2_000_000


def _validate_semantic_threshold(threshold: float) -> None:
    """Reject thresholds cosine similarity cannot use, before importing extras."""
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("Semantic threshold must be between 0 and 1.")


def _semantic_stored_score(score: float, threshold: float) -> float | None:
    """Clamp a cosine into [-1, 1] and store it with :func:`stored_similarity`.

    The shared rule keeps an admitted pair at or above *threshold* and does
    not round a cosine below 1 up to 1. Non-finite scores are dropped.
    """
    if not math.isfinite(score):
        return None
    clamped = max(-1.0, min(1.0, float(score)))
    if clamped < threshold:
        return None
    return stored_similarity(clamped, threshold)


def _content_limit(model: Any) -> int | None:
    """Tokens the model can embed, excluding special tokens. ``None`` skips windows."""
    tokenizer = getattr(model, "tokenizer", None)
    max_seq = getattr(model, "max_seq_length", None)
    if tokenizer is None or max_seq is None:
        return None
    try:
        limit = int(max_seq)
    except (TypeError, ValueError):
        return None
    counter = getattr(tokenizer, "num_special_tokens_to_add", None)
    special = 0
    if counter is not None:
        try:
            special = int(counter(False))
        except TypeError:
            try:
                special = int(counter())
            except (TypeError, ValueError):
                special = 0
        except ValueError:
            special = 0
    limit -= special
    if limit < 1:
        return None
    return limit


def _token_ids(tokenizer: Any, text: str) -> list[Any] | None:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        return None
    try:
        # verbose=False avoids a false "indexing errors" warning on long text
        # we only tokenize so it can be split into windows.
        ids = encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        try:
            ids = encode(text, add_special_tokens=False)
        except TypeError:
            ids = encode(text)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", [])
    try:
        return list(ids)
    except TypeError:
        return None


def _sliding(ids: list[Any], window: int, stride: int) -> list[list[Any]]:
    if len(ids) <= window:
        return [ids]
    chunks: list[list[Any]] = []
    start = 0
    while start < len(ids):
        chunks.append(ids[start : start + window])
        if start + window >= len(ids):
            break
        start += stride
    return chunks


def _prepare_text(model: Any, text: str) -> tuple[list[str], list[Any] | None]:
    """Return passage strings and the token ids they were cut from.

    Short strings, including empty and whitespace-only text, stay intact so
    their embedding matches a single ``encode`` call. Token ids are ``None``
    when the text cannot be tokenized. An empty list is a real tokenization
    of blank text and is not treated as a contained passage.
    """
    limit = _content_limit(model)
    tokenizer = getattr(model, "tokenizer", None)
    if limit is None or tokenizer is None:
        return [text], None
    ids = _token_ids(tokenizer, text)
    if ids is None or len(ids) <= limit:
        return [text], ids
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        return [text], ids
    window = min(_PASSAGE_WINDOW, limit)
    stride = min(_PASSAGE_STRIDE, window)
    segments: list[str] = []
    for chunk in _sliding(ids, window, stride):
        piece = decode(chunk)
        segments.append(piece if isinstance(piece, str) else str(piece))
    return (segments or [text]), ids


def _contains_token_span(haystack: list[Any], needle: list[Any]) -> bool:
    """True when every token of *needle* sits in *haystack* as one slice."""
    if not needle or len(needle) > len(haystack):
        return False
    span = len(needle)
    if haystack[:span] == needle or haystack[-span:] == needle:
        return True
    first = needle[0]
    start = 1
    last = len(haystack) - span
    while start <= last:
        try:
            found = haystack.index(first, start, last + 1)
        except ValueError:
            return False
        if haystack[found : found + span] == needle:
            return True
        start = found + 1
    return False


def _pair_contains(eval_ids: list[Any] | None, train_ids: list[Any] | None) -> bool:
    """True when one text's tokens are a proper subspan of the other's.

    Equal lengths are document identity and stay on the embedding path.
    Blank token lists are not a passage contained in something else. A span
    shorter than :data:`_MIN_CONTAINED_TOKENS` is not a passage either: one
    common word is a subspan of almost any document.
    """
    if not eval_ids or not train_ids or len(eval_ids) == len(train_ids):
        return False
    if len(eval_ids) < len(train_ids):
        if len(eval_ids) < _MIN_CONTAINED_TOKENS:
            return False
        return _contains_token_span(train_ids, eval_ids)
    if len(train_ids) < _MIN_CONTAINED_TOKENS:
        return False
    return _contains_token_span(eval_ids, train_ids)


def _score_contained_span() -> float:
    """Score a verbatim contained passage below whole-document agreement.

    Identical tokens embed to the same vector, so the span cosine is 1.
    The pair is stored at the identity cap even when the threshold is higher.
    A threshold above the cap then leaves the pair unmatched. Promoting the
    score to 1.0 would claim the shorter text is the same document.
    """
    return _IDENTITY_CAP


def _words(text: str) -> list[str]:
    return text.lower().split()


def _one_word_off(shorter: list[str], longer: list[str]) -> bool:
    """True when *shorter* matches one contiguous slice of *longer* except one word."""
    span = len(shorter)
    if span < _MIN_NEAR_WORDS or span >= len(longer):
        return False
    last = len(longer) - span
    for start in range(last + 1):
        diffs = 0
        matched = True
        for offset in range(span):
            if longer[start + offset] != shorter[offset]:
                diffs += 1
                if diffs > 1:
                    matched = False
                    break
        if matched and diffs == 1:
            return True
    return False


def _texts_near_contained(left: str, right: str) -> bool:
    """True when the shorter text is a contained passage with one word changed.

    Equal lengths stay on the embedding path. Two long documents that differ
    by one word are not a short passage inside a longer one.
    """
    left_words = _words(left)
    right_words = _words(right)
    if len(left_words) < len(right_words):
        return _one_word_off(left_words, right_words)
    if len(right_words) < len(left_words):
        return _one_word_off(right_words, left_words)
    return False


def _split_passage_pairs(
    train_texts: list[str], eval_texts: list[str]
) -> set[tuple[int, int]]:
    """Training rows that together tile an evaluation passage.

    Each piece is a contiguous run of at least :data:`_MIN_SPLIT_PIECE`
    words. A cover that uses only one document is verbatim containment and
    is left to that path. The step budget skips pathological repetition
    instead of hanging the scan.
    """
    train_words = [_words(text) for text in train_texts]
    positions: dict[str, list[tuple[int, int]]] = {}
    for doc_index, words in enumerate(train_words):
        for pos, word in enumerate(words):
            positions.setdefault(word, []).append((doc_index, pos))

    pairs: set[tuple[int, int]] = set()
    steps = 0
    for eval_index, eval_text in enumerate(eval_texts):
        words = _words(eval_text)
        length = len(words)
        if length < _MIN_SPLIT_EVAL_WORDS:
            continue
        can_cover = [False] * (length + 1)
        can_cover[length] = True
        choice: list[tuple[int, int] | None] = [None] * length
        aborted = False
        for start in range(length - 1, -1, -1):
            found = False
            for doc_index, pos in positions.get(words[start], ()):
                steps += 1
                if steps > _SPLIT_STEP_BUDGET:
                    aborted = True
                    break
                document = train_words[doc_index]
                limit = min(len(document) - pos, length - start)
                matched = 0
                while matched < limit and document[pos + matched] == words[start + matched]:
                    matched += 1
                    steps += 1
                    if steps > _SPLIT_STEP_BUDGET:
                        aborted = True
                        break
                if aborted:
                    break
                piece = _MIN_SPLIT_PIECE
                while piece <= matched:
                    if can_cover[start + piece]:
                        can_cover[start] = True
                        choice[start] = (piece, doc_index)
                        found = True
                        break
                    piece += 1
                if found:
                    break
            if aborted:
                break
        if aborted or not can_cover[0]:
            continue
        docs: set[int] = set()
        cursor = 0
        while cursor < length:
            picked = choice[cursor]
            if picked is None:
                docs = set()
                break
            piece, doc_index = picked
            docs.add(doc_index)
            cursor += piece
        if len(docs) < 2:
            continue
        for doc_index in docs:
            pairs.add((eval_index, doc_index))
    return pairs


def _adjust_semantic_score(
    eval_text: str,
    train_text: str,
    eval_ids: list[Any] | None,
    train_ids: list[Any] | None,
    cosine: float,
    *,
    split_pair: bool,
) -> float:
    """Prefer a lexical passage score over a diluted window cosine."""
    if _pair_contains(eval_ids, train_ids):
        return _score_contained_span()
    base = cosine if math.isfinite(cosine) else -math.inf
    if _texts_near_contained(eval_text, train_text):
        return min(max(base, _NEAR_CONTAINED_SCORE), _IDENTITY_CAP)
    if split_pair:
        return min(max(base, _SPLIT_COVER_SCORE), _IDENTITY_CAP)
    return cosine


def _encode_matrix(model: Any, texts: list[str], batch_size: int, show_progress: bool) -> Any:
    import numpy as np

    matrix = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _pair_evidence(eval_mat: Any, train_mat: Any) -> tuple[float, bool]:
    """Max cosine, and whether every window on both sides clears ``_SAME_WINDOW``."""
    import numpy as np

    evidence = -math.inf
    eval_best = np.full(eval_mat.shape[0], -math.inf, dtype=np.float64)
    windows_match = True
    for start in range(0, train_mat.shape[0], _SIM_CHUNK):
        sims = eval_mat @ train_mat[start : start + _SIM_CHUNK].T
        evidence = max(evidence, float(np.max(sims)))
        eval_best = np.maximum(eval_best, np.max(sims, axis=1))
        if not bool(np.all(np.max(sims, axis=0) >= _SAME_WINDOW)):
            windows_match = False
    fully_same = windows_match and bool(np.all(eval_best >= _SAME_WINDOW))
    return evidence, fully_same


def _append_match(
    matches: list[MatchRecord],
    *,
    eval_idx: int,
    train_idx: int,
    eval_texts: list[str],
    train_texts: list[str],
    raw_score: float,
    threshold: float,
) -> None:
    stored = _semantic_stored_score(raw_score, threshold)
    if stored is None:
        return
    matches.append(
        MatchRecord(
            eval_index=eval_idx,
            train_index=train_idx,
            eval_text=eval_texts[eval_idx],
            train_text=train_texts[train_idx],
            score=stored,
            method="semantic",
        )
    )


def _faiss_matches(
    model: Any,
    train_texts: list[str],
    eval_texts: list[str],
    train_ids: list[list[Any] | None],
    eval_ids: list[list[Any] | None],
    split_pairs: set[tuple[int, int]],
    *,
    threshold: float,
    batch_size: int,
    show_progress: bool,
) -> list[MatchRecord]:
    """Every training row at or above *threshold*. The index order is not assumed."""
    import faiss  # type: ignore[import-untyped]

    logger.info("Encoding %d train texts ...", len(train_texts))
    train_embs = _encode_matrix(model, train_texts, batch_size, show_progress)
    logger.info("Encoding %d eval texts ...", len(eval_texts))
    eval_embs = _encode_matrix(model, eval_texts, batch_size, show_progress)

    index = faiss.IndexFlatIP(train_embs.shape[1])
    index.add(train_embs)
    # k covers the whole training set. A top-10 cap hid extra copies of one row.
    scores_all, indices_all = index.search(eval_embs, len(train_texts))

    matches: list[MatchRecord] = []
    for eval_idx in tqdm(
        range(len(eval_texts)),
        desc="Semantic scan",
        disable=not show_progress,
        leave=False,
    ):
        seen: set[int] = set()
        for rank in range(len(train_texts)):
            train_idx = int(indices_all[eval_idx][rank])
            if train_idx < 0 or train_idx in seen:
                continue
            seen.add(train_idx)
            raw_score = _adjust_semantic_score(
                eval_texts[eval_idx],
                train_texts[train_idx],
                eval_ids[eval_idx],
                train_ids[train_idx],
                float(scores_all[eval_idx][rank]),
                split_pair=(eval_idx, train_idx) in split_pairs,
            )
            _append_match(
                matches,
                eval_idx=eval_idx,
                train_idx=train_idx,
                eval_texts=eval_texts,
                train_texts=train_texts,
                raw_score=raw_score,
                threshold=threshold,
            )
    return matches


def _passage_matches(
    model: Any,
    train_texts: list[str],
    eval_texts: list[str],
    train_segments: list[list[str]],
    eval_segments: list[list[str]],
    train_ids: list[list[Any] | None],
    eval_ids: list[list[Any] | None],
    split_pairs: set[tuple[int, int]],
    *,
    threshold: float,
    batch_size: int,
    show_progress: bool,
) -> list[MatchRecord]:
    """Score each document by its best passage, and cap identity when windows disagree."""
    train_spans: list[tuple[int, int]] = []
    flat_train: list[str] = []
    for segments in train_segments:
        train_spans.append((len(flat_train), len(flat_train) + len(segments)))
        flat_train.extend(segments)
    eval_spans: list[tuple[int, int]] = []
    flat_eval: list[str] = []
    for segments in eval_segments:
        eval_spans.append((len(flat_eval), len(flat_eval) + len(segments)))
        flat_eval.extend(segments)

    logger.info("Encoding %d train passages ...", len(flat_train))
    train_emb = _encode_matrix(model, flat_train, batch_size, show_progress)
    logger.info("Encoding %d eval passages ...", len(flat_eval))
    eval_emb = _encode_matrix(model, flat_eval, batch_size, show_progress)

    matches: list[MatchRecord] = []
    for eval_idx in tqdm(
        range(len(eval_texts)),
        desc="Semantic scan",
        disable=not show_progress,
        leave=False,
    ):
        eval_start, eval_end = eval_spans[eval_idx]
        eval_mat = eval_emb[eval_start:eval_end]
        ranked: list[tuple[float, int]] = []
        for train_idx, (train_start, train_end) in enumerate(train_spans):
            evidence, fully_same = _pair_evidence(eval_mat, train_emb[train_start:train_end])
            # Fixed windows dilute a short passage. Lexical containment,
            # a one-word edit, or a split across documents replaces that cosine.
            cosine = evidence if fully_same else min(evidence, _IDENTITY_CAP)
            raw = _adjust_semantic_score(
                eval_texts[eval_idx],
                train_texts[train_idx],
                eval_ids[eval_idx],
                train_ids[train_idx],
                cosine,
                split_pair=(eval_idx, train_idx) in split_pairs,
            )
            if raw >= threshold:
                ranked.append((raw, train_idx))
        for raw, train_idx in sorted(ranked, key=lambda item: (-item[0], item[1])):
            _append_match(
                matches,
                eval_idx=eval_idx,
                train_idx=train_idx,
                eval_texts=eval_texts,
                train_texts=train_texts,
                raw_score=raw,
                threshold=threshold,
            )
    return matches


def find_semantic_matches(
    train_texts: list[str],
    eval_texts: list[str],
    *,
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[MatchRecord]:
    """Return every pair whose passage cosine is at least *threshold*.

    Texts that fit in the model window are embedded as a single string, so
    short, empty, and whitespace inputs keep their original scores. Longer
    texts are split into overlapping passages. A score of 1.0 means those
    passages agree; a shared prefix alone cannot report document identity.

    A verbatim passage of at least six tokens contained in the other text is
    scored from that token span, including an evaluation row that is exactly
    the ending of a longer training row. The span cosine is 1. It is reported
    at the identity cap, which is below 1, including when *threshold* is
    higher than the cap. A higher threshold does not relabel the shorter
    text as the same document.

    One changed word inside such a passage, and a passage whose contiguous
    pieces sit in different training rows, stay at or above the default
    threshold without using the identity cap. A single common word does not.

    Parameters
    ----------
    train_texts, eval_texts:
        Lists of raw text strings.
    threshold:
        Minimum cosine similarity to report. Must be between 0 and 1.
    model_name:
        Any model supported by ``sentence-transformers``.
    batch_size:
        Encoding batch size.
    show_progress:
        Show tqdm progress bars.

    Raises
    ------
    ValueError
        If *threshold* is not a finite number in ``[0, 1]``.
    ImportError
        If the semantic extras are not installed or broken.
    """
    _validate_semantic_threshold(threshold)
    require()

    if not train_texts or not eval_texts:
        return []

    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    logger.info("Loading model '%s' ...", model_name)
    model = SentenceTransformer(model_name)
    prepared_train = [_prepare_text(model, text) for text in train_texts]
    prepared_eval = [_prepare_text(model, text) for text in eval_texts]
    train_segments = [segments for segments, _ids in prepared_train]
    eval_segments = [segments for segments, _ids in prepared_eval]
    train_ids = [ids for _segments, ids in prepared_train]
    eval_ids = [ids for _segments, ids in prepared_eval]
    split_pairs = _split_passage_pairs(train_texts, eval_texts)
    windowed = any(
        len(segments) != 1 or segments[0] != text
        for segments, text in zip(train_segments, train_texts)
    ) or any(
        len(segments) != 1 or segments[0] != text
        for segments, text in zip(eval_segments, eval_texts)
    )
    if windowed:
        matches = _passage_matches(
            model,
            train_texts,
            eval_texts,
            train_segments,
            eval_segments,
            train_ids,
            eval_ids,
            split_pairs,
            threshold=threshold,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    else:
        matches = _faiss_matches(
            model,
            train_texts,
            eval_texts,
            train_ids,
            eval_ids,
            split_pairs,
            threshold=threshold,
            batch_size=batch_size,
            show_progress=show_progress,
        )

    logger.info("Semantic scan complete: %d matches above %.2f", len(matches), threshold)
    return matches
