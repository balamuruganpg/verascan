"""Semantic similarity detection via sentence-transformers + FAISS.

This module is **optional** — install with ``pip install verascan[semantic]``.

When TensorFlow is also installed, sentence-transformers can fail to import
due to Keras version conflicts. We handle this gracefully: if the import
fails for *any* reason, we report clear instructions rather than a cryptic
traceback.
"""

from __future__ import annotations

import logging

from tqdm import tqdm

import verascan._env  # noqa: F401
from verascan.report import MatchRecord

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


def find_semantic_matches(
    train_texts: list[str],
    eval_texts: list[str],
    *,
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[MatchRecord]:
    """Return semantically similar pairs above *threshold* cosine similarity.

    Parameters
    ----------
    train_texts, eval_texts:
        Lists of raw text strings.
    threshold:
        Minimum cosine similarity to report (0 -- 1).
    model_name:
        Any model supported by ``sentence-transformers``.
    batch_size:
        Encoding batch size.
    show_progress:
        Show tqdm progress bars.

    Raises
    ------
    ImportError
        If the semantic extras are not installed or broken.
    """
    require()

    import faiss  # type: ignore[import-untyped]
    import numpy as np
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    if not train_texts or not eval_texts:
        return []

    logger.info("Loading model '%s' ...", model_name)
    model = SentenceTransformer(model_name)

    # Encode training set.
    logger.info("Encoding %d train texts ...", len(train_texts))
    train_embs = model.encode(
        train_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Encode eval set.
    logger.info("Encoding %d eval texts ...", len(eval_texts))
    eval_embs = model.encode(
        eval_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Build FAISS index (inner product on L2-normalised vectors = cosine sim).
    dim = train_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(train_embs.astype(np.float32))

    # Search — retrieve top-k candidates per eval example.
    k = min(10, len(train_texts))
    scores_all, indices_all = index.search(eval_embs.astype(np.float32), k)

    matches: list[MatchRecord] = []
    for eval_idx in tqdm(
        range(len(eval_texts)),
        desc="Semantic scan",
        disable=not show_progress,
        leave=False,
    ):
        for rank in range(k):
            train_idx = int(indices_all[eval_idx][rank])
            score = float(scores_all[eval_idx][rank])
            if train_idx < 0:
                continue
            if score < threshold:
                break  # scores are sorted descending
            matches.append(
                MatchRecord(
                    eval_index=eval_idx,
                    train_index=train_idx,
                    eval_text=eval_texts[eval_idx],
                    train_text=train_texts[train_idx],
                    score=round(score, 4),
                    method="semantic",
                )
            )

    logger.info("Semantic scan complete: %d matches above %.2f", len(matches), threshold)
    return matches
