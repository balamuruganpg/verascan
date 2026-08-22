"""Detection engines — exact, fuzzy, and semantic.

Semantic matching is optional.  Use :func:`semantic_available` to check
whether the heavy dependencies are installed before calling semantic methods.
"""

from __future__ import annotations


def semantic_available() -> bool:
    """Check whether semantic matching dependencies are usable."""
    from verascan.engines.semantic import is_available

    return is_available()


__all__ = ["semantic_available"]
