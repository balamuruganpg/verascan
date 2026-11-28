"""Shared text folding for exact and fuzzy comparison.

A zero-width character or a confusable letter should not hide a copy from
the default scan. Format characters are read two ways: removed, and treated
as spaces. Callers match a pair when either reading agrees.
"""

from __future__ import annotations

import unicodedata

# Lowercase Cyrillic letters that spoof Latin, applied after ``str.lower``.
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a",  # а
        "\u0435": "e",  # е
        "\u043e": "o",  # о
        "\u0440": "p",  # р
        "\u0441": "c",  # с
        "\u0443": "y",  # у
        "\u0445": "x",  # х
        "\u0456": "i",  # і
        "\u0458": "j",  # ј
        "\u0455": "s",  # ѕ
        "\u04bb": "h",  # һ
        "\u0501": "d",  # ԁ
        "\u051b": "q",  # ԛ
        "\u051d": "w",  # ԝ
        "\u0475": "v",  # ѵ
        "\u04cf": "i",  # ӏ
    }
)


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower().translate(_CONFUSABLES)


def _without_format(text: str) -> str:
    return "".join(char for char in text if unicodedata.category(char) != "Cf")


def _format_as_space(text: str) -> str:
    return "".join(" " if unicodedata.category(char) == "Cf" else char for char in text)


def _collapse(text: str) -> str:
    return " ".join(text.split())


def match_keys(text: str) -> tuple[str, str]:
    """Return the delete-format and space-format readings of *text*.

    The two strings are equal when *text* has no format characters. A
    zero-width space inside a word agrees on the first reading. A zero-width
    space used instead of a space agrees on the second.
    """
    folded = _fold(text)
    return _collapse(_without_format(folded)), _collapse(_format_as_space(folded))
