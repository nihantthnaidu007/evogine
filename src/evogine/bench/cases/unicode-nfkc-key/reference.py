"""Reference: display keys under NFKC normalization + casefold.

Compatibility-equivalent spellings collapse to one key. Deterministic:
fixed unicode tables only, no wall clock, no randomness, no locale reads.
"""

from __future__ import annotations

import unicodedata

SEPARATOR = "·"


def display_key(parts: list[str]) -> str:
    """Join NFKC-normalized, casefolded parts with the key separator."""
    return SEPARATOR.join(
        unicodedata.normalize("NFKC", part).casefold() for part in parts
    )
