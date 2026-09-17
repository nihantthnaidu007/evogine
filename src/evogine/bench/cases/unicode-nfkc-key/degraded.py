"""Degraded v1: casefold without normalization.

v1 read "case-insensitive matching" and stopped there. ASCII inputs — the
entire launch corpus — collapse correctly. Compatibility forms (fullwidth,
ligatures) survive casefold untouched, so visually identical names get
different keys.
"""

from __future__ import annotations

SEPARATOR = "·"


def display_key(parts: list[str]) -> str:
    """Join casefolded parts with the key separator (no NFKC step)."""
    return SEPARATOR.join(part.casefold() for part in parts)
