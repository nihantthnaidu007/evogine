"""Degraded server: decodes the v1 wire format without verification."""

from __future__ import annotations

from .client import WIRE_VERSION


def decode_batch(payload: dict) -> list[int]:
    """Decode a v1 payload; refuses non-v1 versions, verifies nothing."""
    if not isinstance(payload, dict) or payload.get("ver") != WIRE_VERSION:
        raise ValueError("unsupported payload schema version")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("payload items must be a list")
    return list(items)
