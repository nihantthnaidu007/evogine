"""Reference server: verifies the v2 wire format before decoding."""

from __future__ import annotations

from .client import WIRE_VERSION, checksum


def decode_batch(payload: dict) -> list[int]:
    """Decode a v2 payload, refusing wrong versions and broken checksums."""
    if not isinstance(payload, dict) or payload.get("ver") != WIRE_VERSION:
        raise ValueError("unsupported payload schema version")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("payload items must be a list")
    if payload.get("checksum") != checksum(list(items)):
        raise ValueError("payload checksum mismatch")
    return list(items)
