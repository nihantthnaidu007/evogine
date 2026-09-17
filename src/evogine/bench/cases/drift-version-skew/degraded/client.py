"""Degraded client: encodes the v1 wire format (tagged, no checksum)."""

from __future__ import annotations

WIRE_VERSION = 1


def encode_batch(items: list[int]) -> dict:
    """Encode ``items`` into the v1 wire payload."""
    return {"ver": WIRE_VERSION, "items": list(items)}
