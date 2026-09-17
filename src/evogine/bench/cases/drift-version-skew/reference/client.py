"""Reference client: encodes the v2 wire format (tagged + checksummed)."""

from __future__ import annotations

WIRE_VERSION = 2
_MODULUS = 2**32


def checksum(items: list[int]) -> int:
    """The v2 checksum: item sum mod 2^32."""
    return sum(items) % _MODULUS


def encode_batch(items: list[int]) -> dict:
    """Encode ``items`` into the v2 wire payload."""
    return {"ver": WIRE_VERSION, "items": list(items), "checksum": checksum(list(items))}
