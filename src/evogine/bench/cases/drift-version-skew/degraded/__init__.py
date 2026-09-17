"""Degraded v1: a self-consistent pair that predates the shared schema.

The v1 pair speaks version 1 with no checksum, and each side is internally
consistent — client encodes, server decodes, the facade round-trips. The
seam is the decay: v2 payloads are refused, v1 payloads the v2 contract
forbids are accepted without verification.
"""

from __future__ import annotations

from .client import encode_batch, WIRE_VERSION
from .server import decode_batch


class Pipeline:
    """The client/server facade (v1 semantics throughout)."""

    def __init__(self) -> None:
        self.last_payload: dict | None = None

    def send(self, items: list[int]) -> list[int]:
        """Encode ``items`` client-side and decode the payload server-side."""
        self.last_payload = encode_batch(list(items))
        return decode_batch(self.last_payload)

    def raw_decode(self, payload: dict) -> list[int]:
        """Decode a payload exactly as the (v1) server would."""
        return decode_batch(payload)


__all__ = ["Pipeline", "WIRE_VERSION", "encode_batch", "decode_batch"]
