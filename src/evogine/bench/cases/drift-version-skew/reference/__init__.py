"""Reference: one shared v2 wire schema across the client/server seam."""

from __future__ import annotations

from .client import encode_batch, WIRE_VERSION
from .server import decode_batch


class Pipeline:
    """The client/server facade: encode client-side, decode server-side."""

    def __init__(self) -> None:
        self.last_payload: dict | None = None

    def send(self, items: list[int]) -> list[int]:
        """Encode ``items`` client-side and decode the payload server-side."""
        self.last_payload = encode_batch(list(items))
        return decode_batch(self.last_payload)

    def raw_decode(self, payload: dict) -> list[int]:
        """Decode a payload exactly as the server would."""
        return decode_batch(payload)


__all__ = ["Pipeline", "WIRE_VERSION", "encode_batch", "decode_batch"]
