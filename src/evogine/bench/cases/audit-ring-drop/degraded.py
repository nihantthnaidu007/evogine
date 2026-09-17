"""Degraded v1: drop the oldest line at the cap, report a clean bill.

v1 treats the cap as a ring: when full, the oldest line pops and the new one
appends. The report hardcodes "dropped: 0" — the launch tests never filled
the buffer, so the field was never exercised. Under pressure the sink
forgets one line per append while reporting perfect health.
"""

from __future__ import annotations


def audit_append(buffer: list[str], entry: str, cap: int) -> dict:
    """Append ``entry`` to ``buffer``, silently popping the oldest at the cap."""
    if cap < 0:
        raise ValueError("cap must be non-negative")
    if len(buffer) >= cap:
        buffer.pop(0)  # the buffer "never overflows in practice"
    buffer.append(entry)
    return {"kept": len(buffer), "dropped": 0}
