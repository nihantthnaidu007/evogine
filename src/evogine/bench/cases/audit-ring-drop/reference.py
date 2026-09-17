"""Reference: an audit sink that refuses at capacity instead of forgetting.

Appending to a full buffer raises OverflowError; the caller must rotate the
sink or raise the cap. The report counts what THIS call dropped (always 0
when the append succeeded). Deterministic: no wall clock, no randomness.
"""

from __future__ import annotations


def audit_append(buffer: list[str], entry: str, cap: int) -> dict:
    """Append ``entry`` to ``buffer`` under the capacity cap.

    Raises OverflowError when the buffer already holds ``cap`` entries —
    audit lines are never silently discarded.
    """
    if cap < 0:
        raise ValueError("cap must be non-negative")
    if len(buffer) >= cap:
        raise OverflowError("audit buffer at capacity; rotate or extend the cap")
    buffer.append(entry)
    return {"kept": len(buffer), "dropped": 0}
