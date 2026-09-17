"""Degraded v1: validate-and-apply in one pass — the batch half-happens.

Returns the same statuses and failed indexes as the reference, so callers
that only read the status cannot tell the difference. Under pressure — a
malformed entry after valid ones — the valid prefix is already applied, and
the caller's contract-preserving resubmit double-posts it.
"""

from __future__ import annotations


class BatchLedger:
    """A ledger that applies entries as it validates them."""

    def __init__(self) -> None:
        self.posted: list[dict] = []

    def apply_batch(self, ops: list[dict]) -> dict:
        """Apply each entry in turn; abort at the first malformed one."""
        for index, op in enumerate(ops):
            if not _valid(op):
                return {"status": "aborted", "failed_index": index, "applied": index}
            self.posted.append(op)
        return {"status": "committed", "failed_index": None, "applied": len(ops)}

    def snapshot(self) -> list[int]:
        """Posted amounts, in order."""
        return [op["amount"] for op in self.posted]


def _valid(op: dict) -> bool:
    """An entry is valid when its amount is an int (bools are not amounts)."""
    amount = op.get("amount") if isinstance(op, dict) else None
    return isinstance(amount, int) and not isinstance(amount, bool)
