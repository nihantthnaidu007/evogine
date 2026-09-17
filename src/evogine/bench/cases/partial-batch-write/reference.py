"""Reference: atomic batch posting — all entries validate, or none apply."""

from __future__ import annotations


class BatchLedger:
    """A ledger whose ``apply_batch`` is all-or-nothing."""

    def __init__(self) -> None:
        self.posted: list[dict] = []

    def apply_batch(self, ops: list[dict]) -> dict:
        """Validate every entry, then apply; abort with nothing applied."""

        for index, op in enumerate(ops):
            if not _valid(op):
                return {"status": "aborted", "failed_index": index, "applied": 0}
        self.posted.extend(ops)
        return {"status": "committed", "failed_index": None, "applied": len(ops)}

    def snapshot(self) -> list[int]:
        """Posted amounts, in order."""
        return [op["amount"] for op in self.posted]


def _valid(op: dict) -> bool:
    """An entry is valid when its amount is an int (bools are not amounts)."""
    amount = op.get("amount") if isinstance(op, dict) else None
    return isinstance(amount, int) and not isinstance(amount, bool)
