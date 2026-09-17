"""Reference account replica: reconciliation replays by sequence order."""

from __future__ import annotations

from .ledger import replay_ops


class Account:
    """A replica that keeps pending writes and reconciles on demand."""

    def __init__(self) -> None:
        self._ops: list[dict] = []

    def record(self, seq: int, set_balance: int) -> None:
        """Accept one modeled write; arrival order is irrelevant here."""
        self._ops.append({"seq": seq, "set_balance": set_balance})

    def reconcile(self) -> int:
        """Fold pending writes by sequence order; repeatable and idempotent."""
        return replay_ops(list(self._ops))

    def pending(self) -> int:
        """How many writes await reconciliation."""
        return len(self._ops)
