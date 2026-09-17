"""Degraded account replica: reconciliation replays arrival order."""

from __future__ import annotations

from .ledger import replay_ops


class Account:
    """A replica that folds writes in the order record() was called."""

    def __init__(self) -> None:
        self._ops: list[dict] = []

    def record(self, seq: int, set_balance: int) -> None:
        """Accept one modeled write; seq is stored but never used to order."""
        self._ops.append({"seq": seq, "set_balance": set_balance})

    def reconcile(self) -> int:
        """Fold pending writes in arrival order; repeatable, possibly wrong."""
        return replay_ops(list(self._ops))

    def pending(self) -> int:
        """How many writes await reconciliation."""
        return len(self._ops)
