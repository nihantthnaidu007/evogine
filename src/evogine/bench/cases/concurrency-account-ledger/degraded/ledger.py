"""Degraded ledger: folds writes in arrival order — v1's quiet assumption."""


def replay_ops(ops: list[dict]) -> int:
    """Fold writes in the order they were recorded; last arrival wins."""
    balance = 0
    for op in ops:
        balance = op["set_balance"]
    return balance
