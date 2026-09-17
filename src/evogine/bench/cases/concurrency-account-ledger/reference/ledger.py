"""Reference ledger: folds writes by sequence order — the durable contract."""


def replay_ops(ops: list[dict]) -> int:
    """Fold modeled concurrent writes in sequence order; last write wins."""
    balance = 0
    for op in sorted(ops, key=lambda entry: entry["seq"]):
        balance = op["set_balance"]
    return balance
