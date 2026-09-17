"""Visible happy-path tests: writes recorded in sequence order.

Every test here records writes in exactly sequence order, where arrival
order and sequence order coincide — the reorder case is simply absent. The
degraded version passes this suite.
"""

from candidate import Account, replay_ops


def test_writes_in_sequence_order():
    account = Account()
    account.record(1, 100)
    account.record(2, 250)
    assert account.reconcile() == 250


def test_reconcile_is_repeatable():
    account = Account()
    account.record(1, 100)
    account.record(2, 250)
    assert account.reconcile() == account.reconcile()


def test_empty_account_reconciles_to_zero():
    assert Account().reconcile() == 0


def test_replay_ops_direct():
    assert replay_ops([{"seq": 1, "set_balance": 5}]) == 5


def test_pending_counts_records():
    account = Account()
    account.record(1, 100)
    assert account.pending() == 1
