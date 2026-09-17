"""Visible happy-path tests: clean batches, and an abort on entry zero.

An abort on the first entry applies nothing under either implementation, and
clean batches commit identically — the mid-batch failure is simply absent.
The degraded version passes this suite.
"""

from candidate import BatchLedger


def test_clean_batch_commits():
    ledger = BatchLedger()
    result = ledger.apply_batch([{"amount": 10}, {"amount": -4}])
    assert result["status"] == "committed"
    assert ledger.snapshot() == [10, -4]


def test_abort_on_first_entry_applies_nothing():
    ledger = BatchLedger()
    result = ledger.apply_batch([{"amount": "x"}, {"amount": 10}])
    assert result["status"] == "aborted"
    assert result["failed_index"] == 0
    assert ledger.snapshot() == []


def test_empty_batch_commits_nothing():
    ledger = BatchLedger()
    result = ledger.apply_batch([])
    assert result["status"] == "committed"
    assert ledger.snapshot() == []


def test_non_int_amount_is_rejected():
    ledger = BatchLedger()
    result = ledger.apply_batch([{"amount": True}])
    assert result["status"] == "aborted"
    assert result["failed_index"] == 0
    assert ledger.snapshot() == []
