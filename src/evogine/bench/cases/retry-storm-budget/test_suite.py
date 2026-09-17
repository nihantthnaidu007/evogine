"""Visible happy-path tests: one storm, one denial.

The suite storms the budget and asserts the denial — the behavior launch
review asked for. It never advances the logical clock past the window, so
recovery is simply untested. The degraded version passes this suite.
"""

from candidate import RetryBudget


def test_attempts_within_budget_are_allowed():
    budget = RetryBudget(3, 1000)
    assert budget.record_attempt(0) is True
    assert budget.record_attempt(100) is True
    assert budget.record_attempt(200) is True


def test_fourth_attempt_in_window_is_denied():
    budget = RetryBudget(3, 1000)
    for stamp in (0, 100, 200):
        budget.record_attempt(stamp)
    assert budget.record_attempt(300) is False


def test_window_load_counts_recorded_attempts():
    budget = RetryBudget(3, 1000)
    budget.record_attempt(0)
    budget.record_attempt(100)
    assert budget.window_load() == 2


def test_invalid_construction_raises():
    try:
        RetryBudget(0, 1000)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for max_attempts=0")
