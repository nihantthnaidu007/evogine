"""Visible happy-path tests: the launch coverage the v1 team shipped.

Every date here is day 15 or shorter — the month-end pressure case is simply
absent, which is the point. The degraded version passes this suite.
"""

from candidate import billing_day


def test_add_one_month_mid_month():
    assert billing_day("2026-01-15", 1) == "2026-02-15"


def test_zero_months_is_identity():
    assert billing_day("2026-03-15", 0) == "2026-03-15"


def test_cross_year_addition():
    assert billing_day("2025-11-15", 3) == "2026-02-15"


def test_negative_months_mid_month():
    assert billing_day("2026-03-15", -1) == "2026-02-15"


def test_malformed_date_raises():
    try:
        billing_day("2026-13-01", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for month 13")
