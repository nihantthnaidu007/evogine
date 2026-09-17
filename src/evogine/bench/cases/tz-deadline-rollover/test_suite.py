"""Visible happy-path tests: whole-hour locales, away from midnight.

Every timestamp here is at least two hours from a local midnight crossing
under its whole-hour offset, and every offset is a whole number of hours —
the half-hour pressure case is simply absent. The degraded version passes
this suite.
"""

from candidate import local_date


def test_utc_is_identity():
    assert local_date("2026-06-01T09:15:00", 0) == "2026-06-01"


def test_positive_whole_hour_offset():
    assert local_date("2026-06-01T09:15:00", 60) == "2026-06-01"


def test_negative_whole_hour_offset():
    assert local_date("2026-06-01T09:15:00", -120) == "2026-06-01"


def test_large_offset_crosses_date():
    assert local_date("2026-06-01T20:00:00", 600) == "2026-06-02"


def test_malformed_timestamp_raises():
    try:
        local_date("2026-06-01 09:15:00", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for space-separated timestamp")
