"""Degraded v1: preserves the day of month without knowing month lengths.

Passes every launch test (they only used mid-month days). Under pressure —
month-end signup dates — the date constructor hits an out-of-range day and
raises, aborting the renewal batch instead of billing on the last day of the
month.
"""

from __future__ import annotations

from datetime import date


def billing_day(due: str, months: int) -> str:
    """Advance an ISO date by ``months`` months, keeping the same day.

    v1 assumption: every month has the signup day. Malformed dates raise
    ValueError.
    """
    parts = due.split("-")
    if len(parts) != 3:
        raise ValueError(f"due must be an ISO date: {due!r}")
    year, month, day = (int(part) for part in parts)
    date(year, month, day)
    total = year * 12 + (month - 1) + months
    new_year, index = divmod(total, 12)
    new_month = index + 1
    return date(new_year, new_month, day).isoformat()
