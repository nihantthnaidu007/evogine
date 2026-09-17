"""Reference: month arithmetic with end-of-month clamping.

Adds ``months`` whole calendar months to an ISO date and clamps the day to
the target month's last day (Jan 31 + 1 month is Feb 28, or Feb 29 in a leap
year). Deterministic: stdlib date math only, no wall clock, no randomness.
"""

from __future__ import annotations

import calendar
from datetime import date


def billing_day(due: str, months: int) -> str:
    """Return the due date advanced by ``months`` calendar months.

    The day of month is clamped to the target month's last day. Negative
    ``months`` moves into the past. Malformed dates raise ValueError.
    """
    year, month, day = _parse(due)
    total = year * 12 + (month - 1) + months
    new_year, index = divmod(total, 12)
    new_month = index + 1
    last = calendar.monthrange(new_year, new_month)[1]
    return date(new_year, new_month, min(day, last)).isoformat()


def _parse(due: str) -> tuple[int, int, int]:
    parts = due.split("-")
    if len(parts) != 3:
        raise ValueError(f"due must be an ISO date: {due!r}")
    year, month, day = (int(part) for part in parts)
    date(year, month, day)  # validates; raises ValueError out of range
    return year, month, day
