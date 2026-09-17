"""Reference: UTC timestamp to local calendar date under a fixed offset.

The offset is in whole minutes (may be negative, may include a 30-minute
component). The local date is the date part after shifting. Deterministic:
stdlib datetime only, no wall clock, no randomness.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def local_date(utc_iso: str, offset_minutes: int) -> str:
    """Return the local calendar date (YYYY-MM-DD) for a UTC timestamp.

    ``utc_iso`` is a naive UTC timestamp like ``2026-06-01T18:50:00``;
    ``offset_minutes`` is the locale's fixed offset from UTC in minutes.
    Malformed timestamps raise ValueError.
    """
    moment = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%S")
    shifted = moment + timedelta(minutes=offset_minutes)
    return shifted.date().isoformat()
