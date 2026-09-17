"""Degraded v1: whole-hour offsets only, via floor division.

The renderer predates half-hour locales: it converts the minute offset to
hours with floor division and shifts by whole hours. Launch tests used
whole-hour zones away from midnight, so it passed. Under pressure — a
half-hour zone near a local-midnight crossing — it lands on the wrong
calendar date, silently.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def local_date(utc_iso: str, offset_minutes: int) -> str:
    """Return the local calendar date, shifting by whole hours only.

    v1 assumption: offsets are whole hours; minutes beyond that are rounding
    noise. Malformed timestamps raise ValueError.
    """
    moment = datetime.strptime(utc_iso, "%Y-%m-%dT%H:%M:%S")
    shifted = moment + timedelta(hours=offset_minutes // 60)
    return shifted.date().isoformat()
