"""Degraded v1: a window that never slides.

The budget counts every attempt ever recorded and never prunes. Within the
first window it matches the reference exactly; after a storm it denies
everything forever, because the count can only grow.
"""

from __future__ import annotations


class RetryBudget:
    """At most ``max_attempts`` attempts, ever (v1 window in name only)."""

    def __init__(self, max_attempts: int, window_ms: int) -> None:
        if max_attempts < 1 or window_ms < 1:
            raise ValueError("max_attempts and window_ms must be positive")
        self.max_attempts = max_attempts
        self.window_ms = window_ms
        self._attempts: list[int] = []

    def record_attempt(self, at_ms: int) -> bool:
        """Record an attempt at logical time ``at_ms``; True when allowed."""
        if len(self._attempts) >= self.max_attempts:
            return False
        self._attempts.append(at_ms)
        return True

    def window_load(self) -> int:
        """Attempts the budget currently counts."""
        return len(self._attempts)
