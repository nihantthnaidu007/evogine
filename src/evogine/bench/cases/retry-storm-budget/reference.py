"""Reference: retry allowance over a sliding window.

``record_attempt`` prunes attempts that fell out of the window before
deciding, so a storm exhausts the budget only for as long as the window
remains full. Deterministic: logical millisecond timestamps, no wall clock,
no randomness.
"""

from __future__ import annotations


class RetryBudget:
    """At most ``max_attempts`` attempts inside any ``window_ms`` window."""

    def __init__(self, max_attempts: int, window_ms: int) -> None:
        if max_attempts < 1 or window_ms < 1:
            raise ValueError("max_attempts and window_ms must be positive")
        self.max_attempts = max_attempts
        self.window_ms = window_ms
        self._attempts: list[int] = []

    def record_attempt(self, at_ms: int) -> bool:
        """Record an attempt at logical time ``at_ms``; True when allowed."""
        self._attempts = [
            stamp for stamp in self._attempts if at_ms - stamp < self.window_ms
        ]
        if len(self._attempts) >= self.max_attempts:
            return False
        self._attempts.append(at_ms)
        return True

    def window_load(self) -> int:
        """Attempts still inside the window."""
        return len(self._attempts)
