"""Reference: a bounded queue with backpressure under load spikes.

An offer is admitted only when committed load plus its weight stays within
capacity; otherwise it is shed (False). load() is the honest committed load.
Deterministic: integer weights, no wall clock, no randomness.
"""

from __future__ import annotations


class ShedQueue:
    """A capacity-bounded queue that sheds offers instead of overcommitting."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._load = 0
        self._items: list[str] = []

    def offer(self, item: str, weight: int) -> bool:
        """Admit ``item`` at ``weight``; False (shed) when it would overflow."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        if self._load + weight > self.capacity:
            return False
        self._load += weight
        self._items.append(item)
        return True

    def load(self) -> int:
        """The committed load."""
        return self._load

    def drain(self) -> list[str]:
        """Return every committed item in admission order and reset."""
        items, self._items, self._load = self._items, [], 0
        return items
