"""Degraded v1: admit everything, pin the gauge at capacity.

v1 treats capacity as advisory and clamps the gauge with min(load, capacity)
so the dashboard never shows a number it cannot explain. Below the cap the
two agree exactly. Under a spike the queue overcommits while reporting a
tidy 100%.
"""

from __future__ import annotations


class ShedQueue:
    """A queue that admits everything and clamps its gauge at capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self.capacity = capacity
        self._load = 0
        self._items: list[str] = []

    def offer(self, item: str, weight: int) -> bool:
        """Admit any non-negative weight; capacity is advisory."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self._load += weight
        self._items.append(item)
        return True

    def load(self) -> int:
        """The committed load, clamped to capacity for reporting."""
        return min(self._load, self.capacity)

    def drain(self) -> list[str]:
        """Return every admitted item in admission order and reset."""
        items, self._items, self._load = self._items, [], 0
        return items
