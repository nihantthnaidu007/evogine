"""Degraded v1: the counter wraps modulo the space.

v1 read "bounded counter" as "modulo arithmetic". For the first full sweep
of the space the two agree exactly; past the boundary the counter wraps to
the bottom and the source reissues ids it already emitted — silently.
"""

from __future__ import annotations

COUNTER_BITS = 20
COUNTER_SPACE = 1 << COUNTER_BITS
MAX_COUNTER = COUNTER_SPACE - 1


def pack_id(epoch: int, counter: int) -> int:
    """Pack an epoch tag and in-range counter into one id."""
    if not 0 <= counter <= MAX_COUNTER:
        raise ValueError("counter out of range")
    return (epoch << COUNTER_BITS) | counter


class IdSource:
    """Ids for one epoch; wraps modulo the counter space."""

    def __init__(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self._counter = -1

    def next_id(self) -> int | None:
        """The next id for this epoch; wraps instead of reporting exhaustion."""
        self._counter = (self._counter + 1) % COUNTER_SPACE
        return pack_id(self.epoch, self._counter)

    def skip(self, count: int) -> None:
        """Consume ``count`` ids without emitting; wraps modulo the space."""
        self._counter = (self._counter + count) % COUNTER_SPACE

    def counter(self) -> int:
        """The last consumed counter position (-1 before the first id)."""
        return self._counter
