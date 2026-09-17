"""Reference: epoch-tagged ids with a saturating bounded counter.

The counter space is 20 bits. next_id returns None once the space is
exhausted; it never wraps, so emitted ids stay unique per epoch.
Deterministic: integer arithmetic only, no wall clock, no randomness.
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
    """Ids for one epoch; saturates instead of wrapping."""

    def __init__(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
        self._counter = -1

    def next_id(self) -> int | None:
        """The next id for this epoch, or None when the space is exhausted."""
        if self._counter >= MAX_COUNTER:
            return None
        self._counter += 1
        return pack_id(self.epoch, self._counter)

    def skip(self, count: int) -> None:
        """Consume ``count`` ids without emitting; saturates at the cap."""
        self._counter = min(self._counter + count, MAX_COUNTER)

    def counter(self) -> int:
        """The last consumed counter position (-1 before the first id)."""
        return self._counter
