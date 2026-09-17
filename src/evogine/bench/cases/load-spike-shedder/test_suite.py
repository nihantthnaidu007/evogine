"""Visible happy-path tests: load under the cap.

Every offer here fits, so shedding never triggers and the gauge clamp never
binds — the spike case is simply absent. The degraded version passes this
suite.
"""

from candidate import ShedQueue


def test_offers_under_capacity_are_admitted():
    queue = ShedQueue(10)
    assert queue.offer("a", 4) is True
    assert queue.offer("b", 4) is True
    assert queue.load() == 8


def test_drain_returns_items_in_order():
    queue = ShedQueue(10)
    queue.offer("a", 2)
    queue.offer("b", 3)
    assert queue.drain() == ["a", "b"]
    assert queue.load() == 0


def test_zero_weight_offer_is_admitted():
    queue = ShedQueue(8)
    assert queue.offer("heartbeat", 0) is True


def test_negative_capacity_raises():
    try:
        ShedQueue(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative capacity")
