"""Visible happy-path tests: appends below the capacity cap.

Every call here stays under the cap, where the ring behavior is
indistinguishable from refusal — the boundary case is simply absent. The
degraded version passes this suite.
"""

from candidate import audit_append


def test_append_below_cap():
    buffer = ["a"]
    report = audit_append(buffer, "b", 3)
    assert report["kept"] == 2
    assert buffer == ["a", "b"]


def test_append_to_empty():
    buffer = []
    report = audit_append(buffer, "a", 3)
    assert report["kept"] == 1


def test_append_up_to_cap_minus_one():
    buffer = ["a", "b"]
    report = audit_append(buffer, "c", 3)
    assert report["kept"] == 3


def test_negative_cap_raises():
    try:
        audit_append([], "a", -1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative cap")
