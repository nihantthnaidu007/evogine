"""Visible happy-path tests: the first few ids of an epoch.

Everything here stays at the bottom of the counter space — the exhaustion
boundary is 1,048,576 ids away and the visible suite never approaches it.
The degraded version passes this suite.
"""

from candidate import IdSource, pack_id


def test_first_ids_of_an_epoch():
    source = IdSource(7)
    assert source.next_id() == pack_id(7, 0)
    assert source.next_id() == pack_id(7, 1)


def test_counter_tracks_consumption():
    source = IdSource(7)
    source.next_id()
    source.next_id()
    assert source.counter() == 1


def test_skip_advances_without_emitting():
    source = IdSource(7)
    source.skip(10)
    assert source.next_id() == pack_id(7, 10)


def test_negative_epoch_raises():
    try:
        IdSource(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative epoch")
