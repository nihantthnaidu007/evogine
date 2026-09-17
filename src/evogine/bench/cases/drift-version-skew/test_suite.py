"""Visible happy-path tests: single-side payloads through the facade.

Every test here exercises one self-consistent side — encode/decode through
the same package — so the version skew at the cross-version seam is simply
absent. Both the reference and the v1 pair pass this suite.
"""

from candidate import Pipeline
from candidate.client import encode_batch
from candidate.server import decode_batch


def test_facade_round_trips():
    assert Pipeline().send([1, 2, 3]) == [1, 2, 3]


def test_encode_carries_items():
    assert encode_batch([5])["items"] == [5]


def test_direct_encode_decode_round_trip():
    assert decode_batch(encode_batch([9, 10])) == [9, 10]


def test_empty_batch_round_trips():
    assert Pipeline().send([]) == []
