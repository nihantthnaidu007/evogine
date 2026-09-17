"""Visible happy-path tests: ASCII names, the launch corpus.

Every part here is plain ASCII, where casefold and NFKC+casefold agree —
the compatibility-form pressure case is simply absent. The degraded version
passes this suite.
"""

from candidate import display_key


def test_two_ascii_parts_join():
    assert display_key(["Ada", "Lovelace"]) == "ada·lovelace"


def test_case_insensitive():
    assert display_key(["ADA", "lovelace"]) == "ada·lovelace"


def test_single_part():
    assert display_key(["Grace"]) == "grace"


def test_empty_parts_join_to_separator():
    assert display_key([]) == ""
