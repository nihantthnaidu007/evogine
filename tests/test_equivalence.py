"""Equivalence tests: strict typed canonicalization, host-side outcome
comparison, oracle-free witnesses, and the fail-closed replay statuses —
exercised through the local executor (the same replay driver runs inside
Docker in production)."""

import pytest

from evogine.gates import sandbox as sbx
from evogine.gates.equivalence import (
    ReplayCrash,
    ReplayError,
    ReplayTimeout,
    canon,
    canon_outcome,
    check_function_equivalence,
    check_unit_equivalence,
)

LOCAL = sbx.LocalSubprocessSandbox()

# ============================================================ canon


def test_canon_typed_equality():
    assert canon(3) == canon(3)
    assert canon("a") == canon("a")
    assert canon(3.0) == canon(3.0)
    assert canon([1, 2]) == canon([1, 2])
    assert canon({"k": [1, {"x": None}]}) == canon({"k": [1, {"x": None}]})


def test_canon_bool_is_not_int():
    """True == 1 in Python; canon must not let that pass for equivalence."""
    assert canon(True) != canon(1)


def test_canon_sequences_stay_ordered():
    # Containers canonicalize structurally (JSON heritage): tuples and lists
    # compare equal as sequences, but ORDER is observable and preserved.
    assert canon((1, 2)) == canon([1, 2])
    assert canon([1, 2]) != canon([2, 1])


def test_canon_nan_is_nan_identity():
    assert canon(float("nan")) == canon(float("nan"))
    assert canon(float("inf")) == canon(float("inf"))


def test_canon_falls_back_to_type_and_repr():
    assert canon(complex(1, 2)) == canon(complex(1, 2))
    assert canon(complex(1, 2)) != canon(complex(1, 3))


# ============================================================ canon_outcome


def test_canon_outcome_return_and_raise():
    assert canon_outcome(("ok", 5)) == canon_outcome(("ok", 5))
    assert canon_outcome(("ok", 5)) != canon_outcome(("ok", 6))
    assert canon_outcome(("ok", 5)) != canon_outcome(("raise", "ValueError", "boom"))
    assert canon_outcome(("raise", "ValueError", "boom")) != canon_outcome(
        ("raise", "KeyError", "boom")
    )


def test_canon_outcome_message_is_part_of_the_observable():
    assert canon_outcome(("raise", "ValueError", "a")) != canon_outcome(
        ("raise", "ValueError", "b")
    )


def test_canon_outcome_raises_shape_is_passthrough():
    """Non-ok outcomes are already canonical; they pass through unchanged."""
    outcome = ("raise", "ValueError", "boom")
    assert canon_outcome(outcome) == outcome


# ==================================================== function equivalence


def _fn_cases():
    return [
        {"args": [2, 3], "kwargs": {}, "output": 5},
        {"args": [0, 0], "kwargs": {}, "output": 0},
        {"args": [-1, 1], "kwargs": {}, "output": 0},
    ]


def test_function_equivalent_refactor_accepted():
    candidate = "def add(a, b):\n    return a + b\n"
    ok, witness, stats = check_function_equivalence(
        candidate, "add", _fn_cases(), LOCAL, timeout=5.0
    )
    assert ok is True
    assert witness is None
    assert stats["cases"] == 3


def test_function_divergence_rejected_with_oracle_free_witness():
    candidate = "def add(a, b):\n    return a + b + 1\n"
    ok, witness, stats = check_function_equivalence(
        candidate, "add", _fn_cases(), LOCAL, timeout=5.0
    )
    assert ok is False
    assert stats["cases"] == 3
    assert witness is not None
    assert witness["sequence"] == "golden"
    assert witness["call_index"] == 0
    assert "output" not in witness  # no oracle key at all
    assert "withheld" in witness["original"]  # oracle side never leaves the host
    assert witness["candidate"] is not None


def test_function_raising_candidate_is_replay_error():
    """A candidate whose call raises escapes the driver: no verifiable frame,
    surfaced fail-closed as ReplayError (never a comparison against a lie)."""
    candidate = "def add(a, b):\n    raise ValueError('no')\n"
    with pytest.raises(ReplayError):
        check_function_equivalence(candidate, "add", _fn_cases(), LOCAL, timeout=5.0)


def test_function_aborting_candidate_is_replay_crash():
    candidate = "import os\n\ndef add(a, b):\n    os.abort()\n"
    with pytest.raises(ReplayCrash):
        check_function_equivalence(candidate, "add", _fn_cases(), LOCAL, timeout=5.0)


def test_function_missing_entry_is_replay_error():
    with pytest.raises(ReplayError):
        check_function_equivalence(
            "def other():\n    return 1\n", "add", _fn_cases(), LOCAL, timeout=5.0
        )


def test_function_timeout_is_replay_timeout():
    candidate = "def add(a, b):\n    while True:\n        pass\n"
    with pytest.raises(ReplayTimeout):
        check_function_equivalence(candidate, "add", _fn_cases(), LOCAL, timeout=1.0)


# ====================================================== unit equivalence


def _unit_files(impl):
    """A tiny two-module unit: the public Calc class delegates to helpers."""
    return {
        "__init__.py": "from calc import Calc\n",
        "calc.py": (
            "from helpers import scale\n\n\n"
            "class Calc:\n"
            "    def value(self, n):\n"
            "        return scale(n)\n"
        ),
        "helpers.py": impl,
    }


BASELINE_UNIT = _unit_files("def scale(n):\n    return n * 2\n")

# Behaviorally identical, different internals (n + n instead of n * 2).
EQUIV_UNIT = _unit_files("def scale(n):\n    return n + n\n")

# A regression the corpus must catch.
WORSE_UNIT = _unit_files("def scale(n):\n    return n * 3\n")

# Corpus contract: build_corpus() returns (name, ctor_args, calls) triples;
# each call is (method, args) or (method, args, kwargs).
CORPUS = (
    "def build_corpus():\n"
    "    return [\n"
    "        ('first', (), [('value', [1])]),\n"
    "        ('basic', (), [('value', [4]), ('value', [0])]),\n"
    "    ]\n"
)


def test_unit_equivalent_accepted():
    ok, witness, stats = check_unit_equivalence(
        BASELINE_UNIT, EQUIV_UNIT, "Calc", CORPUS, LOCAL, timeout=5.0
    )
    assert ok is True
    assert witness is None
    assert stats["sequences"] == 2
    # __init__ plus value calls across both sequences (1 + 2) are compared.
    assert stats["calls_compared"] == 5


def test_unit_divergence_rejected_with_oracle_free_witness():
    ok, witness, stats = check_unit_equivalence(
        BASELINE_UNIT, WORSE_UNIT, "Calc", CORPUS, LOCAL, timeout=5.0
    )
    assert ok is False
    assert witness is not None
    assert witness["sequence"] == "first"  # __init__ compared clean; value diverges
    assert "withheld" in witness["original"]
    assert stats["calls_compared"] == 2


def test_unit_candidate_crash_is_replay_crash():
    # The abort happens at unit import inside the container -> process dies.
    bad = {
        "__init__.py": "import os\nos.abort()\n",
        "calc.py": "class Calc:\n    pass\n",
    }
    with pytest.raises(ReplayCrash):
        check_unit_equivalence(BASELINE_UNIT, bad, "Calc", CORPUS, LOCAL, timeout=5.0)


def test_unit_missing_entry_class_is_replay_error():
    with pytest.raises(ReplayError):
        check_unit_equivalence(BASELINE_UNIT, EQUIV_UNIT, "Absent", CORPUS, LOCAL, timeout=5.0)


def test_unit_malformed_corpus_is_value_error():
    with pytest.raises(ValueError):
        check_unit_equivalence(
            BASELINE_UNIT,
            EQUIV_UNIT,
            "Calc",
            "def build_corpus():\n    return ['not a triple']\n",
            LOCAL,
            timeout=5.0,
        )
