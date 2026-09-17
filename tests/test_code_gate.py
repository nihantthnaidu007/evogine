"""Code-gate tests: the four-stage pipeline end to end through the local
executor — acceptance of an equivalent improvement, the never-worse baseline
rule (seed learn_0011), oracle-leak rejection, equivalence/timeout/test/sandbox
failures, multi-file scope, and verdict payload contracts. Docker-only
isolation behavior is exercised in test_sandbox.py."""

import pytest

from evogine.gates import sandbox as sbx
from evogine.gates import verifier
from evogine.gates.code_gate import (
    CodeGate,
    GateInput,
    GateVerdict,
    GoldenCase,
    compare_to_baseline,
    find_oracle_leak,
)

LOCAL = sbx.LocalSubprocessSandbox()
GATE = CodeGate(executor=LOCAL)

# The baseline carries one redundant round-trip; the good candidate removes it
# (strictly fewer bytecode-instruction ops). All golden outputs are long enough
# reprs that the leak scanner would see them if a candidate embedded them.
BASELINE_SRC = "def add(a, b):\n    result = a + b\n    return result\n"

GOOD_SRC = "def add(a, b):\n    return a + b\n"

WORSE_SRC = "def add(a, b):\n    x = a\n    y = b\n    total = x + y\n    return total\n"

# Passes the suite (it special-cases both suite inputs) yet diverges on the
# held-out goldens — and embeds none of their literals.
DIVERGENT_SRC = (
    "def add(a, b):\n"
    "    if a == 2 and b == 3:\n"
    "        return 5\n"
    "    if a == 0 and b == 0:\n"
    "        return 0\n"
    "    return a + b + 1\n"
)

# Passes the suite AND would pass the golden replay (the outputs are right!) —
# the only defense is the host-side leak scan, which is exactly what this
# asserts.
LEAKY_SRC = (
    "def add(a, b):\n"
    "    if (a, b) == (111111, 222222):\n"
    "        return 333333\n"
    "    if (a, b) == (700000, 300000):\n"
    "        return 1000000\n"
    "    if (a, b) == (5000, 6000):\n"
    "        return 11000\n"
    "    return a + b\n"
)

TIMEOUT_SRC = (
    "def add(a, b):\n"
    "    if a > 100:\n"
    "        import time\n"
    "        time.sleep(30)\n"
    "    return a + b\n"
)

CRASH_SRC = "import os\n\n\ndef add(a, b):\n    os.abort()\n"

FAILING_SRC = "def add(a, b):\n    return a - b\n"

SUITE = (
    "from candidate import add\n"
    "\n"
    "\n"
    "def test_two_plus_three():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_zeros():\n"
    "    assert add(0, 0) == 0\n"
)

_GOLDEN = ((111111, 222222, 333333), (700000, 300000, 1000000), (5000, 6000, 11000))
GOLDEN_CASES = tuple(GoldenCase(args=(a, b), output=out) for a, b, out in _GOLDEN)


def _single(candidate_source: str, *, timeout: float = 3.0):
    gate_input = GateInput.single_function(
        baseline_source=BASELINE_SRC,
        candidate_source=candidate_source,
        entry_point="add",
        test_suite=SUITE,
        golden_cases=GOLDEN_CASES,
    )
    return GATE.evaluate(gate_input, timeout=timeout)


# ------------------------------------------------------------------ stage flow


def test_gate_accepts_equivalent_improvement():
    res = _single(GOOD_SRC)
    assert res.verdict is GateVerdict.PASS_TESTS
    assert res.accepted is True
    assert res.gate is None
    assert res.baseline_ops > 0 and res.candidate_ops > 0
    assert res.candidate_ops < res.baseline_ops
    assert res.returned_files == {"candidate.py": GOOD_SRC}
    payload = res.to_dict()
    assert payload["returned"] == "candidate"
    assert payload["disclosure"] == verifier.OPS_DISCLOSURE


def test_gate_rejects_tests_pass_but_worse():
    """Seed learn_0011: passing tests is NOT passing the gate. An equivalent
    candidate that does not beat the baseline is rejected and the BASELINE is
    what survives."""
    res = _single(WORSE_SRC)
    assert res.verdict is GateVerdict.PASS_TESTS  # behaviorally sound...
    assert res.accepted is False  # ...yet rejected by the baseline step
    assert res.gate == "baseline"
    assert "never-worse" in res.reason
    assert res.baseline_ops > 0 and res.candidate_ops > 0
    assert res.returned_files == {"candidate.py": BASELINE_SRC}
    assert res.to_dict()["returned"] == "baseline"


def test_gate_rejects_oracle_leak():
    """A candidate that embeds the withheld expected outputs is rejected
    fail-closed BEFORE its replay can bless it — no ops measured."""
    res = _single(LEAKY_SRC)
    assert res.verdict is GateVerdict.FAIL_EQUIVALENCE
    assert res.gate == "equivalence"
    assert "oracle leak" in res.reason
    assert res.accepted is False
    assert res.candidate_ops is None
    assert res.returned_files == {"candidate.py": BASELINE_SRC}


def test_gate_rejects_divergent_candidate():
    res = _single(DIVERGENT_SRC)
    assert res.verdict is GateVerdict.FAIL_EQUIVALENCE
    assert res.gate == "equivalence"
    assert res.accepted is False
    assert res.witness is not None
    assert "withheld" in res.witness["original"]  # oracle-free feedback
    assert res.witness["sequence"] == "golden"


def test_gate_rejects_timeout():
    res = _single(TIMEOUT_SRC, timeout=2.0)
    assert res.verdict is GateVerdict.FAIL_TIMEOUT
    assert res.accepted is False
    assert res.returned_files == {"candidate.py": BASELINE_SRC}


def test_gate_rejects_failing_tests():
    res = _single(FAILING_SRC)
    assert res.verdict is GateVerdict.FAIL_TESTS
    assert res.gate == "tests"
    assert res.reason.startswith("pytest:")
    assert res.accepted is False


def test_gate_rejects_crashing_candidate():
    res = _single(CRASH_SRC)
    assert res.verdict is GateVerdict.FAIL_SANDBOX
    assert res.accepted is False


# ------------------------------------------------------------------ multi-file


def _unit(impl):
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


BASELINE_UNIT = _unit("def scale(n):\n    return n * 2\n")
EQUIV_UNIT = _unit("def scale(n):\n    return n + n\n")
WORSE_UNIT = _unit("def scale(n):\n    return n * 3\n")

CORPUS = (
    "def build_corpus():\n"
    "    return [\n"
    "        ('first', (), [('value', [1])]),\n"
    "        ('basic', (), [('value', [4]), ('value', [0])]),\n"
    "    ]\n"
)


def _multi(candidate_files):
    return GATE.evaluate(
        GateInput.multi_file(
            baseline_files=BASELINE_UNIT,
            candidate_files=candidate_files,
            entry_class="Calc",
            corpus_source=CORPUS,
        ),
        timeout=5.0,
    )


def test_multi_file_gate_accepts_equivalent_unit():
    res = _multi(EQUIV_UNIT)
    assert res.verdict is GateVerdict.PASS_TESTS
    assert res.accepted is True
    # Multi-file objective score is equivalence itself (no cross-file ops).
    assert res.baseline_ops is None and res.candidate_ops is None
    assert res.returned_files == EQUIV_UNIT


def test_multi_file_gate_rejects_divergent_unit():
    res = _multi(WORSE_UNIT)
    assert res.verdict is GateVerdict.FAIL_EQUIVALENCE
    assert res.accepted is False
    assert res.returned_files == BASELINE_UNIT
    assert res.witness is not None


# --------------------------------------------------------------- input shapes


def test_gate_input_requires_golden_cases():
    with pytest.raises(ValueError):
        GateInput.single_function(
            baseline_source=BASELINE_SRC,
            candidate_source=GOOD_SRC,
            entry_point="add",
            test_suite=SUITE,
            golden_cases=(),
        )


def test_gate_input_requires_unit_init():
    with pytest.raises(ValueError):
        GateInput.multi_file(
            baseline_files={"calc.py": "class Calc:\n    pass\n"},
            candidate_files=EQUIV_UNIT,
            entry_class="Calc",
            corpus_source=CORPUS,
        )


# ------------------------------------------------------------------ pure logic


def test_find_oracle_leak_detects_repr_and_json_forms():
    cases = (GoldenCase(args=(), output="SECRETANSWER"),)
    assert find_oracle_leak("x = 'SECRETANSWER'", cases) is not None
    list_cases = (GoldenCase(args=(), output=[1, 2, 3, 4]),)
    assert find_oracle_leak("data = [1, 2, 3, 4]", list_cases) is not None


def test_find_oracle_leak_ignores_short_outputs_and_clean_source():
    # len < 4 outputs are too collision-prone to scan for (accepted residual).
    short = (GoldenCase(args=(), output=5),)
    assert find_oracle_leak("return 5", short) is None
    cases = (GoldenCase(args=(), output=1234567),)
    assert find_oracle_leak("def add(a, b):\n    return a + b\n", cases) is None


def test_compare_to_baseline_single_function_strictly_better():
    ok, _ = compare_to_baseline("single-function", 10, 12, equivalent=True)
    assert ok is True
    # Equal is not better: the incumbent is kept.
    ok, reason = compare_to_baseline("single-function", 12, 12, equivalent=True)
    assert ok is False and "never-worse" in reason
    ok, _ = compare_to_baseline("single-function", 12, 10, equivalent=True)
    assert ok is False
    ok, _ = compare_to_baseline("single-function", None, 10, equivalent=True)
    assert ok is False


def test_compare_to_baseline_unmeasurable_baseline_vacuous_bar():
    """An incumbent that crashed on the replay inputs cannot anchor the bar."""
    ok, reason = compare_to_baseline("single-function", 12, None, equivalent=True)
    assert ok is True and "unmeasurable" in reason
    # The vacuous bar still requires equivalence: without it, fail closed.
    ok, _ = compare_to_baseline("single-function", 12, None, equivalent=False)
    assert ok is False


def test_compare_to_baseline_multi_file_rides_on_equivalence():
    ok, _ = compare_to_baseline("multi-file", None, None, equivalent=True)
    assert ok is True
    ok, reason = compare_to_baseline("multi-file", None, None, equivalent=False)
    assert ok is False and "not equivalent" in reason
