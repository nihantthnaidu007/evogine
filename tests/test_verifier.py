"""Verifier tests: deterministic bytecode-instruction counting through the
executor boundary (the local subprocess runs the same driver as Docker, just
unsandboxed), plus fail-closed error paths and the disclosure constant.
Docker-only isolation properties are exercised in test_sandbox.py."""

import pytest

from evogine.gates import sandbox as sbx
from evogine.gates import verifier
from evogine.gates.verifier import VerifierError, VerifierTimeout, measure_ops

LOCAL = sbx.LocalSubprocessSandbox()

# Deterministic loop-heavy baseline and a leaner equivalent candidate.
BASELINE_SRC = """\
def total(n):
    acc = 0
    i = 0
    while i < n:
        acc = acc + i
        i = i + 1
    return acc
"""

CANDIDATE_SRC = """\
def total(n):
    acc = 0
    for i in range(n):
        acc = acc + i
    return acc
"""

CASES = [{"args": [10], "kwargs": {}}, {"args": [25], "kwargs": {}}]


# ---------------------------------------------------------------- happy path


def test_measure_ops_positive_and_stable():
    ops = measure_ops(BASELINE_SRC, "total", CASES, LOCAL, timeout=5.0)
    assert isinstance(ops, int) and ops > 0


def test_measure_ops_deterministic_across_runs():
    a = measure_ops(BASELINE_SRC, "total", CASES, LOCAL, timeout=5.0)
    b = measure_ops(BASELINE_SRC, "total", CASES, LOCAL, timeout=5.0)
    assert a == b


def test_measure_ops_distinguishes_improvement():
    lean = measure_ops(CANDIDATE_SRC, "total", CASES, LOCAL, timeout=5.0)
    fat = measure_ops(BASELINE_SRC, "total", CASES, LOCAL, timeout=5.0)
    assert lean < fat


def test_measure_ops_ignores_other_module_code():
    """Extra module-level work (executed before the hook arms) and functions
    that are never called must not enter the count — only the entry point's
    own execution is traced."""
    noisy = "CONST = [i * i for i in range(100)]\n\n" + BASELINE_SRC
    noisy += "\ndef helper(x):\n    return x * 999\n"
    base = measure_ops(BASELINE_SRC, "total", CASES, LOCAL, timeout=5.0)
    noisy_ops = measure_ops(noisy, "total", CASES, LOCAL, timeout=5.0)
    assert noisy_ops == base


# ------------------------------------------------------------- fail-closed


def test_missing_function_is_error():
    with pytest.raises(VerifierError):
        measure_ops(BASELINE_SRC, "absent", CASES, LOCAL, timeout=5.0)


def test_syntax_error_is_error():
    with pytest.raises(VerifierError):
        measure_ops("def f(:\n    return 1", "f", CASES, LOCAL, timeout=5.0)


def test_infinite_loop_times_out():
    with pytest.raises(VerifierTimeout):
        measure_ops(
            "def f(n):\n    while True:\n        n = n + 1\n    return n",
            "f",
            CASES,
            LOCAL,
            timeout=1.0,
        )


def test_raising_call_is_error():
    """A measurement call that raises is an unverifiable run, not a score."""
    with pytest.raises(VerifierError):
        measure_ops(
            "def f(n):\n    raise ValueError('no')\n",
            "f",
            CASES,
            LOCAL,
            timeout=5.0,
        )


def test_disarmed_monitor_is_rejected():
    """A candidate that disarms the monitoring hook mid-run (clearing the
    local instruction events on its own code) yields an error frame
    (fail-closed), never a partial or flattering count."""
    src = (
        "import sys\n"
        "def f(n):\n"
        "    sys.monitoring.set_local_events(0, f.__code__, 0)\n"
        "    return n + 1\n"
    )
    with pytest.raises(VerifierError):
        measure_ops(src, "f", CASES, LOCAL, timeout=5.0)


def test_driver_tool_id_takeover_is_rejected():
    """A candidate that claims the driver's tool id at import time breaks the
    driver's registration; the run then emits no verifiable frame."""
    src = "import sys\nsys.monitoring.use_tool_id(0, 'imposter')\n\ndef f(n):\n    return n + 1\n"
    with pytest.raises(VerifierError):
        measure_ops(src, "f", CASES, LOCAL, timeout=5.0)


# -------------------------------------------------------------- disclosure


def test_disclosure_is_present_and_explicit():
    disclosure = verifier.OPS_DISCLOSURE
    assert "bytecode" in disclosure.lower()
    assert "builtin" in disclosure.lower()
