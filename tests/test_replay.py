"""Replay tests: the warm arm's test-suite contract (replay.py).

An applier may REPLACE the recorded visible test suite when it builds its
patch (a learning can carry a regression suite of its own); the replay must
actually run the swapped suite and reflect its verdict in the evidence.
Pinned here:

  - a patch carrying ``test_suite`` swaps the suite the warm arm runs, and
    the swapped suite's verdict is what the evidence records;
  - a patch with ``test_suite=None`` keeps the recorded suite.

Fixtures run the real code gate through the local executor (fixture code
only, the test_code_gate.py pattern); the Docker boundary is exercised by
the docker-gates CI job, not here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.gates import sandbox as sbx
from evogine.gates.code_gate import CodeGate, GateInput, GateVerdict, GoldenCase
from evogine.gates.replay import AppliedPatch, RecordedRun, replay_one

LOCAL = sbx.LocalSubprocessSandbox()
GATE = CodeGate(executor=LOCAL)

BASELINE_SRC = "def add(a, b):\n    result = a + b\n    return result\n"
GOOD_SRC = "def add(a, b):\n    return a + b\n"

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

# A DIFFERENT visible suite: fails every honest add (the recorded suite passes
# it). A warm arm reporting fail_tests for the GOOD candidate proves THIS
# suite ran, not the recorded one.
FAILING_SUITE = (
    "from candidate import add\n"
    "\n"
    "\n"
    "def test_deliberately_wrong():\n"
    "    assert add(2, 3) == 6\n"
)

_GOLDEN = ((111111, 222222, 333333), (700000, 300000, 1000000), (5000, 6000, 11000))
GOLDEN_CASES = tuple(GoldenCase(args=(a, b), output=out) for a, b, out in _GOLDEN)


def make_learning() -> LearningRecord:
    return LearningRecord(
        id="learn_replay_suite_0001",
        type="failure-pattern",
        title="Swap the visible suite",
        body=RecordBody(
            signal="n/a",
            failure_mode="n/a",
            detection="counterfactual replay",
            fix_strategy="n/a",
        ),
        status="ephemeral",
        confidence=0.9,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="rollout",
            source="run_1",
            note="test learning",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 9, 17, tzinfo=UTC),
        promoted_at=None,
    )


def recorded_good_run(run_id: str) -> RecordedRun:
    """Record one accepted run the honest way: the real gate's own payload."""
    gate_input = GateInput.single_function(
        baseline_source=BASELINE_SRC,
        candidate_source=GOOD_SRC,
        entry_point="add",
        test_suite=SUITE,
        golden_cases=GOLDEN_CASES,
    )
    cold = GATE.evaluate(gate_input, timeout=15.0)
    assert cold.accepted is True  # fixture sanity: the good candidate lands
    return RecordedRun(
        run_id=run_id,
        case_id="case_add",
        gate_input=gate_input,
        cold=cold,
        recorded_at=datetime.now(UTC),
        note="fixture run",
    )


def test_applier_suite_swap_is_applied_and_reflected_in_evidence():
    """A patch carrying test_suite REPLACES the recorded suite for the warm
    arm, and the swapped suite's verdict is what the evidence records: the
    GOOD candidate under the swapped suite fails tests (had the recorded suite
    run, the warm arm would have accepted)."""
    run = recorded_good_run("run_suite_swap")

    def suite_swapper(learning: LearningRecord, recorded: RecordedRun) -> AppliedPatch | None:
        return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC}, test_suite=FAILING_SUITE)

    result = replay_one(run, make_learning(), suite_swapper, GATE, repeats=3, timeout=5.0)
    assert result.applied is True
    assert set(result.warm_verdicts) == {GateVerdict.FAIL_TESTS.value}
    assert result.warm is not None
    assert result.warm.gate == "tests"
    assert result.deltas == (-1.0, -1.0, -1.0)  # accepted cold, rejected warm x3
    assert result.material_regression is True


def test_applier_without_suite_keeps_recorded_suite():
    """``test_suite=None`` keeps the recorded suite (AppliedPatch's documented
    default). The warm candidate below passes the goldens but FAILS the
    recorded suite, so a fail_tests verdict proves the recorded suite still
    ran — dropping it would have produced a never-worse pass_tests rejection
    instead."""
    victim = (
        "def add(a, b):\n"
        "    if a == 0 and b == 0:\n"
        "        return 1\n"
        "    return a + b\n"
    )
    run = recorded_good_run("run_suite_kept")

    def suite_keeper(learning: LearningRecord, recorded: RecordedRun) -> AppliedPatch | None:
        return AppliedPatch(candidate_files={"candidate.py": victim})

    result = replay_one(run, make_learning(), suite_keeper, GATE, repeats=3, timeout=5.0)
    assert result.applied is True
    assert set(result.warm_verdicts) == {GateVerdict.FAIL_TESTS.value}
    assert result.warm is not None
    assert result.warm.gate == "tests"
