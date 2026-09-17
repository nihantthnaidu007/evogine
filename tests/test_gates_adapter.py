"""CodeGateAdapter: the production CandidateGate against the real CodeGate.

The Docker executor is the production boundary; these tests run the identical
pipeline through LocalSubprocessSandbox — fast, deterministic, no Docker —
with the real case corpus.  The loop-level integration (scripted replies
routed through this adapter inside the swarm loop) lives in
``test_bench_harness.py``; here the adapter is exercised directly on accepted
and rejected candidate trees.
"""

from __future__ import annotations

from evogine.bench.harness import to_swarm_task
from evogine.bench.task import load_task
from evogine.gates.sandbox import LocalSubprocessSandbox
from evogine.swarm.gates_adapter import CodeGateAdapter

#: One adapter shared by the module: LocalSubprocessSandbox is stateless.
LOCAL_GATE = CodeGateAdapter(LocalSubprocessSandbox())

# Single-file cases evaluate fastest and cover the distinct verdict paths:
# scenario-kind equivalence for retention, golden-kind for the oracle scan.
CASE_ID = "counter-overflow-wrap"
GOLDEN_CASE_ID = "billing-day-offset"


def test_reference_implementation_is_accepted() -> None:
    """The case's clean target passes the gate (scenario equivalence: no ops)."""
    task = load_task(CASE_ID)
    outcome = LOCAL_GATE.evaluate(task.reference_files, to_swarm_task(task))
    assert outcome.verdict == "pass_tests"
    assert outcome.visible_passed == 1
    assert outcome.properties_passed == 1
    assert outcome.failure_detail is None
    # Multi-row scenario equivalence carries no bytecode-ops score.
    assert outcome.ops == {}


def test_golden_reference_ops_are_measured() -> None:
    """Single-function golden replay measures the never-worse ops bar.

    billing-day-offset's degraded baseline crashes on the month-end golden
    input, so the incumbent has no measurable ops and the accepted candidate
    carries only its own count — the vacuous-bar branch, not a symmetric
    comparison.
    """
    task = load_task(GOLDEN_CASE_ID)
    outcome = LOCAL_GATE.evaluate(task.reference_files, to_swarm_task(task))
    assert outcome.verdict == "pass_tests"
    assert "candidate_ops" in outcome.ops
    assert "baseline_ops" not in outcome.ops


def test_noop_candidate_is_rejected() -> None:
    """Re-emitting the degraded baseline fails equivalence, not the loop."""
    task = load_task(CASE_ID)
    outcome = LOCAL_GATE.evaluate(task.baseline_files, to_swarm_task(task))
    assert outcome.verdict == "fail_equivalence"
    assert outcome.properties_passed == 0
    # The reflector gets an oracle-free reason for the rejection.
    assert outcome.failure_detail


def test_oracle_leak_candidate_is_rejected() -> None:
    """A candidate embedding a golden expected output fails the oracle scan."""
    task = load_task(GOLDEN_CASE_ID)
    expected = task.properties["properties"][0]["rows"][0]["expected"]
    assert isinstance(expected, str) and len(expected) >= 5
    leaky = dict(task.reference_files)
    module = next(iter(leaky))
    leaky[module] = leaky[module] + f"\n\n_GOLDEN = {expected!r}\n"
    outcome = LOCAL_GATE.evaluate(leaky, to_swarm_task(task))
    # The scan rides the equivalence stage (fail-closed rejection).
    assert outcome.verdict == "fail_equivalence"
    assert "oracle leak" in (outcome.failure_detail or "")
