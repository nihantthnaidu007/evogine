"""Memory-gate tests: counterfactual replay promotion, rejection, and demotion.

Fixtures run the FULL code gate pipeline through the trusted local executor
(fixture code only, the test_code_gate.py pattern); Docker-boundary replay
tests skip when the sandbox image is unavailable so CI stays green. The
promotion materiality rule is seed learn_0009's: 1.4826 x MAD of centered
repeats, paired delta strictly above the floor — a replayed verdict change
or nothing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evogine.engine.bootstrap import load_seeds
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.store import Engine, InvalidTransitionError
from evogine.gates import sandbox as sbx
from evogine.gates.code_gate import (
    CodeGate,
    GateInput,
    GateResult,
    GateVerdict,
    GoldenCase,
)
from evogine.gates.memory_gate import (
    SCAN_AUDIT_ACTION,
    MemoryGate,
    MemoryGateConfig,
    find_learning_oracle_leak,
)
from evogine.gates.replay import (
    MAD_TO_FLOOR,
    AppliedPatch,
    RecordedRun,
    ReplayEvidence,
    RunLog,
    UnknownRunError,
    flip_is_material,
    paired_delta_and_floor,
    regression_is_material,
    replay_one,
)

LOCAL = sbx.LocalSubprocessSandbox()
GATE = CodeGate(executor=LOCAL)
SEEDS = Path(__file__).resolve().parents[1] / "src" / "evogine" / "seeds"

# The same fixture family as test_code_gate.py: the baseline carries one
# redundant round-trip; GOOD beats it on ops; BROKEN fails the suite. All
# golden outputs are long reprs so the oracle-leak scanner would see them.
BASELINE_SRC = "def add(a, b):\n    result = a + b\n    return result\n"
GOOD_SRC = "def add(a, b):\n    return a + b\n"
BROKEN_SRC = "def add(a, b):\n    return a - b\n"
WORSE_SRC = "def add(a, b):\n    x = a\n    y = b\n    total = x + y\n    return total\n"

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


# ----------------------------------------------------------------- fixtures


def make_engine(tmp_path: Path, **kwargs) -> Engine:
    return Engine(tmp_path / "engine", **kwargs)


def make_run_log(tmp_path: Path) -> RunLog:
    return RunLog(tmp_path / "runs" / "runs.jsonl")


def make_learning(**overrides) -> LearningRecord:
    fields: dict = {
        "id": "learn_replay_0001",
        "type": "failure-pattern",
        "title": "Sign-flipped arithmetic in add helpers",
        "body": RecordBody(
            signal="add returns a - b",
            failure_mode="wrong operand order",
            detection="counterfactual replay flips the gate verdict",
            fix_strategy="return a + b",
        ),
        "status": "ephemeral",
        "confidence": 0.9,
        "provenance": RecordProvenance(
            proposed_by="reflector",
            origin="rollout",
            source="run_1",
            note="test learning",
        ),
        "counterfactual": RecordCounterfactual(),
        "created": datetime(2026, 9, 17, tzinfo=UTC),
        "promoted_at": None,
    }
    fields.update(overrides)
    return LearningRecord(**fields)


def record_gate_run(
    run_log: RunLog,
    *,
    run_id: str,
    candidate_src: str,
    case_id: str = "case_add",
    baseline_src: str = BASELINE_SRC,
) -> RecordedRun:
    """Record one past run the honest way: run the real gate, keep the payload.

    This is the verifier-side recording contract replay.py documents — the
    cold arm is the code gate's own verdict payload, captured at run time.
    """
    gate_input = GateInput.single_function(
        baseline_source=baseline_src,
        candidate_source=candidate_src,
        entry_point="add",
        test_suite=SUITE,
        golden_cases=GOLDEN_CASES,
    )
    cold = GATE.evaluate(gate_input)
    run = RecordedRun(
        run_id=run_id,
        case_id=case_id,
        gate_input=gate_input,
        cold=cold,
        recorded_at=datetime.now(UTC),
        note="fixture run",
    )
    return run_log.record(run)


def gate_for(run_log: RunLog, applier, **kwargs) -> MemoryGate:
    return MemoryGate(run_log=run_log, applier=applier, executor=LOCAL, **kwargs)


def fixing_applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
    """The learning's fix, applied where it belongs: flips a - b to a + b."""
    source = run.gate_input.candidate_files["candidate.py"]
    if "return a - b" in source:
        return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC})
    return None


def overzealous_applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
    """The same learning applied everywhere it does not belong: on an already-
    correct candidate it adds a redundant round-trip that forfeits the
    never-worse comparison — the regression the demotion path exists for."""
    source = run.gate_input.candidate_files["candidate.py"]
    if "return a - b" in source:
        return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC})
    if "return a + b" in source:
        return AppliedPatch(candidate_files={"candidate.py": BASELINE_SRC})
    return None


def make_flaky_applier(flips: int):
    """Applies the fix for the first ``flips`` calls, then cannot apply at all
    (returns None) — the instability the per-case floor must price out."""
    calls = {"n": 0}

    def applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
        source = run.gate_input.candidate_files["candidate.py"]
        if "return a - b" not in source:
            return None
        calls["n"] += 1
        if calls["n"] <= flips:
            return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC})
        return None

    return applier


def poison_applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
    raise AssertionError("memory gate consulted the applier; it must not have been")


# ------------------------------------------------------- pure floor math


def test_floor_rule_is_seed_learn_0009():
    assert MAD_TO_FLOOR == 1.4826  # phase 0's per-case noise floor constant
    # Consistent flips: MAD 0 -> floor 0 -> any positive delta is material.
    center, floor = paired_delta_and_floor([1.0, 1.0, 1.0])
    assert center == 1.0
    assert floor == 0.0
    assert flip_is_material([1.0, 1.0, 1.0])


def test_unstable_flip_prices_out_below_its_own_floor():
    # deltas split across repeats: median 0.5, MAD 0.5, floor 0.7413 -> noise
    center, floor = paired_delta_and_floor([1.0, 1.0, 0.0, 0.0])
    assert center == 0.5
    assert floor == pytest.approx(1.4826 * 0.5)
    assert not flip_is_material([1.0, 1.0, 0.0, 0.0])
    assert not flip_is_material([0.0, 0.0, 0.0, 0.0])


def test_regression_floor_mirrors_the_flip_floor():
    assert regression_is_material([-1.0, -1.0, -1.0])
    assert not regression_is_material([-1.0, 0.0, 0.0])
    assert not regression_is_material([1.0, 1.0, 1.0])
    assert not flip_is_material([-1.0, -1.0, -1.0])


def test_floor_refuses_single_repeat_and_bad_multiplier():
    with pytest.raises(ValueError, match="at least 2"):
        paired_delta_and_floor([1.0])
    with pytest.raises(ValueError, match="multiplier"):
        paired_delta_and_floor([1.0, 1.0], multiplier=0.0)


def test_config_validates_repeats_and_multiplier():
    with pytest.raises(ValueError, match="repeats"):
        MemoryGateConfig(repeats=1)
    with pytest.raises(ValueError, match="multiplier"):
        MemoryGateConfig(mad_multiplier=0.0)


# ------------------------------------------------------- run log contract


def test_run_log_round_trips_single_function(tmp_path):
    run_log = make_run_log(tmp_path)
    recorded = record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    assert recorded.cold.verdict is GateVerdict.FAIL_TESTS
    assert recorded.cold.accepted is False

    loaded = run_log.get("run_1")
    assert loaded == recorded
    assert loaded.gate_input.baseline_files["candidate.py"] == BASELINE_SRC
    assert loaded.gate_input.candidate_files["candidate.py"] == BROKEN_SRC
    assert loaded.gate_input.entry_point == "add"
    assert loaded.gate_input.test_suite == SUITE
    assert loaded.gate_input.golden_cases == GOLDEN_CASES
    assert loaded.recorded_at == recorded.recorded_at
    assert [run.run_id for run in run_log.runs()] == ["run_1"]


def test_run_log_round_trips_multi_file(tmp_path):
    run_log = make_run_log(tmp_path)
    unit = {
        "__init__.py": "from calc import Calc\n",
        "calc.py": "class Calc:\n    def value(self, n):\n        return n + 1\n",
    }
    baseline = {
        "__init__.py": "from calc import Calc\n",
        "calc.py": "class Calc:\n    def value(self, n):\n        return n\n",
    }
    corpus = "def build_corpus():\n    return [('inc', (), [('value', [1])])]\n"
    gate_input = GateInput.multi_file(
        baseline_files=baseline,
        candidate_files=unit,
        entry_class="Calc",
        corpus_source=corpus,
        test_suite=None,
    )
    cold = GATE.evaluate(gate_input)
    run = run_log.record(
        RecordedRun(run_id="mf_1", case_id="case_unit", gate_input=gate_input, cold=cold)
    )
    loaded = run_log.get("mf_1")
    assert loaded.gate_input.scope == "multi-file"
    assert loaded.gate_input.entry_class == "Calc"
    assert loaded.gate_input.corpus_source == corpus
    assert loaded.gate_input.baseline_files == baseline
    assert loaded.gate_input.candidate_files == unit
    assert loaded == run


def test_run_log_rejects_duplicates_and_corrupt_lines(tmp_path):
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    with pytest.raises(ValueError, match="already recorded"):
        record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)

    log_file = tmp_path / "runs" / "runs.jsonl"
    log_file.write_text(log_file.read_text(encoding="utf-8") + "{not json}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt run log"):
        run_log.runs()
    with pytest.raises(RuntimeError, match="corrupt run log"):
        run_log.get("run_1")  # a damaged log surfaces; it never silently shrinks


def test_run_log_refuses_non_json_golden_output(tmp_path):
    run_log = make_run_log(tmp_path)
    gate_input = GateInput.single_function(
        baseline_source=BASELINE_SRC,
        candidate_source=BROKEN_SRC,
        entry_point="add",
        test_suite=SUITE,
        golden_cases=(GoldenCase(args=(), output=object()),),
    )
    with pytest.raises(ValueError, match="JSON-serializable"):
        run_log.record(
            RecordedRun(
                run_id="run_x", case_id="c", gate_input=gate_input, cold=GATE.evaluate(gate_input)
            )
        )


def test_unknown_run_id_raises(tmp_path):
    run_log = make_run_log(tmp_path)
    with pytest.raises(UnknownRunError):
        run_log.get("no_such_run")


# ------------------------------------------------------- promotion path


def test_material_flip_promotes_with_evidence_on_the_record(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning())
    engine.queue_replay(record.id, ["run_1"])

    gate = gate_for(run_log, fixing_applier)
    result = gate.decide(engine, record.id)

    assert result.status == "promoted"
    assert result.promoted_at is not None
    assert result.counterfactual.verdict_change is True
    assert result.counterfactual.replay_run_ids == ["run_1"]
    evidence = json.loads(result.counterfactual.evidence or "{}")
    assert evidence["material_flips"] == 1
    assert evidence["flipped_run_ids"] == ["run_1"]
    per_run = evidence["per_run"][0]
    assert per_run["cold_verdict"] == "fail_tests"
    assert per_run["cold_accepted"] is False
    assert set(per_run["warm_verdicts"]) == {"pass_tests"}
    assert per_run["material_flip"] is True
    assert evidence["floor_rule"].startswith("1.4826 x MAD")
    # Promotion is the only path into retrieval, and the learning is now there.
    assert [r.id for r in engine.retrieve("add helper", k=5)] == [record.id]


def test_promotion_audit_trail_shows_measurement_and_decision(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning())
    engine.queue_replay(record.id, ["run_1"])
    gate = gate_for(run_log, fixing_applier)
    gate.decide(engine, record.id)

    replay_entries = engine.audit.find("replay", record.id)
    promote_entries = engine.audit.find("promote", record.id)
    assert len(replay_entries) == 1
    assert replay_entries[0].actor == "memory-gate"
    assert "material verdict flip on 1 of 1" in (replay_entries[0].reason or "")
    assert len(promote_entries) == 1
    assert promote_entries[0].actor == "engine"
    # the audit never carries the oracle: no golden output literal may leak
    assert "333333" not in (replay_entries[0].evidence or "")


def test_review_returns_true_for_material_flip(tmp_path):
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    gate = gate_for(run_log, fixing_applier)
    assert gate.review(make_learning(), ["run_1"]) is True


def test_never_worse_rejection_flips_to_accepted_better(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    # The cold arm is a behaviorally sound candidate the never-worse rule
    # rejected (worse ops than baseline): verdict pass_tests, accepted False.
    run = record_gate_run(run_log, run_id="run_worse", candidate_src=WORSE_SRC)
    assert run.cold.verdict is GateVerdict.PASS_TESTS
    assert run.cold.accepted is False
    assert run.cold.gate == "baseline"

    record = engine.propose(make_learning(id="learn_replay_0002"))
    engine.queue_replay(record.id, ["run_worse"])

    def slimming_applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
        source = run.gate_input.candidate_files["candidate.py"]
        if "total = x + y" in source:
            return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC})
        return None

    gate = gate_for(run_log, slimming_applier)
    result = gate.decide(engine, record.id)
    assert result.status == "promoted"
    evidence = json.loads(result.counterfactual.evidence or "{}")
    assert evidence["flipped_run_ids"] == ["run_worse"]
    assert evidence["per_run"][0]["cold_verdict"] == "pass_tests"


# ------------------------------------------------------- rejection path


def test_no_flip_stays_ephemeral_with_recorded_reason(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_0003"))
    engine.queue_replay(record.id, ["run_1"])

    def cosmetic_applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
        # applies the learning "fix" that does not fix: candidate still fails
        source = run.gate_input.candidate_files["candidate.py"]
        if "return a - b" in source:
            return AppliedPatch(candidate_files={"candidate.py": "# patched\n" + BROKEN_SRC})
        return None

    gate = gate_for(run_log, cosmetic_applier)
    result = gate.decide(engine, record.id)

    assert result.status == "ephemeral"
    assert result.counterfactual.verdict_change is None  # failed evidence is stripped
    rejected = engine.audit.find("promote-rejected", record.id)
    assert len(rejected) == 1
    assert "no verdict change" in (rejected[0].reason or "")
    # the gate's own measurement is on the trail too — rejections are not silent
    replay_entry = engine.audit.find("replay", record.id)[0]
    assert "no material verdict flip" in (replay_entry.reason or "")
    assert json.loads(replay_entry.evidence or "{}")["material_flips"] == 0
    assert gate.review(make_learning(id="learn_replay_0003"), ["run_1"]) is False


def test_flaky_flip_priced_out_by_the_floor(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_0004"))
    engine.queue_replay(record.id, ["run_1"])

    # Two of four warm repeats apply the fix; the rest cannot apply at all:
    # deltas [1, 1, 0, 0] -> median 0.5 < floor 0.7413 -> noise, not evidence.
    gate = gate_for(run_log, make_flaky_applier(flips=2), config=MemoryGateConfig(repeats=4))
    result = gate.decide(engine, record.id)

    assert result.status == "ephemeral"
    replay_entry = engine.audit.find("replay", record.id)[0]
    evidence = json.loads(replay_entry.evidence or "{}")
    per_run = evidence["per_run"][0]
    assert per_run["warm_verdicts"].count("pass_tests") == 2
    assert per_run["warm_verdicts"].count("not_applied") == 2
    assert per_run["median_delta"] == 0.5
    assert per_run["floor"] == pytest.approx(1.4826 * 0.5)
    assert per_run["material_flip"] is False
    assert evidence["material_flips"] == 0
    rejected = engine.audit.find("promote-rejected", record.id)
    assert len(rejected) == 1


def test_learning_applicable_to_no_run_rejects_without_evidence(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=WORSE_SRC)
    record = engine.propose(make_learning(id="learn_replay_0005"))
    engine.queue_replay(record.id, ["run_1"])

    gate = gate_for(run_log, fixing_applier)  # fix matches BROKEN_SRC, not WORSE_SRC
    result = gate.decide(engine, record.id)

    assert result.status == "ephemeral"
    replay_entry = engine.audit.find("replay", record.id)[0]
    per_run = json.loads(replay_entry.evidence or "{}")["per_run"][0]
    assert per_run["applied"] is False
    assert per_run["deltas"] == []
    assert per_run["median_delta"] is None
    rejected = engine.audit.find("promote-rejected", record.id)
    assert len(rejected) == 1


def test_empty_replay_rejects_without_evidence(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record = engine.propose(make_learning(id="learn_replay_0006"))
    gate = gate_for(run_log, fixing_applier)
    assert gate.decide(engine, record.id).status == "ephemeral"
    assert engine.audit.find("promote-rejected", record.id) != []


def test_unknown_queued_run_fails_loudly_before_any_mutation(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_0007"))
    engine.queue_replay(record.id, ["run_1", "ghost_run"])

    gate = gate_for(run_log, fixing_applier)
    with pytest.raises(UnknownRunError, match="ghost_run"):
        gate.decide(engine, record.id)
    # nothing was decided and nothing was silently skipped
    assert engine.get(record.id).status == "candidate"
    assert engine.audit.find("promote", record.id) == []
    assert engine.audit.find("promote-rejected", record.id) == []


def test_engine_materiality_threshold_still_applies_above_the_floor(tmp_path):
    # The engine-level minimum is separate: one flipped run clears the
    # per-case floor but not a materiality_threshold of 2 replayed runs.
    engine = make_engine(tmp_path, materiality_threshold=2)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_0008"))
    engine.queue_replay(record.id, ["run_1"])

    gate = gate_for(run_log, fixing_applier)
    result = gate.decide(engine, record.id)

    assert result.status == "ephemeral"
    rejected = engine.audit.find("promote-rejected", record.id)
    assert len(rejected) == 1
    assert "materiality threshold" in (rejected[0].reason or "")


def test_decide_refuses_promoted_record_via_the_engine(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_0009"))
    engine.queue_replay(record.id, ["run_1"])
    gate = gate_for(run_log, fixing_applier)
    gate.decide(engine, record.id)

    with pytest.raises(InvalidTransitionError, match="already promoted"):
        gate.decide(engine, record.id)


# ------------------------------------------------------- demotion path


def test_regression_flip_demotes_promoted_learning(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    # A later run that was fine on its own: GOOD beats the baseline, accepted.
    record_gate_run(run_log, run_id="run_2", candidate_src=GOOD_SRC)
    record = engine.propose(make_learning(id="learn_replay_0010"))
    engine.queue_replay(record.id, ["run_1"])

    gate = gate_for(run_log, overzealous_applier)
    assert gate.decide(engine, record.id).status == "promoted"

    # Later regression: the learning applied where it does not belong turns
    # accepted work into a never-worse rejection — verified by replay.
    assert gate.check_for_regression(engine, record.id, "run_2") is True
    demoted = engine.get(record.id)
    assert demoted.status == "dropped"
    entries = engine.audit.find("demote", record.id)
    assert len(entries) == 1
    assert "run_2" in (entries[0].reason or "")
    assert "regression" in (entries[0].reason or "")


def test_no_regression_flip_keeps_the_learning_promoted(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record_gate_run(run_log, run_id="run_2", candidate_src=GOOD_SRC)
    record = engine.propose(make_learning(id="learn_replay_0011"))
    engine.queue_replay(record.id, ["run_1"])
    gate = gate_for(run_log, fixing_applier)  # the fix does not touch GOOD_SRC
    assert gate.decide(engine, record.id).status == "promoted"

    assert gate.check_for_regression(engine, record.id, "run_2") is False
    assert engine.get(record.id).status == "promoted"
    assert engine.audit.find("demote", record.id) == []


def test_regression_check_ignores_non_promoted_records(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=GOOD_SRC)
    record = engine.propose(make_learning(id="learn_replay_0012"))
    gate = gate_for(run_log, overzealous_applier)
    assert gate.check_for_regression(engine, record.id, "run_1") is False
    assert engine.get(record.id).status == "ephemeral"


# ------------------------------------------------------- the seed pack


def test_seed_records_are_promoted_without_consulting_the_gate(tmp_path):
    engine = make_engine(tmp_path)
    # The poison applier proves the negative: if anything in bootstrap or
    # retrieval routed the seeds through this gate, the test explodes.
    poison_gate = gate_for(make_run_log(tmp_path), poison_applier)

    loaded = load_seeds(engine, SEEDS)

    assert len(loaded) == 12
    assert all(record.status == "promoted" for record in loaded)
    assert {record.id for record in loaded} == {f"learn_{i:04d}" for i in range(1, 13)}
    for record in loaded:
        # provenance, not the gate, is where a seed's promotion story lives
        assert "human" in record.provenance.note.lower()
        assert record.provenance.proposed_by == "human" or record.provenance.origin == "seed"
        assert record.counterfactual.verdict_change is None
        assert record.promoted_at is not None
    actions = {entry.action for entry in engine.audit.entries()}
    assert actions == {"bootstrap-load"}  # no replay, no promote, no demote
    # the warm engine serves the seeds even though the gate was never consulted
    assert len(engine.retrieve("dead code cleanup", k=5)) == 5
    assert poison_gate is not None  # constructed, consulted never


# ------------------------------------------------------- Docker boundary


def _docker_image_present() -> bool:
    rc, _out, _err = sbx._docker(["image", "inspect", sbx.IMAGE])
    return rc == 0


_docker_mark = pytest.mark.skipif(not _docker_image_present(), reason="sandbox image not built")


@_docker_mark
def test_docker_replay_promotes_through_the_real_boundary(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_docker1"))
    engine.queue_replay(record.id, ["run_1"])

    gate = MemoryGate(run_log=run_log, applier=fixing_applier)  # default: Docker
    result = gate.decide(engine, record.id)
    assert result.status == "promoted"
    assert json.loads(result.counterfactual.evidence or "{}")["material_flips"] == 1


@_docker_mark
def test_docker_replay_rejects_without_a_flip(tmp_path):
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_replay_docker2"))
    engine.queue_replay(record.id, ["run_1"])

    def cosmetic_applier(learning: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
        source = run.gate_input.candidate_files["candidate.py"]
        if "return a - b" in source:
            return AppliedPatch(candidate_files={"candidate.py": "# patched\n" + BROKEN_SRC})
        return None

    gate = MemoryGate(run_log=run_log, applier=cosmetic_applier)  # default: Docker
    assert gate.decide(engine, record.id).status == "ephemeral"
    assert len(engine.audit.find("promote-rejected", record.id)) == 1


# ------------------------------------------------------- evidence shape


def test_replay_evidence_counterfactual_and_summary_agree(tmp_path):
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    gate = gate_for(run_log, fixing_applier)
    evidence = gate.evaluate(make_learning(), ["run_1"])

    assert isinstance(evidence, ReplayEvidence)
    assert evidence.verdict_change is True
    assert evidence.flipped_run_ids == ["run_1"]
    counterfactual = evidence.to_counterfactual()
    assert counterfactual.verdict_change is True
    assert counterfactual.replay_run_ids == ["run_1"]
    assert json.loads(counterfactual.evidence or "")["replayed_runs"] == 1


def test_replay_one_reports_cold_history_and_warm_repeats(tmp_path):
    run_log = make_run_log(tmp_path)
    run = record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    learning = make_learning()
    result = replay_one(run, learning, fixing_applier, GATE, repeats=3, timeout=5.0)
    assert result.applied is True
    assert result.cold_verdict == "fail_tests"
    assert result.cold_accepted is False
    assert result.deltas == (1.0, 1.0, 1.0)
    assert result.median_delta == 1.0
    assert result.floor == 0.0
    assert result.material_flip is True
    assert result.material_regression is False
    assert result.warm is not None and result.warm.accepted is True


# --------------------------------------------- promotion-time oracle-leak scan


def test_planted_oracle_leak_in_a_learning_rejects_promotion(tmp_path):
    """A learning carrying a golden output literal never promotes (finding 6)."""
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    leaky = make_learning(
        id="learn_leak_0001",
        body=RecordBody(
            signal="add returns a - b; golden output for (5000, 6000) is 11000",
            failure_mode="wrong operand order",
            detection="counterfactual replay flips the gate verdict",
            fix_strategy="return a + b",
        ),
    )
    record = engine.propose(leaky)
    engine.queue_replay(record.id, ["run_1"])

    gate = gate_for(run_log, fixing_applier)
    result = gate.decide(engine, record.id)

    # Fail closed: the leaky learning is back to ephemeral, never promoted.
    assert result.status == "ephemeral"
    assert result.promoted_at is None
    scan_entries = engine.audit.find(SCAN_AUDIT_ACTION, record.id)
    assert len(scan_entries) == 1
    assert scan_entries[0].actor == "memory-gate"
    assert "golden case" in (scan_entries[0].reason or "")
    # The engine's own rejection is audited too, carrying the scan's reason.
    rejected = engine.audit.find("promote-rejected", record.id)
    assert len(rejected) == 1
    assert "oracle-leak scan" in (rejected[0].reason or "")
    # The scan runs before the replay: no measurement is spent on a leaky draft.
    assert engine.audit.find("replay", record.id) == []


def test_oracle_scan_error_rejects_promotion_fail_closed(tmp_path, monkeypatch):
    """A broken scanner rejects — unscanned text never promotes."""
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning(id="learn_scanerr_0001"))
    engine.queue_replay(record.id, ["run_1"])

    def broken_scanner(record, runs):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr("evogine.gates.memory_gate.find_learning_oracle_leak", broken_scanner)
    gate = gate_for(run_log, fixing_applier)
    result = gate.decide(engine, record.id)

    assert result.status == "ephemeral"
    scan_entries = engine.audit.find(SCAN_AUDIT_ACTION, record.id)
    assert len(scan_entries) == 1
    assert "scan error" in (scan_entries[0].reason or "")
    assert "fail-closed" in (scan_entries[0].reason or "")
    assert engine.audit.find("replay", record.id) == []


def test_clean_learning_still_promotes_under_the_scan(tmp_path):
    """The scan changes nothing for a leak-free learning."""
    engine = make_engine(tmp_path)
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    record = engine.propose(make_learning())
    engine.queue_replay(record.id, ["run_1"])

    gate = gate_for(run_log, fixing_applier)
    result = gate.decide(engine, record.id)

    assert result.status == "promoted"
    assert result.promoted_at is not None
    assert engine.audit.find(SCAN_AUDIT_ACTION, record.id) == []
    # the promotion trail is the usual replay + promote pair
    assert len(engine.audit.find("replay", record.id)) == 1
    assert len(engine.audit.find("promote", record.id)) == 1


def test_learning_leak_scan_covers_body_fields_title_and_provenance(tmp_path):
    run_log = make_run_log(tmp_path)
    record_gate_run(run_log, run_id="run_1", candidate_src=BROKEN_SRC)
    run = run_log.get("run_1")

    def learning_with(field: str, text: str) -> LearningRecord:
        body = RecordBody(signal="", failure_mode="", detection="", fix_strategy="")
        if field == "title":
            return make_learning(id="learn_cover_0001", title=text, body=body)
        return make_learning(id="learn_cover_0001", body=body.model_copy(update={field: text}))

    for field in ("title", "signal", "failure_mode", "detection", "fix_strategy"):
        leaky = learning_with(field, "the golden output is 333333, memorize it")
        assert find_learning_oracle_leak(leaky, [run]) is not None, field
    for provenance_field in ("source", "note"):
        leaky = learning_with("signal", "")
        leaky = leaky.model_copy(
            update={
                "provenance": leaky.provenance.model_copy(
                    update={provenance_field: "echo 333333 back"}
                )
            }
        )
        assert find_learning_oracle_leak(leaky, [run]) is not None, provenance_field
    # A learning that never carries a golden literal scans clean.
    assert find_learning_oracle_leak(make_learning(), [run]) is None


def test_learning_leak_scan_needs_golden_material_to_scan_against(tmp_path):
    """Multi-file runs carry no golden I/O oracle: nothing to scan against."""
    gate_input = GateInput(
        scope="multi-file",
        baseline_files={"__init__.py": ""},
        candidate_files={"unit.py": "class X: ..."},
        entry_class="X",
        corpus_source="corpus",
    )
    cold = GateResult(verdict=GateVerdict.PASS_TESTS, accepted=True, gate=None, reason="")
    run = RecordedRun(run_id="run_m", case_id="c", gate_input=gate_input, cold=cold)
    leaky_body = RecordBody(signal="333333", failure_mode="", detection="", fix_strategy="")
    leaky = make_learning(body=leaky_body)
    assert find_learning_oracle_leak(leaky, [run]) is None
