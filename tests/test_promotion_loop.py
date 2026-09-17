"""The wired promotion loop: recorded runs, replay decisions, audit trail.

These tests close the v2 audit's integration gap end to end: the production
run path (``run_task`` with :class:`PromotionWiring`) records every gate
evaluation as a replayable run, the reflector's landed learning is queued
against the run's baseline+candidate run ids, and the memory gate decides
from counterfactual replay evidence. The loop's gate seam runs as a fast
taxonomy-speaking fake; the memory gate replays through the REAL code gate
on the trusted local executor (the test_memory_gate.py fixture family), so
promotion evidence is measured, not asserted into existence. CI makes no
API calls: every model client is scripted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pytest import CaptureFixture

from evogine.cli import main
from evogine.engine.store import Engine
from evogine.gates import sandbox as sbx
from evogine.gates.code_gate import GateInput, GateVerdict, GoldenCase
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import AppliedPatch, RunLog
from evogine.swarm.promotion import (
    PromotionWiring,
    finish_promotion,
    gate_result_from_outcome,
    record_candidate_run,
    recorded_run_id,
)
from evogine.swarm.roles import ModelResponse
from evogine.swarm.runner import GateOutcome, SwarmConfig, SwarmTask, run_task

LOCAL = sbx.LocalSubprocessSandbox()

# The add fixture family (test_code_gate.py / test_memory_gate.py): the
# baseline carries one redundant round-trip, GOOD beats it on ops, BROKEN
# fails the suite. Golden outputs are long reprs so the oracle-leak scanner
# would see them.
BASELINE_SRC = "def add(a, b):\n    result = a + b\n    return result\n"
GOOD_SRC = "def add(a, b):\n    return a + b\n"
BROKEN_SRC = "def add(a, b):\n    return a - b\n"

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

_LEARNING_DRAFT = json.dumps(
    {
        "type": "failure-pattern",
        "title": "add helper arithmetic repair",
        "confidence": 0.9,
        "body": {
            "signal": "the candidate subtracts instead of adding",
            "failure_mode": "wrong operand application",
            "detection": "code gate rejection on the case",
            "fix_strategy": "return the sum of the operands",
        },
    }
)


# ----------------------------------------------------------------- fixtures


def make_task(tmp_path: Path, **overrides: Any) -> SwarmTask:
    """The add-family swarm task with gate-real properties and a visible suite."""
    suite = tmp_path / "test_suite.py"
    suite.write_text(SUITE)
    fields: dict[str, Any] = {
        "case_id": "case_add",
        "description": "Repair an add helper under arithmetic decay pressure",
        "scenario_brief": "add must sum its arguments",
        "baseline_files": {"candidate.py": BASELINE_SRC},
        "decay_axis": "arithmetic",
        "suite_path": suite,
        "properties": {
            "entry": "add",
            "properties": [
                {
                    "kind": "golden_io",
                    "id": "p_add",
                    "rows": [
                        {"args": [2, 3], "expected": 5},
                        {"args": [700000, 300000], "expected": 1000000},
                    ],
                }
            ],
        },
    }
    fields.update(overrides)
    return SwarmTask(**fields)


def make_config(tmp_path: Path, run_id: str, **overrides: Any) -> SwarmConfig:
    fields: dict[str, Any] = {
        "runs_dir": tmp_path / "runs",
        "run_id": run_id,
        "generations": 1,
        "candidates_per_generation": 1,
    }
    fields.update(overrides)
    return SwarmConfig(**fields)


def make_wiring(tmp_path: Path, applier: Any) -> PromotionWiring:
    run_log = RunLog(tmp_path / "engine" / "runlog.jsonl")
    memory_gate = MemoryGate(run_log=run_log, applier=applier, executor=LOCAL)
    return PromotionWiring(run_log=run_log, memory_gate=memory_gate)


@dataclass
class TaxonomyGate:
    """The loop's fast fake gate, speaking the code gate's verdict taxonomy.

    Recording reconstructs :class:`GateResult` payloads from outcomes, so this
    fake speaks the same encoding the production adapter does: accepted iff
    every visible check passed, rejections carry a failing stage.
    """

    candidates_pass: bool = False
    calls: list[dict[str, str]] = field(default_factory=list)

    def evaluate(self, files: Mapping[str, str], task: SwarmTask) -> GateOutcome:
        self.calls.append(dict(files))
        # extract_code strips builder replies, so match on stripped sources.
        source = files.get("candidate.py", "").strip()
        if source == BASELINE_SRC.strip():
            # The degraded incumbent fails its own hidden property.
            return GateOutcome(
                verdict="fail_tests",
                failure_detail="hidden property p_add expected 5",
                visible_passed=0,
                visible_total=1,
                properties_passed=0,
                properties_total=1,
                gate_stage="tests",
            )
        if self.candidates_pass and source == GOOD_SRC.strip():
            return GateOutcome(
                verdict="pass_tests",
                failure_detail=None,
                visible_passed=1,
                visible_total=1,
                properties_passed=1,
                properties_total=1,
                ops={"baseline_ops": 4, "candidate_ops": 2},
            )
        return GateOutcome(
            verdict="fail_tests",
            failure_detail="add returns a - b",
            visible_passed=0,
            visible_total=1,
            properties_passed=0,
            properties_total=1,
            gate_stage="tests",
        )


class ScriptedRoles:
    """The roles' fake model: builder, verifier, and reflector replies by shape."""

    def __init__(self, builder_reply: str) -> None:
        self._builder_reply = builder_reply

    def generate(
        self, *, messages: list[dict[str, str]], model_config: Any
    ) -> ModelResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if system.startswith("You are the repair builder"):
            return ModelResponse(text=self._builder_reply)
        if system.startswith("You are the independent verifier"):
            return ModelResponse(text="VERDICT: fail\nscripted review is not the gate")
        if system.startswith("You are the reflector"):
            return ModelResponse(text=_LEARNING_DRAFT)
        raise RuntimeError("fake model received an unrecognized role prompt")


def fixing_applier(learning: Any, run: Any) -> AppliedPatch | None:
    """The learning's fix, applied where it belongs: flips a - b to a + b."""
    source = run.gate_input.candidate_files["candidate.py"]
    if "return a - b" in source:
        return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC})
    return None


def cosmetic_applier(learning: Any, run: Any) -> AppliedPatch | None:
    """Applies the learning 'fix' that does not fix: the candidate still fails."""
    source = run.gate_input.candidate_files["candidate.py"]
    if "return a - b" in source:
        return AppliedPatch(candidate_files={"candidate.py": "# patched\n" + BROKEN_SRC})
    return None


def regressing_applier(learning: Any, run: Any) -> AppliedPatch | None:
    """On an already-correct candidate, forfeits the never-worse comparison."""
    source = run.gate_input.candidate_files["candidate.py"]
    if "return a - b" in source:
        return AppliedPatch(candidate_files={"candidate.py": GOOD_SRC})
    if "return a + b" in source:
        return AppliedPatch(candidate_files={"candidate.py": BASELINE_SRC})
    return None


# ------------------------------------------------- the compounding path


def test_promoted_learning_is_retrievable_in_a_subsequent_run(tmp_path: Path) -> None:
    """The test that would have caught the v2 audit's gap: a gated learning
    promoted through the wired path, then retrieved by the next run."""
    engine = Engine(tmp_path / "engine")
    task = make_task(tmp_path)
    wiring = make_wiring(tmp_path, fixing_applier)
    # Candidates fail -> the reflector proposes -> the wiring queues and decides.
    model = ScriptedRoles(builder_reply=BROKEN_SRC)

    report = run_task(
        task, engine, model, TaxonomyGate(), make_config(tmp_path, "run_a"), promotion=wiring
    )

    promotion = report["promotion"]
    assert promotion["decision"] == "promoted"
    learning_id = promotion["learning_id"]
    assert learning_id is not None
    # Baseline AND candidate evaluations were recorded and all queued.
    assert promotion["queued_run_ids"] == [
        "run_a::warm::case_add::baseline",
        "run_a::warm::case_add::candidate::generation-0::candidate-0",
    ]
    record = engine.get(learning_id)
    assert record.status == "promoted"
    # The decision carries measured evidence: a material counterfactual flip.
    assert record.counterfactual.verdict_change is True
    assert record.counterfactual.replay_run_ids == promotion["queued_run_ids"]
    assert len(engine.audit.find("replay", learning_id)) == 1
    assert len(engine.audit.find("promote", learning_id)) == 1

    # The next run retrieves the promoted learning — the compounding path.
    later_wiring = PromotionWiring(
        run_log=wiring.run_log,
        memory_gate=MemoryGate(run_log=wiring.run_log, applier=fixing_applier, executor=LOCAL),
    )
    report2 = run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=GOOD_SRC),  # candidates pass; nothing new proposed
        TaxonomyGate(candidates_pass=True),
        make_config(tmp_path, "run_b"),
        promotion=later_wiring,
    )
    assert learning_id in report2["retrieved_ids"]
    assert report2["promotion"]["learning_id"] is None
    assert report2["promotion"]["decision"] is None
    assert report2["promotion"]["demoted"] == []


def test_rejected_learning_is_not_promoted_and_leaves_evidence(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    task = make_task(tmp_path)
    wiring = make_wiring(tmp_path, cosmetic_applier)

    report = run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=BROKEN_SRC),
        TaxonomyGate(),
        make_config(tmp_path, "run_a"),
        promotion=wiring,
    )

    promotion = report["promotion"]
    assert promotion["decision"] == "ephemeral"
    learning_id = promotion["learning_id"]
    assert learning_id is not None
    assert engine.get(learning_id).status == "ephemeral"
    # The rejection is on the audit trail, not silence.
    assert len(engine.audit.find("promote-rejected", learning_id)) == 1
    replay_entry = engine.audit.find("replay", learning_id)[0]
    assert json.loads(replay_entry.evidence or "{}")["material_flips"] == 0
    # And NOT retrievable: the compounding path never sees it.
    assert learning_id not in [record.id for record in engine.retrieve(task.context, k=5)]


def test_regression_demotes_promoted_knowledge_with_evidence(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    task = make_task(tmp_path)
    promoting = make_wiring(tmp_path, fixing_applier)
    first = run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=BROKEN_SRC),
        TaxonomyGate(),
        make_config(tmp_path, "run_a"),
        promotion=promoting,
    )
    learning_id = first["promotion"]["learning_id"]
    assert learning_id is not None

    # A later run where the learning, applied to an accepted candidate, makes
    # it worse: cold accepted, warm rejected — the demotion path.
    regressing = make_wiring(tmp_path, regressing_applier)
    report = run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=GOOD_SRC),
        TaxonomyGate(candidates_pass=True),
        make_config(tmp_path, "run_b"),
        promotion=regressing,
    )

    assert report["promotion"]["demoted"] == [learning_id]
    assert engine.get(learning_id).status == "dropped"
    demote_entries = engine.audit.find("demote", learning_id)
    assert len(demote_entries) == 1
    # The demotion reason carries the measured regression, not vibes.
    assert "median paired delta" in (demote_entries[0].reason or "")
    assert learning_id not in [record.id for record in engine.retrieve(task.context, k=5)]


# ------------------------------------------------- the append-only run log


def test_run_log_is_append_only_across_runs(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    task = make_task(tmp_path)
    wiring = make_wiring(tmp_path, fixing_applier)
    log_path = wiring.run_log.path

    run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=BROKEN_SRC),
        TaxonomyGate(),
        make_config(tmp_path, "run_a"),
        promotion=wiring,
    )
    after_first = log_path.read_text()
    runs_after_first = wiring.run_log.runs()

    run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=GOOD_SRC),
        TaxonomyGate(candidates_pass=True),
        make_config(tmp_path, "run_b"),
        promotion=wiring,
    )

    # Append-only: earlier evidence is a byte-prefix of the grown log, never
    # rewritten by subsequent runs.
    assert log_path.read_text().startswith(after_first)
    assert len(wiring.run_log.runs()) > len(runs_after_first)
    # Duplicate recorded ids are a caller bug, refused loudly.
    with pytest.raises(ValueError, match="already recorded"):
        run_task(
            task,
            engine,
            ScriptedRoles(builder_reply=BROKEN_SRC),
            TaxonomyGate(),
            make_config(tmp_path, "run_a"),
            promotion=wiring,
        )


def test_unwired_run_writes_no_run_log(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    task = make_task(tmp_path)
    config = make_config(tmp_path, "run_a")
    report = run_task(
        task, engine, ScriptedRoles(builder_reply=BROKEN_SRC), TaxonomyGate(), config
    )
    assert "promotion" not in report
    assert not (tmp_path / "engine" / "runlog.jsonl").exists()


def test_collapsed_candidate_records_no_replay_run(tmp_path: Path) -> None:
    """A candidate whose gate evaluation collapsed has no re-evaluable input:
    only the baseline is recorded, nothing is proposed (the collapsed 0/0
    outcome is retained under pre-existing best-of-N semantics, so the
    reflector is not reached), and the wiring fabricates no evidence."""

    @dataclass
    class CollapsingGate(TaxonomyGate):
        def evaluate(self, files: Mapping[str, str], task: SwarmTask) -> GateOutcome:
            if files == task.baseline_files:
                return super().evaluate(files, task)
            raise RuntimeError("candidate evaluation exploded")

    engine = Engine(tmp_path / "engine")
    task = make_task(tmp_path)
    wiring = make_wiring(tmp_path, fixing_applier)

    report = run_task(
        task,
        engine,
        ScriptedRoles(builder_reply=BROKEN_SRC),
        CollapsingGate(),
        make_config(tmp_path, "run_a"),
        promotion=wiring,
    )

    promotion = report["promotion"]
    assert promotion["queued_run_ids"] == ["run_a::warm::case_add::baseline"]
    assert promotion["learning_id"] is None
    assert promotion["decision"] is None


# ------------------------------------------------- pure pieces


def test_recorded_run_ids_are_deterministic_and_unique(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    config = make_config(tmp_path, "run_a")
    baseline = recorded_run_id(config, task, "baseline")
    assert baseline == recorded_run_id(config, task, "baseline")
    candidate = recorded_run_id(config, task, "candidate", generation=0, candidate_index=0)
    assert candidate != baseline
    other_arm = recorded_run_id(make_config(tmp_path, "run_a", arm="cold"), task, "baseline")
    assert other_arm != baseline


def test_gate_result_from_outcome_maps_the_taxonomy() -> None:
    gate_input = GateInput.single_function(
        baseline_source=BASELINE_SRC,
        candidate_source=BROKEN_SRC,
        entry_point="add",
        test_suite=SUITE,
        golden_cases=GOLDEN_CASES,
    )
    accepted = GateOutcome(
        verdict="pass_tests",
        failure_detail=None,
        visible_passed=1,
        visible_total=1,
        properties_passed=1,
        properties_total=1,
        ops={"baseline_ops": 4, "candidate_ops": 2},
    )
    result = gate_result_from_outcome(accepted, gate_input=gate_input)
    assert result.verdict is GateVerdict.PASS_TESTS
    assert result.accepted is True
    assert result.returned_files == {"candidate.py": BROKEN_SRC}

    rejected = GateOutcome(
        verdict="fail_equivalence",
        failure_detail="golden replay diverged",
        visible_passed=0,
        visible_total=1,
        properties_passed=0,
        properties_total=1,
        gate_stage="equivalence",
    )
    result = gate_result_from_outcome(rejected, gate_input=gate_input)
    assert result.verdict is GateVerdict.FAIL_EQUIVALENCE
    assert result.accepted is False
    assert result.gate == "equivalence"
    assert result.returned_files == {"candidate.py": BASELINE_SRC}


def test_foreign_verdict_vocabulary_fails_loudly() -> None:
    gate_input = GateInput.single_function(
        baseline_source=BASELINE_SRC,
        candidate_source=BROKEN_SRC,
        entry_point="add",
        test_suite=SUITE,
        golden_cases=GOLDEN_CASES,
    )
    foreign = GateOutcome(
        verdict="pass",  # a verdict outside the code gate's taxonomy
        failure_detail=None,
        visible_passed=1,
        visible_total=1,
        properties_passed=1,
        properties_total=1,
    )
    with pytest.raises(ValueError):
        gate_result_from_outcome(foreign, gate_input=gate_input)


def test_wiring_requires_the_real_engine(tmp_path: Path) -> None:
    wiring = make_wiring(tmp_path, fixing_applier)

    class RetrievalOnly:
        def retrieve(self, task_context: str, k: int = 5) -> list[Any]:
            return []

        def propose(self, draft: Any) -> Any:
            raise AssertionError("not reached")

    with pytest.raises(TypeError, match="git-backed Engine"):
        finish_promotion(wiring, RetrievalOnly(), None, [], [])


def test_candidate_recording_skips_missing_files(tmp_path: Path) -> None:
    wiring = make_wiring(tmp_path, fixing_applier)
    task = make_task(tmp_path)
    config = make_config(tmp_path, "run_a")
    outcome = GateOutcome(
        verdict="fail",
        failure_detail="extraction failed: JSONDecodeError",
        visible_passed=0,
        visible_total=0,
        properties_passed=0,
        properties_total=0,
    )
    recorded = record_candidate_run(
        wiring, task, config, outcome, None, generation=0, candidate_index=0
    )
    assert recorded is None


# ------------------------------------------------- the CLI surfaces


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "evogine.yaml"
    config.write_text(
        "\n".join(
            [
                f"runs_dir: {tmp_path / 'runs'}",
                f"engine_root: {tmp_path / 'engine'}",
                "executor: local",
                "model:",
                "  provider: scripted",
                "  name: scripted-model",
                "generations: 1",
                "candidates_per_generation: 1",
            ]
        )
        + "\n"
    )
    return config


CASE_ID = "counter-overflow-wrap"


def test_evolve_runs_the_wired_loop_end_to_end(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    """The full loop under the deterministic scripted model: recorded runs,
    a queued learning, an honest rejection, and the audit trail — no API."""
    config = _write_config(tmp_path)
    assert main(["evolve", "--case", CASE_ID, "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["case_id"] == CASE_ID
    promotion = report["promotion"]
    # The scripted no-op repair is honestly rejected — evidence, not vibes.
    assert promotion["decision"] == "ephemeral"
    assert promotion["learning_id"] is not None
    assert len(promotion["queued_run_ids"]) == 2  # baseline + candidate

    run_log = Path(report["run_log_path"])
    entries = [json.loads(line) for line in run_log.read_text().splitlines()]
    assert {entry["run_id"] for entry in entries} == set(promotion["queued_run_ids"])

    engine = Engine(Path(report["engine_root"]))
    learning_id = promotion["learning_id"]
    assert engine.get(learning_id).status == "ephemeral"
    assert len(engine.audit.find("promote-rejected", learning_id)) == 1


def test_run_cli_still_writes_no_run_log(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The benchmark surface stays unwired: ``run`` never grows a run log."""
    config = _write_config(tmp_path)
    assert main(["run", "--case", CASE_ID, "--arm", "warm", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "promotion" not in report["arms"]["warm"]
    assert not (tmp_path / "engine" / "runlog.jsonl").exists()
