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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import CaptureFixture

from evogine.bench.harness import ModelResponse as BenchModelResponse
from evogine.bench.harness import SwarmModelBridge, to_swarm_task
from evogine.bench.task import load_task
from evogine.cli import ScriptedModel, main
from evogine.engine.records import (
    LearningRecord,
    RecordBody,
    RecordCounterfactual,
    RecordProvenance,
)
from evogine.engine.store import Engine
from evogine.gates import sandbox as sbx
from evogine.gates.code_gate import CodeGate, GateInput, GateResult, GateVerdict, GoldenCase
from evogine.gates.memory_gate import MemoryGate
from evogine.gates.replay import AppliedPatch, RecordedRun, RunLog, replay_one
from evogine.swarm.gates_adapter import CodeGateAdapter
from evogine.swarm.promotion import (
    BuilderApplier,
    PromotionWiring,
    finish_promotion,
    gate_result_from_outcome,
    record_candidate_run,
    recorded_run_id,
)
from evogine.swarm.roles import BUILDER_PACKAGE_NOTE, Builder, ModelClient, ModelResponse
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

    def generate(self, *, messages: list[dict[str, str]], model_config: Any) -> ModelResponse:
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
    report = run_task(task, engine, ScriptedRoles(builder_reply=BROKEN_SRC), TaxonomyGate(), config)
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


def test_evolve_runs_the_wired_loop_end_to_end(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
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


# ------------------------------------------------- the package-layout loop
#
# The quality audit's two highest-value coverage holes: no test drives a
# multi-file package candidate through the wired loop (runner extraction,
# gates_adapter staging, the production applier's re-keying all untested),
# and the production BuilderApplier's success paths never execute — every
# promotion test above stubs the applier by hand. These close both, on the
# real corpus case and the trusted local executor.


@dataclass
class RepairThenApplyScriptedModel:
    """A scripted builder whose repair reply and apply reply differ.

    The CLI's :class:`ScriptedModel` routes both the repair builder's propose
    call and the learning applier's apply call to one ``builder_reply`` — both
    role prompts open "You are the repair builder". A package-layout promotion
    needs them to differ: the repair builder re-emits the degraded tree (the
    gate rejects it, so the reflector proposes), while the applier returns the
    FIXED tree so the warm replay can flip. The applier call is distinguished
    by its prompt's unique "You are applying" clause.
    """

    repair_reply: str
    apply_reply: str
    apply_prompts: list[dict[str, str]] = field(default_factory=list)

    def generate(self, *, messages: list[dict[str, str]], model_config: Any) -> BenchModelResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if "You are applying" in system:
            self.apply_prompts.append(dict(messages[0]))
            return BenchModelResponse(extracted_code=self.apply_reply)
        return ScriptedModel(builder_reply=self.repair_reply).generate(
            messages=messages, model_config=model_config
        )


class CapturingModel:
    """A roles-level ModelClient that records every prompt it is handed."""

    def __init__(self, inner: ModelClient) -> None:
        self._inner = inner
        self.prompts: list[list[dict[str, str]]] = []

    def generate(self, *, messages: list[dict[str, str]], model_config: Any) -> ModelResponse:
        self.prompts.append([dict(m) for m in messages])
        return self._inner.generate(messages=messages, model_config=model_config)


# A minimal package unit for the applier unit tests. Recorded runs key
# candidate files by unprefixed unit path (the staging convention), so the
# fixtures mirror that shape rather than any bench case.
UNIT_BASELINE = {
    "__init__.py": "from .ledger import fold\n\n__all__ = ['fold']\n",
    "ledger.py": (
        "def fold(ops):\n    last = 0\n    for op in ops:\n        last = op[1]\n    return last\n"
    ),
}
UNIT_FIXED = {
    "__init__.py": UNIT_BASELINE["__init__.py"],
    "ledger.py": "def fold(ops):\n    return sorted(ops, key=lambda op: op[0])[-1][1]\n",
}


def make_learning() -> LearningRecord:
    """A queued learning record, shaped like the reflector's proposals."""
    return LearningRecord(
        id="learn_pkg",
        type="failure-pattern",
        title="fold pending writes by sequence number",
        body=RecordBody(
            signal="reconcile folds writes in arrival order",
            failure_mode="last-arrival-wins instead of last-sequence-wins",
            detection="equivalence witness on the reorder scenario",
            fix_strategy="sort the pending writes by sequence before folding",
        ),
        status="candidate",
        confidence=0.9,
        provenance=RecordProvenance(
            proposed_by="reflector",
            origin="swarm-runner",
            source="test",
            note="promotion-loop test fixture",
        ),
        counterfactual=RecordCounterfactual(),
        created=datetime(2026, 1, 1, tzinfo=UTC),
    )


def make_single_file_run() -> RecordedRun:
    """A recorded single-function run (the add fixture family)."""
    return RecordedRun(
        run_id="run_a::warm::case_add::candidate::generation-0::candidate-0",
        case_id="case_add",
        gate_input=GateInput.single_function(
            baseline_source=BASELINE_SRC,
            candidate_source=BROKEN_SRC,
            entry_point="add",
            test_suite=SUITE,
            golden_cases=GOLDEN_CASES,
        ),
        cold=GateResult(
            verdict=GateVerdict.FAIL_TESTS,
            accepted=False,
            gate="tests",
            reason="add returns a - b",
        ),
    )


def make_multi_file_run() -> RecordedRun:
    """A recorded multi-file run: candidate files keyed by unit path."""
    return RecordedRun(
        run_id="run_pkg::warm::case_pkg::candidate::generation-0::candidate-0",
        case_id="case_pkg",
        gate_input=GateInput.multi_file(
            baseline_files=dict(UNIT_BASELINE),
            candidate_files=dict(UNIT_BASELINE),
            entry_class="Ledger",
            corpus_source="def build_corpus():\n    return []\n",
        ),
        cold=GateResult(
            verdict=GateVerdict.FAIL_EQUIVALENCE,
            accepted=False,
            gate="equivalence",
            reason="reorder scenario diverged from the reference",
        ),
    )


@dataclass
class RecordingGate:
    """A gate stub for replay: records the warm inputs it was handed."""

    accepted: bool = True
    seen: list[GateInput] = field(default_factory=list)

    def evaluate(self, gate_input: GateInput, *, timeout: float = 10.0) -> GateResult:
        self.seen.append(gate_input)
        return GateResult(
            verdict=GateVerdict.PASS_TESTS if self.accepted else GateVerdict.FAIL_TESTS,
            accepted=self.accepted,
            gate=None if self.accepted else "tests",
            reason="",
        )


def test_builder_applier_single_file_reply_becomes_the_module_patch() -> None:
    """The single-file success path: the scripted reply text IS the patch.

    Every promotion-loop test above stubs the applier; this drives the
    production :class:`BuilderApplier` through the real Builder + scripted
    model seam the CLI wires.
    """
    builder = Builder(SwarmModelBridge(ScriptedModel(builder_reply=GOOD_SRC)), "scripted-model")
    patch = BuilderApplier(builder)(make_learning(), make_single_file_run())
    # extract_code strips the reply text; the patch carries the stripped module.
    assert patch == AppliedPatch(candidate_files={"candidate.py": GOOD_SRC.strip()})


def test_builder_applier_single_file_treats_any_reply_as_source() -> None:
    """Single-file asymmetry: a reply cannot fail to parse — the text IS the
    module, and the replayed gate (not the parser) judges it."""
    builder = Builder(
        SwarmModelBridge(ScriptedModel(builder_reply="return None  # not code")),
        "scripted-model",
    )
    patch = BuilderApplier(builder)(make_learning(), make_single_file_run())
    assert patch is not None
    assert patch.candidate_files == {"candidate.py": "return None  # not code"}


def test_builder_applier_multi_file_reply_is_rekeyed_and_fed_to_replay() -> None:
    """The multi-file success path: parse the mapping, re-key it via
    ``_unit_path`` onto the recorded run's unit paths, and hand the patch to
    replay unchanged — where the cold-rejected run flips to accepted."""
    scripted_reply = json.dumps(
        {
            "files": {
                "candidate/__init__.py": UNIT_FIXED["__init__.py"],  # prefixed: re-keyed
                "ledger.py": UNIT_FIXED["ledger.py"],  # unprefixed: kept
            }
        }
    )
    builder = Builder(
        SwarmModelBridge(ScriptedModel(builder_reply=scripted_reply)), "scripted-model"
    )
    run = make_multi_file_run()
    learning = make_learning()

    patch = BuilderApplier(builder)(learning, run)
    assert patch == AppliedPatch(candidate_files=UNIT_FIXED)

    gate = RecordingGate(accepted=True)
    # RecordingGate implements exactly the evaluate seam replay_one uses
    # (same signature as CodeGate.evaluate) but is a stub, not a CodeGate
    # subclass — the cast documents that seam without wiring a real executor.
    result = replay_one(
        run, learning, BuilderApplier(builder), cast(CodeGate, gate), repeats=2
    )
    assert result.applied is True
    assert result.warm_verdicts == ("pass_tests", "pass_tests")
    assert result.material_flip is True  # cold rejected, warm accepted, MAD 0
    # The re-keyed patch — not the raw reply keys — is what replay evaluated.
    assert gate.seen[0].candidate_files == UNIT_FIXED


def test_builder_applier_prompts_the_recorded_candidate_multi_file() -> None:
    """The applier's builder call renders the recorded unit files and the
    multi-file reply contract — the prompt seam every warm repeat rides on."""
    inner = CapturingModel(
        SwarmModelBridge(ScriptedModel(builder_reply=json.dumps({"files": UNIT_FIXED})))
    )
    BuilderApplier(Builder(inner, "scripted-model"))(make_learning(), make_multi_file_run())

    assert len(inner.prompts) == 1
    system = inner.prompts[0][0]["content"]
    user = inner.prompts[0][1]["content"]
    assert "You are applying" in system
    # multi_file=True: the JSON reply contract rides in the user template.
    assert BUILDER_PACKAGE_NOTE in user
    assert UNIT_BASELINE["ledger.py"] in user


def test_builder_applier_rejects_malformed_multi_file_replies() -> None:
    """The mapping-validation rejection path: a reply that cannot parse into a
    string-keyed mapping returns ``None`` — replay records ``not_applied``,
    the failure surfaces in the evidence, nothing crashes or fabricates."""
    bad_replies = [
        "def not_json():\n    return None\n",  # no JSON at all
        '["not", "an", "object"]',  # JSON, not an object
        '{"verdict": "fine"}',  # missing the "files" key
        '{"files": ["ledger.py"]}',  # files is a list
        '{"files": {"ledger.py": 404}}',  # non-string source
        '{"files": {"ledger.py": null}}',  # non-string source (None)
    ]
    for reply in bad_replies:
        builder = Builder(SwarmModelBridge(ScriptedModel(builder_reply=reply)), "scripted-model")
        patch = BuilderApplier(builder)(make_learning(), make_multi_file_run())
        assert patch is None, reply


def test_package_candidate_drives_the_full_promotion_loop(tmp_path: Path) -> None:
    """A multi-file package candidate through the whole wired loop, on a real
    corpus case — the audit's untested path end to end.

    The scripted repair builder re-emits the degraded tree with unprefixed
    unit paths: the runner's package extraction normalizes them under
    ``candidate/``, the gate adapter's staging strips the prefix back off, and
    the real code gate rejects the candidate on equivalence. The reflector
    proposes, and the memory gate replays through the PRODUCTION
    BuilderApplier, whose scripted apply reply carries ``candidate/``-prefixed
    keys — the re-keyed patch must land on the recorded run's unit paths for
    the warm arm to accept the fix and promote the learning.
    """
    task = to_swarm_task(load_task("concurrency-account-ledger"))
    assert task.layout == "package"
    engine = Engine(tmp_path / "engine")
    run_log = RunLog(tmp_path / "engine" / "runlog.jsonl")
    model = RepairThenApplyScriptedModel(
        repair_reply=json.dumps(
            {
                "files": {
                    # Unprefixed: the runner's extraction re-prefixes under candidate/.
                    name.removeprefix("candidate/"): source
                    for name, source in task.baseline_files.items()
                }
            }
        ),
        apply_reply=json.dumps(
            {
                # Prefixed: the production applier must re-key onto unit paths.
                "files": dict(sorted(task.reference_files.items()))
            }
        ),
    )
    builder = Builder(SwarmModelBridge(model), "scripted-model")
    wiring = PromotionWiring(
        run_log=run_log,
        memory_gate=MemoryGate(run_log=run_log, applier=BuilderApplier(builder), executor=LOCAL),
    )

    report = run_task(
        task,
        engine,
        SwarmModelBridge(model),
        CodeGateAdapter(executor=LOCAL),
        make_config(tmp_path, "run_pkg"),
        promotion=wiring,
    )

    # Extraction + staging: the candidate run recorded the unprefixed unit
    # tree, cold-rejected by the real gate on equivalence.
    promotion = report["promotion"]
    assert promotion["queued_run_ids"] == [
        "run_pkg::warm::concurrency-account-ledger::baseline",
        "run_pkg::warm::concurrency-account-ledger::candidate::generation-0::candidate-0",
    ]
    candidate_run = run_log.get(promotion["queued_run_ids"][1])
    assert candidate_run.gate_input.scope == "multi-file"
    assert set(candidate_run.gate_input.candidate_files) == {
        "__init__.py",
        "account.py",
        "ledger.py",
    }
    assert candidate_run.cold.verdict is GateVerdict.FAIL_EQUIVALENCE

    # The production applier re-keys the prefixed reply onto unit paths.
    learning_id = promotion["learning_id"]
    assert learning_id is not None
    learning = engine.get(learning_id)
    patch = BuilderApplier(builder)(learning, candidate_run)
    assert patch == AppliedPatch(
        candidate_files={
            name.removeprefix("candidate/"): source for name, source in task.reference_files.items()
        }
    )
    # The applier prompt carried the multi-file reply contract — the seam the
    # warm repeats rode on.
    assert model.apply_prompts
    assert "You are applying" in model.apply_prompts[0]["content"]

    # The decision is measured, not asserted: both replayed runs flipped
    # fail -> pass under the re-keyed patch, above a zero floor.
    assert promotion["decision"] == "promoted"
    record = engine.get(learning_id)
    assert record.status == "promoted"
    assert record.counterfactual.verdict_change is True
    assert record.counterfactual.replay_run_ids == promotion["queued_run_ids"]
    evidence = json.loads(record.counterfactual.evidence or "{}")
    per_run = {entry["run_id"]: entry for entry in evidence["per_run"]}
    assert all(entry["material_flip"] is True for entry in per_run.values())
    assert per_run[candidate_run.run_id]["warm_verdicts"] == ["pass_tests"] * 3


def test_run_cli_still_writes_no_run_log(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """The benchmark surface stays unwired: ``run`` never grows a run log."""
    config = _write_config(tmp_path)
    assert main(["run", "--case", CASE_ID, "--arm", "warm", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "promotion" not in report["arms"]["warm"]
    assert not (tmp_path / "engine" / "runlog.jsonl").exists()
