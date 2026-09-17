"""Harness wiring tests: the benchmark arms run through the REAL loop.

Every test drives ``bench.harness.run`` — the swarm runner, not an inline
evaluation path — so the wiring under test is the production one:

- scripted builder replies (the existing FakeModel approach) routed through
  :class:`~evogine.swarm.gates_adapter.CodeGateAdapter` with the local
  sandbox — tiny deterministic candidates, no Docker, no API calls;
- a spy gate proving the harness's arm selection reaches the adapter seam;
- cold suppression and warm retrieval against a real, seeded engine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from evogine.bench.harness import (
    Arm,
    HarnessConfig,
    ModelConfig,
    ModelResponse,
    NullEngine,
    StorelessProposalError,
    SwarmModelBridge,
    run,
)
from evogine.bench.task import load_task
from evogine.engine.bootstrap import load_seeds
from evogine.engine.store import Engine
from evogine.gates.sandbox import LocalSubprocessSandbox
from evogine.swarm.gates_adapter import CodeGateAdapter
from evogine.swarm.roles import ModelConfig as SwarmModelConfig
from evogine.swarm.roles import ModelResponse as SwarmModelResponse
from evogine.swarm.runner import GateOutcome, SwarmTask, serialize_candidate_files

SEEDS_DIR = Path(__file__).resolve().parents[1] / "src" / "evogine" / "seeds"

CASE_ID = "counter-overflow-wrap"

#: The header the loop's builder prompt uses for retrieved knowledge.
KNOWLEDGE_HEADER = "Promoted knowledge to apply:"

REFLECTOR_JSON = json.dumps(
    {
        "type": "failure-pattern",
        "title": "Scripted harness run: no learning to record",
        "confidence": 0.3,
        "body": {
            "signal": "The scripted candidate did not change the outcome",
            "failure_mode": "Smoke run without a real model client",
            "detection": "Gate verdict on the case",
            "fix_strategy": "Run with a real model client for meaningful repairs",
        },
    }
)

VERIFIER_PASS = "VERDICT: pass\nLooks consistent with the visible contract."


class ScriptedBenchModel:
    """Deterministic bench ModelClient routed on the role system prompts.

    The builder reply is a fixed candidate tree (serialized files); the
    verifier passes so candidates always reach the gate; the reflector files
    a minimal draft.  Builder system prompts are captured so tests can assert
    exactly what knowledge each arm injected.
    """

    def __init__(self, builder_reply: str) -> None:
        self._builder_reply = builder_reply
        self.builder_prompts: list[str] = []
        self.seeds: list[str] = []

    def generate(
        self, *, messages: list[dict[str, str]], model_config: ModelConfig
    ) -> ModelResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        self.seeds.append(model_config.seed)
        if system.startswith("You are the repair builder"):
            # Knowledge renders into the user message, not the system prompt.
            self.builder_prompts.append("\n".join(m["content"] for m in messages))
            return ModelResponse(extracted_code=self._builder_reply)
        if system.startswith("You are the independent verifier"):
            return ModelResponse(extracted_code=VERIFIER_PASS)
        if system.startswith("You are the reflector"):
            return ModelResponse(extracted_code=REFLECTOR_JSON)
        raise RuntimeError("scripted model received an unrecognized role prompt")


class RecordingGate:
    """Spy CandidateGate: records calls, accepts everything.

    Because the runner routes the baseline AND every candidate through the
    gate, the call count pins the arm wiring exactly.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], str]] = []

    def evaluate(self, files: Mapping[str, str], task: SwarmTask) -> GateOutcome:
        self.calls.append((dict(files), task.case_id))
        return GateOutcome(
            verdict="pass_tests",
            failure_detail=None,
            visible_passed=1,
            visible_total=1,
            properties_passed=1,
            properties_total=1,
        )


def _harness_config(
    model: ScriptedBenchModel,
    tmp_path: Path,
    *,
    arms: tuple[Arm, ...] = ("cold",),
    runs_dir: Path | None = None,
) -> HarnessConfig:
    return HarnessConfig(
        builder=model,
        model="scripted",
        generations=1,
        candidates_per_generation=1,
        arms=arms,
        runs_dir=runs_dir if runs_dir is not None else tmp_path / "runs",
    )


def _rollout_arms(path: Path) -> list[str]:
    return [json.loads(line)["config"]["arm"] for line in path.read_text().splitlines()]


def test_reference_candidate_flows_through_real_gate_and_is_retained(
    tmp_path: Path,
) -> None:
    """Builder reply -> runner -> real sandboxed gate -> retained candidate."""
    task = load_task(CASE_ID)
    model = ScriptedBenchModel(serialize_candidate_files(task.reference_files))
    config = _harness_config(model, tmp_path)
    report = run(
        task,
        None,
        config,
        CodeGateAdapter(LocalSubprocessSandbox()),
    )
    arm_report = report["arms"]["cold"]
    assert report["case_id"] == CASE_ID
    assert arm_report["retained"] == "candidate"
    assert arm_report["best"] is not None
    assert arm_report["candidates"][0]["verdict"] == "pass_tests"
    # The baseline still flowed through the loop and was recorded.
    assert arm_report["baseline"]["verdict"]


def test_noop_candidate_falls_back_to_baseline(tmp_path: Path) -> None:
    """A no-op repair (degraded baseline re-emitted) is rejected; baseline wins."""
    task = load_task(CASE_ID)
    model = ScriptedBenchModel(serialize_candidate_files(task.baseline_files))
    config = _harness_config(model, tmp_path)
    report = run(
        task,
        None,
        config,
        CodeGateAdapter(LocalSubprocessSandbox()),
    )
    arm_report = report["arms"]["cold"]
    assert arm_report["retained"] == "baseline"
    assert arm_report["best"] is None
    assert arm_report["candidates"][0]["verdict"] == "fail_equivalence"


def test_cold_arm_suppresses_retrieval(tmp_path: Path) -> None:
    """Even with a seeded engine, the cold arm's builder sees no knowledge."""
    task = load_task(CASE_ID)
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    model = ScriptedBenchModel(serialize_candidate_files(task.reference_files))
    config = _harness_config(model, tmp_path)
    report = run(
        task,
        engine,
        config,
        CodeGateAdapter(LocalSubprocessSandbox()),
    )
    arm_report = report["arms"]["cold"]
    assert arm_report["retrieved_records"] == 0
    assert model.builder_prompts
    assert all(KNOWLEDGE_HEADER not in prompt for prompt in model.builder_prompts)


def test_warm_arm_retrieves_seeded_knowledge(tmp_path: Path) -> None:
    """The warm arm injects promoted records into the builder prompt."""
    task = load_task(CASE_ID)
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    model = ScriptedBenchModel(serialize_candidate_files(task.reference_files))
    config = _harness_config(model, tmp_path, arms=("warm",))
    report = run(
        task,
        engine,
        config,
        CodeGateAdapter(LocalSubprocessSandbox()),
    )
    arm_report = report["arms"]["warm"]
    assert arm_report["retrieved_records"] > 0
    assert any(KNOWLEDGE_HEADER in prompt for prompt in model.builder_prompts)


def test_arm_selection_reaches_the_adapter(tmp_path: Path) -> None:
    """Both arms run the loop: the gate sees baseline + candidate per arm."""
    task = load_task(CASE_ID)
    model = ScriptedBenchModel(serialize_candidate_files(task.reference_files))
    config = _harness_config(
        model,
        tmp_path,
        arms=("cold", "warm"),
        runs_dir=tmp_path / "shared-runs",
    )
    gate = RecordingGate()
    report = run(task, Engine(tmp_path / "engine"), config, gate)
    assert set(report["arms"]) == {"cold", "warm"}
    assert report["rollouts_path"] == str(tmp_path / "shared-runs" / "rollouts.jsonl")
    # The runner routes the baseline and each candidate through the seam:
    # 2 calls per arm for 1 generation x 1 candidate.
    assert len(gate.calls) == 4
    assert {case_id for _, case_id in gate.calls} == {CASE_ID}
    # Both arms' records land in the one shared phase0 file, honestly labeled.
    arms_seen = _rollout_arms(Path(str(report["rollouts_path"])))
    assert "baseline" in arms_seen
    assert "cold" in arms_seen
    assert "warm" in arms_seen


def test_warm_arm_without_engine_raises(tmp_path: Path) -> None:
    """A warm arm without an engine raises instead of degrading to a cold arm."""
    task = load_task(CASE_ID)
    config = _harness_config(ScriptedBenchModel("code"), tmp_path, arms=("warm",))
    try:
        run(task, None, config, RecordingGate())
    except ValueError as exc:
        assert "warm arm requires an engine" in str(exc)
    else:
        raise AssertionError("expected ValueError for warm arm without engine")


def test_empty_arms_raise(tmp_path: Path) -> None:
    task = load_task(CASE_ID)
    config = _harness_config(ScriptedBenchModel("code"), tmp_path, arms=())
    try:
        run(task, None, config, RecordingGate())
    except ValueError as exc:
        assert "no arms" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty arms")


def test_null_engine_retrieves_nothing_and_lands_no_proposal() -> None:
    engine = NullEngine()
    assert engine.retrieve("context", k=3) == []
    try:
        engine.propose({"type": "failure-pattern"})
    except StorelessProposalError as exc:
        assert "no proposal store" in str(exc)
    else:
        raise AssertionError("expected StorelessProposalError from NullEngine.propose")


def test_bridge_maps_bench_protocol_to_swarm_protocol() -> None:
    captured: dict[str, ModelConfig] = {}

    class StubBenchModel:
        def generate(
            self, *, messages: list[dict[str, str]], model_config: ModelConfig
        ) -> ModelResponse:
            captured["config"] = model_config
            return ModelResponse(extracted_code="CODE", input_tokens=3, output_tokens=4)

    bridge = SwarmModelBridge(StubBenchModel())
    response = bridge.generate(
        messages=[{"role": "user", "content": "hi"}],
        model_config=SwarmModelConfig(
            model="m", temperature=0.1, thinking=True, seed="s", max_tokens=99
        ),
    )
    assert isinstance(response, SwarmModelResponse)
    assert response.text == "CODE"
    assert response.input_tokens == 3
    assert response.output_tokens == 4
    # Latency is measured at the bridge (the bench protocol carries none).
    assert response.latency_ms >= 0.0
    # The determinism block flows through; max_tokens stays the roles' concern.
    assert captured["config"].model == "m"
    assert captured["config"].temperature == 0.1
    assert captured["config"].thinking is True
    assert captured["config"].seed == "s"
