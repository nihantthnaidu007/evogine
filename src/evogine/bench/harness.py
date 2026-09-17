"""Warm-versus-cold benchmark harness wiring the REAL loop end to end.

The harness runs each benchmark case once per selected arm — cold (retrieval
suppressed: no engine knowledge reaches the builder) and warm (promoted
knowledge retrieved for the task context) — by delegating every arm to the
swarm task loop (:func:`evogine.swarm.runner.run_task`).  There is no separate
inline generation/evaluation path anymore: the harness bridges the benchmark
seams onto the loop and the loop does the work.

Injection points, explicitly:

- **Engine retrieval** — the ``engine`` argument serves the warm arm; the
  cold arm retrieves through :class:`NullEngine`, which is structurally
  empty and lands no proposals rather than silently succeeding.  A warm arm
  without an engine raises rather than degrading into a second cold arm.
- **Candidate generation** — ``config.builder`` behind the benchmark
  :class:`ModelClient` interface, bridged to the swarm roles' protocol
  (:class:`SwarmModelBridge`); unit tests run scripted fakes and CI makes no
  API calls.
- **Gate invocation** — the optional ``gate`` argument; the default
  :class:`~evogine.swarm.gates_adapter.CodeGateAdapter` routes every
  candidate through the sandboxed code gate (pytest, oracle-leak scan,
  differential equivalence, never-worse) with the Docker executor.  Pass a
  local-executor adapter in trusted test runs.

Every rollout — baseline, candidates, verifier reviews, reflector
proposals — is appended to ``runs/rollouts.jsonl`` in the phase0 schema by
the loop itself, tagged with the arm that produced it.

Determinism follows the learn_0012 seed playbook: thinking off, temperature
pinned, and a per-record seed derived and recorded on every rollout.  Arms of
the same run share ``run_id``, so cold and warm candidates at the same
``(generation, candidate_index)`` are seed-paired and only retrieval differs.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

from evogine.bench.task import BenchTask, load_task
from evogine.engine.records import LearningRecord
from evogine.swarm.gates_adapter import CodeGateAdapter
from evogine.swarm.roles import ModelConfig as SwarmModelConfig
from evogine.swarm.roles import ModelResponse as SwarmModelResponse
from evogine.swarm.runner import (
    BaselineReport,
    CandidateGate,
    EngineLike,
    LoopReport,
    SwarmConfig,
    SwarmTask,
    extract_candidate_files,  # noqa: F401 — re-exported
    run_task,
    serialize_candidate_files,  # noqa: F401 — re-exported
)

Arm = Literal["cold", "warm"]

#: The determinism playbook recorded on every rollout's config block.
PLAYBOOK = "learn_0012"


def derive_seed(run_id: str, case_id: str, generation: int, candidate_index: int) -> str:
    """Derive the per-record seed (learn_0012): sha256 of the record identity.

    Kept for benchmark-report callers; the loop derives and records the
    per-record seed itself (a shorter digest of the same identity).
    """
    payload = f"{run_id}|{case_id}|{generation}|{candidate_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ModelConfig:
    """Determinism block sent with every model call (learn_0012 playbook)."""

    model: str
    temperature: float
    thinking: bool
    seed: str


@dataclass(frozen=True)
class ModelResponse:
    """What a ModelClient returns: candidate code plus token accounting."""

    extracted_code: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelClient(Protocol):
    """The builder seam: every model call goes through this interface, so unit
    tests run scripted fakes and CI makes no API calls."""

    def generate(
        self, *, messages: list[dict[str, str]], model_config: ModelConfig
    ) -> ModelResponse: ...


class Retriever(Protocol):
    """What the harness needs of the engine: promoted-record retrieval."""

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]: ...


class StorelessProposalError(RuntimeError):
    """Raised when a proposal lands on an engine that backs no store."""


class NullEngine:
    """Cold-arm suppression: retrieves nothing and backs no proposal store.

    ``propose`` raising is the honest behavior — the loop records the
    reflector's draft as rejected instead of inventing a landing place.
    """

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        return []

    def propose(self, draft: LearningRecord | Mapping[str, Any]) -> LearningRecord:
        raise StorelessProposalError("the cold arm backs no proposal store")


@dataclass(frozen=True)
class HarnessConfig:
    """Harness configuration; defaults follow the learn_0012 playbook."""

    builder: ModelClient
    model: str = "unset-model"
    generations: int = 1
    candidates_per_generation: int = 4
    temperature: float = 0.0
    thinking: bool = False
    arms: tuple[Arm, ...] = ("cold", "warm")
    retrieval_k: int = 5
    runs_dir: Path = Path("runs")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def model_config(self, seed: str) -> ModelConfig:
        return ModelConfig(
            model=self.model, temperature=self.temperature, thinking=self.thinking, seed=seed
        )


class SwarmModelBridge:
    """Present the benchmark builder behind the swarm roles' ModelClient.

    The bench protocol's reply carries the extracted candidate; the roles'
    protocol wants the raw reply text their parsers consume.  Token accounting
    flows through; ``max_tokens`` is the roles' sizing config and has no bench
    counterpart to forward.
    """

    def __init__(self, builder: ModelClient) -> None:
        self._builder = builder

    def generate(
        self, *, messages: list[dict[str, str]], model_config: SwarmModelConfig
    ) -> SwarmModelResponse:
        response = self._builder.generate(
            messages=messages,
            model_config=ModelConfig(
                model=model_config.model,
                temperature=model_config.temperature,
                thinking=model_config.thinking,
                seed=model_config.seed,
            ),
        )
        return SwarmModelResponse(
            text=response.extracted_code,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


class _StorelessEngine:
    """Retrieval-only EngineLike: proposals are recorded as unlandable."""

    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        return self._retriever.retrieve(task_context, k)

    def propose(self, draft: LearningRecord | Mapping[str, Any]) -> LearningRecord:
        raise StorelessProposalError(
            "engine exposes retrieval only; the reflector's draft cannot land"
        )


def _engine_like(engine: Retriever) -> EngineLike:
    """Adapt the harness's Retriever seam to the loop's EngineLike.

    A real Engine (retrieve + propose) passes through unchanged; a
    retrieval-only retriever is wrapped so a failed arm records its reflector
    draft as unlandable rather than dropping it.
    """
    if callable(getattr(engine, "propose", None)):
        # The runtime duck-check above is the test; the cast asserts it.
        return cast(EngineLike, engine)
    return _StorelessEngine(engine)


def to_swarm_task(task: BenchTask) -> SwarmTask:
    """Project the benchmark task onto the loop's SwarmTask (same fields)."""
    return SwarmTask(
        case_id=task.case_id,
        description=task.description,
        scenario_brief=task.scenario_brief,
        baseline_files=dict(task.baseline_files),
        decay_axis=task.decay_axis,
        layout=task.layout,
        suite_path=task.suite_path,
        properties=dict(task.properties),
        reference_files=dict(task.reference_files),
    )


def _swarm_config(config: HarnessConfig) -> SwarmConfig:
    """Map the harness config onto the loop's; one model name for all roles."""
    return SwarmConfig(
        builder_model=config.model,
        verifier_model=config.model,
        reflector_model=config.model,
        generations=config.generations,
        candidates_per_generation=config.candidates_per_generation,
        temperature=config.temperature,
        thinking=config.thinking,
        retrieval_k=config.retrieval_k,
        runs_dir=Path(config.runs_dir),
        run_id=config.run_id,
    )


class ArmReport(LoopReport):
    """One arm's loop report, labeled with the arm that produced it."""

    arm: str


class BenchReport(TypedDict):
    """Top-level benchmark report: paired arm reports over one case.

    Serialized verbatim by the CLI's ``run`` command, so every key here is a
    public contract of the benchmark.
    """

    run_id: str
    case_id: str
    decay_axis: str
    baseline: BaselineReport
    arms: dict[str, ArmReport]
    rollouts_path: str


def run(
    task: BenchTask,
    engine: Retriever | None,
    config: HarnessConfig,
    gate: CandidateGate | None = None,
) -> BenchReport:
    """Run one benchmark task through the real loop, once per selected arm.

    Each arm delegates to :func:`evogine.swarm.runner.run_task` — retrieval,
    builder, verifier, the sandboxed code gate, retention, and the reflector
    all run for real; rollouts land in the loop's phase0 file tagged with the
    arm.  The report maps each arm to its loop report, with the degraded
    baseline and the rollouts path surfaced at the top level.
    """
    if not config.arms:
        raise ValueError("harness config selects no arms")
    swarm_task = to_swarm_task(task)
    swarm_config = _swarm_config(config)
    active_gate = gate if gate is not None else CodeGateAdapter()
    model = SwarmModelBridge(config.builder)

    # Resolve each arm's retriever up front so the warm guard lives in one
    # place: a warm arm without an engine raises here, and the loop never
    # sees an Optional.
    retrievers: dict[str, Retriever] = dict.fromkeys(config.arms, NullEngine())
    if "warm" in retrievers:
        if engine is None:
            raise ValueError("the warm arm requires an engine; pass one or drop 'warm' from arms")
        retrievers["warm"] = engine

    arms_report: dict[str, ArmReport] = {}
    first: ArmReport | None = None
    for arm in config.arms:
        report = run_task(
            swarm_task,
            _engine_like(retrievers[arm]),
            model,
            active_gate,
            replace(swarm_config, arm=arm),
        )
        arms_report[arm] = {**report, "arm": arm}
        if first is None:
            first = arms_report[arm]

    if first is None:  # pragma: no cover — arms non-empty is checked above
        raise RuntimeError("unreachable: no arm reports were produced")
    return {
        "run_id": config.run_id,
        "case_id": task.case_id,
        "decay_axis": task.decay_axis,
        "baseline": first["baseline"],
        "arms": arms_report,
        "rollouts_path": first["rollouts_path"],
    }


def load_benchmark_case(case_id: str) -> BenchTask:
    """Load a case as a benchmark task (kept here so callers need one import)."""
    return load_task(case_id)
