"""Warm-versus-cold benchmark harness with variance-aware reporting.

The harness runs each benchmark case twice — cold (retrieval suppressed:
no engine knowledge reaches the builder) and warm (promoted knowledge
retrieved for the task context) — under decay-pressure conditions such as
concurrency, retries, time zones, locale, overflow, partial failure,
integration drift, and load spikes. Hidden host-side invariants and
oracle-leak protection live in the case definitions and the gate, not here.

Injection points, explicitly:

- **Engine retrieval** — the ``engine`` argument serves the warm arm; the
  cold arm retrieves through :class:`NullEngine`, which is structurally
  empty rather than a silent failure.
- **Candidate generation** — ``config.builder`` behind the
  :class:`ModelClient` interface; unit tests run a FakeModel, CI makes no
  API calls.
- **Gate invocation** — the optional ``gate`` argument; the default
  :class:`PropertiesGate` is the local stand-in for the sandboxed code gate.

Retention is best-of-N with a never-worse fallback: a candidate is retained
only if it passes the visible suite and scores at least as many hidden
property rows as the degraded baseline; otherwise the baseline is kept.
Every rollout — baseline and candidates alike — is appended to
``runs/rollouts.jsonl`` in the phase0 schema.

Determinism follows the learn_0012 seed playbook: thinking off, temperature
pinned, and a per-record seed ``sha256(run_id|case_id|generation|
candidate_index)`` derived and recorded on every rollout. Arms of the same
run share ``run_id``, so cold and warm candidates are seed-paired and only
retrieval differs.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from evogine.bench.gate import Gate, GateOutcome, PropertiesGate
from evogine.bench.prompts import build_builder_messages
from evogine.bench.rollouts import RolloutWriter, build_rollout
from evogine.bench.task import BenchTask, load_task
from evogine.engine.records import LearningRecord

Arm = Literal["cold", "warm"]

#: The determinism playbook recorded on every rollout's config block.
PLAYBOOK = "learn_0012"


def derive_seed(run_id: str, case_id: str, generation: int, candidate_index: int) -> str:
    """Derive the per-record seed (learn_0012): sha256 of the record identity.

    Arms of one run share ``run_id``, so the cold and warm candidate at the
    same ``(generation, candidate_index)`` are seed-paired — the comparison
    isolates retrieval as the only difference.
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
    tests run a FakeModel and CI makes no API calls."""

    def generate(
        self, *, messages: list[dict[str, str]], model_config: ModelConfig
    ) -> ModelResponse: ...


class Retriever(Protocol):
    """What the harness needs of the engine: promoted-record retrieval."""

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]: ...


class NullEngine:
    """Cold-arm retrieval suppression: returns nothing and backs no store."""

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]:
        return []


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


def extract_candidate_files(extracted_code: str, layout: str) -> dict[str, str]:
    """Turn a builder reply into candidate-tree files.

    Single-file replies are the module source itself; package replies are a
    ``{"files": {relative path: source}}`` JSON object whose paths are
    normalized under ``candidate/``.
    """
    if layout == "single":
        return {"candidate.py": extracted_code}
    if layout == "package":
        payload = json.loads(extracted_code)
        files = payload["files"]
        if not isinstance(files, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in files.items()
        ):
            raise ValueError(
                'multi-file reply must be {"files": {"<relative path>": "<source>"}}'
            )
        return {
            key if key.startswith("candidate/") else f"candidate/{key}": value
            for key, value in files.items()
        }
    raise ValueError(f"unknown case layout: {layout!r}")


def serialize_candidate_files(files: Mapping[str, str]) -> str:
    """Serialize candidate files for a rollout record's ``extracted_code``."""
    if set(files) == {"candidate.py"}:
        return files["candidate.py"]
    return json.dumps({"files": dict(sorted(files.items()))})


def run(
    task: BenchTask,
    engine: Retriever | None,
    config: HarnessConfig,
    gate: Gate | None = None,
) -> dict[str, object]:
    """Run one benchmark task cold and warm and return a variance-aware report.

    The report maps each arm to its candidate outcomes plus the retained
    result, with the degraded baseline and every rollout recorded for audit.
    A warm arm without an engine raises rather than silently degrading into
    a second cold arm.
    """
    # Resolve each arm's retriever up front so the warm guard and the type
    # narrowing live in one place: a warm arm without an engine raises here,
    # and the loop never sees an Optional.
    null_engine = NullEngine()
    retrievers: dict[str, Retriever] = dict.fromkeys(config.arms, null_engine)
    if "warm" in retrievers:
        if engine is None:
            raise ValueError(
                "the warm arm requires an engine; pass one or drop 'warm' from arms"
            )
        retrievers["warm"] = engine
    gate = gate or PropertiesGate()
    writer = RolloutWriter(Path(config.runs_dir) / "rollouts.jsonl")

    baseline_outcome = gate.evaluate(task.baseline_files, task)
    writer.append(
        build_rollout(
            stage="baseline",
            case_id=task.case_id,
            run_id=config.run_id,
            generation=None,
            candidate_index=None,
            model="baseline",
            config={"arm": "baseline", "seed": None, "playbook": PLAYBOOK},
            prompt_messages=[],
            extracted_code=serialize_candidate_files(task.baseline_files),
            verdict=baseline_outcome.verdict,
            failure_detail=baseline_outcome.failure_detail,
            ops=baseline_outcome.ops,
            tokens={"input_tokens": None, "output_tokens": None},
        )
    )

    arms_report: dict[str, object] = {}
    for arm in config.arms:
        retriever = retrievers[arm]
        records = retriever.retrieve(_task_context(task), k=config.retrieval_k)
        messages = build_builder_messages(task, records)
        candidates: list[dict[str, object]] = []
        best: dict[str, object] | None = None
        best_key: tuple[int, int, int] | None = None
        for generation in range(config.generations):
            for candidate_index in range(config.candidates_per_generation):
                seed = derive_seed(config.run_id, task.case_id, generation, candidate_index)
                response = config.builder.generate(
                    messages=messages, model_config=config.model_config(seed)
                )
                outcome = _evaluate_reply(response, task, gate)
                writer.append(
                    build_rollout(
                        stage="candidate",
                        case_id=task.case_id,
                        run_id=config.run_id,
                        generation=generation,
                        candidate_index=candidate_index,
                        model=config.model,
                        config={
                            "arm": arm,
                            "seed": seed,
                            "temperature": config.temperature,
                            "thinking": config.thinking,
                            "playbook": PLAYBOOK,
                        },
                        prompt_messages=messages,
                        extracted_code=response.extracted_code,
                        verdict=outcome.verdict,
                        failure_detail=outcome.failure_detail,
                        ops=outcome.ops,
                        tokens={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    )
                )
                eligible = (
                    outcome.visible_failures == 0
                    and outcome.properties_passed >= baseline_outcome.properties_passed
                )
                key = (outcome.properties_passed, -generation, -candidate_index)
                if eligible and (best_key is None or key > best_key):
                    best_key = key
                    best = {
                        "generation": generation,
                        "candidate_index": candidate_index,
                        "verdict": outcome.verdict,
                        "properties_passed": outcome.properties_passed,
                        "properties_total": outcome.properties_total,
                    }
                candidates.append(
                    {
                        "generation": generation,
                        "candidate_index": candidate_index,
                        "verdict": outcome.verdict,
                        "properties_passed": outcome.properties_passed,
                        "properties_total": outcome.properties_total,
                    }
                )
        arms_report[arm] = {
            "arm": arm,
            "retrieved_records": len(records),
            "candidates": candidates,
            "best": best,
            "retained": "candidate" if best is not None else "baseline",
            "baseline_properties_passed": baseline_outcome.properties_passed,
        }

    return {
        "run_id": config.run_id,
        "case_id": task.case_id,
        "decay_axis": task.decay_axis,
        "baseline": {
            "verdict": baseline_outcome.verdict,
            "visible_failures": baseline_outcome.visible_failures,
            "properties_passed": baseline_outcome.properties_passed,
            "properties_total": baseline_outcome.properties_total,
        },
        "arms": arms_report,
        "rollouts_path": str(writer.path),
    }


def load_benchmark_case(case_id: str) -> BenchTask:
    """Load a case as a benchmark task (kept here so callers need one import)."""
    return load_task(case_id)


def _evaluate_reply(response: ModelResponse, task: BenchTask, gate: Gate) -> GateOutcome:
    """Evaluate a builder reply; a reply that breaks extraction is a failing
    candidate, not a harness crash."""
    try:
        files = extract_candidate_files(response.extracted_code, task.layout)
        return gate.evaluate(files, task)
    except Exception as exc:
        return GateOutcome(
            verdict="fail",
            failure_detail=f"extraction failed: {type(exc).__name__}: {exc}",
            visible_passed=0,
            visible_total=0,
            properties_passed=0,
            properties_total=0,
        )


def _task_context(task: BenchTask) -> str:
    """The retrieval context a warm engine is queried with for this task."""
    return f"{task.case_id} {task.decay_axis} {task.description}"
