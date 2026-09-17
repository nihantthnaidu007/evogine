"""The swarm task loop: retrieve, generate, gate, retain, reflect.

Per the v1 blueprint lifecycle (``art_bGBG4tqU``), one task run:

1. **Baseline** — the degraded baseline is evaluated by the gate first; its
   property score is the never-worse bar every candidate must clear.
2. **Retrieve (warm arm)** — the engine returns promoted records for the task
   context; they are the only engine knowledge the builder ever sees.
3. **Builder candidates** — best-of-N candidates across generations, up to
   ``max_parallel_builders`` at a time; every call is a rollout record.
4. **Verifier + code gate** — the verifier reviews each candidate as an
   independent voice (it never sees the gate's outcome); the code gate is the
   objective authority and alone decides what may land.
5. **Retention** — best-of-N: a candidate is retained only when the gate
   passes it and it scores at least as many properties as the baseline;
   otherwise the baseline is kept (never-worse fallback).
6. **Reflector** — on failure (no candidate retained), the reflector diagnoses
   the failure pattern from the recorded outcomes and its draft lands in the
   engine via ``Engine.propose`` — status ``ephemeral``, retrievable by nobody
   until the memory gate proves it.

Every step appends a rollout record in the phase0 schema (plus
``reasoning_content`` and ``latency_ms``) to ``runs/rollouts.jsonl``. The
determinism playbook is learn_0012: thinking off, temperature pinned, and a
per-record seed ``sha256(run_id|case_id|generation|candidate_index)[:8]``
derived and recorded on every candidate and verifier record.

Seams, explicitly: the model (:class:`evogine.swarm.roles.ModelClient`), the
gate (:class:`CandidateGate` — the sandboxed code gate adapts here), and the
engine (retrieve + propose). Tests run fakes; CI makes no API calls.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

from evogine.engine.records import LearningRecord
from evogine.swarm.roles import (
    Builder,
    ModelClient,
    Reflector,
    RoleCall,
    Verifier,
    extract_code,
    parse_learning_draft,
    parse_verdict,
)
from evogine.swarm.rollouts import RolloutWriter, build_rollout

#: The determinism playbook recorded on every rollout's config block.
PLAYBOOK = "learn_0012"


@dataclass(frozen=True)
class SwarmTask:
    """One swarm task: generator-visible material plus host-side references.

    ``description``, ``scenario_brief``, and ``baseline_files`` are the
    generator-visible material role prompts may render. ``suite_path``,
    ``properties``, and ``reference_files`` are host-side only — the gate's
    inputs, never a prompt's. Hidden properties and gate failure details must
    never reach a builder prompt (the runner's tests pin that path).
    """

    case_id: str
    description: str
    scenario_brief: str
    baseline_files: dict[str, str]
    decay_axis: str = "general"
    layout: str = "single"  # "single" or "package"
    suite_path: Path | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    reference_files: dict[str, str] = field(default_factory=dict)

    @property
    def context(self) -> str:
        """The retrieval context a warm engine is queried with for this task."""
        return f"{self.case_id} {self.decay_axis} {self.description} {self.scenario_brief}"


@dataclass(frozen=True)
class GateOutcome:
    """Result of evaluating one candidate tree.

    Structural twin of the benchmark gate's outcome, carrying everything the
    loop records: verdict, failure detail, visible/property counts, and the
    ops counts for the rollout record.
    """

    verdict: str
    failure_detail: str | None
    visible_passed: int
    visible_total: int
    properties_passed: int
    properties_total: int
    ops: Mapping[str, int] = field(default_factory=dict)

    @property
    def visible_failures(self) -> int:
        return self.visible_total - self.visible_passed


class CandidateGate(Protocol):
    """The loop's objective authority: evaluates candidate files and never
    trusts the builder. The sandboxed code gate adapts here."""

    def evaluate(self, files: Mapping[str, str], task: SwarmTask) -> GateOutcome: ...


class EngineLike(Protocol):
    """What the loop needs of the engine: warm retrieval and the proposal path.

    The reflector's draft lands ONLY through ``propose`` — proposals never
    bypass the engine's lifecycle, so they land ``ephemeral`` by construction.
    """

    def retrieve(self, task_context: str, k: int = 5) -> list[LearningRecord]: ...

    def propose(self, draft: LearningRecord | Mapping[str, Any]) -> LearningRecord: ...


@dataclass(frozen=True)
class SwarmConfig:
    """Swarm configuration; defaults follow the learn_0012 playbook.

    Role sizing is task-scaled: per-role model names and the builder
    parallelism cap are config, not code.
    """

    builder_model: str = "claude-sonnet-4-5"
    verifier_model: str = "claude-sonnet-4-5"
    reflector_model: str = "claude-sonnet-4-5"
    builder_max_tokens: int = 4096
    verifier_max_tokens: int = 2048
    reflector_max_tokens: int = 2048
    max_parallel_builders: int = 1
    generations: int = 1
    candidates_per_generation: int = 4
    temperature: float = 0.0  # learn_0012: temperature pinned
    thinking: bool = False  # learn_0012: thinking off
    retrieval_k: int = 5
    # Which benchmark arm this loop run serves ("warm"|"cold"); recorded on
    # every candidate/reflect rollout so paired arms are distinguishable.
    arm: str = "warm"
    runs_dir: Path = Path("runs")
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def derive_seed(run_id: str, case_id: str, generation: int, candidate_index: int) -> str:
    """Derive the per-record seed (learn_0012): sha256 of the record identity.

    Arms of one run share ``run_id``, so candidates at the same
    ``(generation, candidate_index)`` are seed-paired across arms — the
    comparison isolates retrieval as the only difference.
    """
    payload = f"{run_id}|{case_id}|{generation}|{candidate_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass
class _CandidateResult:
    """One candidate unit: builder call, extraction, gate outcome, review."""

    generation: int
    candidate_index: int
    seed: str
    builder: RoleCall
    extracted: str
    outcome: GateOutcome
    verifier: RoleCall  # every candidate gets a review — never optional
    verifier_verdict: str = "unknown"


class BaselineReport(TypedDict):
    """The degraded incumbent's gate outcome, as the loop reports it."""

    verdict: str
    visible_failures: int
    properties_passed: int
    properties_total: int


class CandidateReport(TypedDict):
    """One candidate's gate outcome, as the loop reports it."""

    generation: int
    candidate_index: int
    verdict: str
    properties_passed: int
    properties_total: int
    verifier_verdict: str


class BestReport(TypedDict):
    """The retained candidate's position and objective counts."""

    generation: int
    candidate_index: int
    properties_passed: int
    properties_total: int


class LoopReport(TypedDict):
    """Variance-aware summary returned by :func:`run_task`.

    The report is serialized verbatim into the harness and CLI JSON output,
    so every key here is a public contract of the loop.
    """

    run_id: str
    case_id: str
    retrieved_records: int
    retrieved_ids: list[str]
    baseline: BaselineReport
    candidates: list[CandidateReport]
    best: BestReport | None
    retained: Literal["candidate", "baseline"]
    proposal: dict[str, object] | None
    rollouts_path: str


def run_task(
    task: SwarmTask,
    engine: EngineLike,
    model: ModelClient,
    gate: CandidateGate,
    config: SwarmConfig | None = None,
) -> LoopReport:
    """Run one task through the swarm loop and return a variance-aware report.

    The gate is a required argument — a loop without an objective authority
    must not exist. Every model-calling step is recorded to
    ``runs/rollouts.jsonl`` in the phase0 schema; the report summarizes the
    baseline, candidates, retention, and any learning proposal.
    """
    config = config or SwarmConfig()
    writer = RolloutWriter(Path(config.runs_dir) / "rollouts.jsonl")

    # 1. Baseline: the never-worse bar.
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
            ops=dict(baseline_outcome.ops),
            tokens={"input_tokens": None, "output_tokens": None},
            reasoning_content="",
            latency_ms=None,
        )
    )

    # 2. Retrieve (warm arm): the only engine knowledge the builder sees.
    knowledge = engine.retrieve(task.context, k=config.retrieval_k)

    builder = Builder(
        model,
        config.builder_model,
        temperature=config.temperature,
        thinking=config.thinking,
        max_tokens=config.builder_max_tokens,
    )
    verifier = Verifier(
        model,
        config.verifier_model,
        temperature=config.temperature,
        thinking=config.thinking,
        max_tokens=config.verifier_max_tokens,
    )
    reflector = Reflector(
        model,
        config.reflector_model,
        temperature=config.temperature,
        thinking=config.thinking,
        max_tokens=config.reflector_max_tokens,
    )

    # 3+4. Candidates: builder -> verifier -> gate, bounded by the parallel cap.
    results = _generate_candidates(task, knowledge, builder, verifier, gate, config)

    # 5. Retention: best-of-N with a never-worse fallback to the baseline.
    eligible = [
        result
        for result in results
        if result.outcome.visible_failures == 0
        and result.outcome.properties_passed >= baseline_outcome.properties_passed
    ]
    best: _CandidateResult | None = None
    if eligible:
        best = max(
            eligible,
            key=lambda r: (r.outcome.properties_passed, -r.generation, -r.candidate_index),
        )

    for result in results:
        _write_candidate_records(writer, task, config, result)

    # 6. Reflector: on failure, diagnose and land the learning as ephemeral.
    proposal: dict[str, object] | None = None
    if best is None:
        proposal = _reflect(task, engine, reflector, baseline_outcome, results, config, writer)

    baseline_report: BaselineReport = {
        "verdict": baseline_outcome.verdict,
        "visible_failures": baseline_outcome.visible_failures,
        "properties_passed": baseline_outcome.properties_passed,
        "properties_total": baseline_outcome.properties_total,
    }
    candidates_report: list[CandidateReport] = [
        {
            "generation": result.generation,
            "candidate_index": result.candidate_index,
            "verdict": result.outcome.verdict,
            "properties_passed": result.outcome.properties_passed,
            "properties_total": result.outcome.properties_total,
            "verifier_verdict": result.verifier_verdict,
        }
        for result in results
    ]
    best_report: BestReport | None = None
    if best is not None:
        best_report = {
            "generation": best.generation,
            "candidate_index": best.candidate_index,
            "properties_passed": best.outcome.properties_passed,
            "properties_total": best.outcome.properties_total,
        }
    retained: Literal["candidate", "baseline"] = "candidate" if best is not None else "baseline"
    return {
        "run_id": config.run_id,
        "case_id": task.case_id,
        "retrieved_records": len(knowledge),
        "retrieved_ids": [record.id for record in knowledge],
        "baseline": baseline_report,
        "candidates": candidates_report,
        "best": best_report,
        "retained": retained,
        "proposal": proposal,
        "rollouts_path": str(writer.path),
    }


def _generate_candidates(
    task: SwarmTask,
    knowledge: list[LearningRecord],
    builder: Builder,
    verifier: Verifier,
    gate: CandidateGate,
    config: SwarmConfig,
) -> list[_CandidateResult]:
    """Run every (generation, candidate_index) unit, parallelism bounded.

    ``knowledge`` is the warm retrieval result and goes into every builder
    prompt. Results come back in deterministic index order regardless of pool
    size, so rollout records append in generation/index order.
    """
    generations = range(config.generations)
    indexes = range(config.candidates_per_generation)

    def unit(generation: int, candidate_index: int) -> _CandidateResult:
        seed = derive_seed(config.run_id, task.case_id, generation, candidate_index)
        builder_call = builder.propose(task, knowledge, seed=seed)
        extracted = extract_code(builder_call.response.text)
        outcome = _evaluate_reply(extracted, task, gate)
        verifier_call = verifier.review(task, extracted, seed=seed)
        return _CandidateResult(
            generation=generation,
            candidate_index=candidate_index,
            seed=seed,
            builder=builder_call,
            extracted=extracted,
            outcome=outcome,
            verifier=verifier_call,
            verifier_verdict=parse_verdict(verifier_call.response.text),
        )

    max_workers = max(1, config.max_parallel_builders)
    if max_workers == 1:
        return [unit(generation, index) for generation in generations for index in indexes]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(unit, generation, index) for generation in generations for index in indexes
        ]
        return [future.result() for future in futures]


def _evaluate_reply(extracted: str, task: SwarmTask, gate: CandidateGate) -> GateOutcome:
    """Evaluate a builder reply; a reply that breaks extraction is a failing
    candidate, not a loop crash."""
    try:
        files = extract_candidate_files(extracted, task.layout)
        return gate.evaluate(files, task)
    except Exception as exc:  # noqa: BLE001 — gate is a plugin seam; any failure is a recorded verdict
        return GateOutcome(
            verdict="fail",
            failure_detail=f"extraction failed: {type(exc).__name__}: {exc}",
            visible_passed=0,
            visible_total=0,
            properties_passed=0,
            properties_total=0,
        )


def _write_candidate_records(
    writer: RolloutWriter, task: SwarmTask, config: SwarmConfig, result: _CandidateResult
) -> None:
    """Append the builder and verifier rollout records for one candidate."""
    determinism = {
        "arm": config.arm,
        "seed": result.seed,
        "temperature": config.temperature,
        "thinking": config.thinking,
        "playbook": PLAYBOOK,
    }
    writer.append(
        build_rollout(
            stage="candidate",
            case_id=task.case_id,
            run_id=config.run_id,
            generation=result.generation,
            candidate_index=result.candidate_index,
            model=config.builder_model,
            config=dict(determinism),
            prompt_messages=result.builder.messages,
            extracted_code=result.extracted,
            verdict=result.outcome.verdict,
            failure_detail=result.outcome.failure_detail,
            ops=dict(result.outcome.ops),
            tokens={
                "input_tokens": result.builder.response.input_tokens,
                "output_tokens": result.builder.response.output_tokens,
            },
            reasoning_content=result.builder.response.reasoning_content,
            latency_ms=result.builder.response.latency_ms,
        )
    )
    review_text = result.verifier.response.text
    writer.append(
        build_rollout(
            stage="verifier",
            case_id=task.case_id,
            run_id=config.run_id,
            generation=result.generation,
            candidate_index=result.candidate_index,
            model=config.verifier_model,
            config=dict(determinism),
            prompt_messages=result.verifier.messages,
            extracted_code=result.extracted,
            verdict=result.verifier_verdict,
            failure_detail=review_text if result.verifier_verdict == "fail" else None,
            ops={},
            tokens={
                "input_tokens": result.verifier.response.input_tokens,
                "output_tokens": result.verifier.response.output_tokens,
            },
            reasoning_content=result.verifier.response.reasoning_content,
            latency_ms=result.verifier.response.latency_ms,
        )
    )


def _reflect(
    task: SwarmTask,
    engine: EngineLike,
    reflector: Reflector,
    baseline_outcome: GateOutcome,
    results: list[_CandidateResult],
    config: SwarmConfig,
    writer: RolloutWriter,
) -> dict[str, object] | None:
    """Diagnose the failed attempt and land the draft via ``engine.propose``.

    The proposal is the reflector's only write path: it lands ``ephemeral``
    or not at all — a parse or validation failure is a recorded rollout, never
    a direct store write.
    """
    call = reflector.diagnose(task, _render_evidence(baseline_outcome, results), seed="")
    base_fields: dict[str, Any] = {
        "stage": "reflector",
        "case_id": task.case_id,
        "run_id": config.run_id,
        "generation": None,
        "candidate_index": None,
        "model": config.reflector_model,
        "prompt_messages": call.messages,
        "extracted_code": call.response.text,
        "ops": {},
        "tokens": {
            "input_tokens": call.response.input_tokens,
            "output_tokens": call.response.output_tokens,
        },
        "reasoning_content": call.response.reasoning_content,
        "latency_ms": call.response.latency_ms,
    }
    try:
        draft = parse_learning_draft(call.response.text)
        record = engine.propose(
            {
                "type": draft.get("type", "failure-pattern"),
                "title": draft.get("title", f"Unreflected failure in {task.case_id}"),
                "body": draft.get("body", {}),
                "confidence": draft.get("confidence"),
                "provenance": {
                    "proposed_by": "reflector",
                    "origin": "swarm-runner",
                    "source": f"run {config.run_id}/case {task.case_id}",
                    "note": "drafted by the reflector from recorded rollout outcomes; not yet replayed",
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 — boundary catch by design: a rejected
        # draft is a recorded failure, not a loop crash (same convention as the
        # benchmark gate); parse/coercion/git-store errors all land here.
        writer.append(
            build_rollout(
                **base_fields,
                config={"arm": config.arm, "seed": None, "playbook": PLAYBOOK},
                verdict="fail",
                failure_detail=f"learning proposal rejected: {type(exc).__name__}: {exc}",
            )
        )
        return None
    writer.append(
        build_rollout(
            **base_fields,
            config={
                "arm": config.arm,
                "seed": None,
                "playbook": PLAYBOOK,
                "learning_id": record.id,
            },
            verdict="proposed",
            failure_detail=None,
        )
    )
    return {"learning_id": record.id, "status": record.status}


def _render_evidence(baseline_outcome: GateOutcome, results: list[_CandidateResult]) -> str:
    """Render the recorded outcomes for the reflector's diagnosis prompt.

    Failure details may name hidden property ids — that is the design: the
    reflector is the diagnosis role, and its draft reaches a builder prompt
    only through promotion.
    """
    lines = [
        (
            "baseline: "
            f"verdict={baseline_outcome.verdict} "
            f"properties={baseline_outcome.properties_passed}/{baseline_outcome.properties_total}"
        ),
    ]
    for result in results:
        lines.append(
            f"candidate gen={result.generation} idx={result.candidate_index}: "
            f"verdict={result.outcome.verdict} "
            f"properties={result.outcome.properties_passed}/{result.outcome.properties_total} "
            f"verifier={result.verifier_verdict} "
            f"failure={result.outcome.failure_detail or 'none'}"
        )
    return "\n".join(lines)


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
            raise ValueError('multi-file reply must be {"files": {"<relative path>": "<source>"}}')
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
