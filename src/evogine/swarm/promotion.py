"""The promotion half of the loop: record runs, replay them, decide.

v1 shipped every gate in isolation; nothing in the production run path fed
them (the v2 audit's integration gap). This module is the wiring: the swarm
loop's gate evaluations are recorded as replayable :class:`RecordedRun`
entries in the run log, the reflector's landed learning is queued against
those runs, and the memory gate decides — promote on a material counterfactual
flip above the per-case floor, reject into the audit trail otherwise. Promoted
knowledge is guarded by regression checks against every later recorded run,
with demotion when it stops earning its confidence.

Wiring is opt-in: :func:`evogine.swarm.runner.run_task` behaves exactly as
before unless a :class:`PromotionWiring` is passed. The benchmark harness
never passes one, so frozen benchmark behavior — and the ``evogine run``
path it owns — is untouched. The engine-growing surface is ``evogine evolve``.

Everything here is deterministic and audit-recorded: recorded-run ids derive
from the run's identity (never from wall-clock state), every decision carries
replay evidence ids, and the only writes are the ones the engine's own
append-only audit trail records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from evogine.engine.store import Engine
from evogine.gates.code_gate import GateInput, GateResult, GateVerdict
from evogine.gates.replay import AppliedPatch, RecordedRun
from evogine.swarm.roles import extract_code

if TYPE_CHECKING:
    from evogine.engine.records import LearningRecord
    from evogine.gates.memory_gate import MemoryGate
    from evogine.gates.replay import RunLog
    from evogine.swarm.roles import Builder
    from evogine.swarm.runner import GateOutcome, SwarmConfig, SwarmTask

__all__ = [
    "BuilderApplier",
    "PromotionReport",
    "PromotionWiring",
    "check_promoted_for_regression",
    "finish_promotion",
    "gate_result_from_outcome",
    "record_baseline_run",
    "record_candidate_run",
    "recorded_run_id",
    "require_engine",
]


class PromotionReport(TypedDict):
    """What the wired promotion path did in one loop run.

    ``decision`` is the decided record's lifecycle status — ``"promoted"``
    when the replay evidence carried a material verdict flip, ``"ephemeral"``
    when it did not (the rejection is named ``promote-rejected`` on the audit
    trail). ``None`` when nothing was proposed or there was nothing replayable.
    ``demoted`` lists the learning ids this run's regression checks dropped.
    """

    learning_id: str | None
    queued_run_ids: list[str]
    decision: str | None
    demoted: list[str]


@dataclass(frozen=True)
class PromotionWiring:
    """Opt-in promotion dependencies for :func:`evogine.swarm.runner.run_task`.

    ``run_log`` is the append-only run log the memory gate replays from;
    ``memory_gate`` is the deciding gate, built with the caller's choice of
    learning applier and executor. The run task only forwards them.
    """

    run_log: RunLog
    memory_gate: MemoryGate


# ---------------------------------------------------------------------------
# Recording: loop gate evaluations become replayable runs
# ---------------------------------------------------------------------------


def recorded_run_id(
    config: SwarmConfig,
    task: SwarmTask,
    stage: str,
    *,
    generation: int | None = None,
    candidate_index: int | None = None,
) -> str:
    """Deterministic replay-run id: one recorded run per gate evaluation.

    Uniqueness across runs, arms, and cases is load-bearing — the run log
    refuses duplicate ids as a caller bug, so a retried run must carry a
    fresh ``run_id`` rather than re-record over its evidence.
    """
    parts = [config.run_id, config.arm, task.case_id, stage]
    if generation is not None:
        parts.append(f"generation-{generation}")
    if candidate_index is not None:
        parts.append(f"candidate-{candidate_index}")
    return "::".join(parts)


def gate_result_from_outcome(outcome: GateOutcome, *, gate_input: GateInput) -> GateResult:
    """Reconstruct the replayable cold payload from the loop's outcome encoding.

    The loop's ``CandidateGate`` seam reports the adapter's encoding — an
    accepted candidate counts every visible check passed, a rejection counts
    them failed — and the verdict vocabulary is the code gate's own. Anything
    else cannot be recorded faithfully and fails loudly here: missing or
    foreign evidence is never a silent skip.
    """
    verdict = GateVerdict(outcome.verdict)
    accepted = outcome.visible_total > 0 and outcome.visible_passed == outcome.visible_total
    return GateResult(
        verdict=verdict,
        accepted=accepted,
        gate=outcome.gate_stage,
        reason=outcome.failure_detail or "",
        baseline_ops=outcome.ops.get("baseline_ops"),
        candidate_ops=outcome.ops.get("candidate_ops"),
        returned_files=dict(
            gate_input.candidate_files if accepted else gate_input.baseline_files
        ),
    )


def _gate_input_for(files: Mapping[str, str], task: SwarmTask) -> GateInput:
    # Lazy import: gates_adapter imports the runner at module scope and the
    # runner imports this module — the cycle breaks at this one deferred edge.
    from evogine.swarm.gates_adapter import build_gate_input

    return build_gate_input(files, task)


def _record(
    wiring: PromotionWiring,
    *,
    config: SwarmConfig,
    task: SwarmTask,
    gate_input: GateInput,
    outcome: GateOutcome,
    stage: str,
    generation: int | None = None,
    candidate_index: int | None = None,
) -> str:
    run = RecordedRun(
        run_id=recorded_run_id(
            config, task, stage, generation=generation, candidate_index=candidate_index
        ),
        case_id=task.case_id,
        gate_input=gate_input,
        cold=gate_result_from_outcome(outcome, gate_input=gate_input),
        note=f"swarm loop {stage} evaluation (arm {config.arm})",
    )
    wiring.run_log.record(run)
    return run.run_id


def record_baseline_run(
    wiring: PromotionWiring,
    task: SwarmTask,
    config: SwarmConfig,
    outcome: GateOutcome,
) -> str:
    """Record the loop's baseline evaluation as a replayable run; return its id.

    The GateInput mirrors what the code-gate adapter built for the very same
    evaluation (same pure builder, same inputs), so replay is honest.
    """
    return _record(
        wiring,
        config=config,
        task=task,
        gate_input=_gate_input_for(task.baseline_files, task),
        outcome=outcome,
        stage="baseline",
    )


def record_candidate_run(
    wiring: PromotionWiring,
    task: SwarmTask,
    config: SwarmConfig,
    outcome: GateOutcome,
    files: Mapping[str, str] | None,
    *,
    generation: int,
    candidate_index: int,
) -> str | None:
    """Record one candidate evaluation; ``None`` when there is nothing to record.

    Extraction failures never produced candidate files, so there is no
    re-evaluable input to keep — recorded runs carry real inputs by contract.
    """
    if files is None:
        return None
    return _record(
        wiring,
        config=config,
        task=task,
        gate_input=_gate_input_for(files, task),
        outcome=outcome,
        stage="candidate",
        generation=generation,
        candidate_index=candidate_index,
    )


# ---------------------------------------------------------------------------
# Deciding: regression guard, then queue + replay + decide
# ---------------------------------------------------------------------------


def check_promoted_for_regression(
    wiring: PromotionWiring,
    engine: Engine,
    knowledge: list[LearningRecord],
    run_ids: list[str],
) -> list[str]:
    """Demote promoted knowledge this run's recorded runs regress; return ids.

    ``knowledge`` is what the warm arm retrieved this run — the promoted
    population exactly. The memory gate's own regression semantics decide;
    this is only the production caller the audit found missing.
    """
    demoted: list[str] = []
    for record in knowledge:
        for run_id in run_ids:
            if wiring.memory_gate.check_for_regression(engine, record.id, run_id):
                demoted.append(record.id)
                break  # demoted once — a dropped record no longer regresses
    return demoted


def require_engine(engine: object) -> Engine:
    """Return the real git-backed store, or fail loudly.

    The wired promotion path mutates lifecycle state (queue, promote, drop) —
    a retrieval/proposal-only engine seam cannot carry it.
    """
    if isinstance(engine, Engine):
        return engine
    raise TypeError(
        "promotion wiring requires the git-backed Engine "
        f"(got {type(engine).__name__}, which exposes only retrieve/propose)"
    )


def finish_promotion(
    wiring: PromotionWiring,
    engine: object,
    proposal: Mapping[str, object] | None,
    knowledge: list[LearningRecord],
    recorded_run_ids: list[str],
) -> PromotionReport:
    """Run the promotion phase: guard old knowledge, decide the new learning.

    Called once per loop run, after every gate evaluation is recorded. First
    the regression guard replays this run's recorded runs against promoted
    knowledge; then, if the reflector landed a learning, it is queued against
    the run's baseline+candidate run ids (the design's proposal→candidate
    transition) and the memory gate decides from replay evidence. No recorded
    runs means nothing to replay — the learning stays ephemeral with no
    fabricated evidence, and the report says so.
    """
    store = require_engine(engine)
    demoted = check_promoted_for_regression(wiring, store, knowledge, recorded_run_ids)
    raw_id = proposal.get("learning_id") if proposal else None
    learning_id: str | None = raw_id if isinstance(raw_id, str) else None
    decision: str | None = None
    if learning_id is not None and recorded_run_ids:
        store.queue_replay(learning_id, list(recorded_run_ids))
        decided = wiring.memory_gate.decide(store, learning_id)
        decision = decided.status
    return PromotionReport(
        learning_id=learning_id,
        queued_run_ids=list(recorded_run_ids),
        decision=decision,
        demoted=list(demoted),
    )


# ---------------------------------------------------------------------------
# The production learning applier: the Builder applies the learning
# ---------------------------------------------------------------------------

# gates_adapter's package prefix for candidate files; a literal here avoids
# importing gates_adapter (and with it the runner) at module scope.
_UNIT_PREFIX = "candidate/"


def _unit_path(path: str) -> str:
    """Recorded runs key candidate files by unit path (no package prefix)."""
    return path.removeprefix(_UNIT_PREFIX)


class BuilderApplier:
    """The production :class:`~evogine.gates.replay.LearningApplier`.

    Backed by the swarm's Builder role: one model call per replayed run,
    prompted to apply the learning's fix strategy to the recorded candidate
    source. The call is deterministic under the scripted model and pinned
    temperature, and its seed derives from the learning and run ids so any
    individual replay is reproducible.

    A reply that cannot be parsed into candidate files returns ``None``:
    replay records the ``not_applied`` pseudo-verdict and the per-case floor
    prices the instability out — the failure surfaces in the evidence, it is
    never swallowed.
    """

    def __init__(self, builder: Builder) -> None:
        self._builder = builder

    def __call__(self, record: LearningRecord, run: RecordedRun) -> AppliedPatch | None:
        multi_file = run.gate_input.scope == "multi-file"
        seed = hashlib.sha256(f"apply|{record.id}|{run.run_id}".encode()).hexdigest()[:8]
        call = self._builder.apply_learning(
            record, run.gate_input.candidate_files, multi_file=multi_file, seed=seed
        )
        text = extract_code(call.response.text)
        if not multi_file:
            return AppliedPatch(candidate_files={"candidate.py": text})
        try:
            files = json.loads(text)["files"]
            if not isinstance(files, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in files.items()
            ):
                raise ValueError("not a string-keyed mapping")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        return AppliedPatch(
            candidate_files={_unit_path(key): value for key, value in files.items()}
        )
