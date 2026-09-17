"""Objective gate #2: counterfactual-replay promotion of learned knowledge.

The memory gate keeps agent-generated knowledge ephemeral until proven
useful: a learning is promoted only when counterfactual replay on recorded
past runs (:mod:`evogine.gates.replay`) shows a MATERIAL verdict change —
the learning, applied to a past run's candidate, would have flipped the code
gate's verdict (``fail -> pass``, or a never-worse rejection become
``accepted-better``) above the per-case noise floor. Nothing else counts:
not the proposer's confidence, not reflection quality, not any subjective
signal — a replayed verdict change or nothing.

LIFECYCLE — wired onto the engine's own states and audit log; the gate adds
no storage of its own:

  * ``propose`` lands a draft ``ephemeral`` (the engine's; proposing never
    fails — filtering is this gate's job).
  * ``queue_replay`` names the past runs to replay and marks the record
    ``candidate`` (the engine's).
  * :meth:`MemoryGate.decide` replays those runs and hands the evidence to
    ``engine.promote``. A material flip promotes with the evidence stamped
    on the record; anything else is rejected BY the engine — the record
    returns to ``ephemeral`` and the rejection lands in the audit log.
    The gate records its replay measurement first (actor ``memory-gate``)
    so the trail shows both the measurement and the decision.
  * :meth:`MemoryGate.check_for_regression` is the demotion path: a promoted
    learning whose applied form would consistently flip a later recorded
    run from accepted to rejected is demoted via ``engine.demote`` —
    ``dropped``, reason recorded, kept for audit, never silently forgotten.
  * The seed pack never touches this gate: the twelve seeds are
    human-promoted at bootstrap (``engine.bootstrap``) and their provenance
    notes say so. No seed is promoted, re-decided, or demoted here.

The promotion decision is deliberately conservative, and that is the design:
phase 0's strongest finding was that undirected feedback is not load-bearing,
and a gate that promoted anything the reflector felt good about would rebuild
exactly that failure at engine scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.code_gate import CodeGate, Executor
from evogine.gates.replay import (
    MAD_TO_FLOOR,
    LearningApplier,
    ReplayEvidence,
    RunLog,
    replay_one,
)

#: The engine's audit actor for memory-gate measurements, distinct from the
#: engine's own ``engine`` actor so the trail shows who measured and who ruled.
GATE_ACTOR = "memory-gate"


@dataclass(frozen=True)
class MemoryGateConfig:
    """Gate knobs.

    The floor RULE itself is not configurable away from seed learn_0009: the
    multiplier defaults to :data:`MAD_TO_FLOOR` (``1.4826 x MAD`` of centered
    repeats, paired delta strictly above the floor), and promotion on any
    subjective signal is structurally impossible — there is no field for it.
    """

    repeats: int = 3
    mad_multiplier: float = MAD_TO_FLOOR
    timeout: float = 10.0

    def __post_init__(self) -> None:
        if self.repeats < 2:
            raise ValueError("repeats must be >= 2: a single replay carries no variance estimate")
        if self.mad_multiplier <= 0:
            raise ValueError("mad_multiplier must be positive")


class MemoryGate:
    """Objective verifier for candidate learnings.

    Args:
        run_log: where recorded past runs live — the replay's source of
            truth (:class:`evogine.gates.replay.RunLog`).
        applier: applies a learning to a recorded run's candidate.
            Production callers pass the swarm's builder-backed applier;
            tests pass a deterministic transform. The gate never trusts the
            applier — it re-verifies whatever the applier produced.
        executor: the code gate's execution boundary; defaults to the Docker
            sandbox. Tests may inject the trusted-local executor for fixture
            code only.
        config: thresholds and repeat count (:class:`MemoryGateConfig`).
    """

    def __init__(
        self,
        *,
        run_log: RunLog,
        applier: LearningApplier,
        executor: Executor | None = None,
        config: MemoryGateConfig | None = None,
    ) -> None:
        self._run_log = run_log
        self._applier = applier
        self._gate = CodeGate(executor=executor)
        self._config = config or MemoryGateConfig()

    def evaluate(self, record: LearningRecord, run_ids: list[str]) -> ReplayEvidence:
        """Counterfactual replay of ``record`` against the named past runs.

        Pure measurement — no engine mutation. An unknown run id raises
        (:class:`evogine.gates.replay.UnknownRunError`): missing evidence is
        a loud failure, never a silent skip.
        """
        results = []
        for run_id in run_ids:
            run = self._run_log.get(run_id)
            results.append(
                replay_one(
                    run,
                    record,
                    self._applier,
                    self._gate,
                    repeats=self._config.repeats,
                    multiplier=self._config.mad_multiplier,
                    timeout=self._config.timeout,
                )
            )
        return ReplayEvidence(record_id=record.id, results=tuple(results))

    def review(self, record: LearningRecord, replay_run_ids: list[str]) -> bool:
        """Return ``True`` iff ``record``'s counterfactual replay is material.

        ``replay_run_ids`` name the queued replay runs to inspect. The gate
        promotes only when the replayed verdicts change with the learning
        applied; case-specific materiality thresholds apply.
        """
        return self.evaluate(record, replay_run_ids).verdict_change

    def decide(self, engine: Engine, learning_id: str) -> LearningRecord:
        """Replay the queued runs and let the engine rule on the evidence.

        Reads the run ids the engine's ``queue_replay`` attached, measures
        the counterfactual replay, records the measurement in the audit log
        (actor ``memory-gate``), then hands the evidence to
        ``engine.promote``. A material flip promotes with the evidence
        stamped on the record; anything else is rejected by the engine — the
        record stays ephemeral and the rejection reason lands in the audit
        log. The engine remains the only writer of record status.

        Raises:
            RecordNotFoundError: no record with that learning id.
            evogine.gates.replay.UnknownRunError: a queued run id is missing
                from the run log (fail loudly; nothing is mutated).
        """
        record = engine.get(learning_id)
        run_ids = list(record.counterfactual.replay_run_ids)
        evidence = self.evaluate(record, run_ids)
        engine.audit.append(
            "replay",
            actor=GATE_ACTOR,
            learning_id=learning_id,
            evidence=evidence.summary(),
            reason=(
                f"material verdict flip on {len(evidence.flipped_run_ids)} of "
                f"{len(evidence.results)} replayed run(s)"
                if evidence.verdict_change
                else "no material verdict flip; promotion may not proceed on subjective signal"
            ),
        )
        return engine.promote(learning_id, evidence.to_counterfactual())

    def check_for_regression(self, engine: Engine, learning_id: str, run_id: str) -> bool:
        """Demotion path: demote a promoted learning a later run disproves.

        Replays ``run_id`` with the learning applied; a MATERIAL regression
        flip — the recorded run was accepted, the learning-applied form is
        consistently rejected — demotes via ``engine.demote`` (``dropped``,
        reason recorded, kept for audit). One verified regression is enough:
        confidence is earned repeatedly, and a learning that breaks accepted
        work has already been disproven once.

        Returns ``True`` when a demotion happened. A record that is not
        promoted has nothing to demote: returns False without touching
        anything. An unknown run or learning id raises (loud, no mutation).
        """
        record = engine.get(learning_id)
        if record.status != "promoted":
            return False
        run = self._run_log.get(run_id)
        evidence = self.evaluate(record, [run.run_id])
        result = evidence.results[0]
        if not result.material_regression:
            return False
        engine.demote(
            learning_id,
            f"regression: replay of run '{run.run_id}' (case '{run.case_id}') flips "
            f"{result.cold_verdict}/accepted -> {list(result.warm_verdicts)} with the learning "
            f"applied; median paired delta {result.median_delta} below floor -{result.floor} "
            f"(seed learn_0009)",
        )
        return True
