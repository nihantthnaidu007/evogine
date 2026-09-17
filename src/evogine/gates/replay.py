"""Counterfactual replay: re-running the code gate on recorded past runs.

This module is the memory gate's evidence engine. A :class:`RecordedRun`
preserves everything one past gate evaluation needed — baseline, candidate,
test suite, golden I/O — plus the verdict payload the code gate returned at
the time (the run's ``cold`` arm). Counterfactual replay asks the question
the whole engine exists for: would the verdict have been different had the
agent followed a proposed learning? The learning is applied to the recorded
candidate by a caller-provided :func:`LearningApplier` (the swarm's
builder-backed applier in production; a deterministic transform in tests),
the full code gate pipeline is re-run on the result (the ``warm`` arm), and
the two verdicts are compared.

MATERIALITY (seed learn_0009): a verdict flip is only evidence above the
per-case noise floor. Each replayed run repeats the warm arm ``repeats``
times; the paired deltas (``1.0`` when the gate accepted the applied
candidate, ``0.0`` when not, minus the recorded cold arm's accepted bit) are
centered on their median, and the floor is ``1.4826 x MAD`` of the centered
deltas — phase 0's per-case noise floor. A flip is material only when the
median paired delta is STRICTLY above that floor. A deterministic executor
gives MAD 0 for a real flip, so a consistent flip is material; a flaky one
(deltas split across repeats) prices out below its own floor — without
this, near-floor "wins" are noise wearing a medal. No promotion may ever
rest on model confidence, reflection quality, or any other subjective
signal: a replayed verdict change or nothing.

Run recording contract: verifier-side callers (the swarm runner, the
benchmark harness) append one :class:`RecordedRun` per gate evaluation to a
:class:`RunLog` — an append-only JSONL file under the same discipline as the
engine's audit log (append-only, duplicate ids refused, corrupt lines raise,
nothing silently shrinks). The replay reads recorded runs from a RunLog and
never trusts anything the proposing agent said about the learning. The cold
arm is history, not a re-enactment: replay never re-runs it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from evogine.engine.records import LearningRecord, RecordCounterfactual
from evogine.gates.code_gate import (
    OPS_DISCLOSURE,
    CodeGate,
    GateInput,
    GateResult,
    GateVerdict,
    GoldenCase,
)

#: Seed learn_0009: per-case floors are 1.4826 x MAD of centered repeats.
MAD_TO_FLOOR = 1.4826

#: The objective score behind a paired delta: did the gate accept the candidate?
_ACCEPTED = 1.0
_REJECTED = 0.0

#: Pseudo-verdict recorded for a warm repeat whose applier could not apply
#: the learning at all — visible in the evidence, never silently skipped.
NOT_APPLIED = "not_applied"


class UnknownRunError(LookupError):
    """No recorded run with the requested id exists in the run log."""


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def paired_delta_and_floor(
    deltas: Sequence[float], multiplier: float = MAD_TO_FLOOR
) -> tuple[float, float]:
    """Seed learn_0009's per-case floor, as a pure function.

    Returns ``(median_delta, floor)``: the deltas centered on their median,
    and ``multiplier x MAD`` of the centered deltas. At least two repeats are
    required — a single replay carries no variance estimate, and the gate
    fails closed rather than promote on an unmeasured claim.

    Raises:
        ValueError: fewer than two deltas, or a non-positive multiplier.
    """
    if len(deltas) < 2:
        raise ValueError("materiality needs at least 2 paired repeats; refusing to guess")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    center = _median(deltas)
    centered = [delta - center for delta in deltas]
    mad = _median([abs(value) for value in centered])
    return center, multiplier * mad


def flip_is_material(deltas: Sequence[float], multiplier: float = MAD_TO_FLOOR) -> bool:
    """True iff the median paired delta is strictly above the per-case floor."""
    center, floor = paired_delta_and_floor(deltas, multiplier)
    return center > floor


def regression_is_material(deltas: Sequence[float], multiplier: float = MAD_TO_FLOOR) -> bool:
    """True iff the median paired delta is strictly below minus the floor.

    The demotion mirror of :func:`flip_is_material`: a learning is demoted
    only when its applied form consistently turns accepted runs into
    rejected ones, not on a single bad outcome.
    """
    center, floor = paired_delta_and_floor(deltas, multiplier)
    return center < -floor


@dataclass(frozen=True)
class AppliedPatch:
    """The learning applied to one recorded run's candidate.

    ``candidate_files`` replaces the run's candidate sources wholesale;
    ``test_suite`` of ``None`` keeps the recorded suite. The applier owns the
    transformation — the gate's objectivity lives in re-verifying whatever
    the applier produced, never in judging its craft.
    """

    candidate_files: dict[str, str]
    test_suite: str | None = None


#: Applies a learning to a recorded run's candidate; ``None`` when the
#: learning does not apply to that run (recorded as applied=False, never
#: silently skipped).
LearningApplier = Callable[[LearningRecord, "RecordedRun"], AppliedPatch | None]


@dataclass(frozen=True)
class RecordedRun:
    """One recorded past run: the code gate's input, and the verdict it returned.

    The ``cold`` payload is the run as history recorded it. Golden outputs
    must be JSON-serializable — recorded runs are JSON (the phase0 rollout
    schema), and a non-JSON output refuses to record rather than round-trip
    through a lossy repr.
    """

    run_id: str
    case_id: str
    gate_input: GateInput
    cold: GateResult
    recorded_at: datetime | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the run log."""
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "note": self.note,
            "cold": self.cold.to_dict(),
            "gate_input": _gate_input_to_dict(self.gate_input),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecordedRun:
        """Rebuild a recorded run from :meth:`to_dict` output.

        The rebuilt cold payload preserves the full :meth:`GateResult.to_dict`
        surface — verdict, acceptance, gate, reason, ops counts, disclosure,
        witness, and stats. ``returned_files`` is not serialized (to_dict
        records only the ``returned`` selection); it is reconstructed from the
        gate's own rule — the candidate when accepted, else the baseline —
        using the recorded run's own sources, so a round-tripped run compares
        equal to the original.
        """
        cold_payload = dict(payload["cold"])
        gate_input = _gate_input_from_dict(payload["gate_input"])
        cold = GateResult(
            verdict=GateVerdict(cold_payload["verdict"]),
            accepted=cold_payload["accepted"],
            gate=cold_payload["gate"],
            reason=cold_payload["reason"],
            baseline_ops=cold_payload.get("baseline_ops"),
            candidate_ops=cold_payload.get("candidate_ops"),
            disclosure=cold_payload.get("disclosure", OPS_DISCLOSURE),
            witness=cold_payload.get("witness"),
            stats=cold_payload.get("stats"),
            returned_files=(
                dict(gate_input.candidate_files)
                if cold_payload["accepted"]
                else dict(gate_input.baseline_files)
            ),
        )
        recorded_at = payload.get("recorded_at")
        return cls(
            run_id=str(payload["run_id"]),
            case_id=str(payload["case_id"]),
            gate_input=gate_input,
            cold=cold,
            recorded_at=datetime.fromisoformat(recorded_at) if recorded_at else None,
            note=str(payload.get("note", "")),
        )


def _json_output(output: Any, index: int) -> Any:
    """Pass JSON-serializable golden outputs through; refuse anything else."""
    try:
        json.dumps(output)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"golden case {index} output is not JSON-serializable "
            f"({type(output).__name__}); recorded runs are JSON — refusing to record lossily"
        ) from error
    return output


def _gate_input_to_dict(gate_input: GateInput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scope": gate_input.scope,
        "candidate_files": dict(gate_input.candidate_files),
        "test_suite": gate_input.test_suite,
    }
    if gate_input.scope == "single-function":
        payload["entry_point"] = gate_input.entry_point
        payload["baseline_source"] = gate_input.baseline_files["candidate.py"]
        payload["golden_cases"] = [
            {"args": list(case.args), "kwargs": dict(case.kwargs), "output": _json_output(case.output, i)}
            for i, case in enumerate(gate_input.golden_cases)
        ]
    else:
        payload["baseline_files"] = dict(gate_input.baseline_files)
        payload["entry_class"] = gate_input.entry_class
        payload["corpus_source"] = gate_input.corpus_source
    return payload


def _gate_input_from_dict(payload: Mapping[str, Any]) -> GateInput:
    if payload["scope"] == "single-function":
        cases = tuple(
            GoldenCase(args=tuple(case["args"]), kwargs=dict(case["kwargs"]), output=case["output"])
            for case in payload["golden_cases"]
        )
        return GateInput.single_function(
            baseline_source=payload["baseline_source"],
            candidate_source=payload["candidate_files"]["candidate.py"],
            entry_point=payload["entry_point"],
            test_suite=payload["test_suite"],
            golden_cases=cases,
        )
    return GateInput.multi_file(
        baseline_files=payload["baseline_files"],
        candidate_files=payload["candidate_files"],
        entry_class=payload["entry_class"],
        corpus_source=payload["corpus_source"],
        test_suite=payload["test_suite"],
    )


class RunLog:
    """Append-only JSONL log of recorded past runs — the replay's source of truth.

    Mirrors the engine audit log's discipline: :meth:`record` is the only
    write path, duplicate run ids are refused (caller bug, not a retry), and
    a corrupt line raises instead of silently shrinking the replayable
    history. Reads are full scans; v1 run counts are small and the log lives
    under the engine's data directory.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, run: RecordedRun) -> RecordedRun:
        """Append ``run``; refuses a duplicate ``run_id``."""
        if any(existing.run_id == run.run_id for existing in self.runs()):
            raise ValueError(f"run id '{run.run_id}' is already recorded; run ids are unique")
        with self.path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(run.to_dict(), separators=(",", ":")) + "\n")
        return run

    def get(self, run_id: str) -> RecordedRun:
        """Return the recorded run with ``run_id``.

        Raises:
            UnknownRunError: no recorded run with that id.
        """
        for run in self.runs():
            if run.run_id == run_id:
                return run
        raise UnknownRunError(f"no recorded run with id '{run_id}'")

    def runs(self) -> list[RecordedRun]:
        """Return every recorded run in append order.

        Raises:
            RuntimeError: any line is not a valid recorded run.
        """
        if not self.path.exists():
            return []
        recorded: list[RecordedRun] = []
        with self.path.open("r", encoding="utf-8") as log:
            for line_number, line in enumerate(log, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    recorded.append(RecordedRun.from_dict(json.loads(stripped)))
                except Exception as error:
                    message = f"corrupt run log at {self.path}, line {line_number}: {error}"
                    raise RuntimeError(message) from error
        return recorded


@dataclass(frozen=True)
class RunReplayResult:
    """One replayed run: recorded cold history versus the warm (applied) repeats."""

    run_id: str
    case_id: str
    applied: bool  # False when the applier could not apply the learning here
    cold_accepted: bool
    cold_verdict: str
    warm_verdicts: tuple[str, ...]  # one per warm repeat; empty when not applied
    deltas: tuple[float, ...]
    median_delta: float | None
    floor: float | None
    material_flip: bool  # fail -> pass / worse -> accepted-better, above the floor
    material_regression: bool  # accepted -> rejected, below minus the floor
    warm: GateResult | None = None  # the last warm payload, for diagnosis

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe, oracle-free view for evidence summaries."""
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "applied": self.applied,
            "cold_verdict": self.cold_verdict,
            "cold_accepted": self.cold_accepted,
            "warm_verdicts": list(self.warm_verdicts),
            "deltas": list(self.deltas),
            "median_delta": self.median_delta,
            "floor": self.floor,
            "material_flip": self.material_flip,
            "material_regression": self.material_regression,
        }


def replay_one(
    run: RecordedRun,
    learning: LearningRecord,
    applier: LearningApplier,
    gate: CodeGate,
    *,
    repeats: int,
    multiplier: float = MAD_TO_FLOOR,
    timeout: float = 10.0,
) -> RunReplayResult:
    """Replay one recorded run with ``learning`` applied; measure the flip.

    The cold arm is the recorded verdict. Each warm repeat re-applies the
    learning (the applier may be nondeterministic when the builder is a
    model — the repeats measure the whole warm arm's variance) and re-runs
    the full code gate pipeline on the result; the paired deltas price the
    flip against the per-case floor (seed learn_0009).

    A repeat the applier cannot apply to scores as rejected and records the
    :data:`NOT_APPLIED` pseudo-verdict — an inconsistently applicable
    learning is an unstable one, and the floor prices it out. A run the
    learning never applies to is recorded as ``applied=False`` with no
    deltas: no measurement, no evidence, promotable by no one.
    """
    cold_score = _ACCEPTED if run.cold.accepted else _REJECTED
    scores: list[float] = []
    verdicts: list[str] = []
    last_warm: GateResult | None = None
    for _ in range(repeats):
        patch = applier(learning, run)
        if patch is None:
            scores.append(_REJECTED)
            verdicts.append(NOT_APPLIED)
            continue
        updates: dict[str, Any] = {"candidate_files": dict(patch.candidate_files)}
        if patch.test_suite is not None:
            updates["test_suite"] = patch.test_suite
        warm_input = replace(run.gate_input, **updates)
        warm_result = gate.evaluate(warm_input, timeout=timeout)
        scores.append(_ACCEPTED if warm_result.accepted else _REJECTED)
        verdicts.append(warm_result.verdict.value)
        last_warm = warm_result
    if verdicts and all(verdict == NOT_APPLIED for verdict in verdicts):
        return RunReplayResult(
            run_id=run.run_id,
            case_id=run.case_id,
            applied=False,
            cold_accepted=run.cold.accepted,
            cold_verdict=run.cold.verdict.value,
            warm_verdicts=(),
            deltas=(),
            median_delta=None,
            floor=None,
            material_flip=False,
            material_regression=False,
            warm=None,
        )
    deltas = tuple(score - cold_score for score in scores)
    center, floor = paired_delta_and_floor(deltas, multiplier)
    return RunReplayResult(
        run_id=run.run_id,
        case_id=run.case_id,
        applied=True,
        cold_accepted=run.cold.accepted,
        cold_verdict=run.cold.verdict.value,
        warm_verdicts=tuple(verdicts),
        deltas=deltas,
        median_delta=center,
        floor=floor,
        material_flip=flip_is_material(deltas, multiplier),
        material_regression=regression_is_material(deltas, multiplier),
        warm=last_warm,
    )


@dataclass(frozen=True)
class ReplayEvidence:
    """Assembled counterfactual-replay evidence for one learning.

    ``verdict_change`` is True only when at least one replayed run shows a
    MATERIAL verdict flip — the exact predicate ``Engine.promote`` requires.
    The summary string is oracle-free (the code gate's rejection payloads
    already are) and JSON-encoded for the record's ``counterfactual.evidence``.
    """

    record_id: str
    results: tuple[RunReplayResult, ...]

    @property
    def verdict_change(self) -> bool:
        return any(result.material_flip for result in self.results)

    @property
    def flipped_run_ids(self) -> list[str]:
        return [result.run_id for result in self.results if result.material_flip]

    @property
    def regressed_run_ids(self) -> list[str]:
        return [result.run_id for result in self.results if result.material_regression]

    @property
    def replayed_run_ids(self) -> list[str]:
        return [result.run_id for result in self.results]

    def summary(self) -> str:
        """The JSON evidence string: per-run measurements plus the floor rule."""
        payload = {
            "record_id": self.record_id,
            "replayed_runs": len(self.results),
            "material_flips": len(self.flipped_run_ids),
            "flipped_run_ids": self.flipped_run_ids,
            "regressed_run_ids": self.regressed_run_ids,
            "per_run": [result.to_dict() for result in self.results],
            "floor_rule": (
                f"{MAD_TO_FLOOR} x MAD of centered repeats (seed learn_0009); a flip is "
                "material only when the median paired delta is strictly above the floor"
            ),
        }
        return json.dumps(payload, separators=(",", ":"))

    def to_counterfactual(self) -> RecordCounterfactual:
        """The engine-facing evidence: what ``Engine.promote`` consumes."""
        return RecordCounterfactual(
            replay_run_ids=self.replayed_run_ids,
            verdict_change=self.verdict_change,
            evidence=self.summary(),
        )
