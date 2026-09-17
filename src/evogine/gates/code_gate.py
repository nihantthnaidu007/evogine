"""Objective gate #1: sandboxed behavioral verification of code changes.

The code gate decides whether a builder-proposed change may land, and it
never trusts the proposing agent's own claims. Four stages, in order; the
first failing stage decides. The gate is objective, not advisory.

  1. TESTS (both scopes, when a test suite is given): the candidate and the
     suite run inside the locked-down sandbox (:mod:`evogine.gates.sandbox`)
     under pytest. Accepted residual (carried from Darwin verify_case.py):
     the suite rides WITH the candidate, so a candidate could memorize the
     tests' own answers — the later stages exist precisely because of that.
  2. ORACLE-LEAK SCAN (single-function): a host-side scan of the candidate
     source for the golden expected-output literals. A candidate that embeds
     the answers is rejected fail-closed BEFORE its replay can bless it.
  3. EQUIVALENCE (both scopes): differential equivalence at the public API
     (:mod:`evogine.gates.equivalence`). Single-function: golden I/O replay,
     inputs in, expected outputs compared host-side. Multi-file: corpus
     sequences replayed against the candidate in the sandbox while the
     original replays host-side, per-call outcomes compared under strict
     typed equality.
  4. NEVER-WORSE-THAN-BASELINE (:func:`compare_to_baseline`, seed learn_0011):
     a candidate that passed every behavioral check must still BEAT the
     baseline on the objective score to displace it — bytecode-instruction
     ops for single-function work (:mod:`evogine.gates.verifier`); public-API
     differential equivalence for multi-file work (already enforced by stage
     3; ops comparison across files is not defined in v1). A candidate that
     passes tests but does not beat the baseline is REJECTED, and the gate
     returns the BASELINE as the surviving source.

VERDICT TAXONOMY: ``PASS_TESTS`` / ``FAIL_TESTS`` / ``FAIL_EQUIVALENCE`` /
``FAIL_TIMEOUT`` / ``FAIL_SANDBOX`` describe the BEHAVIORAL outcome;
``GateResult.accepted`` is the decision. A never-worse rejection therefore
reads ``verdict=PASS_TESTS, accepted=False, gate='baseline'`` — the candidate
was behaviorally sound but nothing about it beat the incumbent.

EVERY payload carries the ops counts (when measured) and the disclosure
constant: bytecode-instruction counting is blind to C-level builtin work
(seed learn_0010) — see :data:`OPS_DISCLOSURE`. Rejection reasons and
witnesses are oracle-free by construction: expected outputs and the
original's outcomes are withheld host-side and never appear in feedback.

THREAT MODEL (carried from Darwin's SECURITY.md — read before trusting the
gate): the boundary is the locked-down container, which walls off ACCESS
(host filesystem, network, the oracle). It does NOT split the process, so the
candidate and the trusted in-container driver are co-resident and in-process
interposition (frame-locals serializer hijack, compute-and-lie) remains an
OPEN, documented bypass; the single-frame token extraction only raises the bar
for naive cheats. The gate's compensations are corpus design (golden inputs
not a subset of test inputs), the host-side oracle-leak scan, fail-closed
frame handling, and the never-worse rule.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from evogine.gates import equivalence, verifier
from evogine.gates.sandbox import DockerSandbox, Executor, RunResult, RunStatus

#: Seed learn_0010, surfaced in EVERY verdict payload (see verifier.py).
OPS_DISCLOSURE = verifier.OPS_DISCLOSURE

Scope = Literal["single-function", "multi-file"]


class GateVerdict(str, Enum):
    """The behavioral outcome vocabulary of the code gate."""

    PASS_TESTS = "pass_tests"
    FAIL_TESTS = "fail_tests"
    FAIL_EQUIVALENCE = "fail_equivalence"
    FAIL_TIMEOUT = "fail_timeout"
    FAIL_SANDBOX = "fail_sandbox"


@dataclass(frozen=True)
class GoldenCase:
    """One held-out golden I/O case. ``output`` is ORACLE material: it is
    compared host-side and is never staged into the sandbox."""

    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    output: Any = None

    def inputs(self) -> dict[str, Any]:
        """The input side only — the part that may enter the sandbox."""
        return {"args": list(self.args), "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class GateInput:
    """Everything one gate evaluation needs, for either scope.

    Build via :meth:`single_function` or :meth:`multi_file` so the scope
    union stays validated.
    """

    scope: Scope
    baseline_files: dict[str, str]
    candidate_files: dict[str, str]
    entry_point: str = ""  # single-function: the function name
    test_suite: str | None = None  # pytest suite source, both scopes
    golden_cases: tuple[GoldenCase, ...] = ()  # single-function (oracle)
    entry_class: str = ""  # multi-file: the unit's public class
    corpus_source: str | None = None  # multi-file: build_corpus() source

    @classmethod
    def single_function(
        cls,
        *,
        baseline_source: str,
        candidate_source: str,
        entry_point: str,
        test_suite: str,
        golden_cases: Sequence[GoldenCase],
    ) -> GateInput:
        if not entry_point:
            raise ValueError("single-function scope needs an entry_point")
        if not golden_cases:
            raise ValueError("single-function scope needs at least one golden case")
        return cls(
            scope="single-function",
            baseline_files={"candidate.py": baseline_source},
            candidate_files={"candidate.py": candidate_source},
            entry_point=entry_point,
            test_suite=test_suite,
            golden_cases=tuple(golden_cases),
        )

    @classmethod
    def multi_file(
        cls,
        *,
        baseline_files: Mapping[str, str],
        candidate_files: Mapping[str, str],
        entry_class: str,
        corpus_source: str,
        test_suite: str | None = None,
    ) -> GateInput:
        if not entry_class:
            raise ValueError("multi-file scope needs an entry_class")
        if not corpus_source:
            raise ValueError("multi-file scope needs a corpus_source")
        if "__init__.py" not in baseline_files:
            raise ValueError("baseline unit is missing __init__.py")
        return cls(
            scope="multi-file",
            baseline_files=dict(baseline_files),
            candidate_files=dict(candidate_files),
            entry_class=entry_class,
            corpus_source=corpus_source,
            test_suite=test_suite,
        )


@dataclass(frozen=True)
class GateResult:
    """The verdict payload. ``verdict`` is the behavioral outcome, ``accepted``
    the decision, ``returned_files`` the source the engine should keep (the
    candidate when accepted, else the baseline). Every payload carries the ops
    counts (``None`` when the failing stage precedes measurement) and the
    disclosure; reasons and witnesses are oracle-free by construction."""

    verdict: GateVerdict
    accepted: bool
    gate: str | None  # "tests" | "equivalence" | "baseline" | None when accepted
    reason: str
    baseline_ops: int | None = None
    candidate_ops: int | None = None
    disclosure: str = OPS_DISCLOSURE
    returned_files: dict[str, str] | None = None
    witness: dict[str, Any] | None = None
    stats: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe view for engine logs and downstream verdict consumers."""
        return {
            "verdict": self.verdict.value,
            "accepted": self.accepted,
            "gate": self.gate,
            "reason": self.reason,
            "baseline_ops": self.baseline_ops,
            "candidate_ops": self.candidate_ops,
            "disclosure": self.disclosure,
            "returned": "candidate" if self.accepted else "baseline",
            "witness": self.witness,
            "stats": self.stats,
        }


# The pytest stage command, identical to Darwin verify_case.py's Layer 1.
# -p no:cacheprovider avoids a .pytest_cache write against the read-only root.
_PYTEST_CMD = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_suite.py"]

# Expected-output literals shorter than this are too collision-prone to scan
# for (every source contains small ints); the residual for tiny outputs is
# corpus design, not scanning.
_MIN_LEAK_LITERAL = 4


def find_oracle_leak(candidate_source: str, golden_cases: Sequence[GoldenCase]) -> str | None:
    """Return a description of the first embedded expected output, or None.

    A candidate that hardcodes the golden answers passes both the tests and
    the replay — the only honest defense is to refuse candidates that CARRY
    the oracle in their source. We scan for the output's repr form and (when
    JSON-representable) its json.dumps form. Outputs too short to be
    distinguishable are skipped (accepted residual; see _MIN_LEAK_LITERAL).
    """
    for i, case in enumerate(golden_cases):
        forms = [repr(case.output)]
        try:
            forms.append(json.dumps(case.output, ensure_ascii=False))
        except (TypeError, ValueError):
            pass  # non-JSON outputs have no dumps form to scan for
        for form in forms:
            if len(form) >= _MIN_LEAK_LITERAL and form in candidate_source:
                return f"golden case {i} output appears as a literal in the candidate source"
    return None


def compare_to_baseline(
    scope: Scope,
    candidate_ops: int | None,
    baseline_ops: int | None,
    *,
    equivalent: bool,
) -> tuple[bool, str]:
    """The never-worse rule (seed learn_0011), as a pure decision.

    Single-function: displace the incumbent only on a STRICT ops improvement —
    equal is not better, and "nothing beats it" means the baseline is kept.
    An incumbent whose measurement crashed on the replay inputs cannot anchor
    the bar at all: the equivalent, measurable candidate dominates it and is
    accepted.  Multi-file: the objective score is public-API equivalence
    itself, already enforced upstream; acceptance rides on it (ops comparison
    across files is not defined in v1)."""
    if scope == "multi-file":
        return equivalent, (
            "candidate is behaviorally equivalent at the public API"
            if equivalent
            else "candidate is not equivalent at the public API"
        )
    if candidate_ops is not None and baseline_ops is not None and candidate_ops < baseline_ops:
        return True, (
            f"candidate beats baseline on the objective score: {candidate_ops} < "
            f"{baseline_ops} bytecode-instruction ops"
        )
    if equivalent and candidate_ops is not None and baseline_ops is None:
        return True, (
            "baseline is unmeasurable on the replay inputs (it raised); the "
            f"working candidate is accepted on equivalence with {candidate_ops} ops"
        )
    return False, (
        f"never-worse rejected: candidate ops {candidate_ops} does not beat "
        f"baseline ops {baseline_ops}; the incumbent is kept"
    )


def _run_failure(res: RunResult, failed_verdict: GateVerdict) -> GateVerdict:
    """Map a non-ok executor run to the gate verdict for its stage.

    TIMED_OUT is always FAIL_TIMEOUT and CRASHED always FAIL_SANDBOX (the
    boundary owns those outcomes); any other non-ok run means the stage's own
    check failed (tests failed / not equivalent), with the captured output
    carried in the reason for diagnosis."""
    if res.status == RunStatus.TIMED_OUT:
        return GateVerdict.FAIL_TIMEOUT
    if res.status == RunStatus.CRASHED:
        return GateVerdict.FAIL_SANDBOX
    return failed_verdict


def _detail(res: RunResult) -> str:
    payload = res.stdout.strip() or res.stderr.strip() or res.status.value
    return str(payload)[:500]


class CodeGate:
    """The objective code gate. ``executor`` defaults to the Docker boundary;
    tests may inject a trusted-local executor for fixture code only."""

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor

    def evaluate(self, gate_input: GateInput, *, timeout: float = 10.0) -> GateResult:
        """Run all four stages and return the verdict payload."""
        executor = self._executor if self._executor is not None else DockerSandbox()
        scope = gate_input.scope

        # Stage 1: sandboxed pytest (when a suite is given). Multi-file units
        # stage under candidate/ for the suite's `from candidate import ...`
        # to resolve — the same convention the equivalence driver relies on
        # (candidate_files keys are unprefixed unit paths).
        if gate_input.test_suite is not None:
            suite_files = dict(gate_input.candidate_files)
            if scope == "multi-file":
                suite_files = {
                    f"candidate/{name}": content for name, content in suite_files.items()
                }
            res = executor.run(
                {**suite_files, "test_suite.py": gate_input.test_suite},
                _PYTEST_CMD,
                timeout=timeout,
            )
            if res.status != RunStatus.OK:
                verdict = _run_failure(res, GateVerdict.FAIL_TESTS)
                return self._reject(gate_input, verdict, "tests", f"pytest: {_detail(res)}")

        # Stage 2 (single-function): the host-side oracle-leak scan. It keys on
        # the golden material, not the test suite: a suite-less single-function
        # case that carries golden I/O is scanned exactly like a suite-bearing
        # one (the previous guard skipped the scan whenever test_suite was
        # None, leaving those candidates unscanned). A case with NO golden
        # material has no expected outputs to leak — the scan vacuously passes
        # (accepted residual; there is no oracle to defend).
        if scope == "single-function" and gate_input.golden_cases:
            leak = find_oracle_leak(
                gate_input.candidate_files["candidate.py"], gate_input.golden_cases
            )
            if leak is not None:
                return self._reject(
                    gate_input,
                    GateVerdict.FAIL_EQUIVALENCE,
                    "equivalence",
                    f"oracle leak: {leak} — expected outputs are withheld from "
                    "the sandbox precisely to prevent this; rejected fail-closed",
                )

        # Stage 3: differential equivalence at the public API.
        try:
            if scope == "single-function":
                equivalent, witness, stats = equivalence.check_function_equivalence(
                    gate_input.candidate_files["candidate.py"],
                    gate_input.entry_point,
                    [case.inputs() | {"output": case.output} for case in gate_input.golden_cases],
                    executor,
                    timeout=timeout,
                )
            else:
                equivalent, witness, stats = equivalence.check_unit_equivalence(
                    gate_input.baseline_files,
                    gate_input.candidate_files,
                    gate_input.entry_class,
                    gate_input.corpus_source or "",
                    executor,
                    timeout=timeout,
                )
        except equivalence.ReplayTimeout as exc:
            return self._reject(gate_input, GateVerdict.FAIL_TIMEOUT, "equivalence", str(exc))
        except equivalence.ReplayCrash as exc:
            return self._reject(gate_input, GateVerdict.FAIL_SANDBOX, "equivalence", str(exc))
        except equivalence.ReplayError as exc:
            return self._reject(gate_input, GateVerdict.FAIL_EQUIVALENCE, "equivalence", str(exc))
        if not equivalent:
            return self._reject(
                gate_input,
                GateVerdict.FAIL_EQUIVALENCE,
                "equivalence",
                "behavior mismatch: candidate diverges at the public API "
                "(original outcome withheld)",
                witness=witness,
                stats=stats,
            )

        # Stage 4: never-worse-than-baseline.
        candidate_ops: int | None = None
        baseline_ops: int | None = None
        if scope == "single-function":
            cases_in = [case.inputs() for case in gate_input.golden_cases]
            try:
                candidate_ops = verifier.measure_ops(
                    gate_input.candidate_files["candidate.py"],
                    gate_input.entry_point,
                    cases_in,
                    executor,
                    timeout=timeout,
                )
            except verifier.VerifierTimeout as exc:
                return self._reject(
                    gate_input,
                    GateVerdict.FAIL_TIMEOUT,
                    "baseline",
                    str(exc),
                    candidate_ops=candidate_ops,
                    baseline_ops=baseline_ops,
                )
            except verifier.VerifierError as exc:
                return self._reject(
                    gate_input,
                    GateVerdict.FAIL_SANDBOX,
                    "baseline",
                    str(exc),
                    candidate_ops=candidate_ops,
                    baseline_ops=baseline_ops,
                )
            try:
                baseline_ops = verifier.measure_ops(
                    gate_input.baseline_files["candidate.py"],
                    gate_input.entry_point,
                    cases_in,
                    executor,
                    timeout=timeout,
                )
            except verifier.VerifierCodeError:
                # The incumbent crashed on the replay inputs — the very decay
                # the benchmark measures (e.g. a degraded billing rule raising
                # on month-end dates).  There is no measurable incumbent to be
                # "not worse than", so the ops bar is vacuous: acceptance
                # rides on the equivalence just proven.  Candidate-side or
                # environment faults above stay fail-closed.
                baseline_ops = None
            except verifier.VerifierTimeout as exc:
                return self._reject(
                    gate_input,
                    GateVerdict.FAIL_TIMEOUT,
                    "baseline",
                    str(exc),
                    candidate_ops=candidate_ops,
                    baseline_ops=baseline_ops,
                )
            except verifier.VerifierError as exc:
                return self._reject(
                    gate_input,
                    GateVerdict.FAIL_SANDBOX,
                    "baseline",
                    str(exc),
                    candidate_ops=candidate_ops,
                    baseline_ops=baseline_ops,
                )
        accepted, rule_reason = compare_to_baseline(
            scope, candidate_ops, baseline_ops, equivalent=equivalent
        )
        if accepted:
            return GateResult(
                verdict=GateVerdict.PASS_TESTS,
                accepted=True,
                gate=None,
                reason=rule_reason,
                baseline_ops=baseline_ops,
                candidate_ops=candidate_ops,
                returned_files=dict(gate_input.candidate_files),
                stats=stats,
            )
        return self._reject(
            gate_input,
            GateVerdict.PASS_TESTS,  # behaviorally sound; the baseline step rejected it
            "baseline",
            rule_reason,
            candidate_ops=candidate_ops,
            baseline_ops=baseline_ops,
            stats=stats,
        )

    def _reject(
        self,
        gate_input: GateInput,
        verdict: GateVerdict,
        gate: str,
        reason: str,
        *,
        candidate_ops: int | None = None,
        baseline_ops: int | None = None,
        witness: dict[str, Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> GateResult:
        """A rejection keeps the incumbent: the returned source is the baseline."""
        return GateResult(
            verdict=verdict,
            accepted=False,
            gate=gate,
            reason=reason,
            baseline_ops=baseline_ops,
            candidate_ops=candidate_ops,
            returned_files=dict(gate_input.baseline_files),
            witness=witness,
            stats=stats,
        )
