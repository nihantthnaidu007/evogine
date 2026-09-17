"""The production :class:`~evogine.swarm.runner.CandidateGate` — the code gate.

The swarm loop's retention and reflection logic run against one seam:
``evaluate(files, task) -> GateOutcome``.  This adapter routes every candidate
tree through the sandboxed :class:`~evogine.gates.code_gate.CodeGate` end to
end: a sandboxed pytest stage against the case's visible suite (run by the
gate), a host-side oracle-leak scan, a differential equivalence replay against
the case's hidden properties, and the strict never-worse comparison against
the degraded baseline (seed ``learn_0011``).  Rejections always return the
baseline files.  The adapter surfaces the gate's verdict taxonomy
(``pass_tests``/``fail_tests``/``fail_equivalence``/``fail_timeout``/
``fail_sandbox``) and, where measured, the baseline/candidate bytecode-op
counts as the objective score.

Scope selection is driven by the case itself (see :func:`build_gate_input`):

- **single-function** — golden-I/O rows become the withheld replay oracle and
  never-worse is the bytecode-ops bar.  Rows pinning an *expected raise*
  cannot ride the golden replay (the in-container driver does not catch) and
  are skipped — the visible suite remains their only check.
- **multi-file** — scenario rows become a differential corpus; the case's
  *reference* implementation is the equivalence target and acceptance is
  equivalence itself (cross-file op counts are not defined).  Package trees
  are normalized to unprefixed unit paths, the convention
  :meth:`CodeGate.evaluate` equivalence staging relies on.

The adapter encodes the gate's binary decision onto the loop's counters:
an accepted candidate counts as one visible check and one hidden property
passed; any rejection counts both as failed.  That keeps the runner's
retention rule and the gate's decision exactly aligned — a behaviorally
sound candidate rejected by never-worse (``pass_tests`` with the baseline
returned) still counts as a failure and cannot be retained.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evogine.gates.code_gate import CodeGate, GateInput, GateResult, GoldenCase
from evogine.gates.sandbox import Executor
from evogine.swarm.runner import GateOutcome, SwarmTask

__all__ = [
    "CodeGateAdapter",
    "build_gate_input",
    "corpus_source_from_scenario",
    "golden_cases_from_properties",
]

#: The module every single-file case candidate (and reference) is keyed under.
_BASELINE_MODULE = "candidate.py"

#: Package-layout trees are keyed under this prefix (bench task convention).
_CANDIDATE_PREFIX = "candidate/"


def build_gate_input(files: Mapping[str, str], task: SwarmTask) -> GateInput:
    """Build the code-gate input for a candidate (or baseline) tree.

    Single-function cases evaluate the ``candidate.py`` module directly with
    golden-I/O rows; everything else evaluates the tree as a multi-file unit
    with a differential corpus built from the case's scenario rows.
    """
    rows = _property_rows(task.properties)
    if not rows:
        raise ValueError(
            f"{task.case_id}: case carries no hidden properties; the gate would accept anything"
        )
    kinds = sorted({kind for kind, _ in rows})
    entry = str(task.properties.get("entry", "")).strip()
    if not entry:
        raise ValueError(f"{task.case_id}: case declares no entry point")

    if task.layout == "single":
        if kinds == ["golden_io"]:
            suite = task.suite_path.read_text() if task.suite_path else None
            return GateInput(
                scope="single-function",
                baseline_files={_BASELINE_MODULE: task.baseline_files[_BASELINE_MODULE]},
                candidate_files={_BASELINE_MODULE: files[_BASELINE_MODULE]},
                entry_point=entry,
                test_suite=suite,
                golden_cases=golden_cases_from_properties(task.properties),
            )
        if kinds == ["scenario"]:
            return _multi_file_input(_unit_files(files, task), task, entry)
        raise ValueError(f"{task.case_id}: unsupported single-layout property mix {kinds}")
    if task.layout == "package":
        if kinds != ["scenario"]:
            raise ValueError(
                f"{task.case_id}: package cases support only scenario properties, got {kinds}"
            )
        return _multi_file_input(_unit_files(files, task), task, entry)
    raise ValueError(f"{task.case_id}: unsupported case layout {task.layout!r}")


def golden_cases_from_properties(properties: Mapping[str, Any]) -> tuple[GoldenCase, ...]:
    """Withheld golden rows (expected outputs) — the replay oracle.

    Rows that pin an expected *raise* are skipped: the in-container replay
    driver does not catch, so such rows cannot be expressed as golden cases.
    A golden-I/O case must keep at least one expected-output row.
    """
    cases: list[GoldenCase] = []
    for kind, rows in _property_rows(properties):
        if kind != "golden_io":
            continue
        for row in rows:
            if "expected" not in row:
                continue
            cases.append(
                GoldenCase(
                    args=tuple(row.get("args", ())),
                    kwargs=dict(row.get("kwargs", {})),
                    output=row["expected"],
                )
            )
    if not cases:
        raise ValueError("no golden rows with expected outputs found")
    return tuple(cases)


def corpus_source_from_scenario(properties: Mapping[str, Any]) -> str:
    """Render the case's scenario rows as a differential corpus.

    The corpus drives :func:`evogine.gates.code_gate.check_unit_equivalence`:
    the reference implementation is replayed host-side for each sequence and
    the candidate must reproduce its outcomes.  Sequence names embed the
    property id, so a divergence witness points at the decaying property.
    """
    sequences: list[tuple[str, list[Any], list[list[Any]]]] = []
    for prop in properties.get("properties", []):
        if prop.get("kind") != "scenario":
            continue
        prop_id = str(prop.get("id", "scenario"))
        for index, row in enumerate(prop.get("rows", [])):
            construct = row.get("construct") or {}
            ctor_args = list(construct.get("args", []))
            calls = [
                [step["method"], list(step.get("args", [])), dict(step.get("kwargs", {}))]
                for step in row.get("steps", [])
            ]
            sequences.append((f"{prop_id}#{index}", ctor_args, calls))
    if not sequences:
        raise ValueError("no scenario rows found for the differential corpus")
    # repr of JSON-typed data is a valid Python literal; the runner-side
    # builder converts each triple to the tuple shape the driver validates.
    return (
        "# Generated by evogine.swarm.gates_adapter from the case's hidden scenario rows.\n"
        "def build_corpus():\n"
        f"    sequences = {sequences!r}\n"
        "    return [(name, tuple(ctor_args), calls) for name, ctor_args, calls in sequences]\n"
    )


def _multi_file_input(unit_files: dict[str, str], task: SwarmTask, entry: str) -> GateInput:
    """GateInput for a normalized unit tree against the case's reference."""
    return GateInput.multi_file(
        baseline_files=_unit_files(task.reference_files, task),
        candidate_files=unit_files,
        entry_class=entry,
        corpus_source=corpus_source_from_scenario(task.properties),
        test_suite=task.suite_path.read_text() if task.suite_path else None,
    )


def _unit_files(files: Mapping[str, str], task: SwarmTask) -> dict[str, str]:
    """Normalize a candidate/reference tree into an unprefixed unit package."""
    if task.layout == "package":
        return _strip_candidate_prefix(files)
    return _synthesize_package(files)


def _strip_candidate_prefix(files: Mapping[str, str]) -> dict[str, str]:
    """Unprefix the bench case's ``candidate/`` tree (equivalence staging)."""
    unit: dict[str, str] = {}
    for name, content in files.items():
        if not name.startswith(_CANDIDATE_PREFIX):
            raise ValueError(
                f"package case file {name!r} is missing the {_CANDIDATE_PREFIX!r} prefix"
            )
        unit[name[len(_CANDIDATE_PREFIX) :]] = content
    if "__init__.py" not in unit:
        raise ValueError("package unit is missing __init__.py")
    return unit


def _synthesize_package(files: Mapping[str, str]) -> dict[str, str]:
    """Wrap a single module in a package so both consumers resolve it.

    The visible suite imports the unit as ``candidate`` (helpers included —
    not just the entry class), and the equivalence driver loads the unit as a
    package via ``load_unit('candidate')``.  A wildcard re-export covers both.
    """
    module_name = _BASELINE_MODULE.removesuffix(".py")
    return {
        "__init__.py": f"from .{module_name} import *  # noqa: F401,F403\n",
        _BASELINE_MODULE: files[_BASELINE_MODULE],
    }


def _property_rows(properties: Mapping[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Flatten the case's hidden property groups to (kind, rows) pairs."""
    return [
        (str(prop.get("kind", "")), list(prop.get("rows", [])))
        for prop in properties.get("properties", [])
    ]


def _ops_from_result(result: GateResult) -> dict[str, int]:
    """The objective-score counts, when the gate measured them."""
    ops: dict[str, int] = {}
    if result.baseline_ops is not None:
        ops["baseline_ops"] = result.baseline_ops
    if result.candidate_ops is not None:
        ops["candidate_ops"] = result.candidate_ops
    return ops


def _outcome_from_result(result: GateResult) -> GateOutcome:
    """Encode the gate's binary decision onto the loop's counters.

    Accepted: one visible check and one hidden property passed — the gate is
    the objective authority, not a scorecard.  Rejected: both counted failed,
    with the gate's reason (witnesses included) as the failure detail so the
    reflector can diagnose; this includes never-worse rejections of
    behaviorally sound candidates.
    """
    if result.accepted:
        return GateOutcome(
            verdict=result.verdict.value,
            failure_detail=None,
            visible_passed=1,
            visible_total=1,
            properties_passed=1,
            properties_total=1,
            ops=_ops_from_result(result),
        )
    return GateOutcome(
        verdict=result.verdict.value,
        failure_detail=result.reason,
        visible_passed=0,
        visible_total=1,
        properties_passed=0,
        properties_total=1,
        ops=_ops_from_result(result),
    )


class CodeGateAdapter:
    """CandidateGate backed by the sandboxed :class:`CodeGate`.

    ``executor`` defaults to the Docker sandbox (the production boundary);
    pass :class:`~evogine.gates.sandbox.LocalSubprocessSandbox` for trusted
    deterministic development and test runs.
    """

    def __init__(self, executor: Executor | None = None, *, timeout: float = 10.0) -> None:
        self._gate = CodeGate(executor=executor)
        self._timeout = timeout

    def evaluate(self, files: Mapping[str, str], task: SwarmTask) -> GateOutcome:
        """Route one candidate tree through the full gate pipeline."""
        gate_input = build_gate_input(files, task)
        result = self._gate.evaluate(gate_input, timeout=self._timeout)
        return _outcome_from_result(result)
