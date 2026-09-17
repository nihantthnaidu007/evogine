"""Host-side candidate evaluation: the benchmark's local gate.

The harness never trusts a builder's self-assessment. This module is the
LOCAL evaluation path for the v1 skeleton (no Docker): it materializes
candidate source files under the module name ``candidate``, runs the case's
visible happy-path test suite, and then checks the case's hidden properties
host-side. properties.json is data compared against candidate behavior by
generic checkers — its expected values never enter any prompt (the standing
scan in tests/test_oracle_leak.py enforces this).

Two property kinds cover the corpus:

- ``golden_io`` — call a symbol with JSON arguments, compare the return
  value (JSON-serializable contract) or the raised exception type name.
- ``scenario`` — construct an object, then replay a deterministic method
  sequence, comparing each step's outcome. This is how stateful pressure
  (concurrent writes, load spikes) is pinned without threads or wall clocks.

Evaluation is deterministic end to end: no I/O beyond the candidate temp
tree, no randomness, no wall clock inside any case.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import itertools
import json
import sys
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from evogine.bench.task import BenchTask

CANDIDATE_PACKAGE = "candidate"

_SUITE_COUNTER = itertools.count()


@dataclass(frozen=True)
class GateOutcome:
    """Result of evaluating one candidate against one case."""

    verdict: str  # "pass" or "fail"
    failure_detail: str | None
    visible_passed: int
    visible_total: int
    properties_passed: int
    properties_total: int

    @property
    def visible_failures(self) -> int:
        return self.visible_total - self.visible_passed

    @property
    def ops(self) -> dict[str, int]:
        """Operation counts for the rollout record."""
        return {
            "visible_tests": self.visible_total,
            "visible_failures": self.visible_failures,
            "properties_checked": self.properties_total,
            "properties_failed": self.properties_total - self.properties_passed,
        }


class Gate(Protocol):
    """The evaluation seam the harness calls; the sandboxed code gate adapts here."""

    def evaluate(self, files: Mapping[str, str], task: BenchTask) -> GateOutcome: ...


@contextmanager
def candidate_import_context(files: Mapping[str, str]):
    """Materialize candidate files into a temp tree importable as ``candidate``.

    ``files`` maps candidate-tree relative paths — ``candidate.py`` for
    single-file cases, ``candidate/<module>.py`` for packages — to source
    text. The tree is import-isolated: sys.path and every ``candidate*``
    sys.modules entry are restored on exit, so cases never see each other's
    candidates. Candidate paths that escape the temp tree are rejected.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        for rel, source in files.items():
            path = root / rel
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"candidate path escapes the candidate tree: {rel!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        sys.path.insert(0, tmp)
        try:
            yield root
        finally:
            sys.path.remove(tmp)
            prefixes = (CANDIDATE_PACKAGE, CANDIDATE_PACKAGE + ".")
            for name in [n for n in sys.modules if n.startswith(prefixes)]:
                del sys.modules[name]


class PropertiesGate:
    """Local gate: visible suite + hidden properties against candidate source.

    A candidate verdict is ``pass`` only when the visible suite is clean AND
    every hidden property holds. Any candidate-caused exception (syntax
    error, import failure, crash inside a test or property row) is a ``fail``
    verdict with the error recorded — never a harness crash, never silence.
    """

    def evaluate(self, files: Mapping[str, str], task: BenchTask) -> GateOutcome:
        try:
            with candidate_import_context(files):
                candidate = importlib.import_module(CANDIDATE_PACKAGE)
                visible_passed, visible_total, visible_failures = self._run_visible_suite(
                    task.suite_path
                )
                props_passed, props_total, prop_failures = self._check_properties(
                    candidate, task.properties
                )
        except Exception as exc:  # candidate-caused; recorded as a fail verdict
            return GateOutcome(
                verdict="fail",
                failure_detail=f"candidate failed to evaluate: {type(exc).__name__}: {exc}",
                visible_passed=0,
                visible_total=0,
                properties_passed=0,
                properties_total=0,
            )

        if visible_failures:
            verdict = "fail"
            detail = "visible suite failures: " + "; ".join(visible_failures)
        elif props_passed < props_total:
            verdict = "fail"
            detail = "property failures: " + "; ".join(prop_failures)
        else:
            verdict = "pass"
            detail = None
        return GateOutcome(
            verdict=verdict,
            failure_detail=detail,
            visible_passed=visible_passed,
            visible_total=visible_total,
            properties_passed=props_passed,
            properties_total=props_total,
        )

    def _run_visible_suite(self, suite_path: Path) -> tuple[int, int, list[str]]:
        """Run the case's visible test functions against the ``candidate`` module."""
        name = f"evogine_case_suite_{next(_SUITE_COUNTER)}"
        spec = importlib.util.spec_from_file_location(name, suite_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load visible suite: {suite_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
            tests = [
                (test_name, fn)
                for test_name, fn in inspect.getmembers(module, inspect.isfunction)
                if test_name.startswith("test_")
            ]
            failures = []
            for test_name, fn in tests:
                try:
                    fn()
                except Exception as exc:  # a test failure, not a gate bug
                    failures.append(f"{test_name}: {type(exc).__name__}: {exc}")
        finally:
            sys.modules.pop(name, None)
        return len(tests) - len(failures), len(tests), failures

    def _check_properties(
        self, candidate: ModuleType, properties: dict
    ) -> tuple[int, int, list[str]]:
        """Check every hidden property row; returns (rows passed, rows total, details)."""
        passed = 0
        total = 0
        failures: list[str] = []
        for prop in properties["properties"]:
            entry = _resolve_entry(candidate, prop.get("entry", properties["entry"]))
            kind = prop["kind"]
            if kind not in ("golden_io", "scenario"):
                raise ValueError(f"unknown property kind: {kind!r}")
            for index, row in enumerate(prop["rows"]):
                total += 1
                if kind == "golden_io":
                    detail = _check_golden_row(entry, row)
                else:
                    detail = _check_scenario_row(entry, row)
                if detail is None:
                    passed += 1
                else:
                    failures.append(f"{prop['id']}[{index}]: {detail}")
        return passed, total, failures


def _resolve_entry(candidate: ModuleType, dotted: str) -> Any:
    """Resolve a dotted entry symbol on the candidate module or package."""
    obj: Any = candidate
    for depth, part in enumerate(dotted.split(".")):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            if depth == 0 and isinstance(obj, ModuleType):
                importlib.import_module(f"{obj.__name__}.{part}")
                obj = getattr(obj, part)
            else:
                raise
    return obj


def _check_golden_row(entry: Any, row: dict) -> str | None:
    """Check one call row; None means the row held, a string explains the miss."""
    args = row.get("args", [])
    kwargs = row.get("kwargs", {})
    raises = row.get("raises")
    try:
        actual = entry(*args, **kwargs)
    except Exception as exc:
        name = type(exc).__name__
        if raises is not None:
            return None if name == raises else f"raised {name}, expected {raises}"
        return f"raised {name}: {exc}"
    if raises is not None:
        return f"returned {_short(actual)}, expected raise {raises}"
    if "expected" not in row:
        raise ValueError("property row needs 'expected' or 'raises'")
    return _compare(row["expected"], actual)


def _check_scenario_row(ctor: Any, row: dict) -> str | None:
    """Check one construct-and-call scenario row against the candidate."""
    construct = row.get("construct", {})
    construct_raises = row.get("construct_raises")
    try:
        obj = ctor(*construct.get("args", []), **construct.get("kwargs", {}))
    except Exception as exc:
        name = type(exc).__name__
        if construct_raises is not None:
            return (
                None
                if name == construct_raises
                else f"constructor raised {name}, expected {construct_raises}"
            )
        return f"constructor raised {name}: {exc}"
    if construct_raises is not None:
        return f"constructed, expected constructor raise {construct_raises}"
    for index, step in enumerate(row.get("steps", [])):
        method_name = step["method"]
        method = getattr(obj, method_name, None)
        if method is None or not callable(method):
            return f"step {index}: no callable method {method_name!r}"
        detail = _check_golden_row(method, step)
        if detail is not None:
            return f"step {index} ({method_name}): {detail}"
    return None


def _to_jsonable(value: Any) -> Any:
    """Normalize a candidate result to the JSON result contract.

    Tuples become lists so replay comparison is structural; anything outside
    the contract (sets, custom objects) raises — case contracts return
    JSON-serializable values by design.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    raise TypeError(f"{type(value).__name__} is outside the JSON result contract")


def _compare(expected: Any, actual: Any) -> str | None:
    """Strict JSON equality; None when the result matches the frozen oracle."""
    try:
        actual = _to_jsonable(actual)
    except (TypeError, ValueError) as exc:
        return f"non-JSON-serializable result: {exc}"
    if isinstance(expected, bool) != isinstance(actual, bool):
        return f"expected {expected!r}, got {actual!r} (bool/int mismatch)"
    if expected != actual:
        return f"expected {json.dumps(expected, sort_keys=True)}, got {_short(actual)}"
    return None


def _short(value: Any) -> str:
    """Compact one result for a failure detail line."""
    try:
        text = json.dumps(_to_jsonable(value))
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 120 else text[:117] + "..."
