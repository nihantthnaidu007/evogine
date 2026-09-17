#!/usr/bin/env python3
"""Freeze case properties from the reference implementation.

Usage: ``python tools/freeze_properties.py --case <case_id>`` (or ``--all``).

Reads each case's properties skeleton (rows with inputs, no expectations),
runs the inputs against the case REFERENCE, and writes the golden values
(expected returns, raised exception type names) into ``properties.json``.
The frozen values are the reference's recorded behavior on hand-chosen
pressure inputs — the golden-I/O discipline: choose the inputs, let the
clean implementation define correct output.

Freeze resolves symbols and normalizes results through the same gate code
paths the checker uses, so what freeze writes is exactly what the gate
compares. Re-running after an intentional reference change re-freezes;
drift otherwise surfaces as a tests/test_cases.py failure.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from evogine.bench.gate import (
    _resolve_entry,
    _to_jsonable,
    candidate_import_context,
)
from evogine.bench.task import BenchTask, discover_case_ids, load_task


def freeze_golden_row(entry: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Run one input row against the reference and record its outcome.

    Row inputs are deep-copied before the call: a reference that mutates its
    arguments (an append-style sink) must not corrupt the recorded inputs.
    """
    out = {
        "args": copy.deepcopy(row.get("args", [])),
        "kwargs": copy.deepcopy(row.get("kwargs", {})),
    }
    # Call arguments must be distinct objects from the recorded inputs: a
    # reference that mutates its arguments (append-style sinks) corrupts
    # whatever object it is handed, and the recorded inputs must stay clean.
    call_args = copy.deepcopy(out["args"])
    call_kwargs = copy.deepcopy(out["kwargs"])
    try:
        out["expected"] = _to_jsonable(entry(*call_args, **call_kwargs))
    except Exception as exc:  # noqa: BLE001 — a raised error is the golden value
        out["raises"] = type(exc).__name__
    return out


def freeze_scenario_row(entry: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Replay one construct-and-call row against the reference."""
    construct = row.get("construct", {})
    out: dict[str, Any] = {
        "construct": {
            "args": copy.deepcopy(construct.get("args", [])),
            "kwargs": copy.deepcopy(construct.get("kwargs", {})),
        }
    }
    # Recorded inputs stay pristine even when the reference mutates its
    # arguments: the call gets its own copies.
    construct_args = copy.deepcopy(out["construct"]["args"])
    construct_kwargs = copy.deepcopy(out["construct"]["kwargs"])
    try:
        # Mirror the gate's constructor call exactly.
        obj = entry(*construct_args, **construct_kwargs)
    except Exception as exc:  # noqa: BLE001 — a constructor raise is the golden value
        out["construct_raises"] = type(exc).__name__
        return out
    steps: list[dict[str, Any]] = []
    for step in row.get("steps", []):
        frozen = {
            "method": step["method"],
            "args": copy.deepcopy(step.get("args", [])),
            "kwargs": copy.deepcopy(step.get("kwargs", {})),
        }
        step_args = copy.deepcopy(frozen["args"])
        step_kwargs = copy.deepcopy(frozen["kwargs"])
        method = getattr(obj, step["method"])
        try:
            frozen["expected"] = _to_jsonable(method(*step_args, **step_kwargs))
        except Exception as exc:  # noqa: BLE001 — a step raise is the golden value
            frozen["raises"] = type(exc).__name__
        steps.append(frozen)
    out["steps"] = steps
    return out


def freeze_case(case_id: str) -> None:
    """Freeze one case's properties.json in place."""
    task: BenchTask = load_task(case_id)
    doc = task.properties
    with candidate_import_context(task.reference_files):
        # The reference module exists only at runtime (materialized inside the
        # import context); import_module resolves it through the same import
        # machinery the gate uses, and types as the ModuleType _resolve_entry
        # expects — a static `import candidate` is invisible to the type checker.
        candidate = importlib.import_module("candidate")

        entry = _resolve_entry(candidate, doc["entry"])
        frozen_properties: list[dict[str, Any]] = []
        for prop in doc["properties"]:
            frozen_rows = [
                freeze_golden_row(entry, row)
                if prop["kind"] == "golden_io"
                else freeze_scenario_row(entry, row)
                for row in prop["rows"]
            ]
            frozen_properties.append({"id": prop["id"], "kind": prop["kind"], "rows": frozen_rows})

    payload = {
        "case_id": doc["case_id"],
        "entry": doc["entry"],
        "properties": frozen_properties,
    }
    # Round-trip through JSON so the file holds exactly what the gate compares.
    payload = json.loads(json.dumps(payload, allow_nan=False))
    out_path = task.root / "properties.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = sum(len(prop["rows"]) for prop in frozen_properties)
    print(f"froze {case_id}: {len(frozen_properties)} properties, {rows} rows -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="freeze a single case by id")
    group.add_argument("--all", action="store_true", help="freeze every discovered case")
    args = parser.parse_args()

    ids = discover_case_ids() if args.all else [args.case]
    if not args.all and not ids[0]:
        parser.error("--case requires a case id")
    failures = 0
    for case_id in ids:
        try:
            freeze_case(case_id)
        except Exception as exc:  # noqa: BLE001 — a bad case is reported, not fatal to siblings
            failures += 1
            print(f"FAILED {case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
