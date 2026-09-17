"""Analyze benchmark rollout records into spend, latency, and paired deltas.

Reads one or more ``rollouts.jsonl`` files (the phase0 schema the loop
appends) and reports, per ``case_id x arm``:

- token spend and model-call counts by stage (builder/verifier/reflector),
- the accepted-bit paired deltas warm vs cold over seed-paired candidates
  (arms of one ``run_id`` share ``(generation, candidate_index)`` seeds, so
  only retrieval differs),
- per-case medians and MAD noise floors over run-level outcomes
  (seed learn_0009: medians, never means),
- the learn_0011 rejection counter (candidates that passed tests but were
  still rejected by never-worse), per arm,
- learn_0012 determinism coverage (every candidate/verifier record carries
  the playbook block with a seed).

Usage::

    python tools/analyze_rollouts.py runs/pilot/rollouts.jsonl
    python tools/analyze_rollouts.py runs/pilot/rollouts.jsonl --out results.csv

The CSV (when requested) holds one row per case x arm: medians, MADs, and
paired deltas of the run-level outcome score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

#: Candidate/verifier records must carry this determinism block (learn_0012).
PLAYBOOK = "learn_0012"


def _median_abs_deviation(values: list[float]) -> float:
    """MAD about the median, scaled to a consistent noise floor (1.4826 x MAD)."""
    if len(values) < 2:
        return 0.0
    center = median(values)
    deviations = [abs(value - center) for value in values]
    return 1.4826 * median(deviations)


def load_rollouts(paths: list[Path]) -> list[dict[str, Any]]:
    """Load rollout records from every file; a corrupt line raises, never skips."""
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as error:
                    message = f"corrupt rollout at {path}, line {line_number}: {error}"
                    raise RuntimeError(message) from error
    return records


def _accepted_bit(record: dict[str, Any]) -> int:
    """The gate's binary decision, from the rollout record's encoding.

    The adapter encodes acceptance as ``failure_detail=None``; a rejected
    candidate — including never-worse rejections of behaviorally sound
    repairs — carries the gate's reason in ``failure_detail`` instead.
    """
    return int(record.get("failure_detail") is None)


def _never_worse_rejection(record: dict[str, Any]) -> bool:
    """learn_0011 counter: passed tests but rejected by never-worse."""
    detail = record.get("failure_detail") or ""
    return record.get("verdict") == "pass_tests" and "never-worse" in detail


def paired_deltas(records: list[dict[str, Any]]) -> dict[str, dict[str, list[int]]]:
    """Seed-paired accepted-bit deltas per case: warm minus cold.

    Groups candidate records by (case_id, run_id, generation, candidate_index);
    a pair is complete when both arms recorded the same slot.
    """
    slots: dict[tuple[str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record.get("stage") != "candidate":
            continue
        arm = record.get("config", {}).get("arm")
        key = (
            str(record.get("case_id")),
            str(record.get("run_id")),
            int(record.get("generation") or 0),
            int(record.get("candidate_index") or 0),
        )
        slots[key][str(arm)] = record

    deltas: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"warm": [], "cold": []})
    for (case_id, _run_id, _gen, _idx), arms in sorted(slots.items()):
        if "warm" in arms and "cold" in arms:
            delta = _accepted_bit(arms["warm"]) - _accepted_bit(arms["cold"])
            deltas[case_id]["warm"].append(delta)
    return deltas


def spend_by_stage(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Token spend and call counts by stage across the loaded records."""
    summary: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0}
    )
    for record in records:
        stage = str(record.get("stage"))
        tokens = record.get("tokens") or {}
        summary[stage]["calls"] += 1
        summary[stage]["input_tokens"] += int(tokens.get("input_tokens") or 0)
        summary[stage]["output_tokens"] += int(tokens.get("output_tokens") or 0)
        summary[stage]["latency_ms"] += float(record.get("latency_ms") or 0.0)
    return dict(summary)


def determinism_coverage(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count candidate/verifier records missing the learn_0012 block."""
    checked = missing = 0
    for record in records:
        if record.get("stage") not in ("candidate", "verifier"):
            continue
        checked += 1
        config = record.get("config") or {}
        if config.get("playbook") != PLAYBOOK or not config.get("seed"):
            missing += 1
    return checked, missing


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the per-case/arm analysis dict (also the CSV row source)."""
    cases = sorted({str(record.get("case_id")) for record in records})
    deltas = paired_deltas(records)
    report: dict[str, Any] = {
        "cases": {},
        "spend_by_stage": spend_by_stage(records),
        "determinism": determinism_coverage(records),
    }
    for case_id in cases:
        case_records = [r for r in records if r.get("case_id") == case_id]
        case_report: dict[str, Any] = {"arms": {}}
        for arm in ("cold", "warm"):
            arm_candidates = [
                r
                for r in case_records
                if r.get("stage") == "candidate"
                and (r.get("config") or {}).get("arm") == arm
            ]
            # learn_0009: run-level outcome = accepted candidates per run_id;
            # the median and its MAD floor come from repeats, never means.
            per_run: dict[str, int] = defaultdict(int)
            for r in arm_candidates:
                per_run[str(r.get("run_id"))] += _accepted_bit(r)
            outcomes = [per_run[run_id] for run_id in sorted(per_run)]
            case_report["arms"][arm] = {
                "candidates": len(arm_candidates),
                "accepted": sum(_accepted_bit(r) for r in arm_candidates),
                "per_run_accepted": outcomes,
                "median_outcome": median(outcomes) if outcomes else None,
                "mad_floor": _median_abs_deviation([float(v) for v in outcomes]),
                "never_worse_rejections": sum(_never_worse_rejection(r) for r in arm_candidates),
            }
        case_report["paired_deltas"] = deltas.get(case_id, {}).get("warm", [])
        report["cases"][case_id] = case_report
    return report


def main(argv: list[str] | None = None) -> int:
    """Print the analysis summary; optionally write the results CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", type=Path, help="rollouts.jsonl paths")
    parser.add_argument("--out", type=Path, default=None, help="optional results CSV path")
    args = parser.parse_args(argv)

    records = load_rollouts(args.rollouts)
    report = analyze(records)
    print(json.dumps(report, indent=2, default=str))

    if args.out is not None:
        with args.out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
            writer.writerow(
                [
                    "case_id",
                    "arm",
                    "candidates",
                    "accepted",
                    "median_outcome",
                    "mad_floor",
                    "never_worse_rejections",
                    "paired_deltas_warm_minus_cold",
                ]
            )
            for case_id, case_report in report["cases"].items():
                for arm, arm_report in case_report["arms"].items():
                    writer.writerow(
                        [
                            case_id,
                            arm,
                            arm_report["candidates"],
                            arm_report["accepted"],
                            arm_report["median_outcome"],
                            arm_report["mad_floor"],
                            arm_report["never_worse_rejections"],
                            ";".join(str(d) for d in case_report["paired_deltas"]),
                        ]
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
