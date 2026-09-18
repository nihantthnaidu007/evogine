"""Phase-3 compounding benchmark driver: multi-generation paired runs.

Phase-2 evidence (reports/CROSSCASE.md): warm retrieval was never worse than
cold on 9 of 10 cases with one strict win (drift-version-skew), at +9.4%
token cost. Phase 3 asks whether that advantage COMPOUNDS: the warm arm
retains its learning store across generation blocks while the cold control
cannot (it is stateless by construction).

Design, per the approved phase plan:

- Three cases. The proven discriminator (drift-version-skew) keeps the full
  phase-2 4-repeat noise floor; a saturated control (billing-day-offset) and
  a floor control (audit-ring-drop) run 3 repeats each — their phase-2
  outcome distributions were degenerate (all-identical), and any variance
  appearing at 3 repeats triggers extension rather than analysis.
- A case runs G generation blocks of R paired repeats. Each repeat is one
  both-arms ``evogine run`` invocation, phase-2-identical in every respect:
  same config schema, executor, model, learn_0012 seeds, retrieval_k,
  candidates. The warm arm's engine persists across blocks (one lineage per
  case, bootstrapped from the seed pack — verified to reproduce the phase-2
  warm-start retrieval state); the cold control opens no store.
- Between blocks a scoped promotion pass runs the production promotion path
  (evogine.swarm.promotion) over the block's own records: every recorded
  gate evaluation is reconstructed as a replayable RecordedRun — the
  candidate files are re-evaluated through the real code gate locally and
  the fresh verdict is cross-checked against history before recording —
  then ``finish_promotion`` decides each ephemeral proposal against its own
  invocation's recorded runs. The regression guard is scoped to newly
  promoted records only: the 12 seed records are phase-2-validated on these
  very cases, and guarding them would multiply replay cost without new
  information. The scoping is disclosed in the findings.

Operational rules baked in (phase-2 lessons):

- No polling loops: invocations run sequentially under the CLI's own bounded
  timeouts; a hung request degrades to a typed failed candidate (PR #29).
- Budget guard: after each invocation and pass the driver projects total
  phase spend from actuals; past the stop line it exits loudly.
- Checkpoint-friendly state: records, reports, ledgers, and engine archives
  land on disk at every block boundary for commit and push.

Usage::

    python tools/run_compounding.py --mode scripted            # free dry run
    python tools/run_compounding.py --mode live                # all cases
    python tools/run_compounding.py --mode live --cases drift-version-skew
    python tools/run_compounding.py --analyze-only             # results only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_rollouts import analyze, load_rollouts

from evogine.bench.harness import SwarmModelBridge, to_swarm_task
from evogine.bench.task import load_task
from evogine.cli import ScriptedModel
from evogine.engine.bootstrap import load_seeds
from evogine.engine.records import LearningRecord
from evogine.engine.store import Engine
from evogine.gates.code_gate import CodeGate
from evogine.gates.memory_gate import MemoryGate, MemoryGateConfig
from evogine.gates.replay import RecordedRun, RunLog
from evogine.gates.sandbox import LocalSubprocessSandbox
from evogine.swarm.gates_adapter import build_gate_input
from evogine.swarm.promotion import (
    BuilderApplier,
    PromotionWiring,
    check_promoted_for_regression,
    finish_promotion,
    recorded_run_id,
)
from evogine.swarm.roles import AnthropicModel, Builder, RoleCall
from evogine.swarm.runner import (
    SwarmConfig,
    extract_candidate_files,
    serialize_candidate_files,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "compounding"
SEEDS_DIR = REPO / "src" / "evogine" / "seeds"

#: The code-gate timeout the production loop runs under (the adapter default);
#: the pass's re-evaluations must match it exactly or the cross-check lies.
GATE_TIMEOUT_S = 10.0

#: Budget stop line per case, proportional to its invocation share of 30.
STOP_LINE_TOTAL = 1_300_000
TOTAL_INVOCATIONS = 30
#: Actuals from phase 2: ~27k tokens per both-arms invocation; a promotion
#: decide replays ~5 runs x 3 repeats x ~1.3k tokens.
PER_INVOCATION_PROJECTION = 28_000
PER_PASS_PROJECTION = 20_000

#: Deterministic provenance source format the runner writes on proposals.
_PROVENANCE_RUN_RE = re.compile(r"run (\S+)/case ")


@dataclass(frozen=True)
class CasePlan:
    """One case's phase-3 shape: generation blocks and paired repeats."""

    case_id: str
    generations: int
    repeats: int


CASES: dict[str, CasePlan] = {
    "drift-version-skew": CasePlan("drift-version-skew", generations=3, repeats=4),
    "billing-day-offset": CasePlan("billing-day-offset", generations=3, repeats=3),
    "audit-ring-drop": CasePlan("audit-ring-drop", generations=3, repeats=3),
}


class BudgetStop(RuntimeError):
    """The projected phase spend crossed the approved plan's stop line."""


def phase3_invocation_config(base: Path, provider: str | None, engine_root: Path) -> str:
    """Rewrite the base config per invocation: runs_dir, engine_root, provider.

    Line-rewrite discipline from the phase-2 driver: the produced YAML stays
    byte-similar to the audited base, only the pinned keys change.
    """
    lines: list[str] = []
    for line in base.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("runs_dir:"):
            lines.append("runs_dir: rollout")
        elif line.lstrip().startswith("engine_root:"):
            lines.append(f"engine_root: {engine_root}")
        elif provider and line.strip() in ("provider: anthropic", "provider: scripted"):
            lines.append(f"  provider: {provider}")
        else:
            lines.append(line)
    text = "\n".join(lines) + "\n"
    for required in ("runs_dir: rollout", f"engine_root: {engine_root}"):
        if required not in text:
            raise ValueError(f"config rewrite failed: {required!r} missing")
    return text


def _token_sums(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Total (input, output) tokens over rollout records."""
    tokens_in = sum(int((r.get("tokens") or {}).get("input_tokens") or 0) for r in records)
    tokens_out = sum(int((r.get("tokens") or {}).get("output_tokens") or 0) for r in records)
    return tokens_in, tokens_out


def _accepted(record: dict[str, Any]) -> int:
    """The analyzer's acceptance encoding: failure_detail None means accepted."""
    return int(record.get("failure_detail") is None)


def _degraded_markers(records: list[dict[str, Any]]) -> list[str]:
    """Typed provider-fault evidence in a run's records (disclosed, not polled)."""
    pattern = re.compile(r"APITimeoutError|APIStatusError|APIConnectionError|overloaded|rate.?limit", re.IGNORECASE)
    markers: list[str] = []
    for record in records:
        detail = record.get("failure_detail") or ""
        match = pattern.search(str(detail))
        if match:
            markers.append(f"{record.get('stage')}:{match.group(0)}")
    return markers


def ensure_engine(case_id: str) -> Path:
    """Return the case's lineage engine directory, bootstrapping if fresh."""
    engine_dir = RUNS / "engine" / case_id
    if (engine_dir / "records").is_dir():
        return engine_dir
    engine = Engine(engine_dir, actor="bench-driver")
    loaded = load_seeds(engine, str(SEEDS_DIR))
    print(f"[{case_id}] bootstrapped lineage engine: {len(loaded)} seed records", flush=True)
    return engine_dir


def run_one(
    plan: CasePlan,
    generation: int,
    repeat: int,
    base_config: Path,
    provider: str | None,
) -> dict[str, Any]:
    """Run one (case, generation, repeat) both-arms invocation.

    Returns a manifest row; a failed invocation raises SystemExit with the
    partial records archived (phase-2 discipline: never re-execute completed
    work, never overwrite evidence).
    """
    case = plan.case_id
    report_path = RUNS / "reports" / f"{case}-g{generation}-r{repeat}.json"
    records_file = RUNS / f"rollouts.{case}.g{generation}.jsonl"
    if report_path.exists():
        print(f"[{case} g{generation} r{repeat}] already complete; skipping", flush=True)
        row = dict(json.loads(report_path.read_text(encoding="utf-8")).get("manifest", {}))
        row["_skipped"] = True
        return row

    engine_dir = ensure_engine(case)
    scratch = RUNS / f"scratch-{case}-g{generation}-r{repeat}"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "rollout").mkdir(exist_ok=True)
    cfg_path = scratch / "config.yaml"
    cfg_path.write_text(phase3_invocation_config(base_config, provider, engine_dir), encoding="utf-8")

    venv_exe = Path(sys.executable).parent / "evogine"
    if not venv_exe.exists():
        raise SystemExit("no evogine console script next to this python")
    cmd = [str(venv_exe), "run", "--case", case, "--config", str(cfg_path), "--allow-local-live"]
    print(f"[{case} g{generation} r{repeat}] running:", " ".join(cmd), flush=True)
    started = time.monotonic()
    with (scratch / "stdout.log").open("w", encoding="utf-8") as out, (scratch / "stderr.log").open(
        "w", encoding="utf-8"
    ) as err:
        result = subprocess.run(
            cmd, cwd=scratch, env=dict(os.environ), stdout=out, stderr=err, check=False
        )
    seconds = round(time.monotonic() - started, 1)

    produced = scratch / "rollout" / "rollouts.jsonl"
    if result.returncode != 0:
        if produced.exists():
            archive = RUNS / f"rollouts.preflight.{case}-g{generation}-r{repeat}.jsonl"
            archive.write_text(produced.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[{case} g{generation} r{repeat}] FAILED rc={result.returncode}; "
                  f"partial records archived to {archive.name}", flush=True)
        else:
            print(f"[{case} g{generation} r{repeat}] FAILED rc={result.returncode} (no records)", flush=True)
        raise SystemExit(
            f"invocation {case} g{generation} r{repeat} failed; inspect {scratch}/stderr.log"
        )

    records = load_rollouts([produced])
    with records_file.open("a", encoding="utf-8") as dst:
        for record in records:
            dst.write(json.dumps(record, separators=(",", ":")) + "\n")

    stdout_text = (scratch / "stdout.log").read_text(encoding="utf-8")
    report: dict[str, Any] = {"unparsed": True}
    try:
        parsed = json.loads(stdout_text[stdout_text.index("{"):])
        if isinstance(parsed, dict):
            report = parsed
    except (json.JSONDecodeError, ValueError):
        report = {"unparsed": True, "stdout_chars": len(stdout_text)}
    tokens_in, tokens_out = _token_sums(records)
    warm_accepted = sum(
        _accepted(r) for r in records if r.get("stage") == "candidate"
        and (r.get("config") or {}).get("arm") == "warm"
    )
    cold_accepted = sum(
        _accepted(r) for r in records if r.get("stage") == "candidate"
        and (r.get("config") or {}).get("arm") == "cold"
    )
    row = {
        "run_id": str(records[0].get("run_id")) if records else "",
        "case_id": case,
        "generation": generation,
        "repeat": repeat,
        "records": len(records),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "warm_accepted": warm_accepted,
        "cold_accepted": cold_accepted,
        "degraded": ";".join(_degraded_markers(records)),
        "seconds": seconds,
    }
    report_path.write_text(
        json.dumps({"manifest": row, "report": report}, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"[{case} g{generation} r{repeat}] ok: warm={warm_accepted}/4 cold={cold_accepted}/4 "
        f"tokens={tokens_in + tokens_out} in {seconds}s",
        flush=True,
    )
    shutil.rmtree(scratch, ignore_errors=True)
    row["_skipped"] = False
    return row


class TokenCountingBuilder(Builder):
    """Builder that counts its replay calls — the promotion pass's token feed."""

    def __init__(self, inner: Builder) -> None:
        super().__init__(
            inner.model,
            inner.model_name,
            temperature=inner.temperature,
            thinking=inner.thinking,
            max_tokens=inner.max_tokens,
        )
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def apply_learning(
        self,
        record: LearningRecord,
        candidate_files: Mapping[str, str],
        *,
        multi_file: bool,
        seed: str,
    ) -> RoleCall:
        call = super().apply_learning(record, candidate_files, multi_file=multi_file, seed=seed)
        self.calls += 1
        self.input_tokens += call.response.input_tokens or 0
        self.output_tokens += call.response.output_tokens or 0
        return call


def promotion_pass(plan: CasePlan, generation: int, provider: str, model_name: str) -> dict[str, Any]:
    """Scoped promotion pass after one generation block; returns the ledger entry.

    Production path, scoped two ways (disclosed in the findings): proposals
    are decided against their OWN invocation's recorded runs, and the
    regression guard covers newly promoted records only.
    """
    case = plan.case_id
    engine_dir = ensure_engine(case)
    engine = Engine(engine_dir, actor="bench-promotion")
    # Production convention (_cmd_evolve): the run log lives in the engine dir.
    run_log = RunLog(engine_dir / "runlog.jsonl")
    bench_task = load_task(case)
    task = to_swarm_task(bench_task)
    gate = CodeGate(executor=LocalSubprocessSandbox())

    if provider == "scripted":
        # ScriptedModel speaks the bench protocol; the roles' Builder expects
        # the swarm protocol — the same bridge _cmd_evolve uses.
        inner: Builder = Builder(
            SwarmModelBridge(
                ScriptedModel(
                    builder_reply=serialize_candidate_files(dict(bench_task.baseline_files))
                )
            ),
            model_name,
            temperature=0.0,
            thinking=False,
            max_tokens=4096,
        )
    else:
        inner = Builder(AnthropicModel(), model_name, temperature=0.0, thinking=False, max_tokens=4096)
    counter = TokenCountingBuilder(inner)
    memory_gate = MemoryGate(
        run_log=run_log,
        applier=BuilderApplier(counter),
        executor=LocalSubprocessSandbox(),
        config=MemoryGateConfig(),
    )
    wiring = PromotionWiring(run_log=run_log, memory_gate=memory_gate)

    # 1. Reconstruct the block's gate evaluations as replayable runs.
    block_file = RUNS / f"rollouts.{case}.g{generation}.jsonl"
    records = load_rollouts([block_file])
    existing = {run.run_id for run in run_log.runs()}
    warm_runs: dict[str, dict[str, list[str]]] = {}
    accepted_runs: list[str] = []
    for record in records:
        run_id = str(record.get("run_id"))
        stage = str(record.get("stage"))
        if stage not in ("baseline", "candidate"):
            # Verifier/reflector records are model-call evidence in the
            # verifier's own verdict taxonomy — the live loop records no
            # replayable run for them, so there is nothing to reconstruct.
            continue
        arm = str((record.get("config") or {}).get("arm"))
        files = extract_candidate_files(str(record.get("extracted_code")), task.layout)
        if files is None:
            # Live extraction failed: nothing re-evaluable was recorded then,
            # matching record_candidate_run's no-input-no-record contract.
            print(f"[{case} g{generation}] {run_id}/{stage}: no candidate files "
                  "(live extraction failure); skipping reconstruction", flush=True)
            continue
        gate_input = build_gate_input(files, task)
        fresh = gate.evaluate(gate_input, timeout=GATE_TIMEOUT_S)
        recorded_acceptance = record.get("failure_detail") is None
        if str(fresh.verdict.value) != str(record.get("verdict")) or fresh.accepted != recorded_acceptance:
            raise RuntimeError(
                f"gate nondeterminism on {run_id}/{stage}: recorded "
                f"({record.get('verdict')}, {recorded_acceptance}) vs fresh "
                f"({fresh.verdict.value}, {fresh.accepted}) — stopping, evidence is unsound"
            )
        if stage == "baseline":
            replay_id = recorded_run_id(
                SwarmConfig(run_id=run_id, arm=arm), task, stage
            )
        else:
            replay_id = recorded_run_id(
                SwarmConfig(run_id=run_id, arm=arm),
                task,
                stage,
                generation=int(record.get("generation") or 0),
                candidate_index=int(record.get("candidate_index") or 0),
            )
        if replay_id in existing:
            continue
        run_log.record(
            RecordedRun(
                run_id=replay_id,
                case_id=case,
                gate_input=gate_input,
                cold=fresh,
                note=f"phase-3 block {generation} reconstruction ({provider})",
            )
        )
        existing.add(replay_id)
        if arm == "warm":
            bucket = warm_runs.setdefault(run_id, {"baseline": [], "candidates": []})
            bucket["candidates" if stage == "candidate" else "baseline"].append(replay_id)
        if fresh.accepted and stage == "candidate":
            accepted_runs.append(replay_id)

    # 2. Decide each new ephemeral proposal against its own invocation's runs.
    decided_path = RUNS / "decided" / f"{case}.jsonl"
    decided_ids: set[str] = set()
    if decided_path.exists():
        decided_ids = {
            str(json.loads(line).get("learning_id"))
            for line in decided_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    proposals: list[tuple[str, str]] = []  # (learning_id, source run_id)
    for record_file in sorted((engine_dir / "records").glob("*.json")):
        payload = json.loads(record_file.read_text(encoding="utf-8"))
        if payload.get("status") != "ephemeral":
            continue
        learning_id = str(payload.get("id"))
        if learning_id in decided_ids:
            continue
        source = str((payload.get("provenance") or {}).get("source") or "")
        match = _PROVENANCE_RUN_RE.search(source)
        proposals.append((learning_id, match.group(1) if match else ""))

    decisions: list[dict[str, Any]] = []
    for learning_id, source_run in proposals:
        source_bucket = warm_runs.get(source_run)
        if source_bucket is None:
            print(f"[{case} g{generation}] proposal {learning_id}: source run {source_run!r} "
                  "not in this block; skipping (disclosed)", flush=True)
            continue
        queued = source_bucket["baseline"] + source_bucket["candidates"]
        report = finish_promotion(wiring, engine, {"learning_id": learning_id}, [], queued)
        status = engine.get(learning_id).status
        decisions.append(
            {"learning_id": learning_id, "decision": report["decision"],
             "record_status": status, "queued_run_ids": report["queued_run_ids"]}
        )
        with decided_path.open("a", encoding="utf-8") as dst:
            dst.write(json.dumps({"learning_id": learning_id, "decision": report["decision"]}) + "\n")

    # 3. Regression guard scoped to what this phase created.
    promoted = [engine.get(d["learning_id"]) for d in decisions if d["record_status"] == "promoted"]
    demoted = check_promoted_for_regression(wiring, engine, promoted, accepted_runs)

    entry: dict[str, Any] = {
        "case_id": case,
        "generation": generation,
        "provider": provider,
        "reconstructed_runs": len(run_log.runs()),
        "decisions": decisions,
        "promoted": [record.id for record in promoted],
        "demoted": demoted,
        "replay_tokens": {"calls": counter.calls, "input": counter.input_tokens, "output": counter.output_tokens},
    }
    with (RUNS / "promotion-passes.jsonl").open("a", encoding="utf-8") as dst:
        dst.write(json.dumps(entry) + "\n")
    print(
        f"[{case} g{generation}] promotion pass: {len(decisions)} decided, "
        f"{len(promoted)} promoted, {len(demoted)} demoted, replay tokens "
        f"{counter.input_tokens + counter.output_tokens}",
        flush=True,
    )
    return entry


def _archive_engine(case: str, generation: int) -> None:
    """Tarball the case's lineage engine at a block boundary (loss-proofing)."""
    engine_dir = RUNS / "engine" / case
    archive = RUNS / "engine-archive" / f"engine-{case}-g{generation}.tgz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(engine_dir, arcname=f"engine/{case}")


def spend_snapshot(done: int, remaining: int, pass_count: int, tokens_so_far: int) -> dict[str, Any]:
    """The budget guard's actuals-plus-projection view."""
    projected = tokens_so_far + remaining * PER_INVOCATION_PROJECTION + pass_count * PER_PASS_PROJECTION
    return {
        "tokens_so_far": tokens_so_far,
        "projected_total": projected,
        "invocations_done": done,
        "invocations_remaining": remaining,
        "passes_projected": pass_count,
    }


def _model_name(base_config: Path) -> str:
    """The base config's model.name — the pass's replay builder uses it too."""
    for line in base_config.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("name:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"config {base_config} has no model name")


def _case_tokens_so_far(case: str) -> int:
    """Token actuals from completed reports and promotion passes (resume safety)."""
    tokens = 0
    for path in sorted((RUNS / "reports").glob(f"{case}-g*-r*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8")).get("manifest") or {}
        tokens += int(manifest.get("tokens_in") or 0) + int(manifest.get("tokens_out") or 0)
    passes_file = RUNS / "promotion-passes.jsonl"
    if passes_file.exists():
        for line in passes_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("case_id") != case:
                continue
            replay = entry.get("replay_tokens") or {}
            tokens += int(replay.get("input") or 0) + int(replay.get("output") or 0)
    return tokens


def _pass_generations_done(case: str) -> set[int]:
    """Generations whose promotion pass already wrote a ledger entry."""
    passes_file = RUNS / "promotion-passes.jsonl"
    done: set[int] = set()
    if passes_file.exists():
        for line in passes_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("case_id") == case:
                done.add(int(entry.get("generation") or 0))
    return done


def run_case(
    plan: CasePlan,
    base_config: Path,
    provider: str | None,
    resume: bool,
    skip_final_pass: bool,
) -> None:
    """Run one case's full generation ladder: blocks of repeats + passes."""
    case = plan.case_id
    records_glob = RUNS / f"rollouts.{case}.g*.jsonl"
    if not resume and list(RUNS.glob(f"rollouts.{case}.g*.jsonl")):
        raise SystemExit(
            f"refusing to run: {records_glob.name} pattern already has data "
            "(use --resume to continue an existing run)"
        )
    model_name = _model_name(base_config)
    total_invocations = plan.generations * plan.repeats
    stop_line = STOP_LINE_TOTAL * total_invocations / TOTAL_INVOCATIONS
    tokens_so_far = _case_tokens_so_far(case)
    pass_done = _pass_generations_done(case)
    for generation in range(1, plan.generations + 1):
        repeats_done = sum(
            1 for repeat in range(1, plan.repeats + 1)
            if (RUNS / "reports" / f"{case}-g{generation}-r{repeat}.json").exists()
        )
        if repeats_done == plan.repeats and generation in pass_done:
            continue
        for repeat in range(1, plan.repeats + 1):
            row = run_one(plan, generation, repeat, base_config, provider)
            if not row.get("_skipped"):
                tokens_so_far += int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0)
            done = len(list((RUNS / "reports").glob(f"{case}-g*-r*.json")))
            remaining = total_invocations - done
            remaining_passes = plan.generations - generation + 1
            snapshot = spend_snapshot(done, remaining, remaining_passes, tokens_so_far)
            (RUNS / f"spend-{case}.json").write_text(
                json.dumps({**snapshot, "stop_line": stop_line}, indent=2) + "\n", encoding="utf-8"
            )
            if snapshot["projected_total"] > stop_line:
                raise BudgetStop(
                    f"{case}: projected phase spend {snapshot['projected_total']} tokens "
                    f"exceeds the stop line {stop_line:.0f} — halting per the plan's budget rule"
                )
        _archive_engine(case, generation)
        if generation in pass_done:
            continue
        is_final = generation == plan.generations
        if is_final and skip_final_pass:
            print(f"[{case} g{generation}] final-block promotion pass skipped by flag", flush=True)
            continue
        entry = promotion_pass(plan, generation, provider or "anthropic", model_name)
        replay = entry["replay_tokens"]
        tokens_so_far += int(replay["input"]) + int(replay["output"])
    print(f"[{case}] complete: {plan.generations} generation blocks", flush=True)


def write_results() -> dict[str, Any]:
    """Per-case x generation results, pooled results, and the analysis summary."""
    record_files: list[Path] = []
    for case_id in CASES:
        record_files.extend(sorted(RUNS.glob(f"rollouts.{case_id}.g*.jsonl")))
    records = load_rollouts(record_files)
    summary = analyze(records)

    per_generation_rows: list[list[str]] = []
    for path in record_files:
        slice_report = analyze(load_rollouts([path]))
        generation = path.stem.rsplit(".g", 1)[1]
        for case_id, case_report in slice_report["cases"].items():
            for arm, arm_report in case_report["arms"].items():
                per_generation_rows.append(
                    [case_id, generation, arm] + [
                        str(arm_report[key]) for key in
                        ("candidates", "accepted", "median_outcome", "mad_floor", "never_worse_rejections")
                    ] + [";".join(str(d) for d in case_report["paired_deltas"])]
                )

    with (RUNS / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow([
            "case_id", "generation", "arm", "candidates", "accepted",
            "median_outcome", "mad_floor", "never_worse_rejections", "paired_deltas_warm_minus_cold",
        ])
        writer.writerows(per_generation_rows)
    with (RUNS / "results-pooled.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
        writer.writerow([
            "case_id", "arm", "candidates", "accepted", "median_outcome",
            "mad_floor", "never_worse_rejections", "paired_deltas_warm_minus_cold",
        ])
        for case_id, case_report in summary["cases"].items():
            for arm, arm_report in case_report["arms"].items():
                writer.writerow([case_id, arm] + [
                    str(arm_report[key]) for key in
                    ("candidates", "accepted", "median_outcome", "mad_floor", "never_worse_rejections")
                ] + [";".join(str(d) for d in case_report["paired_deltas"])])

    analysis = {
        "determinism": summary["determinism"],
        "spend_by_stage": summary["spend_by_stage"],
        "cases": summary["cases"],
        "record_files": [path.name for path in record_files],
    }
    (RUNS / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"determinism": summary["determinism"], "spend_by_stage": summary["spend_by_stage"]}, indent=2))
    return analysis


def main(argv: list[str] | None = None) -> int:
    """Entry point: run generation ladders or just analyze existing records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scripted", "live"), default=None)
    parser.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    parser.add_argument("--base-config", type=Path, default=RUNS / "compounding.yaml")
    parser.add_argument(
        "--out",
        default="compounding",
        help="runs/ subdirectory for output; dry runs use 'compounding-dry' "
        "so scripted evidence never mixes with live records",
    )
    parser.add_argument("--resume", action="store_true", help="skip completed invocations")
    parser.add_argument("--skip-final-pass", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)

    global RUNS
    RUNS = REPO / "runs" / args.out

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "reports").mkdir(exist_ok=True)
    (RUNS / "decided").mkdir(exist_ok=True)

    if args.analyze_only:
        write_results()
        return 0
    if args.mode is None:
        parser.error("--mode is required unless --analyze-only")
    if args.mode == "live" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("live mode requires ANTHROPIC_API_KEY in the environment")

    selected = list(CASES.values())
    if args.cases:
        names = [name.strip() for name in args.cases.split(",")]
        unknown = [name for name in names if name not in CASES]
        if unknown:
            parser.error(f"unknown case(s): {unknown}")
        selected = [CASES[name] for name in names]

    for plan in selected:
        run_case(plan, args.base_config, args.mode, args.resume, args.skip_final_pass)
    write_results()
    return 0


if __name__ == "__main__":
    sys.exit(main())
