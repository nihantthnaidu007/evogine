"""Phase-2 cross-case benchmark driver.

Runs every benchmark case x REPEATS paired repeats as one both-arms CLI
invocation each: omitting ``--arm`` keeps cold and warm in one process with
a shared run_id, which is what makes the arms seed-paired under learn_0012;
each invocation auto-generates its own run_id, so repeats are independent.

Isolation: every invocation runs in its own scratch directory (fresh
``runs_dir``), so parallel workers never append to one file and a crashed
attempt cannot pollute the dataset. On success the invocation's records
append to the per-case dataset file ``runs/crosscase/rollouts.<case>.jsonl``
and the scratch is removed; on failure the partial file is archived as
``rollouts.preflight.<case>-r<repeat>.jsonl``.

Usage:
    python tools/run_crosscase.py --cases audit-ring-drop,billing-day-offset \
        [--repeats 4] [--config runs/crosscase/crosscase.yaml] [--provider scripted]

``--provider scripted`` swaps the model provider for free dry-run/driver
validation; live runs use the config's provider (anthropic) untouched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs" / "crosscase"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="comma-separated case ids")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--config", default=str(RUNS / "crosscase.yaml"), help="base config"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="override model.provider (e.g. scripted for a free dry run)",
    )
    return parser.parse_args()


def invocation_config(base: Path, provider_override: str | None) -> str:
    text = base.read_text()
    if provider_override:
        lines = []
        for line in text.splitlines():
            if line.strip() == "provider: anthropic" or line.strip() == "provider: scripted":
                lines.append(f"  provider: {provider_override}")
            else:
                lines.append(line)
        text = "\n".join(lines) + "\n"
    # runs_dir is resolved relative to the driver's cwd; pin it to the
    # per-invocation 'rollout' dir so the produced file lands where the
    # driver collects it, regardless of the base config's runs_dir.
    lines = [
        "runs_dir: rollout" if line.lstrip().startswith("runs_dir:") else line
        for line in text.splitlines()
    ]
    text = "\n".join(lines) + "\n"
    if "runs_dir: rollout" not in text:
        raise ValueError(f"config {base} has no runs_dir key")
    return text


def run_case(case: str, repeats: int, base_config: Path, provider: str | None) -> None:
    case_file = RUNS / f"rollouts.{case}.jsonl"
    if case_file.exists():
        raise SystemExit(
            f"refusing to run: {case_file} already exists "
            "(delete it deliberately to re-run the case)"
        )
    for repeat in range(1, repeats + 1):
        scratch = RUNS / f"scratch-{case}-r{repeat}"
        shutil.rmtree(scratch, ignore_errors=True)
        (scratch / "rollout").mkdir(parents=True)
        cfg_path = scratch / "config.yaml"
        cfg_path.write_text(invocation_config(base_config, provider))

        env = dict(os.environ)
        # The CLI reads ANTHROPIC_API_KEY; tmux workers export it from the
        # injected SECRET_ANTHROPIC_API_KEY before invoking this driver.
        # The console script (not `python -m evogine.cli`, which imports
        # without a __main__ guard and exits 0 as a silent no-op).
        exe = shutil.which("evogine") or str(Path(sys.executable).parent / "evogine")
        cmd = [
            exe,
            "run",
            "--case",
            case,
            "--config",
            str(cfg_path),
            "--allow-local-live",
        ]
        print(f"[{case} r{repeat}] running:", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=scratch, env=env, check=False)

        produced = scratch / "rollout" / "rollouts.jsonl"
        if result.returncode == 0 and produced.exists():
            with produced.open() as src, case_file.open("a") as dst:
                dst.write(src.read())
            print(f"[{case} r{repeat}] ok -> {case_file.name}", flush=True)
            shutil.rmtree(scratch)
        else:
            archive = RUNS / f"rollouts.preflight.{case}-r{repeat}.jsonl"
            if produced.exists():
                shutil.move(str(produced), str(archive))
            print(
                f"[{case} r{repeat}] FAILED rc={result.returncode} "
                f"(partial records archived to {archive.name})",
                flush=True,
            )
            raise SystemExit(result.returncode or 1)


def main() -> None:
    args = parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    base_config = Path(args.config)
    for case in args.cases.split(","):
        case = case.strip()
        if case:
            run_case(case, args.repeats, base_config, args.provider)
    print("driver complete:", args.cases, flush=True)


if __name__ == "__main__":
    main()
