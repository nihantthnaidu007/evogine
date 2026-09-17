"""Relaunch specific (case, repeat) phase-2 invocations after a driver stall.

The phase-2 run stalled on 2026-09-17 when the Anthropic API stopped
returning bodies: the three in-flight CLI processes held established TLS
connections and appended no records for 20+ minutes. The driver's guards
(refuse existing per-case datasets, always start at repeat 1) are correct
for a clean run but block a surgical retry, so this utility reuses the
driver's ``invocation_config`` and append discipline to run exactly the
listed pairs — completed repeats are never re-executed, failures are
archived under the same ``rollouts.preflight.*`` convention.

Usage::

    python tools/rerun_invocations.py  # the four stalled pairs, in order
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_crosscase import RUNS, invocation_config

#: The invocations lost to the 2026-09-17 API stall, in driver order.
PAIRS: list[tuple[str, int]] = [
    ("concurrency-account-ledger", 4),
    ("partial-batch-write", 3),
    ("partial-batch-write", 4),
    ("unicode-nfkc-key", 4),
]


def run_one(case: str, repeat: int, base_config: Path) -> int:
    """Run one (case, repeat) both-arms invocation; append on success."""
    scratch = RUNS / f"scratch-{case}-r{repeat}"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "rollout").mkdir(parents=True)
    cfg_path = scratch / "config.yaml"
    cfg_path.write_text(invocation_config(base_config, None))

    venv_exe = Path(sys.executable).parent / "evogine"
    if not venv_exe.exists():
        raise SystemExit("no evogine console script next to this python")
    cmd = [
        str(venv_exe),
        "run",
        "--case",
        case,
        "--config",
        str(cfg_path),
        "--allow-local-live",
    ]
    print(f"[{case} r{repeat}] running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=scratch, env=dict(os.environ), check=False)

    produced = scratch / "rollout" / "rollouts.jsonl"
    case_file = RUNS / f"rollouts.{case}.jsonl"
    if result.returncode == 0 and produced.exists():
        with produced.open() as src, case_file.open("a") as dst:
            dst.write(src.read())
        print(f"[{case} r{repeat}] ok -> {case_file.name}", flush=True)
        shutil.rmtree(scratch)
        return 0
    archive = RUNS / f"rollouts.preflight.{case}-r{repeat}.jsonl"
    if produced.exists():
        shutil.move(str(produced), str(archive))
    print(
        f"[{case} r{repeat}] FAILED rc={result.returncode} "
        f"(partial records archived to {archive.name})",
        flush=True,
    )
    return result.returncode or 1


def main() -> None:
    """Run every listed pair sequentially; stop on the first failure."""
    base_config = RUNS / "crosscase.yaml"
    for case, repeat in PAIRS:
        code = run_one(case, repeat, base_config)
        if code != 0:
            raise SystemExit(code)
    print("relaunch complete", flush=True)


if __name__ == "__main__":
    main()
