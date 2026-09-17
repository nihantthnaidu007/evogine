"""Console entry point for the ``evogine`` command.

    evogine run --case <id> [--arm warm|cold] [--config <path>]
    evogine case list
    evogine case inspect <id>

``run`` wires one benchmark case through the whole stack — benchmark task,
swarm loop (:mod:`evogine.swarm.runner`), the sandboxed code-gate adapter
(:mod:`evogine.swarm.gates_adapter`), and the git-backed learning engine —
and writes phase0 rollout records under the configured runs directory.  It is
deliberately thin: full benchmark sweeps are the benchmark task's job.

Without a config file the run uses the deterministic scripted model and the
Docker executor.  A YAML config (``--config``) may select:

    runs_dir: runs                    # rollout records root
    engine_root: runs/engine          # git-backed learning store
    executor: docker                  # or: local (trusted subprocess)
    model:
      provider: scripted              # or: anthropic (needs ANTHROPIC_API_KEY)
      name: claude-sonnet-4-5
    generations: 1
    candidates_per_generation: 1
    temperature: 0.0
    retrieval_k: 5

The ``executor: local`` executor runs model-generated code as host processes
with network access — it is a trusted-subprocess convenience, NOT a security
boundary. Pairing it with the live provider (``model.provider: anthropic``)
raises :class:`UnsafeExecutorConfig` by default; pass ``--allow-local-live``
to opt in deliberately (local dev iteration with a real provider). Live runs
without that explicit opt-out belong in ``executor: docker``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evogine import __version__
from evogine.bench.harness import (
    Arm,
    HarnessConfig,
    ModelClient,
    ModelConfig,
    ModelResponse,
    run,
)
from evogine.bench.task import discover_case_ids, load_task
from evogine.engine.store import Engine
from evogine.gates.sandbox import Executor, LocalSubprocessSandbox
from evogine.swarm.gates_adapter import CodeGateAdapter
from evogine.swarm.roles import AnthropicModel
from evogine.swarm.roles import ModelConfig as SwarmModelConfig
from evogine.swarm.runner import serialize_candidate_files

#: The scripted verifier reply: consistent with a gate-rejected no-op repair.
_VERIFIER_REPLY = (
    "VERDICT: fail\nThe scripted smoke run does not review candidates; the gate decides."
)

#: The scripted reflector reply: a minimal, valid learning draft.
_REFLECTOR_REPLY = json.dumps(
    {
        "type": "failure-pattern",
        "title": "Scripted smoke run: the no-op repair did not beat the baseline",
        "confidence": 0.3,
        "body": {
            "signal": "The gate rejected the scripted no-op repair",
            "failure_mode": "Smoke run without a real model client",
            "detection": "CodeGate verdict on the case",
            "fix_strategy": "Run with the anthropic provider for real repairs",
        },
    }
)


@dataclass(frozen=True)
class CliConfig:
    """Validated ``evogine run`` configuration (see module docstring)."""

    runs_dir: Path = Path("runs")
    engine_root: Path = Path("runs/engine")
    executor: str = "docker"
    model_provider: str = "scripted"
    model_name: str = "claude-sonnet-4-5"
    generations: int = 1
    candidates_per_generation: int = 1
    temperature: float = 0.0
    retrieval_k: int = 5


class UnsafeExecutorConfig(ValueError):
    """A run configuration pairs the local executor with a live provider.

    The local executor runs model-generated code as host processes with
    network access; it is explicitly not a security boundary (v1 security
    review, findings H3/M4). Such a pairing is refused unless the caller
    opts in with ``--allow-local-live``.
    """


def _validate_executor_pairing(config: CliConfig, *, allow_local_live: bool) -> None:
    """Refuse ``executor: local`` + a live provider unless explicitly allowed.

    No-op for every safe pairing (docker with anything, local with the
    scripted model); raises :class:`UnsafeExecutorConfig` for the dangerous
    one so the reason is named at the config layer, before anything runs.
    """
    if config.executor != "local" or config.model_provider != "anthropic":
        return
    if allow_local_live:
        return
    raise UnsafeExecutorConfig(
        f"executor:local paired with the live provider {config.model_provider!r} is"
        " refused by default: the local executor runs model-generated code as host"
        " processes with network access, and local execution is not a security"
        " boundary. Use executor:docker for live runs, or pass --allow-local-live"
        " to accept this deliberately."
    )


def load_config(path: Path) -> CliConfig:
    """Load and validate a run config; unknown keys are rejected, not ignored."""
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"config {path} must be a mapping")
    known = {
        "runs_dir",
        "engine_root",
        "executor",
        "model",
        "generations",
        "candidates_per_generation",
        "temperature",
        "retrieval_k",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"config {path} has unknown keys: {', '.join(unknown)}")
    model = raw.get("model") or {}
    if not isinstance(model, dict) or set(model) - {"provider", "name"}:
        raise ValueError('config model must be a mapping of "provider" and "name"')
    return CliConfig(
        runs_dir=Path(str(raw.get("runs_dir", "runs"))),
        engine_root=Path(str(raw.get("engine_root", "runs/engine"))),
        executor=str(raw.get("executor", "docker")),
        model_provider=str(model.get("provider", "scripted")),
        model_name=str(model.get("name", "claude-sonnet-4-5")),
        generations=int(raw.get("generations", 1)),
        candidates_per_generation=int(raw.get("candidates_per_generation", 1)),
        temperature=float(raw.get("temperature", 0.0)),
        retrieval_k=int(raw.get("retrieval_k", 5)),
    )


class ScriptedModel:
    """Deterministic no-API model for smoke runs; routes by role system prompt.

    The builder re-emits the case's degraded baseline — a no-op repair the
    gate will judge — the verifier files a fail verdict, and the reflector
    drafts a minimal learning.  An unrecognized prompt shape raises rather
    than guessing.
    """

    def __init__(self, builder_reply: str) -> None:
        self._builder_reply = builder_reply

    def generate(
        self, *, messages: list[dict[str, str]], model_config: ModelConfig
    ) -> ModelResponse:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        if system.startswith("You are the repair builder"):
            return ModelResponse(extracted_code=self._builder_reply)
        if system.startswith("You are the independent verifier"):
            return ModelResponse(extracted_code=_VERIFIER_REPLY)
        if system.startswith("You are the reflector"):
            return ModelResponse(extracted_code=_REFLECTOR_REPLY)
        raise RuntimeError("scripted model received an unrecognized role prompt")


class _SwarmToBenchModel:
    """Present the swarm's :class:`AnthropicModel` behind the bench ModelClient."""

    def __init__(self, model: AnthropicModel) -> None:
        self._model = model

    def generate(
        self, *, messages: list[dict[str, str]], model_config: ModelConfig
    ) -> ModelResponse:
        response = self._model.generate(
            messages=messages,
            model_config=SwarmModelConfig(
                model=model_config.model,
                temperature=model_config.temperature,
                thinking=model_config.thinking,
                seed=model_config.seed,
            ),
        )
        return ModelResponse(
            extracted_code=response.text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )


def _build_model(config: CliConfig, baseline_files: dict[str, str]) -> ModelClient:
    """The composition root's model selection for ``run``."""
    if config.model_provider == "scripted":
        # The no-op repair: re-emit the degraded baseline, layout-aware
        # (single: module source; package: {"files": {...}} JSON).
        return ScriptedModel(builder_reply=serialize_candidate_files(baseline_files))
    if config.model_provider == "anthropic":
        return _SwarmToBenchModel(AnthropicModel())
    raise ValueError(
        f"unknown model provider {config.model_provider!r} (supported: scripted, anthropic)"
    )


def _executor(name: str) -> Executor | None:
    """Map the config's executor to the gate's executor (None = Docker)."""
    if name == "docker":
        return None
    if name == "local":
        return LocalSubprocessSandbox()
    raise ValueError(f"unknown executor {name!r} (supported: docker, local)")


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one case through the wired loop and print the JSON report."""
    config = load_config(Path(args.config)) if args.config else CliConfig()
    _validate_executor_pairing(config, allow_local_live=bool(args.allow_local_live))
    task = load_task(args.case)
    arms: tuple[Arm, ...] = (args.arm,) if args.arm else ("cold", "warm")

    # The warm arm retrieves through the real engine; cold-only runs open no
    # store at all.
    engine = Engine(config.engine_root, actor="evogine-cli") if "warm" in arms else None
    harness_config = HarnessConfig(
        builder=_build_model(config, dict(task.baseline_files)),
        model=config.model_name,
        generations=config.generations,
        candidates_per_generation=config.candidates_per_generation,
        temperature=config.temperature,
        retrieval_k=config.retrieval_k,
        arms=arms,
        runs_dir=config.runs_dir,
    )
    gate = CodeGateAdapter(executor=_executor(config.executor))
    report = run(task, engine, harness_config, gate)
    print(json.dumps(report, indent=2, default=str))
    return 0


def _cmd_case_list(args: argparse.Namespace) -> int:
    """Print the benchmark case ids, one per line."""
    for case_id in discover_case_ids():
        print(case_id)
    return 0


def _cmd_case_inspect(args: argparse.Namespace) -> int:
    """Print a human-readable summary of one benchmark case."""
    task = load_task(args.case_id)
    lines = [
        f"case: {task.case_id}",
        f"decay_axis: {task.decay_axis}",
        f"difficulty: {task.difficulty}",
        f"layout: {task.layout}",
        f"entry: {task.properties.get('entry')}",
        f"suite: {task.suite_path}",
        f"baseline files: {', '.join(sorted(task.baseline_files))}",
        f"description: {task.description}",
    ]
    for prop in task.properties.get("properties", []):
        lines.append(
            f"property: {prop.get('kind')} id={prop.get('id')} rows={len(prop.get('rows', []))}"
        )
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``evogine`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="evogine",
        description="Self-evolving agentic engine: run one benchmark case through the real loop.",
    )
    parser.add_argument("--version", action="version", version=f"evogine {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run one benchmark case cold and/or warm")
    run_parser.add_argument("--case", required=True, help="benchmark case id")
    run_parser.add_argument(
        "--arm", choices=["warm", "cold"], default=None, help="single arm (default: both)"
    )
    run_parser.add_argument("--config", default=None, help="YAML config path")
    run_parser.add_argument(
        "--allow-local-live",
        action="store_true",
        help=(
            "allow executor:local with a live provider (anthropic); refused by "
            "default because local execution is not a security boundary"
        ),
    )
    run_parser.set_defaults(func=_cmd_run)

    case_parser = sub.add_parser("case", help="inspect the benchmark case corpus")
    case_sub = case_parser.add_subparsers(dest="case_command", required=True)
    list_parser = case_sub.add_parser("list", help="list case ids")
    list_parser.set_defaults(func=_cmd_case_list)
    inspect_parser = case_sub.add_parser("inspect", help="inspect one case")
    inspect_parser.add_argument("case_id", help="benchmark case id")
    inspect_parser.set_defaults(func=_cmd_case_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``evogine`` CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Any = args.func
    return int(func(args))
