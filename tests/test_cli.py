"""CLI smoke tests: ``evogine run`` / ``case list`` / ``case inspect``.

The run tests execute a real benchmark case through the whole wired stack —
bench task, swarm loop, sandboxed code-gate adapter, engine — using the
deterministic scripted model and the local sandbox executor: no Docker, no
API calls. The pairing tests cover the config-layer refusal of
``executor: local`` + the live ``anthropic`` provider and its explicit
``--allow-local-live`` opt-out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from evogine import cli as cli_module
from evogine.cli import CliConfig, UnsafeExecutorConfig, _validate_executor_pairing, main

CASE_ID = "counter-overflow-wrap"


def _write_config(tmp_path: Path) -> Path:
    """A minimal local-executor, scripted-provider config rooted in tmp_path."""
    config = tmp_path / "evogine.yaml"
    config.write_text(
        "\n".join(
            [
                f"runs_dir: {tmp_path / 'runs'}",
                f"engine_root: {tmp_path / 'engine'}",
                "executor: local",
                "model:",
                "  provider: scripted",
                "  name: scripted-model",
                "generations: 1",
                "candidates_per_generation: 1",
            ]
        )
        + "\n"
    )
    return config


def test_case_list_prints_the_corpus(capsys: CaptureFixture[str]) -> None:
    assert main(["case", "list"]) == 0
    out = capsys.readouterr().out
    assert "counter-overflow-wrap" in out
    assert "tz-deadline-rollover" in out


def test_case_inspect_prints_case_fields(capsys: CaptureFixture[str]) -> None:
    assert main(["case", "inspect", CASE_ID]) == 0
    out = capsys.readouterr().out
    assert f"case: {CASE_ID}" in out
    assert "decay_axis: overflow" in out
    assert "difficulty: medium" in out


def test_run_cold_executes_the_case_and_writes_rollouts(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    config = _write_config(tmp_path)
    assert main(["run", "--case", CASE_ID, "--arm", "cold", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["case_id"] == CASE_ID
    assert set(report["arms"]) == {"cold"}
    rollouts = Path(str(report["rollouts_path"]))
    assert rollouts.exists()
    # Phase0 rollout records landed, stages intact.
    records = [json.loads(line) for line in rollouts.read_text().splitlines()]
    assert any(record["stage"] == "baseline" for record in records)
    assert any(record["stage"] == "candidate" for record in records)


def test_run_warm_opens_the_engine(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    config = _write_config(tmp_path)
    assert main(["run", "--case", CASE_ID, "--arm", "warm", "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    # An empty store retrieves nothing — honestly empty, not fabricated.
    assert report["arms"]["warm"]["retrieved_records"] == 0
    assert (tmp_path / "engine").exists()


def test_run_rejects_unknown_config_keys(tmp_path: Path) -> None:
    config = tmp_path / "bogus.yaml"
    config.write_text(f"runs_dir: {tmp_path / 'runs'}\nbogus_key: 1\n")
    with pytest.raises(ValueError, match="unknown keys"):
        main(["run", "--case", CASE_ID, "--arm", "cold", "--config", str(config)])


def test_run_rejects_unknown_model_provider(tmp_path: Path) -> None:
    config = tmp_path / "magic.yaml"
    config.write_text(
        f"runs_dir: {tmp_path / 'runs'}\nexecutor: local\nmodel:\n  provider: magic\n"
    )
    with pytest.raises(ValueError, match="unknown model provider"):
        main(["run", "--case", CASE_ID, "--arm", "cold", "--config", str(config)])


def _write_local_live_config(tmp_path: Path) -> Path:
    """The dangerous pairing: local executor plus the live anthropic provider."""
    config = tmp_path / "local-live.yaml"
    config.write_text(
        "\n".join(
            [
                f"runs_dir: {tmp_path / 'runs'}",
                f"engine_root: {tmp_path / 'engine'}",
                "executor: local",
                "model:",
                "  provider: anthropic",
                "  name: claude-sonnet-4-5",
                "generations: 1",
                "candidates_per_generation: 1",
            ]
        )
        + "\n"
    )
    return config


def test_run_refuses_local_executor_with_live_provider_by_default(
    tmp_path: Path,
) -> None:
    """The dangerous pairing is refused at the config layer, before anything runs."""
    config = _write_local_live_config(tmp_path)
    with pytest.raises(UnsafeExecutorConfig) as excinfo:
        main(["run", "--case", CASE_ID, "--arm", "cold", "--config", str(config)])
    message = str(excinfo.value)
    assert "not a security boundary" in message
    assert "executor:docker" in message
    assert "--allow-local-live" in message


def test_run_allows_local_live_pairing_with_explicit_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--allow-local-live opts in deliberately; the run proceeds past validation."""
    config = _write_local_live_config(tmp_path)
    # Stub the harness so the opt-in pairing is exercised without API calls:
    # reaching the stub proves validation accepted the pairing.
    calls: list[tuple[object, ...]] = []

    def _stub_run(*args: object, **kwargs: object) -> dict[str, str]:
        calls.append(args)
        return {"stub": "report"}

    monkeypatch.setattr(cli_module, "run", _stub_run)
    assert (
        main(
            [
                "run",
                "--case",
                CASE_ID,
                "--arm",
                "cold",
                "--config",
                str(config),
                "--allow-local-live",
            ]
        )
        == 0
    )
    assert len(calls) == 1


def test_docker_executor_with_live_provider_is_unaffected() -> None:
    """Docker is the executor for live runs — no refusal, no opt-out needed."""
    config = CliConfig(executor="docker", model_provider="anthropic")
    # Passing without raising is the assertion.
    _validate_executor_pairing(config, allow_local_live=False)


def test_local_executor_with_scripted_provider_is_unaffected() -> None:
    """The scripted model is not live: the local-executor pairing stays allowed."""
    config = CliConfig(executor="local", model_provider="scripted")
    # Passing without raising is the assertion.
    _validate_executor_pairing(config, allow_local_live=False)
