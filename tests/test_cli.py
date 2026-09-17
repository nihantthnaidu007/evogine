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
from evogine.bench.harness import ModelConfig
from evogine.cli import (
    CliConfig,
    ScriptedModel,
    UnsafeExecutorConfig,
    _build_model,
    _executor,
    _validate_executor_pairing,
    load_config,
    main,
)

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


# ---------------------------------------------------------------------------
# Fail-closed validation paths: bad input raises, never silently coerced.
# ---------------------------------------------------------------------------


def test_load_config_rejects_non_mapping_document(tmp_path: Path) -> None:
    """A YAML scalar or list is not a config: refuse it instead of guessing keys."""
    config = tmp_path / "scalar.yaml"
    config.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_config(config)


@pytest.mark.parametrize("unknown_key", ["typo_key", "model_name", "generations_"])
def test_load_config_rejects_unknown_keys(tmp_path: Path, unknown_key: str) -> None:
    """Unknown keys are rejected, not ignored — a silently dropped setting lies."""
    config = tmp_path / "unknown-key.yaml"
    config.write_text(f"runs_dir: runs\n{unknown_key}: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys") as excinfo:
        load_config(config)
    assert unknown_key in str(excinfo.value)


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("model: claude-x", id="model-is-a-scalar"),
        pytest.param(
            "model:\n  provider: scripted\n  surprise: 1", id="model-has-unknown-subkey"
        ),
    ],
)
def test_load_config_rejects_malformed_model_block(tmp_path: Path, raw: str) -> None:
    """The model block must be exactly {provider, name}: anything else is refused."""
    config = tmp_path / "model.yaml"
    config.write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match='must be a mapping of "provider" and "name"'):
        load_config(config)


_SCRIPTED_CONFIG = ModelConfig(
    model="scripted-model", temperature=0.0, thinking=False, seed="test-seed"
)


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="no-system-prompt"),
        pytest.param(
            [{"role": "system", "content": "You are the mysterious oracle"}],
            id="unrecognized-role",
        ),
        pytest.param(
            [{"role": "user", "content": "You are the repair builder"}],
            id="role-text-outside-system",
        ),
    ],
)
def test_scripted_model_refuses_unrecognized_role_prompts(
    messages: list[dict[str, str]],
) -> None:
    """The scripted model routes by role system prompt; anything else fails closed."""
    model = ScriptedModel(builder_reply="x")
    with pytest.raises(RuntimeError, match="unrecognized role"):
        model.generate(messages=messages, model_config=_SCRIPTED_CONFIG)


@pytest.mark.parametrize("provider", ["magic", "openai", "", "Scripted"])
def test_build_model_rejects_unknown_providers(provider: str) -> None:
    """Only scripted and anthropic are supported; anything else is refused."""
    config = CliConfig(model_provider=provider)
    with pytest.raises(ValueError, match="unknown model provider"):
        _build_model(config, baseline_files={})


@pytest.mark.parametrize("name", ["magic", "ssh", "", "Local"])
def test_executor_rejects_unknown_names(name: str) -> None:
    """Only docker and local are supported; anything else is refused."""
    with pytest.raises(ValueError, match="unknown executor"):
        _executor(name)
