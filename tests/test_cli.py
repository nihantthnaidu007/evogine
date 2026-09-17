"""CLI smoke tests: ``evogine run`` / ``case list`` / ``case inspect``.

The run tests execute a real benchmark case through the whole wired stack —
bench task, swarm loop, sandboxed code-gate adapter, engine — using the
deterministic scripted model and the local sandbox executor: no Docker, no
API calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture

from evogine.cli import main

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
