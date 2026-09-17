"""Bootstrap tests: the 12 real seed records load, commit, and retrieve."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from evogine.engine.bootstrap import load_seeds
from evogine.engine.store import Engine

SEEDS_DIR = Path(__file__).resolve().parents[1] / "src" / "evogine" / "seeds"


def git_subjects(engine: Engine) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(engine.data_dir), "log", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.splitlines()


def test_seed_pack_ships_exactly_12_records() -> None:
    files = sorted(SEEDS_DIR.glob("*.json"))
    assert len(files) == 12
    ids = {json.loads(f.read_text(encoding="utf-8"))["id"] for f in files}
    assert ids == {f"learn_{i:04d}" for i in range(1, 13)}


def test_bootstrap_loads_all_12_seeds(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    loaded = load_seeds(engine, SEEDS_DIR)
    assert len(loaded) == 12
    assert all(record.status == "promoted" for record in loaded)
    assert {record.id for record in loaded} == {f"learn_{i:04d}" for i in range(1, 13)}
    for record in loaded:
        assert record.promoted_at is not None


def test_every_seed_carries_provenance(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    for record in load_seeds(engine, SEEDS_DIR):
        assert record.provenance.proposed_by, f"{record.id} lacks provenance"
        assert record.provenance.origin, f"{record.id} lacks origin"
        assert record.provenance.note, f"{record.id} lacks provenance note"


def test_retrieve_returns_all_seeds_after_bootstrap(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    matched = engine.retrieve("failure pattern playbook check", k=20)
    fallback = engine.retrieve("a task with no tag overlap at all", k=20)
    assert {r.id for r in matched} == {f"learn_{i:04d}" for i in range(1, 13)}
    assert {r.id for r in fallback} == {f"learn_{i:04d}" for i in range(1, 13)}
    confidences = [r.confidence for r in fallback]
    assert confidences == sorted(confidences, reverse=True)


def test_retrieve_respects_k_over_seeds(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    assert len(engine.retrieve("failure pattern playbook check", k=5)) == 5


def test_engine_construction_loads_promoted_seeds_into_retrieval(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    reopened = Engine(tmp_path / "engine")
    results = reopened.retrieve("failure pattern playbook check", k=20)
    assert {r.id for r in results} == {f"learn_{i:04d}" for i in range(1, 13)}


def test_bootstrap_is_a_single_committed_mutation(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    proc = subprocess.run(
        ["git", "-C", str(engine.data_dir), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "1"
    assert git_subjects(engine)[0] == "bootstrap: load 12 promoted seed record(s)"
    loads = engine.audit.find("bootstrap-load")
    assert {entry.learning_id for entry in loads} == {f"learn_{i:04d}" for i in range(1, 13)}


def test_bootstrap_commits_seed_files_into_the_store(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    stored = sorted((Path(engine.data_dir) / "records").glob("*.json"))
    assert len(stored) == 12
    first = json.loads(stored[0].read_text(encoding="utf-8"))
    assert first["status"] == "promoted"
    assert first["provenance"]["proposed_by"]


def test_bootstrap_refuses_non_promoted_seeds(tmp_path: Path) -> None:
    fake_seed = tmp_path / "fake-seeds" / "learn_bad.json"
    fake_seed.parent.mkdir(parents=True)
    fake_seed.write_text(
        json.dumps({
            "id": "learn_bad",
            "type": "playbook",
            "title": "not promoted",
            "body": {"signal": "", "failure_mode": "", "detection": "", "fix_strategy": ""},
            "status": "ephemeral",
            "confidence": 0.0,
            "provenance": {"proposed_by": "x", "origin": "x", "source": "", "note": ""},
            "counterfactual": {"replay_run_ids": [], "verdict_change": None, "evidence": None},
            "created": "2026-09-17",
            "promoted_at": None,
        }),
        encoding="utf-8",
    )
    engine = Engine(tmp_path / "engine")
    with pytest.raises(ValueError, match="promoted"):
        load_seeds(engine, fake_seed.parent)


def test_bootstrap_refuses_invalid_seed_files(tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad-seeds"
    bad_dir.mkdir()
    (bad_dir / "learn_broken.json").write_text("{not a record", encoding="utf-8")
    engine = Engine(tmp_path / "engine")
    with pytest.raises(ValueError, match="learn_broken"):
        load_seeds(engine, bad_dir)


def test_bootstrap_refuses_duplicate_load(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "engine")
    load_seeds(engine, SEEDS_DIR)
    with pytest.raises(ValueError, match="already exists"):
        load_seeds(engine, SEEDS_DIR)


def test_bootstrap_empty_dir_is_a_no_op(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    engine = Engine(tmp_path / "engine")
    assert load_seeds(engine, empty) == []
